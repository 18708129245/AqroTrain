// ============================================================================
//  ProjectManager.cpp —— 工程 / 模块 / 图像池 / 样本状态 / 标注读写
// ============================================================================
#include "project/ProjectManager.h"

#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>
#include <QSet>

// Win32 / CRT / 外壳 API：用于把「创建目录」拆成多种策略并取回真实错误码。
// 只在 .cpp 里引入，避免 windows.h 的宏污染其它头文件。
#ifndef NOMINMAX
#  define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <direct.h>
#include <shlobj.h>

#pragma comment(lib, "shell32.lib")

#include <opencv2/imgproc.hpp>

#include "core/ImageIO.h"
#include "core/JsonStore.h"
#include "core/Metrics.h"

namespace aq {

namespace {

/// 目录的权限摘要（rwx 形式），用于判断是否被只读属性 / ACL 限制
QString permString(const QString &path)
{
    const QFile::Permissions p = QFileInfo(path).permissions();
    QString s;
    s += (p & QFileDevice::ReadOwner)  ? QLatin1Char('r') : QLatin1Char('-');
    s += (p & QFileDevice::WriteOwner) ? QLatin1Char('w') : QLatin1Char('-');
    s += (p & QFileDevice::ExeOwner)   ? QLatin1Char('x') : QLatin1Char('-');
    return s;
}

/// 是否带 Windows「只读」属性。
/// 只对**已存在**的路径判断——不存在的路径权限位全为空，会被误判成只读。
bool looksReadOnlyImpl(const QString &path)
{
    const QFileInfo fi(path);
    if (!fi.exists())
        return false;
    const QFile::Permissions p = fi.permissions();
    return !(p & QFileDevice::WriteUser) && !(p & QFileDevice::WriteOwner);
}

/// 把 Qt 路径转成 Win32 宽字符串路径（'/' → '\'）
QString toWin32Path(const QString &path)
{
    QString p = QDir::toNativeSeparators(path);
    return p;
}

/// 需要长路径前缀时给出 `\\?\` 形式。
/// Windows 的 MAX_PATH 限制是 260；超过后必须用 `\\?\` 前缀，
/// 且该形式要求绝对路径、反斜杠分隔。这里留出余量，从 250 字符起就启用。
QString withLongPathPrefix(const QString &nativePath)
{
    if (nativePath.startsWith(QLatin1String("\\\\?\\")))
        return nativePath;
    if (nativePath.length() < 250)
        return nativePath;
    if (nativePath.startsWith(QLatin1String("\\\\")))
        return QStringLiteral("\\\\?\\UNC\\") + nativePath.mid(2);
    return QStringLiteral("\\\\?\\") + nativePath;
}

} // namespace

// ===========================================================================
//  静态工具
// ===========================================================================
QString ProjectManager::permissionString(const QString &path)
{
    return permString(path);
}

QString ProjectManager::describeWin32Error(quint32 code)
{
    switch (code) {
    case 0:   return QStringLiteral("无错误");
    case 2:   return QStringLiteral("找不到指定的文件（ERROR_FILE_NOT_FOUND）");
    case 3:   return QStringLiteral("找不到指定的路径（ERROR_PATH_NOT_FOUND）：上级目录不存在");
    case 5:   return QStringLiteral("拒绝访问（ERROR_ACCESS_DENIED）："
                                    "被 ACL 或安全软件/组策略拦截");
    case 32:  return QStringLiteral("文件被另一进程占用（ERROR_SHARING_VIOLATION）");
    case 80:  return QStringLiteral("文件已存在（ERROR_FILE_EXISTS）");
    case 123: return QStringLiteral("文件名、目录名或卷标语法不正确（ERROR_INVALID_NAME）："
                                    "名称里可能含非法字符或结尾是空格/点");
    case 145: return QStringLiteral("目录不是空的（ERROR_DIR_NOT_EMPTY）");
    case 183: return QStringLiteral("文件已存在（ERROR_ALREADY_EXISTS）：目录已建好");
    case 206: return QStringLiteral("文件名或扩展名太长（ERROR_FILENAME_EXCED_RANGE）：超过 MAX_PATH");
    case 267: return QStringLiteral("目录名无效（ERROR_DIRECTORY）");
    case 1223: return QStringLiteral("操作被用户取消（ERROR_CANCELLED）");
    default:  return QStringLiteral("Win32 错误码 %1").arg(code);
    }
}

bool ProjectManager::createOneLevel(const QString &path, QVector<DirCreateAttempt> *attempts,
                                   QString *err)
{
    if (attempts)
        attempts->clear();

    // 已经存在（含并发创建）直接算成功
    const QFileInfo before(path);
    if (before.exists())
        return before.isDir();

    const QString native = toWin32Path(path);
    const QString leafDir = QFileInfo(native).absolutePath();
    const QString leafName = QFileInfo(native).fileName();
    if (leafName.isEmpty()) {
        if (err) *err = QStringLiteral("路径的最后一段为空：%1").arg(path);
        return false;
    }

    auto record = [attempts](const QString &strategy, bool ok, quint32 code) {
        if (!attempts)
            return;
        DirCreateAttempt a;
        a.strategy   = strategy;
        a.ok         = ok;
        a.win32Error = code;
        a.message    = ok ? QString() : ProjectManager::describeWin32Error(code);
        attempts->append(a);
    };

    auto succeeded = [&path]() {
        const QFileInfo after(path);
        return after.exists() && after.isDir();
    };

    QString lastError;

    // --- 策略 1：Win32 CreateDirectoryW（含长路径前缀） -------------------
    {
        const QString target = withLongPathPrefix(leafDir) + QLatin1Char('\\') + leafName;
        ::SetLastError(0);
        const BOOL ok = ::CreateDirectoryW(reinterpret_cast<const wchar_t *>(target.utf16()), nullptr);
        const quint32 code = ok ? 0u : quint32(::GetLastError());
        const bool good = (ok != FALSE) || code == 183u || succeeded();
        record(QStringLiteral("Win32 CreateDirectoryW%1")
                   .arg(target.startsWith(QLatin1String("\\\\?\\"))
                            ? QStringLiteral("（长路径 \\\\?\\）")
                            : QString()),
               good, good ? 0u : code);
        if (good)
            return true;
        lastError = QStringLiteral("CreateDirectoryW → %1").arg(describeWin32Error(code));
    }

    // --- 策略 2：QDir::mkdir（Qt 自己的实现，可能与上面的错误路径不同）---
    {
        const QString dir = QFileInfo(native).absolutePath();
        const bool ok = QDir(dir).mkdir(leafName);
        record(QStringLiteral("Qt QDir::mkdir"), ok || succeeded(), ok ? 0u : 5u);
        if (ok || succeeded())
            return true;
        lastError += QStringLiteral("；QDir::mkdir → 失败");
    }

    // --- 策略 3：CRT _wmkdir -------------------------------------------
    {
        const QString target = withLongPathPrefix(toWin32Path(QDir::cleanPath(path)));
        ::_wmkdir(reinterpret_cast<const wchar_t *>(target.utf16()));
        const bool ok = succeeded();
        record(QStringLiteral("CRT _wmkdir"), ok, ok ? 0u : quint32(::GetLastError()));
        if (ok)
            return true;
        lastError += QStringLiteral("；_wmkdir → 失败");
    }

    // --- 策略 4：外壳 API SHCreateDirectoryExW（自带重试语义）-----------
    {
        const QString target = toWin32Path(QDir::cleanPath(path));
        const int rc = ::SHCreateDirectoryExW(nullptr,
                                             reinterpret_cast<const wchar_t *>(target.utf16()),
                                             nullptr);
        // 返回 ERROR_SUCCESS / ERROR_ALREADY_EXISTS / ERROR_FILE_EXISTS 都算成功
        const bool ok = (rc == ERROR_SUCCESS || rc == ERROR_ALREADY_EXISTS || rc == ERROR_FILE_EXISTS)
                        || succeeded();
        record(QStringLiteral("外壳 SHCreateDirectoryExW"), ok, ok ? 0u : quint32(rc));
        if (ok)
            return true;
        lastError += QStringLiteral("；SHCreateDirectoryExW → %1").arg(describeWin32Error(quint32(rc)));
    }

    // --- 策略 5：QDir::mkpath（最后手段：允许它把中间层也建出来）--------
    {
        const bool ok = QDir().mkpath(QDir::cleanPath(path));
        record(QStringLiteral("Qt QDir::mkpath"), ok || succeeded(), ok ? 0u : 5u);
        if (ok || succeeded())
            return true;
        lastError += QStringLiteral("；QDir::mkpath → 失败");
    }

    if (err)
        *err = lastError;
    return false;
}

bool ProjectManager::removeDirRecursive(const QString &path)
{
    if (path.trimmed().isEmpty())
        return false;

    const QString native = withLongPathPrefix(QDir::toNativeSeparators(QDir::cleanPath(path)));
    const wchar_t *w = reinterpret_cast<const wchar_t *>(native.utf16());

    const DWORD attr = ::GetFileAttributesW(w);
    if (attr == INVALID_FILE_ATTRIBUTES)
        return true;   // 不存在即视为已删除

    if (!(attr & FILE_ATTRIBUTE_DIRECTORY)) {
        ::SetFileAttributesW(w, FILE_ATTRIBUTE_NORMAL);
        return ::DeleteFileW(w) != 0;
    }

    // 先递归清空子项
    WIN32_FIND_DATAW fd {};
    const QString pattern = native + QStringLiteral("\\*");
    HANDLE h = ::FindFirstFileW(reinterpret_cast<const wchar_t *>(pattern.utf16()), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            const QString name = QString::fromWCharArray(fd.cFileName);
            if (name == QLatin1String(".") || name == QLatin1String(".."))
                continue;
            const QString child = native + QLatin1Char('\\') + name;
            if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
                removeDirRecursive(child);
            } else {
                const wchar_t *cw = reinterpret_cast<const wchar_t *>(child.utf16());
                // 只读文件直接删会失败，先清掉属性
                ::SetFileAttributesW(cw, FILE_ATTRIBUTE_NORMAL);
                if (!::DeleteFileW(cw)) {
                    // 仍失败则再试一次（有些环境需要先去掉只读位再删）
                    ::SetFileAttributesW(cw, FILE_ATTRIBUTE_NORMAL);
                    ::DeleteFileW(cw);
                }
            }
        } while (::FindNextFileW(h, &fd));
        ::FindClose(h);
    }

    ::SetFileAttributesW(w, FILE_ATTRIBUTE_NORMAL);
    if (::RemoveDirectoryW(w))
        return true;

    // 目录本身是只读属性时 RemoveDirectory 可能失败，清掉属性后再试
    ::SetFileAttributesW(w, FILE_ATTRIBUTE_NORMAL);
    return ::RemoveDirectoryW(w) != 0;
}

bool ProjectManager::probeWritable(const QString &dir, QString *err)
{
    const QFileInfo fi(dir);
    if (!fi.exists() || !fi.isDir()) {
        if (err) *err = QStringLiteral("目录不存在：%1").arg(QDir::toNativeSeparators(dir));
        return false;
    }

    // 用「真的写一个临时文件」探测：只读属性 / isWritable() / ACL 的结论
    // 在实测中会互相矛盾，只有真正写入才能反映本进程是否被允许。
    const QString probe = fi.absolutePath() + QLatin1Char('/')
                          + QStringLiteral(".aqrotrain_write_probe_%1.tmp")
                                .arg(quintptr(QCoreApplication::applicationPid()));
    QFile f(probe);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (err) {
            *err = QStringLiteral("无法在 %1 中新建文件（%2）")
                       .arg(QDir::toNativeSeparators(dir), f.errorString());
        }
        return false;
    }
    const bool wrote = f.write("probe", 5) == 5;
    f.close();
    QFile::remove(probe);

    if (!wrote) {
        if (err) *err = QStringLiteral("写入 %1 失败").arg(QDir::toNativeSeparators(probe));
        return false;
    }
    return true;
}

bool ProjectManager::looksReadOnly(const QString &path)
{
    return looksReadOnlyImpl(path);
}

bool ProjectManager::clearReadOnlyAttribute(const QString &dir, QString *err)
{
    const QFileInfo fi(dir);
    if (!fi.exists() || !fi.isDir()) {
        if (err) *err = QStringLiteral("目录不存在：%1").arg(QDir::toNativeSeparators(dir));
        return false;
    }

    QFile::Permissions wanted = fi.permissions();
    wanted |= QFileDevice::ReadOwner | QFileDevice::ReadUser;
    wanted |= QFileDevice::WriteOwner | QFileDevice::WriteUser;
    wanted |= QFileDevice::ExeOwner | QFileDevice::ExeUser;

    if (!QFile::setPermissions(dir, wanted)) {
        if (err) *err = QStringLiteral("无法修改目录属性：%1").arg(QDir::toNativeSeparators(dir));
        return false;
    }
    return true;
}

bool ProjectManager::ensureDir(const QString &path, QString *err,
                               QVector<DirCreateAttempt> *attempts)
{
    const QString clean = QDir::cleanPath(QDir::fromNativeSeparators(path));
    if (clean.isEmpty() || clean == QLatin1String(".")) {
        if (err) *err = QStringLiteral("目录路径为空。");
        return false;
    }

    const QFileInfo target(clean);
    if (target.exists()) {
        if (target.isDir())
            return true;
        if (err) {
            *err = QStringLiteral("该路径已被同名文件占用，无法创建目录：\n%1")
                       .arg(QDir::toNativeSeparators(clean));
        }
        return false;
    }

    // 从盘符起逐层创建：单层创建语义明确，
    // 不会像 mkpath 那样因某一层判定异常就整体失败且无法定位。
    const QStringList parts = clean.split(QLatin1Char('/'), QString::SkipEmptyParts);
    QString cur;
    for (int i = 0; i < parts.size(); ++i) {
        if (i == 0)
            cur = parts.at(0);
        else
            cur += QLatin1Char('/') + parts.at(i);

        // 盘符（C: / F:）本身不需要创建
        if (i == 0 && parts.at(0).endsWith(QLatin1Char(':')))
            continue;

        const QFileInfo fi(cur);
        if (fi.exists()) {
            if (!fi.isDir()) {
                if (err) {
                    *err = QStringLiteral("路径中的「%1」已存在且不是目录。")
                               .arg(QDir::toNativeSeparators(cur));
                }
                return false;
            }
            continue;
        }

        QString levelErr;
        if (createOneLevel(cur, attempts, &levelErr))
            continue;

        // 单层创建失败：拼出可诊断的说明
        const QString parentPath = QFileInfo(cur).absolutePath();
        const QFileInfo parentInfo(parentPath);

        QString reason;
        if (!parentInfo.exists())
            reason = QStringLiteral("上级目录不存在");
        else if (!parentInfo.isDir())
            reason = QStringLiteral("上级路径不是目录");
        else if (looksReadOnlyImpl(parentPath))
            reason = QStringLiteral("上级目录带「只读」属性。Windows 下目录被标记只读时，"
                                    "部分环境会拒绝在其中创建子目录");
        else if (!parentInfo.isWritable())
            reason = QStringLiteral("上级目录没有写权限");

        // 关键判据：能不能在这个上级目录里写文件？
        // 能写文件却建不了目录 → 不是普通权限问题（多为安全软件/策略拦截）；
        // 连文件都写不了 → 目录真的不可写。
        QString probeErr;
        const bool writable = probeWritable(parentPath, &probeErr);

        if (reason.isEmpty()) {
            if (writable)
                reason = QStringLiteral("本进程可以往上级目录写文件，但**无法在其中创建子目录**。"
                                        "这通常不是普通权限问题，而是安全软件／组策略"
                                        "专门拦截了目录创建");
            else
                reason = QStringLiteral("本进程无法往上级目录写入（探测：%1）").arg(probeErr);
        }

        if (err) {
            const QString yes = QStringLiteral("是");
            const QString no  = QStringLiteral("否");
            QString msg;
            msg += QStringLiteral("无法创建目录：%1\n\n").arg(QDir::toNativeSeparators(clean));
            msg += QStringLiteral("失败层级：%1\n").arg(QDir::toNativeSeparators(cur));
            msg += QStringLiteral("原因：%1\n").arg(reason);
            if (!levelErr.isEmpty())
                msg += QStringLiteral("最后一个 API 的返回：%1\n").arg(levelErr);
            msg += QStringLiteral("上级目录：%1\n").arg(QDir::toNativeSeparators(parentPath));
            msg += QStringLiteral("　（存在=%1，是目录=%2，可写=%3，权限=%4，写文件探测=%5）")
                       .arg(parentInfo.exists() ? yes : no,
                            parentInfo.isDir() ? yes : no,
                            parentInfo.isWritable() ? yes : no,
                            permString(parentPath),
                            writable ? QStringLiteral("成功") : QStringLiteral("失败"));
            if (attempts && !attempts->isEmpty()) {
                msg += QStringLiteral("\n各 API 尝试结果：\n");
                for (const DirCreateAttempt &a : *attempts) {
                    msg += QStringLiteral("　· %1：%2%3\n")
                               .arg(a.strategy,
                                    a.ok ? QStringLiteral("成功") : QStringLiteral("失败"),
                                    a.ok ? QString()
                                         : QStringLiteral("（错误码 %1：%2）")
                                               .arg(a.win32Error)
                                               .arg(a.message));
                }
            }
            *err = msg;
        }
        return false;
    }

    return QFileInfo(clean).isDir();
}

// ===========================================================================
//  构造 / 析构
// ===========================================================================
ProjectManager::ProjectManager(QObject *parent) : QObject(parent) {}
ProjectManager::~ProjectManager() = default;

// ===========================================================================
//  工程
// ===========================================================================
bool ProjectManager::createProject(const QString &root, const QString &name,
                                   ImageFormat fmt, QString *err)
{
    if (root.trimmed().isEmpty()) {
        if (err) *err = QStringLiteral("工程路径不能为空。");
        return false;
    }
    const QString clean = QDir::cleanPath(QDir::fromNativeSeparators(root));

    // 允许使用已存在但为空的目录
    QDir dir(clean);
    if (dir.exists()) {
        const QFileInfoList entries =
            dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden);
        if (!entries.isEmpty()) {
            if (err) {
                *err = QStringLiteral("目标目录已存在且非空，请改用新目录或在其中新建工程：\n%1")
                           .arg(QDir::toNativeSeparators(clean));
            }
            return false;
        }
    }

    Project p;
    p.setRoot(clean);
    p.setName(name.isEmpty() ? QFileInfo(clean).fileName() : name);
    p.setImageFormat(fmt);
    p.setCreated(QDateTime::currentDateTime().toString(Qt::ISODate));

    if (!ensureDir(p.root(), err))
        return false;
    if (!ensureDir(p.imagesDir(), err))
        return false;
    if (!ensureDir(p.thumbsDir(), err))
        return false;
    if (!ensureDir(p.modulesDir(), err))
        return false;
    if (!p.save(err))
        return false;

    m_project       = p;
    m_modules.clear();
    m_currentModule = -1;
    m_samples.clear();
    m_images.clear();
    m_open = true;

    emit projectOpened();
    emit modulesChanged();
    emit samplesChanged();
    return true;
}

bool ProjectManager::openProject(const QString &root, QString *err)
{
    const QString clean = QDir::cleanPath(QDir::fromNativeSeparators(root));

    Project p;
    if (!Project::load(clean, &p, err))
        return false;

    if (!ensureDir(p.root(), err))
        return false;
    if (!ensureDir(p.imagesDir(), err))
        return false;
    if (!ensureDir(p.thumbsDir(), err))
        return false;
    if (!ensureDir(p.modulesDir(), err))
        return false;

    m_project = p;
    m_open    = true;

    // 载入全部模块
    m_modules.clear();
    const QVector<Project::ModuleRef> refs = m_project.moduleRefs();
    for (const Project::ModuleRef &r : refs) {
        Module m;
        QString modErr;
        const QString dir = m_project.moduleDir(r.id);
        if (Module::load(dir, &m, &modErr)) {
            m.setRoot(dir);
            m.ensureDefaultClasses();
        } else {
            // module.json 缺失或损坏：按工程记录重建一个最小可用的模块描述
            m = Module();
            m.setId(r.id);
            m.setName(r.name);
            m.setType(r.type);
            m.setOrder(r.order);
            m.setRoot(dir);
            m.ensureDefaultClasses();
        }
        m.setOrder(r.order);
        m.trainRef().task       = m.type();
        m.trainRef().datasetRoot = dir;
        m.trainRef().outputDir   = m.modelsDir();
        m_modules.append(m);
        createModuleDirs(m, nullptr);
    }

    rebuildImageList();

    m_currentModule = -1;
    if (!m_modules.isEmpty()) {
        QString modErr;
        setCurrentModule(0, &modErr);
    } else {
        m_samples.clear();
    }

    emit projectOpened();
    emit modulesChanged();
    emit samplesChanged();
    return true;
}

void ProjectManager::closeProject()
{
    if (!m_open)
        return;

    Q_UNUSED(saveAll(nullptr));

    m_modules.clear();
    m_currentModule = -1;
    m_samples.clear();
    m_images.clear();
    m_project = Project();
    m_open    = false;

    emit projectClosed();
}

bool ProjectManager::saveAll(QString *err)
{
    if (!m_open)
        return true;

    if (!m_project.save(err))
        return false;
    if (m_currentModule >= 0)
        return saveCurrentModule(err);
    return true;
}

// ===========================================================================
//  模块
// ===========================================================================
const Module &ProjectManager::moduleAt(int index) const
{
    static const Module kEmpty;
    if (index < 0 || index >= m_modules.size())
        return kEmpty;
    return m_modules.at(index);
}

const Module *ProjectManager::currentModule() const
{
    if (m_currentModule < 0 || m_currentModule >= m_modules.size())
        return nullptr;
    return &m_modules.at(m_currentModule);
}

QString ProjectManager::makeModuleId() const
{
    QSet<QString> used;
    for (const Module &m : m_modules)
        used.insert(m.id());

    for (int i = 1; i < 100000; ++i) {
        const QString id = QStringLiteral("m%1").arg(i);
        if (!used.contains(id))
            return id;
    }
    return QStringLiteral("m%1").arg(QDateTime::currentMSecsSinceEpoch());
}

bool ProjectManager::createModuleDirs(const Module &m, QString *err) const
{
    if (!ensureDir(m.root(), err))
        return false;
    if (!ensureDir(m.labelsDir(), err))
        return false;
    if (!ensureDir(m.modelsDir(), err))
        return false;
    if (!ensureDir(m.resultsDir(), err))
        return false;
    if (!ensureDir(m.logsDir(), err))
        return false;
    return true;
}

bool ProjectManager::addModule(TaskType type, const QString &name, QString *err, int *index)
{
    if (!m_open) {
        if (err) *err = QStringLiteral("尚未打开工程。");
        return false;
    }

    Module m;
    m.setId(makeModuleId());
    m.setName(name.isEmpty() ? taskTypeName(type) : name);
    m.setType(type);
    m.setOrder(m_modules.size());
    m.setRoot(m_project.moduleDir(m.id()));
    m.ensureDefaultClasses();

    m.trainRef().task        = type;
    m.trainRef().datasetRoot = m.root();
    m.trainRef().outputDir   = m.modelsDir();
    m.trainRef().numClasses  = m.targetClassCount();

    if (!createModuleDirs(m, err))
        return false;
    if (!m.save(err))
        return false;

    // 新模块从零开始划分：全部图片为「未划分」
    SampleIndex si;
    si.syncWith(QStringList());
    if (!si.save(m.samplesFilePath(), err))
        return false;

    m_modules.append(m);

    Project::ModuleRef ref;
    ref.id    = m.id();
    ref.name  = m.name();
    ref.type  = m.type();
    m_project.appendModuleRef(ref);
    if (!m_project.save(err))
        return false;

    if (index)
        *index = m_modules.size() - 1;

    emit modulesChanged();
    return true;
}

bool ProjectManager::removeModule(int index, QString *err)
{
    if (!m_open) {
        if (err) *err = QStringLiteral("尚未打开工程。");
        return false;
    }
    if (index < 0 || index >= m_modules.size()) {
        if (err) *err = QStringLiteral("模块下标越界：%1").arg(index);
        return false;
    }

    // 串联删除：连同其后所有模块的数据、模型与测试结果
    for (int i = index; i < m_modules.size(); ++i) {
        const QString dir = m_modules.at(i).root();
        if (QDir(dir).exists() && !removeDirRecursive(dir)) {
            if (err) {
                *err = QStringLiteral("无法删除模块目录（可能被占用）：\n%1")
                           .arg(QDir::toNativeSeparators(dir));
            }
            return false;
        }
    }

    m_modules.remove(index, m_modules.size() - index);
    m_project.removeModuleRefsFrom(index);

    if (!m_project.save(err))
        return false;

    // 当前模块被删掉了就退到最后一个可用模块
    if (m_modules.isEmpty()) {
        m_currentModule = -1;
        m_samples.clear();
    } else {
        m_currentModule = qMin(m_currentModule, m_modules.size() - 1);
        if (m_currentModule < 0)
            m_currentModule = 0;
        QString loadErr;
        setCurrentModule(m_currentModule, &loadErr);
    }

    emit modulesChanged();
    emit currentModuleChanged(m_currentModule);
    emit samplesChanged();
    return true;
}

bool ProjectManager::setCurrentModule(int index, QString *err)
{
    if (index < 0 || index >= m_modules.size()) {
        if (err) *err = QStringLiteral("模块下标越界：%1").arg(index);
        return false;
    }

    // 切换前先把上一个模块的状态落盘，避免丢失划分
    if (m_currentModule >= 0 && m_currentModule < m_modules.size()) {
        Q_UNUSED(m_modules[m_currentModule].save(nullptr));
        Q_UNUSED(m_samples.save(m_modules[m_currentModule].samplesFilePath(), nullptr));
    }

    m_currentModule = index;

    // 重新从磁盘载入该模块（保证与外部修改一致）
    Module m;
    QString loadErr;
    if (Module::load(m_modules.at(index).root(), &m, &loadErr)) {
        m.setRoot(m_modules.at(index).root());
        m.ensureDefaultClasses();
        m.setOrder(index);
        m.trainRef().task        = m.type();
        m.trainRef().datasetRoot = m.root();
        m.trainRef().outputDir   = m.modelsDir();
        m_modules[index] = m;
    }
    createModuleDirs(m_modules.at(index), nullptr);

    // 载入样本状态并与图像池对齐
    m_samples.clear();
    Q_UNUSED(m_samples.load(m_modules.at(index).samplesFilePath(), nullptr));
    QStringList keys;
    keys.reserve(m_images.size());
    for (const QString &p : m_images)
        keys << QFileInfo(p).fileName();
    if (m_samples.syncWith(keys))
        Q_UNUSED(m_samples.save(m_modules.at(index).samplesFilePath(), nullptr));

    emit currentModuleChanged(index);
    emit classesChanged();
    emit samplesChanged();
    return true;
}

bool ProjectManager::saveCurrentModule(QString *err)
{
    if (!m_open || m_currentModule < 0 || m_currentModule >= m_modules.size())
        return true;

    Module &m = m_modules[m_currentModule];
    m.trainRef().task        = m.type();
    m.trainRef().datasetRoot = m.root();
    m.trainRef().outputDir   = m.modelsDir();
    m.trainRef().numClasses  = m.targetClassCount();

    if (!m.save(err))
        return false;
    return m_samples.save(m.samplesFilePath(), err);
}

bool ProjectManager::renameCurrentModule(const QString &name, QString *err)
{
    if (!m_open || m_currentModule < 0 || m_currentModule >= m_modules.size()) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    const QString trimmed = name.trimmed();
    if (trimmed.isEmpty()) {
        if (err) *err = QStringLiteral("模块名称不能为空。");
        return false;
    }

    m_modules[m_currentModule].setName(trimmed);

    // 模块名同时记录在 project.json 的模块引用里，两处都要更新
    QVector<Project::ModuleRef> refs = m_project.moduleRefs();
    if (m_currentModule < refs.size()) {
        refs[m_currentModule].name = trimmed;
        m_project.setModuleRefs(refs);
    }

    if (!saveCurrentModule(err))
        return false;
    if (!m_project.save(err))
        return false;

    emit modulesChanged();
    return true;
}

bool ProjectManager::setCurrentConfig(const TrainConfig &train, const TestConfig &test, QString *err)
{
    if (!m_open || m_currentModule < 0 || m_currentModule >= m_modules.size()) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    Module &m = m_modules[m_currentModule];
    m.setTrain(train);
    m.setTest(test);
    return saveCurrentModule(err);
}

bool ProjectManager::setCurrentTestConfig(const TestConfig &test, QString *err)
{
    if (!m_open || m_currentModule < 0 || m_currentModule >= m_modules.size()) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    m_modules[m_currentModule].setTest(test);
    return saveCurrentModule(err);
}

bool ProjectManager::setCurrentClasses(const QVector<ClassInfo> &classes, QString *err)
{
    if (!m_open || m_currentModule < 0 || m_currentModule >= m_modules.size()) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    Module &m = m_modules[m_currentModule];
    m.setClasses(classes);
    m.ensureDefaultClasses();
    if (!saveCurrentModule(err))
        return false;
    emit classesChanged();
    return true;
}

// ===========================================================================
//  图像池
// ===========================================================================
QString ProjectManager::imageKey(int index) const
{
    if (index < 0 || index >= m_images.size())
        return QString();
    return QFileInfo(m_images.at(index)).fileName();
}

int ProjectManager::indexOfKey(const QString &imageKey) const
{
    for (int i = 0; i < m_images.size(); ++i) {
        if (QFileInfo(m_images.at(i)).fileName() == imageKey)
            return i;
    }
    return -1;
}

QSize ProjectManager::imageSize(int index, bool *ok) const
{
    if (ok)
        *ok = false;
    if (index < 0 || index >= m_images.size())
        return QSize();

    const cv::Mat img = imageio::imreadUnicode(m_images.at(index), cv::IMREAD_COLOR);
    if (img.empty())
        return QSize();
    if (ok)
        *ok = true;
    return QSize(img.cols, img.rows);
}

void ProjectManager::rebuildImageList()
{
    m_images = imageio::listImages(m_project.imagesDir());
}

/// 生成一个在 images/ 下不冲突的文件名（重名自动追加序号）
static QString uniqueFileName(const QString &dir, const QString &base, const QString &ext)
{
    QString name = base + QLatin1Char('.') + ext;
    int n = 1;
    while (QFileInfo::exists(dir + QLatin1Char('/') + name)) {
        name = QStringLiteral("%1_%2.%3").arg(base).arg(n).arg(ext);
        ++n;
    }
    return name;
}

QString ProjectManager::importOneImage(const QString &srcPath, QString *err)
{
    const cv::Mat img = imageio::imreadUnicode(srcPath, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        if (err) *err = QStringLiteral("无法读取（可能已损坏或格式不受支持）");
        return QString();
    }

    // 统一转成工程图片格式；BMP/PNG/TIF 保留原始位深，灰度图仍写灰度
    cv::Mat toWrite = img;
    const ImageFormat fmt = m_project.imageFormat();
    if (fmt == ImageFormat::JPG && img.channels() == 4)
        cv::cvtColor(img, toWrite, cv::COLOR_BGRA2BGR);

    const QString ext  = imageFormatExt(fmt);
    const QString base = QFileInfo(srcPath).completeBaseName();
    const QString name = uniqueFileName(m_project.imagesDir(), base, ext);
    const QString dst  = m_project.imagesDir() + QLatin1Char('/') + name;

    if (!imageio::imwriteUnicode(dst, toWrite)) {
        if (err) *err = QStringLiteral("写入目标文件失败");
        return QString();
    }
    return name;
}

bool ProjectManager::importImages(const QStringList &files, QString *err, int *added,
                                  QStringList *failed)
{
    if (!m_open) {
        if (err) *err = QStringLiteral("尚未打开工程。");
        return false;
    }
    if (added)
        *added = 0;

    QStringList problems;
    int okCount = 0;

    for (const QString &f : files) {
        QString oneErr;
        const QString name = importOneImage(f, &oneErr);
        if (name.isEmpty()) {
            problems << QStringLiteral("%1：%2").arg(QFileInfo(f).fileName(), oneErr);
            continue;
        }
        ++okCount;
    }

    if (okCount > 0) {
        rebuildImageList();
        QStringList keys;
        keys.reserve(m_images.size());
        for (const QString &p : m_images)
            keys << QFileInfo(p).fileName();
        if (m_samples.syncWith(keys) && m_currentModule >= 0)
            Q_UNUSED(m_samples.save(m_modules.at(m_currentModule).samplesFilePath(), nullptr));
        emit samplesChanged();
    }

    if (added)
        *added = okCount;
    if (failed)
        *failed = problems;

    if (problems.isEmpty())
        return okCount > 0;
    if (err)
        *err = problems.join(QLatin1Char('\n'));
    return okCount > 0;
}

bool ProjectManager::importFolder(const QString &dir, QString *err, int *added, QStringList *failed)
{
    const QStringList files = imageio::listImages(dir);
    if (files.isEmpty()) {
        if (err) *err = QStringLiteral("该文件夹下没有可导入的图像：\n%1")
                            .arg(QDir::toNativeSeparators(dir));
        return false;
    }
    return importImages(files, err, added, failed);
}

bool ProjectManager::removeImage(int index, QString *err)
{
    if (!m_open || index < 0 || index >= m_images.size()) {
        if (err) *err = QStringLiteral("图像下标无效。");
        return false;
    }

    const QString path = m_images.at(index);
    const QString key  = QFileInfo(path).fileName();

    // 先删各模块下这张图的标注与结果，再删图像本身
    for (const Module &m : m_modules) {
        QFile::remove(m.labelPath(key));
        QFile::remove(m.maskPath(key));
        QFile::remove(m.resultPath(key));
    }

    if (!QFile::remove(path)) {
        if (err) {
            *err = QStringLiteral("无法删除图像文件：\n%1")
                       .arg(QDir::toNativeSeparators(path));
        }
        return false;
    }

    m_images.removeAt(index);
    m_samples.remove(key);
    if (m_currentModule >= 0)
        Q_UNUSED(m_samples.save(m_modules.at(m_currentModule).samplesFilePath(), nullptr));

    emit samplesChanged();
    return true;
}

bool ProjectManager::rescanImages(QString *err)
{
    if (!m_open) {
        if (err) *err = QStringLiteral("尚未打开工程。");
        return false;
    }
    rebuildImageList();
    QStringList keys;
    keys.reserve(m_images.size());
    for (const QString &p : m_images)
        keys << QFileInfo(p).fileName();
    if (m_samples.syncWith(keys) && m_currentModule >= 0)
        Q_UNUSED(m_samples.save(m_modules.at(m_currentModule).samplesFilePath(), nullptr));
    emit samplesChanged();
    return true;
}

// ===========================================================================
//  样本状态
// ===========================================================================
SampleState ProjectManager::sampleState(int index) const
{
    return m_samples.state(imageKey(index));
}

bool ProjectManager::setSampleState(int index, const SampleState &s)
{
    const QString key = imageKey(index);
    if (key.isEmpty())
        return false;
    m_samples.setState(key, s);
    if (m_currentModule >= 0)
        Q_UNUSED(m_samples.save(m_modules.at(m_currentModule).samplesFilePath(), nullptr));
    emit samplesChanged();
    return true;
}

bool ProjectManager::setSplit(const QVector<int> &indexes, Split split)
{
    if (!m_open)
        return false;
    for (int i : indexes) {
        const QString key = imageKey(i);
        if (key.isEmpty())
            continue;
        SampleState s = m_samples.state(key);
        s.split = split;
        m_samples.setState(key, s);
    }
    if (m_currentModule >= 0)
        Q_UNUSED(m_samples.save(m_modules.at(m_currentModule).samplesFilePath(), nullptr));
    emit samplesChanged();
    return true;
}

bool ProjectManager::setOk(const QVector<int> &indexes, bool ok)
{
    if (!m_open)
        return false;
    for (int i : indexes) {
        const QString key = imageKey(i);
        if (key.isEmpty())
            continue;
        SampleState s = m_samples.state(key);
        s.ok = ok;
        m_samples.setState(key, s);
    }
    if (m_currentModule >= 0)
        Q_UNUSED(m_samples.save(m_modules.at(m_currentModule).samplesFilePath(), nullptr));
    emit samplesChanged();
    return true;
}

DatasetStats ProjectManager::stats() const
{
    DatasetStats st;
    st.total     = m_images.size();
    st.annotated = m_samples.annotatedCount();
    st.train     = m_samples.count(Split::Train);
    st.test      = m_samples.count(Split::Test);
    st.okImages  = m_samples.okCount();
    return st;
}

// ===========================================================================
//  标注读写
// ===========================================================================
bool ProjectManager::loadAnnotation(int index, QVector<AnnoObject> *objects, int *classId,
                                    AnnoState *state, QString *err) const
{
    const Module *m = currentModule();
    if (!m) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    const QString key = imageKey(index);
    if (key.isEmpty()) {
        if (err) *err = QStringLiteral("图像下标无效。");
        return false;
    }
    return anno::readLabel(m->labelPath(key), m->maskPath(key), m->type(),
                          objects, classId, state, m->test().minArea, err);
}

bool ProjectManager::saveAnnotation(int index, const QVector<AnnoObject> &objects, int classId,
                                    AnnoState state, QString *err)
{
    const Module *m = currentModule();
    if (!m) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    const QString key = imageKey(index);
    if (key.isEmpty()) {
        if (err) *err = QStringLiteral("图像下标无效。");
        return false;
    }

    bool sizeOk = false;
    const QSize size = imageSize(index, &sizeOk);
    if (!sizeOk) {
        if (err) *err = QStringLiteral("无法读取图像尺寸，标注未保存。");
        return false;
    }

    if (!anno::writeLabel(m->labelPath(key), m->maskPath(key), m->type(),
                          objects, classId, state, size, err))
        return false;

    SampleState s = m_samples.state(key);
    s.state = state;
    m_samples.setState(key, s);
    Q_UNUSED(m_samples.save(m->samplesFilePath(), nullptr));

    emit samplesChanged();
    return true;
}

bool ProjectManager::removeAnnotationFile(int index, QString *err)
{
    const Module *m = currentModule();
    if (!m) {
        if (err) *err = QStringLiteral("尚未选择模块。");
        return false;
    }
    const QString key = imageKey(index);
    if (key.isEmpty()) {
        if (err) *err = QStringLiteral("图像下标无效。");
        return false;
    }

    bool ok = true;
    if (QFileInfo::exists(m->labelPath(key)) && !QFile::remove(m->labelPath(key)))
        ok = false;
    if (QFileInfo::exists(m->maskPath(key)) && !QFile::remove(m->maskPath(key)))
        ok = false;

    SampleState s = m_samples.state(key);
    s.state = AnnoState::None;
    m_samples.setState(key, s);
    Q_UNUSED(m_samples.save(m->samplesFilePath(), nullptr));

    if (!ok && err)
        *err = QStringLiteral("部分标注文件无法删除（可能被占用）。");

    emit samplesChanged();
    return ok;
}

bool ProjectManager::savePrediction(int index, const PredictionResult &pr, QString *err) const
{
    const Module *m = currentModule();
    if (!m)
        return false;
    const QString key = imageKey(index);
    if (key.isEmpty())
        return false;

    QJsonObject root;
    root.insert(QStringLiteral("task"), taskTypeKey(pr.task));

    if (pr.task == TaskType::Classification) {
        root.insert(QStringLiteral("classId"), pr.classId);
        root.insert(QStringLiteral("score"), double(pr.score));
        QJsonArray arr;
        for (float s : pr.scores)
            arr.append(double(s));
        root.insert(QStringLiteral("scores"), arr);
    } else if (pr.task == TaskType::Detection) {
        QJsonArray arr;
        for (const DetectionBox &b : pr.boxes) {
            QJsonObject o;
            o.insert(QStringLiteral("classId"), b.classId);
            o.insert(QStringLiteral("score"), double(b.score));
            o.insert(QStringLiteral("rect"), jsonio::rectToJson(b.rect));
            arr.append(o);
        }
        root.insert(QStringLiteral("boxes"), arr);
    } else {
        if (!pr.mask.empty()) {
            const QString maskFile = QFileInfo(key).completeBaseName() + QStringLiteral("_pred.png");
            const QString maskPath = m->resultsDir() + QLatin1Char('/') + maskFile;
            if (!imageio::imwriteUnicode(maskPath, pr.mask)) {
                if (err) *err = QStringLiteral("写入预测掩码失败：%1").arg(maskFile);
                return false;
            }
            root.insert(QStringLiteral("maskFile"), maskFile);
        }
    }

    return jsonio::writeObject(m->resultPath(key), root, err);
}

bool ProjectManager::loadPrediction(int index, PredictionResult *pr, QString *err) const
{
    if (pr)
        *pr = PredictionResult();

    const Module *m = currentModule();
    if (!m || !pr)
        return false;
    const QString key = imageKey(index);
    if (key.isEmpty())
        return false;

    QJsonObject root;
    if (!jsonio::readObject(m->resultPath(key), &root, err))
        return false;

    bool ok = false;
    pr->task = taskTypeFromKey(jsonio::str(root, QStringLiteral("task"), taskTypeKey(m->type())), &ok);

    if (pr->task == TaskType::Classification) {
        pr->classId = jsonio::integer(root, QStringLiteral("classId"), -1);
        pr->score   = float(jsonio::num(root, QStringLiteral("score"), 0.0));
        const QJsonArray arr = root.value(QStringLiteral("scores")).toArray();
        for (int i = 0; i < arr.size(); ++i)
            pr->scores.push_back(float(arr.at(i).toDouble()));
    } else if (pr->task == TaskType::Detection) {
        const QJsonArray arr = root.value(QStringLiteral("boxes")).toArray();
        for (int i = 0; i < arr.size(); ++i) {
            const QJsonObject o = arr.at(i).toObject();
            DetectionBox b;
            b.classId = jsonio::integer(o, QStringLiteral("classId"), 0);
            b.score   = float(jsonio::num(o, QStringLiteral("score"), 0.0));
            b.rect    = jsonio::rectFromJson(o.value(QStringLiteral("rect")).toArray());
            pr->boxes.push_back(b);
        }
    } else {
        const QString maskFile = jsonio::str(root, QStringLiteral("maskFile"));
        if (!maskFile.isEmpty()) {
            pr->mask = imageio::imreadUnicode(m->resultsDir() + QLatin1Char('/') + maskFile,
                                              cv::IMREAD_GRAYSCALE);
        }
    }
    return true;
}

bool ProjectManager::saveMetrics(const Metrics &mt, QString *err) const
{
    const Module *m = currentModule();
    if (!m)
        return false;
    return jsonio::writeObject(m->metricsPath(), metrics::metricsToJson(mt), err);
}

Metrics ProjectManager::loadMetrics() const
{
    const Module *m = currentModule();
    if (!m)
        return Metrics();
    QJsonObject o;
    if (!jsonio::readObject(m->metricsPath(), &o, nullptr))
        return Metrics();
    return metrics::metricsFromJson(o);
}

bool ProjectManager::hasModel() const
{
    const Module *m = currentModule();
    return m && QFileInfo::exists(m->modelPath());
}

// ===========================================================================
//  训练视图
// ===========================================================================
QVector<TrainSample> ProjectManager::trainSamples() const
{
    QVector<TrainSample> out;
    const Module *m = currentModule();
    if (!m)
        return out;

    for (int i = 0; i < m_images.size(); ++i) {
        const QString key = QFileInfo(m_images.at(i)).fileName();
        const SampleState s = m_samples.state(key);
        if (s.split != Split::Train)
            continue;
        if (s.state == AnnoState::None)
            continue;   // 未标注的图不参与训练

        TrainSample t;
        t.imageKey  = key;
        t.imagePath = m_images.at(i);
        t.labelPath = m->labelPath(key);
        t.maskPath  = (m->type() == TaskType::Segmentation) ? m->maskPath(key) : QString();

        if (m->type() == TaskType::Classification) {
            QVector<AnnoObject> objs;
            AnnoState st = AnnoState::None;
            if (anno::readLabel(m->labelPath(key), QString(), m->type(), &objs, &t.classId, &st,
                                m->test().minArea, nullptr)) {
                // 分类任务的 classId 已由 readLabel 填入
            } else {
                continue;
            }
        }
        out.append(t);
    }
    return out;
}

QVector<TrainSample> ProjectManager::testSamples() const
{
    QVector<TrainSample> out;
    const Module *m = currentModule();
    if (!m)
        return out;

    for (int i = 0; i < m_images.size(); ++i) {
        const QString key = QFileInfo(m_images.at(i)).fileName();
        const SampleState s = m_samples.state(key);
        if (s.split != Split::Test)
            continue;

        TrainSample t;
        t.imageKey  = key;
        t.imagePath = m_images.at(i);
        t.labelPath = m->labelPath(key);
        t.maskPath  = (m->type() == TaskType::Segmentation) ? m->maskPath(key) : QString();
        t.classId   = -1;
        if (m->type() == TaskType::Classification && s.state != AnnoState::None) {
            QVector<AnnoObject> objs;
            AnnoState st = AnnoState::None;
            Q_UNUSED(anno::readLabel(m->labelPath(key), QString(), m->type(), &objs, &t.classId, &st,
                                     m->test().minArea, nullptr));
        }
        out.append(t);
    }
    return out;
}

QVector<int> ProjectManager::unannotatedIndexes() const
{
    QVector<int> out;
    for (int i = 0; i < m_images.size(); ++i) {
        if (m_samples.state(imageKey(i)).state == AnnoState::None)
            out.append(i);
    }
    return out;
}

void ProjectManager::ensureSampleEntry(int index)
{
    const QString key = imageKey(index);
    if (key.isEmpty() || m_samples.contains(key))
        return;
    m_samples.setState(key, SampleState());
}

} // namespace aq
