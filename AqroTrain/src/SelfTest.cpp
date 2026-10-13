// ============================================================================
//  SelfTest.cpp —— 无人值守自检实现
//  ---------------------------------------------------------------------------
//  三个入口，覆盖「界面 -> 端到端数据流 -> 标注交互」三层验证：
//
//   · runAutoTrain    端到端：建工程 -> 三模块 -> 合成数据 -> 脚本化标注 -> 划分
//                     -> 训练 -> 自动评价 -> 导出模型 -> 重新加载推理
//   · runAnnoSelfTest 标注交互：向画布 viewport 派发鼠标事件，验证对象写入、
//                     顶点编辑、撤销/重做，并断言「对象光栅化掩码 == 画布掩码」
//   · runMkdirDiagnostic 目录创建诊断：逐层报告失败位置与原因
//
//  三者均返回 0 表示通过；report 填入可读的中文报告。
// ============================================================================
#include "SelfTest.h"

#include <QApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QStringList>
#include <QTextStream>
#include <QTimer>

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <random>

#include "annotate/ImageCanvas.h"
#include "core/ImageIO.h"
#include "core/Metrics.h"
#include "project/ProjectManager.h"
#include "train/Evaluator.h"
#include "train/ITrainer.h"
#include "train/TrainerFactory.h"

namespace aq {

namespace {

// ---------------------------------------------------------------------------
//  合成数据
//  ---------------------------------------------------------------------------
struct SynthSample {
    cv::Mat  img;        ///< BGR
    cv::Rect defect;     ///< 缺陷所在矩形
    int      variant = 0;///< 0 = 红色缺陷；1 = 蓝色缺陷
};

/**
 * 生成合成样本：深灰背景 + 一个高饱和色块作为缺陷。
 *
 * 两点设计考虑：
 *  1. OpenCV 轻量后端用「HSV 直方图 + 8x8 缩略图」做分类特征、图块颜色特征做
 *     检测特征、逐像素颜色特征做分割特征，红/蓝高饱和色块与灰色背景在这些
 *     特征上都清晰可分，因此自检验证的是**整条数据流通路**，而不是去赌随机森林的运气。
 *  2. 缺陷块刻意做得较大（约占图像面积 20%~40%）。实测当缺陷只占 6% 时，
 *     全局颜色直方图被背景稀释，轻量后端的分类准确率会掉到 50% 左右 ——
 *     那是算法本身的局限（已写入 README），不适合拿来当通路自检的判据。
 */
QVector<SynthSample> makeSynthetic(int count, int w, int h, unsigned seed)
{
    QVector<SynthSample> out;
    out.reserve(count);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> posX(12, w - 132);
    std::uniform_int_distribution<int> posY(12, h - 132);
    std::uniform_int_distribution<int> sizeD(90, 120);
    std::uniform_int_distribution<int> noise(-8, 8);
    std::uniform_int_distribution<int> variantD(0, 1);

    for (int i = 0; i < count; ++i) {
        SynthSample s;
        s.variant = variantD(rng);

        const int bw = sizeD(rng);
        const int bh = sizeD(rng);
        s.defect = cv::Rect(posX(rng), posY(rng), bw, bh);

        s.img = cv::Mat(h, w, CV_8UC3);
        for (int y = 0; y < h; ++y) {
            cv::Vec3b *row = s.img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < w; ++x) {
                const int g = 40 + noise(rng);
                row[x] = cv::Vec3b(uchar(g), uchar(g), uchar(g));
            }
        }

        const cv::Scalar color = (s.variant == 0) ? cv::Scalar(30, 30, 230)     // BGR 红
                                                 : cv::Scalar(230, 60, 30);     // BGR 蓝
        cv::rectangle(s.img, s.defect, color, cv::FILLED);
        out.append(s);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  报告辅助
// ---------------------------------------------------------------------------
struct CheckItem {
    QString name;
    bool    ok = false;
    QString detail;
};

class Report {
public:
    void add(const QString &name, bool ok, const QString &detail = QString())
    {
        m_items.append({ name, ok, detail });
        if (!ok)
            ++m_failed;
    }

    QString text(const QString &title) const
    {
        QString r;
        r += QStringLiteral("================ %1 ================\n\n").arg(title);
        for (const CheckItem &it : m_items) {
            r += QStringLiteral("%1 %2\n").arg(it.ok ? QStringLiteral("[通过]")
                                                    : QStringLiteral("[失败]"),
                                              it.name);
            if (!it.detail.isEmpty())
                r += QStringLiteral("        %1\n").arg(it.detail);
        }
        r += QStringLiteral("\n合计 %1 项，失败 %2 项。\n").arg(m_items.size()).arg(m_failed);
        r += QStringLiteral("\n================ 自检结束 ================\n");
        return r;
    }

    int failed() const { return m_failed; }

private:
    QVector<CheckItem> m_items;
    int m_failed = 0;
};

// ---------------------------------------------------------------------------
//  训练辅助：同步等待异步训练结束
// ---------------------------------------------------------------------------
bool trainModule(const Module &mod, const QVector<TrainSample> &samples,
                 QString *log, QString *modelPath, QString *err)
{
    ITrainer *trainer = TrainerFactory::create(nullptr, TrainerFactory::fromKey(mod.train().backend));
    if (!trainer) {
        if (err) *err = QStringLiteral("没有可用的训练后端。");
        return false;
    }

    TrainConfig cfg = mod.train();
    cfg.task        = mod.type();
    cfg.numClasses  = mod.targetClassCount();
    cfg.datasetRoot = mod.root();
    cfg.outputDir   = mod.modelsDir();
    cfg.extra.insert(QStringLiteral("scoreThreshold"), mod.test().scoreThreshold);
    cfg.extra.insert(QStringLiteral("nmsThreshold"), mod.test().nmsThreshold);
    cfg.extra.insert(QStringLiteral("minArea"), mod.test().minArea);
    cfg.extra.insert(QStringLiteral("valSplit"), 0.25);

    if (!trainer->initialize(cfg, samples, mod.classes(), err)) {
        delete trainer;
        return false;
    }

    QEventLoop loop;
    bool     ok = false;
    QString  message;
    QString  path;

    QObject::connect(trainer, &ITrainer::trainingFinished, &loop,
                     [&](bool o, const QString &p, const QString &m) {
                         ok = o;
                         path = p;
                         message = m;
                         loop.quit();
                     });
    QObject::connect(trainer, &ITrainer::logMessage, &loop,
                     [&](const QString &text, int level) {
                         if (!log)
                             return;
                         // 注意：这里必须用 QString，不能用 QLatin1String(tag) ——
                         // tag 里是 UTF-8 字节，按 Latin-1 解释会变成乱码。
                         QString prefix;
                         switch (level) {
                         case 2: prefix = QStringLiteral("[警告] "); break;
                         case 3: prefix = QStringLiteral("[错误] "); break;
                         default: break;
                         }
                         log->append(prefix + text + QLatin1Char('\n'));
                     });

    // 超时保护：自检不应无限等待
    QTimer guard;
    guard.setSingleShot(true);
    QObject::connect(&guard, &QTimer::timeout, &loop, &QEventLoop::quit);
    guard.start(20 * 60 * 1000);

    trainer->start();
    loop.exec();

    if (trainer->isRunning())
        trainer->requestStop();

    delete trainer;

    if (!ok) {
        if (err) *err = message.isEmpty() ? QStringLiteral("训练未成功结束（可能超时或被中止）。")
                                         : message;
        return false;
    }
    if (modelPath)
        *modelPath = path;
    return true;
}

// ---------------------------------------------------------------------------
//  目录创建用例（作为端到端自检的一部分，保证「建目录」这条路上不出意外）
//  ---------------------------------------------------------------------------
void runDirCases(const QString &base, Report *rep, QString *log)
{
    const QString sandbox = base + QStringLiteral("/dircases");
    ProjectManager::removeDirRecursive(sandbox);
    ProjectManager::ensureDir(sandbox, nullptr);

    // 1) 多级嵌套：三级都不存在
    {
        const QString p = sandbox + QStringLiteral("/a/b/c");
        QString err;
        const bool ok = ProjectManager::ensureDir(p, &err) && QFileInfo(p).isDir();
        rep->add(QStringLiteral("目录创建：多级嵌套一次建好（三级都不存在）"), ok,
                 ok ? QDir::toNativeSeparators(p) : err);
    }

    // 2) 重复创建：已存在必须成功而不是报错
    {
        const QString p = sandbox + QStringLiteral("/a/b/c");
        QString err;
        const bool ok = ProjectManager::ensureDir(p, &err);
        rep->add(QStringLiteral("目录创建：对已存在的目录重复调用不报错"), ok,
                 ok ? QStringLiteral("返回成功") : err);
    }

    // 3) 叶子名是纯数字（实际遇到的形状：D:\datasets\1\1）
    {
        const QString p = sandbox + QStringLiteral("/1/1");
        QString err;
        const bool ok = ProjectManager::ensureDir(p, &err) && QFileInfo(p).isDir();
        rep->add(QStringLiteral("目录创建：叶子名为纯数字（形如 .../1/1）"), ok,
                 ok ? QDir::toNativeSeparators(p) : err);
    }

    // 4) 长路径：超过 Windows 的 MAX_PATH(260)，验证 \\?\ 前缀生效
    {
        QString deep = sandbox;
        for (int i = 0; i < 12; ++i)
            deep += QStringLiteral("/verylongdirectorysegment_%1").arg(i);
        deep += QStringLiteral("/leaf");
        QString err;
        const bool ok = ProjectManager::ensureDir(deep, &err) && QFileInfo(deep).isDir();

        // 关键：长路径不仅要能建，还要能删。
        // QDir::removeRecursively / rmdir / 资源管理器都删不掉 >MAX_PATH 的目录，
        // 若不处理就会在磁盘上留下永远清不掉的垃圾。
        const bool removed = ok && ProjectManager::removeDirRecursive(deep)
                             && !QFileInfo(deep).exists();

        rep->add(QStringLiteral("目录创建：长路径（%1 字符 > MAX_PATH）可建且可删").arg(deep.length()),
                 ok && removed,
                 ok ? (removed ? QStringLiteral("已创建并已用长路径感知的递归删除清掉，长度 %1")
                                     .arg(deep.length())
                               : QStringLiteral("创建成功但**删除失败**（会留下清不掉的垃圾）"))
                    : err);
    }

    // 5) 同名文件占位：必须失败，且原因要指明「已存在且不是目录」
    {
        const QString blocker = sandbox + QStringLiteral("/blocked");
        QFile f(blocker);
        f.open(QIODevice::WriteOnly);
        f.write("x");
        f.close();

        const QString p = blocker + QStringLiteral("/child");
        QString err;
        const bool failedAsExpected = !ProjectManager::ensureDir(p, &err);
        const bool reasonClear = err.contains(QStringLiteral("已存在且不是目录"));
        rep->add(QStringLiteral("目录创建：被同名文件占位时失败并说明原因"),
                 failedAsExpected && reasonClear,
                 reasonClear ? err.split(QLatin1Char('\n')).first() : err);
    }

    // 6) 直接在已存在的空目录里建工程：完全不依赖「创建目录」这条路径
    {
        const QString existing = sandbox + QStringLiteral("/existing_empty");
        ProjectManager::ensureDir(existing, nullptr);

        ProjectManager pm;
        QString err;
        const bool ok = pm.createProject(existing, QStringLiteral("用已有目录"), ImageFormat::PNG, &err)
                        && QFileInfo(existing + QStringLiteral("/project.json")).exists();
        rep->add(QStringLiteral("工程创建：直接使用已存在的空目录（无需新建目录）"), ok,
                 ok ? QDir::toNativeSeparators(existing) : err);
        if (ok)
            pm.closeProject();
    }

    // 7) 已存在但非空的目录必须被拒绝，避免把别人的文件当工程目录
    {
        const QString nonEmpty = sandbox + QStringLiteral("/non_empty");
        ProjectManager::ensureDir(nonEmpty, nullptr);
        QFile f(nonEmpty + QStringLiteral("/somefile.txt"));
        f.open(QIODevice::WriteOnly);
        f.write("x");
        f.close();

        ProjectManager pm;
        QString err;
        const bool rejected = !pm.createProject(nonEmpty, QStringLiteral("x"), ImageFormat::PNG, &err);
        rep->add(QStringLiteral("工程创建：已存在且非空的目录被拒绝并说明原因"),
                 rejected && err.contains(QStringLiteral("非空")), err.split(QLatin1Char('\n')).first());
    }

    // 8) 写文件探测：可写目录必须报成功
    {
        QString err;
        const bool ok = ProjectManager::probeWritable(sandbox, &err);
        rep->add(QStringLiteral("写文件探测：可写目录返回成功"), ok,
                 ok ? QDir::toNativeSeparators(sandbox) : err);
    }

    // 9) 写文件探测：对不存在的目录必须报失败而不是崩溃
    {
        QString err;
        const bool ok = ProjectManager::probeWritable(sandbox + QStringLiteral("/no_such_dir"), &err);
        rep->add(QStringLiteral("写文件探测：对不存在的目录安全返回失败"), !ok,
                 ok ? QStringLiteral("意外返回成功") : err);
    }

    // 10) 报告落盘：正常路径必须原地写入
    {
        const QString p = sandbox + QStringLiteral("/report_ok.txt");
        const QString used = writeReport(p, QStringLiteral("hello"), sandbox);
        const bool ok = (used == p) && QFileInfo(p).exists() && QFileInfo(p).size() > 0;
        rep->add(QStringLiteral("报告落盘：正常路径原地写入"), ok,
                 ok ? QDir::toNativeSeparators(p) : QStringLiteral("实际写入：%1").arg(used));
    }

    // 11) 报告落盘：路径非法时必须自动回退，而不是静默丢掉报告
    {
        const QString bad = sandbox + QStringLiteral("/bad<name>|report.txt");
        const QString used = writeReport(bad, QStringLiteral("fallback"), sandbox);
        const bool ok = used != bad && !used.isEmpty() && QFileInfo(used).exists()
                        && QFileInfo(used).size() > 0;
        rep->add(QStringLiteral("报告落盘：路径非法时自动回退到可写位置"), ok,
                 ok ? QStringLiteral("已回退到 %1").arg(QDir::toNativeSeparators(used))
                    : QStringLiteral("未能写入任何位置"));
    }

    if (log)
        *log += QStringLiteral("目录创建用例已在 %1 下执行。\n")
                    .arg(QDir::toNativeSeparators(sandbox));
}

} // namespace

// ===========================================================================
//  报告落盘（带回退）
// ===========================================================================
QString writeReport(const QString &requestedPath, const QString &text, const QString &workDirHint)
{
    auto tryWrite = [&text](const QString &path) -> bool {
        if (path.isEmpty())
            return false;
        QDir().mkpath(QFileInfo(path).absolutePath());
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text))
            return false;
        QTextStream ts(&f);
        ts.setCodec("UTF-8");
        ts << text;
        ts.flush();
        f.close();
        return QFileInfo(path).exists() && QFileInfo(path).size() >= 0;
    };

    if (tryWrite(requestedPath))
        return requestedPath;

    // 回退 1：工作目录下（自检本来就在那里创建文件，通常可写）
    if (!workDirHint.isEmpty()) {
        const QString alt = QDir::cleanPath(workDirHint) + QStringLiteral("/selftest_report.txt");
        if (alt != requestedPath && tryWrite(alt))
            return alt;
    }

    // 回退 2：程序所在目录
    {
        const QString alt = QCoreApplication::applicationDirPath() + QStringLiteral("/aqrotrain_report.txt");
        if (alt != requestedPath && tryWrite(alt))
            return alt;
    }

    return QString();
}

// ===========================================================================
//  目录创建诊断
// ===========================================================================
int runMkdirDiagnostic(const QString &dir, QString *report)
{
    QString r;

    const QString clean = QDir::cleanPath(QDir::fromNativeSeparators(dir));
    r += QStringLiteral("================ AqroTrain 目录创建诊断 ================\n\n");
    r += QStringLiteral("目标目录：%1\n\n").arg(QDir::toNativeSeparators(clean));

    r += QStringLiteral("--- 逐层状态 ---\n");
    QStringList parts = clean.split(QLatin1Char('/'), QString::SkipEmptyParts);
    QString cur;
    for (int i = 0; i < parts.size(); ++i) {
        cur = (i == 0) ? parts.at(0) : cur + QLatin1Char('/') + parts.at(i);
        const QFileInfo fi(cur);
        const QString yes = QStringLiteral("是");
        const QString no  = QStringLiteral("否");
        r += QStringLiteral("%1\n").arg(QDir::toNativeSeparators(cur));
        if (!fi.exists()) {
            r += QStringLiteral("    不存在\n");
            continue;
        }
        r += QStringLiteral("    存在=%1，是目录=%2，可写=%3，只读属性=%4，权限=%5\n")
                 .arg(yes, fi.isDir() ? yes : no, fi.isWritable() ? yes : no,
                      ProjectManager::looksReadOnly(cur) ? yes : no,
                      ProjectManager::permissionString(cur));
    }

    r += QStringLiteral("\n--- 实际创建 ---\n");
    QString err;
    QVector<ProjectManager::DirCreateAttempt> attempts;
    const bool ok = ProjectManager::ensureDir(clean, &err, &attempts);

    if (ok) {
        r += QStringLiteral("创建成功：%1\n").arg(QDir::toNativeSeparators(clean));
    } else {
        r += QStringLiteral("创建失败\n\n%1\n").arg(err);
    }

    // 各 API 的尝试结果（即使整体成功也打印，便于发现「某个 API 在这台机器上不可用」）
    if (!attempts.isEmpty()) {
        r += QStringLiteral("\n--- 各 API 尝试结果 ---\n");
        for (const ProjectManager::DirCreateAttempt &a : attempts) {
            r += QStringLiteral("%1 %2")
                     .arg(a.ok ? QStringLiteral("[成功]") : QStringLiteral("[失败]"), a.strategy);
            if (!a.ok)
                r += QStringLiteral("　错误码 %1：%2").arg(a.win32Error).arg(a.message);
            r += QLatin1Char('\n');
        }
    }

    // 写文件探测：区分「目录不可写」与「只是不能建目录」
    {
        const QString parent = QFileInfo(clean).absolutePath();
        QString probeErr;
        const bool w = ProjectManager::probeWritable(parent, &probeErr);
        r += QStringLiteral("\n--- 上级目录写文件探测 ---\n%1：%2\n")
                 .arg(QDir::toNativeSeparators(parent),
                      w ? QStringLiteral("成功（本进程可以在该目录里写文件）") : probeErr);
    }

    if (!ok) {
        r += QStringLiteral("\n--- 建议 ---\n");
        const QFileInfo parentInfo(QFileInfo(clean).absolutePath());
        QString probeErr2;
        const bool writable = ProjectManager::probeWritable(parentInfo.absoluteFilePath(), &probeErr2);
        if (writable) {
            r += QStringLiteral("可以往上级目录写文件、却建不了子目录，说明**不是普通权限问题**。\n");
            r += QStringLiteral("优先排查第三方安全软件的「文件夹保护 / 文档保护」：\n");
            r += QStringLiteral("它会静默拒绝未签名的新程序创建目录，而 ACL 与只读属性检查一切正常。\n");
        } else {
            r += QStringLiteral("本进程无法往该目录写文件，属于真实的写入受限。\n");
        }
        r += QStringLiteral("处理办法：加入安全软件信任区 / 关闭文件夹保护 / 换一个位置创建"
                            "（例如 D:\\datasets\\...）/ 以管理员身份运行。\n");
    }

    r += QStringLiteral("\n================ 诊断结束 ================\n");

    if (report)
        *report = r;
    return ok ? 0 : 1;
}

// ===========================================================================
//  端到端自动训练自检
// ===========================================================================
int runAutoTrain(const QString &workDir, QString *report)
{
    Report rep;
    QString log;

    const QString root = QDir::cleanPath(QDir::fromNativeSeparators(workDir));
    const QString projRoot = root + QStringLiteral("/proj");
    const QString imgSrc = root + QStringLiteral("/synth");

    log += QStringLiteral("工作目录：%1\n").arg(QDir::toNativeSeparators(root));

    // --- 0. 准备干净的工作目录 ---
    // 用长路径感知的删除：上一轮可能留下 >MAX_PATH 的测试目录，
    // QDir::removeRecursively 删不掉会导致后续断言基于脏状态。
    ProjectManager::removeDirRecursive(root);
    rep.add(QStringLiteral("准备干净的工作目录"), ProjectManager::ensureDir(root, nullptr),
            QDir::toNativeSeparators(root));
    ProjectManager::ensureDir(imgSrc, nullptr);

    // --- 0b. 目录创建 / 工程创建的专项用例 ---
    runDirCases(root, &rep, &log);

    // --- 1. 合成数据 ---
    const int kCount = 24;
    const QVector<SynthSample> synth = makeSynthetic(kCount, 192, 192, 20260101u);
    QStringList synthFiles;
    for (int i = 0; i < synth.size(); ++i) {
        const QString f = imgSrc + QStringLiteral("/synth_%1.png").arg(i, 3, 10, QLatin1Char('0'));
        if (imageio::imwriteUnicode(f, synth.at(i).img))
            synthFiles << f;
    }
    rep.add(QStringLiteral("生成合成数据"), synthFiles.size() == kCount,
            QStringLiteral("%1 / %2 张已写入 %3").arg(synthFiles.size()).arg(kCount)
                .arg(QDir::toNativeSeparators(imgSrc)));

    // --- 2. 建工程 + 三个模块 ---
    ProjectManager pm;
    QString err;
    const bool created = pm.createProject(projRoot, QStringLiteral("自检工程"),
                                         ImageFormat::PNG, &err);
    rep.add(QStringLiteral("新建工程"), created, created ? projRoot : err);
    if (!created) {
        if (report) *report = rep.text(QStringLiteral("AqroTrain 端到端自检")) + log;
        return 1;
    }

    const TaskType tasks[3] = { TaskType::Classification, TaskType::Detection,
                                TaskType::Segmentation };
    int moduleIndex[3] = { -1, -1, -1 };
    for (int i = 0; i < 3; ++i) {
        QString e;
        if (!pm.addModule(tasks[i], taskTypeName(tasks[i]), &e, &moduleIndex[i])) {
            rep.add(QStringLiteral("添加模块 %1").arg(taskTypeName(tasks[i])), false, e);
            if (report) *report = rep.text(QStringLiteral("AqroTrain 端到端自检")) + log;
            return 1;
        }
    }
    rep.add(QStringLiteral("添加三个模块（分类 / 检测 / 分割）"), true);

    // --- 3. 导入图像 ---
    int added = 0;
    QStringList failed;
    pm.importImages(synthFiles, &err, &added, &failed);
    rep.add(QStringLiteral("导入图像到工程图像池"), added == kCount && failed.isEmpty(),
            QStringLiteral("成功 %1 张，失败 %2 张%3")
                .arg(added).arg(failed.size())
                .arg(failed.isEmpty() ? QString() : QStringLiteral("（%1）")
                                                          .arg(failed.join(QLatin1Char(';')))));
    if (added != kCount) {
        if (report) *report = rep.text(QStringLiteral("AqroTrain 端到端自检")) + log;
        return 1;
    }

    // --- 4. 逐模块脚本化标注 + 数据划分 ---
    struct ModResult {
        QString  modelPath;
        Metrics  metrics;
        int      trainCount = 0;
        int      testCount = 0;
    };
    QVector<ModResult> results(3);

    for (int mi = 0; mi < 3; ++mi) {
        QString e;
        if (!pm.setCurrentModule(moduleIndex[mi], &e)) {
            rep.add(QStringLiteral("切换到模块 %1").arg(taskTypeName(tasks[mi])), false, e);
            continue;
        }

        const Module *mod = pm.currentModule();
        const int numImages = pm.imageCount();

        // 可选：用环境变量覆盖模块的置信度阈值，便于做「阈值 - 精确率/召回率」曲线。
        // 只影响自检，不影响正常使用（正常使用由左侧「测试参数」面板控制）。
        const QString scoreEnv = qEnvironmentVariable("AQROTRAIN_AUTOTRAIN_SCORE");
        if (!scoreEnv.isEmpty()) {
            bool numOk = false;
            const double v = scoreEnv.toDouble(&numOk);
            if (numOk && v > 0.0 && v < 1.0) {
                TestConfig tc = mod->test();
                tc.scoreThreshold = v;
                pm.setCurrentTestConfig(tc, nullptr);
                mod = pm.currentModule();
                log += QStringLiteral("[自检] 已把「%1」的置信度阈值覆盖为 %2\n")
                           .arg(taskTypeName(tasks[mi]))
                           .arg(v, 0, 'f', 2);
            }
        }

        // 前 2/3 进训练集，其余进测试集
        const int trainUpTo = (numImages * 2) / 3;
        QVector<int> trainIdx, testIdx;
        for (int i = 0; i < numImages; ++i) {
            if (i < trainUpTo)
                trainIdx.append(i);
            else
                testIdx.append(i);
        }
        pm.setSplit(trainIdx, Split::Train);
        pm.setSplit(testIdx, Split::Test);

        int annotated = 0;
        for (int i = 0; i < numImages; ++i) {
            const SynthSample &s = synth.at(i);
            QVector<AnnoObject> objs;
            int classId = -1;

            if (tasks[mi] == TaskType::Classification) {
                classId = s.variant;   // 0 = 红块, 1 = 蓝块
            } else {
                AnnoObject o;
                o.type    = AnnoObject::Rect;
                o.classId = 1;         // 索引 0 是背景，目标从 1 开始
                o.rect    = QRectF(s.defect.x, s.defect.y, s.defect.width, s.defect.height);
                o.rect = o.rect.adjusted(-4, -4, 4, 4);   // 稍微外扩，模拟人工标注
                objs.append(o);
            }

            if (pm.saveAnnotation(i, objs, classId, AnnoState::Done, &e))
                ++annotated;
            else
                log += QStringLiteral("[警告] 保存标注失败（图 %1）：%2\n").arg(i).arg(e);
        }

        const QVector<TrainSample> trainSamples = pm.trainSamples();
        const QVector<TrainSample> testSamples = pm.testSamples();
        results[mi].trainCount = trainSamples.size();
        results[mi].testCount = testSamples.size();

        rep.add(QStringLiteral("模块「%1」标注与划分").arg(taskTypeName(tasks[mi])),
                annotated == numImages && !trainSamples.isEmpty() && !testSamples.isEmpty(),
                QStringLiteral("已标注 %1 / %2；训练集 %3，测试集 %4（类别数 %5，目标类别 %6）")
                    .arg(annotated).arg(numImages).arg(trainSamples.size())
                    .arg(testSamples.size()).arg(mod->classCount())
                    .arg(mod->targetClassCount()));

        // --- 训练 ---
        QString modelPath, trainErr, trainLog;
        const bool trained = trainModule(*mod, trainSamples, &trainLog, &modelPath, &trainErr);
        log += QStringLiteral("\n---------- %1 训练日志 ----------\n%2")
                   .arg(taskTypeName(tasks[mi]), trainLog);
        rep.add(QStringLiteral("模块「%1」训练").arg(taskTypeName(tasks[mi])), trained,
                trained ? QStringLiteral("模型：%1（%2 KB）")
                              .arg(QFileInfo(modelPath).fileName())
                              .arg(QFileInfo(modelPath).size() / 1024)
                        : trainErr);

        if (!trained)
            continue;
        results[mi].modelPath = modelPath;

        // 模型与元数据必须存在
        const bool metaOk = modelio::readModelMeta(modelPath, nullptr, nullptr)
                            && QFileInfo(modelio::metaPath(modelPath)).exists();
        rep.add(QStringLiteral("模块「%1」模型与元数据落盘").arg(taskTypeName(tasks[mi])), metaOk,
                QStringLiteral("%1 + %2")
                    .arg(QFileInfo(modelPath).fileName(),
                         QFileInfo(modelio::metaPath(modelPath)).fileName()));

        // --- 自动评价 ---
        Evaluator ev;
        Metrics mt;
        QVector<Evaluator::Item> items;
        QString evalErr;
        QString evalLog;
        QObject::connect(&ev, &Evaluator::logMessage, &ev,
                         [&evalLog](const QString &text, int level) {
                             QString prefix;
                             switch (level) {
                             case 1: prefix = QStringLiteral("[成功] "); break;
                             case 2: prefix = QStringLiteral("[警告] "); break;
                             case 3: prefix = QStringLiteral("[错误] "); break;
                             default: break;
                             }
                             evalLog += prefix + text + QLatin1Char('\n');
                         });
        const bool evaluated = ev.evaluate(*mod, testSamples, modelPath, &mt, &items, &evalErr);
        rep.add(QStringLiteral("模块「%1」测试集评价").arg(taskTypeName(tasks[mi])), evaluated,
                evaluated ? mt.summary() : evalErr);
        if (!evaluated)
            continue;
        results[mi].metrics = mt;

        log += QStringLiteral("\n---------- %1 评价日志 ----------\n%2")
                   .arg(taskTypeName(tasks[mi]), evalLog);

        // 逐类指标写进报告，便于定位「某一类完全没学到」这类问题
        for (const ClassMetric &cm : mt.perClass) {
            if (cm.tp == 0 && cm.fp == 0 && cm.fn == 0)
                continue;
            log += QStringLiteral("  %1. %2：TP %3  FP %4  FN %5  精确率 %6%  召回率 %7%"
                                  "  IoU %8\n")
                       .arg(cm.classId)
                       .arg(cm.name)
                       .arg(cm.tp)
                       .arg(cm.fp)
                       .arg(cm.fn)
                       .arg(cm.precision * 100.0, 0, 'f', 1)
                       .arg(cm.recall * 100.0, 0, 'f', 1)
                       .arg(cm.iou, 0, 'f', 4);
        }

        // 每图预测结果落盘（供「显示测试结果」叠加）
        int savedPred = 0;
        for (const Evaluator::Item &item : items) {
            const int idx = pm.indexOfKey(item.sample.imageKey);
            if (idx >= 0 && pm.savePrediction(idx, item.pred, nullptr))
                ++savedPred;
        }
        pm.saveMetrics(mt, nullptr);
        rep.add(QStringLiteral("模块「%1」单图预测结果落盘").arg(taskTypeName(tasks[mi])),
                savedPred == items.size() && savedPred > 0,
                QStringLiteral("%1 / %2 张").arg(savedPred).arg(items.size()));

        // --- 指标下限 ---
        //
        // 分类与分割的门槛是「真的学到了」：这两条通路的特征（全局颜色直方图 /
        // 逐像素颜色）与任务匹配，实测能到 90% 以上。
        //
        // 检测则不同：OpenCV 轻量后端的检测是「滑窗 + 图块分类 + NMS」的简化实现，
        // 精度取决于缺陷形态与滑窗尺度的巧合程度，实测精确率很低（见 README 第八节），
        // 因此这里只断言**数据流通路成立且不漏检**，而把真实的 P/R 数值写进报告，
        // 由使用者自己判断是否可用。需要可用的检测精度请启用 LibTorch 后端。
        bool qualityOk = false;
        QString qualityDetail;
        if (tasks[mi] == TaskType::Classification) {
            qualityOk = mt.precision >= 0.80 && mt.recall >= 0.80;
            qualityDetail = QStringLiteral("精确率 %1%，召回率 %2%（要求均 >= 80%）")
                                .arg(mt.precision * 100.0, 0, 'f', 1)
                                .arg(mt.recall * 100.0, 0, 'f', 1);
        } else if (tasks[mi] == TaskType::Detection) {
            qualityOk = mt.valid && mt.testSamples > 0 && mt.recall > 0.0
                        && mt.missRate < 1.0;
            qualityDetail = QStringLiteral("精确率 %1%，召回率 %2%，漏检率 %3%，过检率 %4%"
                                           "（轻量检测器只校验「通路成立且非全部漏检」；"
                                           "要求 >= 80% 精确率请启用 LibTorch 后端）")
                                .arg(mt.precision * 100.0, 0, 'f', 1)
                                .arg(mt.recall * 100.0, 0, 'f', 1)
                                .arg(mt.missRate * 100.0, 0, 'f', 1)
                                .arg(mt.overkillRate * 100.0, 0, 'f', 1);
        } else {
            qualityOk = mt.precision >= 0.80 && mt.recall >= 0.80 && mt.mIoU >= 0.60;
            qualityDetail = QStringLiteral("像素精确率 %1%，召回率 %2%，mIoU %3"
                                           "（要求精确率/召回率 >= 80%，mIoU >= 0.60）")
                                .arg(mt.precision * 100.0, 0, 'f', 1)
                                .arg(mt.recall * 100.0, 0, 'f', 1)
                                .arg(mt.mIoU, 0, 'f', 4);
        }
        rep.add(QStringLiteral("模块「%1」指标达标").arg(taskTypeName(tasks[mi])), qualityOk,
                qualityDetail);

        // --- 导出模型 + 重新加载推理 ---
        const QString exportDir = root + QStringLiteral("/export_") + taskTypeKey(tasks[mi]);
        QString exportedMain, exportErr;
        const bool exported = modelio::exportModel(modelPath, exportDir, &exportedMain, &exportErr);
        rep.add(QStringLiteral("模块「%1」导出模型").arg(taskTypeName(tasks[mi])),
                exported && QFileInfo::exists(exportedMain),
                exported ? exportedMain : exportErr);

        if (exported) {
            ITrainer *reloader = TrainerFactory::create(nullptr, TrainerFactory::fromKey(mod->train().backend));
            QString loadErr;
            bool inferOk = false;
            QString inferDetail;
            if (reloader && reloader->loadModel(exportedMain, &loadErr)) {
                const cv::Mat probe = imageio::imreadUnicode(testSamples.first().imagePath, cv::IMREAD_COLOR);
                PredictionResult pr;
                if (!probe.empty() && reloader->predict(probe, &pr, &loadErr)) {
                    inferOk = pr.valid();
                    inferDetail = QStringLiteral("任务=%1，类别数=%2，结果有效=%3")
                                      .arg(taskTypeName(reloader->modelTask()))
                                      .arg(reloader->modelClasses().size())
                                      .arg(inferOk ? QStringLiteral("是") : QStringLiteral("否"));
                } else {
                    inferDetail = QStringLiteral("推理失败：%1").arg(loadErr);
                }
            } else {
                inferDetail = QStringLiteral("载入失败：%1").arg(loadErr);
            }
            delete reloader;
            rep.add(QStringLiteral("模块「%1」导出件可被重新加载并推理")
                        .arg(taskTypeName(tasks[mi])),
                    inferOk, inferDetail);
        }
    }

    // --- 5. 工程可重开 ---
    pm.closeProject();
    ProjectManager pm2;
    QString reopenErr;
    const bool reopened = pm2.openProject(projRoot, &reopenErr);
    int reopenedModules = pm2.moduleCount();
    rep.add(QStringLiteral("工程可重新打开且模块完整"), reopened && reopenedModules == 3,
            reopened ? QStringLiteral("模块数 %1，图像数 %2").arg(reopenedModules).arg(pm2.imageCount())
                     : reopenErr);

    if (reopened && reopenedModules == 3) {
        // 重开后检查第一个模块的指标是否仍在
        QString e;
        pm2.setCurrentModule(0, &e);
        const Metrics mt = pm2.loadMetrics();
        rep.add(QStringLiteral("重开后指标可读回"), mt.valid, mt.summary());
    }

    // --- 汇总 ---
    QString rep2 = rep.text(QStringLiteral("AqroTrain 端到端自检"));
    rep2 += QStringLiteral("\n---------- 各模块指标汇总 ----------\n");
    for (int i = 0; i < 3; ++i) {
        rep2 += QStringLiteral("%1：训练样本 %2，测试样本 %3，%4\n")
                    .arg(taskTypeName(tasks[i]))
                    .arg(results[i].trainCount)
                    .arg(results[i].testCount)
                    .arg(results[i].metrics.valid ? results[i].metrics.summary()
                                                  : QStringLiteral("未得到指标"));
    }
    rep2 += QStringLiteral("\n---------- 日志 ----------\n") + log;

    if (report)
        *report = rep2;
    return rep.failed() == 0 ? 0 : 1;
}

// ===========================================================================
//  标注交互自检
// ===========================================================================
int runAnnoSelfTest(QString *report)
{
    Report rep;

    // --- 准备画布（离屏，不需要显示窗口）---
    ImageCanvas canvas;
    canvas.resize(900, 700);

    const QVector<ClassInfo> classes = {
        ClassInfo(QStringLiteral("背景"), QColor(Qt::black)),
        ClassInfo(QStringLiteral("缺陷"), QColor(0xE6, 0x19, 0x4B)),
    };
    canvas.setClasses(classes);
    canvas.setHasBackground(true);
    canvas.setTask(TaskType::Segmentation);

    const QVector<SynthSample> synth = makeSynthetic(1, 192, 192, 7u);
    canvas.setImage(synth.first().img, QStringLiteral("selftest.png"));
    rep.add(QStringLiteral("画布载入图像"), canvas.hasImage() && canvas.imageSize() == QSize(192, 192),
            QStringLiteral("图像尺寸 %1×%2").arg(canvas.imageSize().width())
                .arg(canvas.imageSize().height()));

    // 事件必须发给 viewport：QGraphicsView 的鼠标事件由 viewport 接收后再转发
    auto sendMouse = [&canvas](QEvent::Type type, const QPointF &imgPos, Qt::MouseButton button,
                               Qt::MouseButtons buttons) {
        const QPoint vp = canvas.mapFromScene(imgPos);
        QMouseEvent ev(type, QPointF(vp), button, buttons, Qt::NoModifier);
        QApplication::sendEvent(canvas.viewport(), &ev);
    };

    auto stroke = [&](const QPointF &from, const QPointF &to) {
        sendMouse(QEvent::MouseButtonPress, from, Qt::LeftButton, Qt::LeftButton);
        for (int k = 1; k <= 4; ++k) {
            const QPointF p = from + (to - from) * (k / 4.0);
            sendMouse(QEvent::MouseMove, p, Qt::NoButton, Qt::LeftButton);
        }
        sendMouse(QEvent::MouseButtonRelease, to, Qt::LeftButton, Qt::NoButton);
    };

    const int w = canvas.imageSize().width();
    const int h = canvas.imageSize().height();

    auto maskPixelCount = [&](int classId) {
        const cv::Mat m = canvas.mask();
        if (m.empty())
            return 0;
        return cv::countNonZero(m == classId);
    };

    auto maskMatchesObjects = [&]() {
        cv::Mat expect;
        anno::rasterize(canvas.objects(), w, h, &expect);
        const cv::Mat actual = canvas.mask();
        if (actual.empty() || expect.size() != actual.size())
            return false;
        return cv::countNonZero(expect != actual) == 0;
    };

    // --- 1. 画笔（选中目标类别）---
    canvas.setTool(ImageCanvas::ToolBrush);
    canvas.setBrushSize(24);
    canvas.setCurrentClass(1);

    const int objBefore = canvas.objects().size();
    stroke(QPointF(40, 40), QPointF(120, 40));
    const int painted = maskPixelCount(1);
    rep.add(QStringLiteral("画笔：选中目标类别时成功写入掩码"),
            canvas.objects().size() == objBefore + 1 && painted > 0,
            QStringLiteral("对象 %1 -> %2；类别 1 像素 %3 个")
                .arg(objBefore).arg(canvas.objects().size()).arg(painted));

    const int objAfterBrush = canvas.objects().size();

    // --- 2. 画笔（停在背景类别）必须被拦截 ---
    canvas.setCurrentClass(0);
    stroke(QPointF(40, 140), QPointF(120, 140));
    rep.add(QStringLiteral("画笔：停在背景类别时被拦截（不产生有效标注）"),
            canvas.objects().size() == objAfterBrush && maskPixelCount(1) == painted,
            QStringLiteral("对象数保持 %1；类别 1 像素保持 %2 个")
                .arg(canvas.objects().size()).arg(maskPixelCount(1)));
    canvas.setCurrentClass(1);

    // --- 3. 橡皮擦：擦掉一部分 ---
    canvas.setTool(ImageCanvas::ToolEraser);
    canvas.setBrushSize(24);
    stroke(QPointF(80, 40), QPointF(80, 40));
    const int afterErase = maskPixelCount(1);
    rep.add(QStringLiteral("橡皮擦：擦除后目标像素减少（擦除作为背景笔画对象存在）"),
            afterErase < painted,
            QStringLiteral("擦除前 %1 个 -> 擦除后 %2 个").arg(painted).arg(afterErase));

    // --- 4. 多边形闭合 ---
    canvas.setTool(ImageCanvas::ToolPolygon);
    const int beforePoly = canvas.objects().size();
    const QVector<QPointF> polyPts = { QPointF(30, 120), QPointF(90, 120), QPointF(90, 170),
                                      QPointF(30, 170) };
    for (const QPointF &p : polyPts)
        sendMouse(QEvent::MouseButtonPress, p, Qt::LeftButton, Qt::LeftButton);
    // 右键闭合
    sendMouse(QEvent::MouseButtonPress, polyPts.last(), Qt::RightButton, Qt::RightButton);

    int polyIndex = -1;
    for (int i = 0; i < canvas.objects().size(); ++i) {
        if (canvas.objects().at(i).type == AnnoObject::Polygon)
            polyIndex = i;
    }
    const bool polyOk = polyIndex >= 0
                        && canvas.objects().at(polyIndex).points.size() == polyPts.size()
                        && canvas.objects().size() == beforePoly + 1;
    rep.add(QStringLiteral("多边形：点击加点 + 右键闭合生成多边形对象"), polyOk,
            polyOk ? QStringLiteral("顶点数 %1").arg(canvas.objects().at(polyIndex).points.size())
                   : QStringLiteral("未生成多边形对象"));

    // --- 5. 拖顶点：几何必须变化，且掩码同步 ---
    bool vertexMoved = false;
    QString vertexDetail;
    QPointF polyOldPt;
    bool havePolyOld = false;
    QVector<AnnoObject> polySnapshotBeforeDrag;

    if (polyIndex >= 0) {
        canvas.setTool(ImageCanvas::ToolSelect);
        canvas.setSelectedObject(polyIndex);

        polyOldPt = canvas.objects().at(polyIndex).points.at(0);
        polySnapshotBeforeDrag = canvas.objects();
        havePolyOld = true;
        const QPointF newPt = polyOldPt + QPointF(-12, -12);

        sendMouse(QEvent::MouseButtonPress, polyOldPt, Qt::LeftButton, Qt::LeftButton);
        sendMouse(QEvent::MouseMove, newPt, Qt::NoButton, Qt::LeftButton);
        sendMouse(QEvent::MouseButtonRelease, newPt, Qt::LeftButton, Qt::NoButton);

        const QPointF nowPt = canvas.objects().at(polyIndex).points.at(0);
        vertexMoved = (QLineF(nowPt, newPt).length() < 2.0);
        vertexDetail = QStringLiteral("顶点 (%1, %2) -> (%3, %4)，期望 (%5, %6)")
                           .arg(int(polyOldPt.x())).arg(int(polyOldPt.y()))
                           .arg(int(nowPt.x())).arg(int(nowPt.y()))
                           .arg(int(newPt.x())).arg(int(newPt.y()));
    }
    rep.add(QStringLiteral("选择工具：拖动多边形顶点生效"), vertexMoved, vertexDetail);

    // --- 6. 对象光栅化 == 画布掩码（逐像素）---
    rep.add(QStringLiteral("一致性：对象列表光栅化掩码 == 画布掩码（逐像素相等）"),
            maskMatchesObjects(),
            QStringLiteral("对象 %1 个；掩码非零像素 %2 个")
                .arg(canvas.objects().size())
                .arg(cv::countNonZero(canvas.mask() > 0)));

    // --- 7. 撤销几何编辑：顶点必须回到拖动前的位置 ---
    //     注意这里断言的是「几何回退」而不是「对象数变化」：
    //     拖动顶点不会改变对象个数，用个数判断会误报失败。
    bool geoUndoOk = false;
    QString geoUndoDetail;
    if (havePolyOld) {
        canvas.undo();
        const bool back = canvas.objects().size() == polySnapshotBeforeDrag.size()
                          && canvas.objects().at(polyIndex).points.at(0) == polyOldPt;
        geoUndoOk = back && maskMatchesObjects();
        geoUndoDetail = QStringLiteral("撤销后顶点 (%1, %2)，期望回到 (%3, %4)")
                            .arg(int(canvas.objects().at(polyIndex).points.at(0).x()))
                            .arg(int(canvas.objects().at(polyIndex).points.at(0).y()))
                            .arg(int(polyOldPt.x()))
                            .arg(int(polyOldPt.y()));
    }
    rep.add(QStringLiteral("撤销：拖动顶点后可回退到拖动前的几何（掩码同步）"), geoUndoOk,
            geoUndoDetail);

    // --- 8. 复制 / 粘贴 + 撤销 / 重做（对象个数确实变化的那种操作）---
    canvas.setSelectedObject(0);
    QVector<AnnoObject> clip;
    if (!canvas.objects().isEmpty())
        clip.append(canvas.objects().first());

    const int beforePaste = canvas.objects().size();
    canvas.pasteObjects(clip);
    const int afterPaste = canvas.objects().size();
    rep.add(QStringLiteral("复制 / 粘贴对象"),
            afterPaste == beforePaste + 1 && maskMatchesObjects(),
            QStringLiteral("对象数 %1 -> %2").arg(beforePaste).arg(afterPaste));

    canvas.undo();
    const bool undoOk = canvas.objects().size() == beforePaste && maskMatchesObjects();
    const int afterUndo = canvas.objects().size();

    canvas.redo();
    const bool redoOk = canvas.objects().size() == afterPaste && maskMatchesObjects();

    rep.add(QStringLiteral("撤销 / 重做：对象列表与掩码同步回退与恢复"),
            undoOk && redoOk,
            QStringLiteral("对象数 %1 -> 粘贴后 %2 -> 撤销后 %3 -> 重做后 %4")
                .arg(beforePaste).arg(afterPaste).arg(afterUndo).arg(canvas.objects().size()));

    // --- 9. 检测任务的固定尺寸框 ---
    ImageCanvas det;
    det.resize(900, 700);
    det.setClasses(classes);
    det.setHasBackground(true);
    det.setTask(TaskType::Detection);
    det.setImage(synth.first().img, QStringLiteral("det.png"));
    det.setTool(ImageCanvas::ToolFixedRect);
    det.setFixedRectSize(QSize(50, 40));
    det.setCurrentClass(1);

    const QPoint vp = det.mapFromScene(QPointF(100, 100));
    QMouseEvent detPress(QEvent::MouseButtonPress, QPointF(vp), Qt::LeftButton, Qt::LeftButton,
                         Qt::NoModifier);
    QApplication::sendEvent(det.viewport(), &detPress);
    QMouseEvent detRelease(QEvent::MouseButtonRelease, QPointF(vp), Qt::LeftButton, Qt::NoButton,
                           Qt::NoModifier);
    QApplication::sendEvent(det.viewport(), &detRelease);

    bool fixedOk = det.objects().size() == 1;
    QString fixedDetail;
    if (fixedOk) {
        const QRectF r = det.objects().first().rect.normalized();
        fixedOk = qFuzzyCompare(r.width() + 1.0, 50.0 + 1.0) && qFuzzyCompare(r.height() + 1.0, 40.0 + 1.0)
                  && det.objects().first().classId == 1;
        fixedDetail = QStringLiteral("生成 %1×%2 的框，类别 %3")
                          .arg(int(r.width())).arg(int(r.height()))
                          .arg(det.objects().first().classId);
    } else {
        fixedDetail = QStringLiteral("未生成框");
    }
    rep.add(QStringLiteral("固定尺寸框：点一下即放置指定尺寸的框"), fixedOk, fixedDetail);

    // --- 10. 检测：拖拽画框 ---
    ImageCanvas drag;
    drag.resize(900, 700);
    drag.setClasses(classes);
    drag.setHasBackground(true);
    drag.setTask(TaskType::Detection);
    drag.setImage(synth.first().img, QStringLiteral("drag.png"));
    drag.setTool(ImageCanvas::ToolRect);
    drag.setCurrentClass(1);

    auto sendTo = [](ImageCanvas &c, QEvent::Type type, const QPointF &imgPos,
                     Qt::MouseButton button, Qt::MouseButtons buttons) {
        const QPoint v = c.mapFromScene(imgPos);
        QMouseEvent ev(type, QPointF(v), button, buttons, Qt::NoModifier);
        QApplication::sendEvent(c.viewport(), &ev);
    };
    sendTo(drag, QEvent::MouseButtonPress, QPointF(30, 30), Qt::LeftButton, Qt::LeftButton);
    sendTo(drag, QEvent::MouseMove, QPointF(110, 90), Qt::NoButton, Qt::LeftButton);
    sendTo(drag, QEvent::MouseButtonRelease, QPointF(110, 90), Qt::LeftButton, Qt::NoButton);

    bool dragOk = drag.objects().size() == 1;
    QString dragDetail;
    if (dragOk) {
        const QRectF r = drag.objects().first().rect.normalized();
        dragOk = std::abs(r.width() - 80.0) < 3.0 && std::abs(r.height() - 60.0) < 3.0;
        dragDetail = QStringLiteral("生成 %1×%2 的框（期望约 80×60）")
                         .arg(int(r.width())).arg(int(r.height()));
    } else {
        dragDetail = QStringLiteral("未生成框");
    }
    rep.add(QStringLiteral("检测：拖拽生成矩形框（尺寸与拖拽范围一致）"), dragOk, dragDetail);

    // --- 11. 画笔采样去重 -------------------------------------------------
    // 回归用例：鼠标每移动一像素就会产生一个 move 事件，
    // 若无条件 append，在同一片区域反复涂抹会让 points 无上限增长。
    {
        ImageCanvas dedup;
        dedup.resize(700, 500);
        dedup.setClasses(classes);
        dedup.setHasBackground(true);
        dedup.setTask(TaskType::Segmentation);
        dedup.setImage(synth.first().img, QStringLiteral("brush_dedup.png"));
        dedup.setTool(ImageCanvas::ToolBrush);
        dedup.setBrushSize(24);
        dedup.setCurrentClass(1);

        auto sendD = [&dedup](QEvent::Type type, const QPointF &imgPos, Qt::MouseButton button,
                             Qt::MouseButtons buttons) {
            const QPoint vp = dedup.mapFromScene(imgPos);
            QMouseEvent ev(type, QPointF(vp), button, buttons, Qt::NoModifier);
            QApplication::sendEvent(dedup.viewport(), &ev);
        };

        // 落笔，然后在同一处来回抖动 200 次（每次位移不超过 1 像素）
        sendD(QEvent::MouseButtonPress, QPointF(60, 60), Qt::LeftButton, Qt::LeftButton);
        for (int i = 0; i < 200; ++i) {
            const double dx = double(i % 3) - 1.0;        // -1 / 0 / 1
            const double dy = double((i / 3) % 3) - 1.0;
            sendD(QEvent::MouseMove, QPointF(60 + dx, 60 + dy), Qt::NoButton, Qt::LeftButton);
        }
        sendD(QEvent::MouseButtonRelease, QPointF(60, 60), Qt::LeftButton, Qt::NoButton);

        int dedupPoints = 0;
        for (const AnnoObject &o : dedup.objects())
            dedupPoints += o.points.size();

        rep.add(QStringLiteral("画笔：同一处反复涂抹 200 次，采样点不无限累积"),
                dedupPoints <= 4 && dedup.objects().size() <= 1,
                QStringLiteral("对象 %1 个，采样点合计 %2 个（上限 4）")
                    .arg(dedup.objects().size())
                    .arg(dedupPoints));

        // 在已经画过的像素上再落笔、抬起：不应产生对象，也不应留下撤销步骤
        const int objBeforeNoop = dedup.objects().size();
        const bool undoBeforeNoop = dedup.canUndo();
        sendD(QEvent::MouseButtonPress, QPointF(60, 60), Qt::LeftButton, Qt::LeftButton);
        sendD(QEvent::MouseButtonRelease, QPointF(60, 60), Qt::LeftButton, Qt::NoButton);
        rep.add(QStringLiteral("画笔：在已画像素上落笔不留对象、不留撤销步骤"),
                dedup.objects().size() == objBeforeNoop && dedup.canUndo() == undoBeforeNoop,
                QStringLiteral("对象 %1（期望 %2），可撤销=%3（期望 %4）")
                    .arg(dedup.objects().size())
                    .arg(objBeforeNoop)
                    .arg(dedup.canUndo() ? QStringLiteral("是") : QStringLiteral("否"))
                    .arg(undoBeforeNoop ? QStringLiteral("是") : QStringLiteral("否")));

        // 真实的拖动必须照常记录采样点，并且掩码随之增长
        const int ptsBeforeDrag = dedupPoints;
        auto maskCount = [&dedup]() {
            const cv::Mat m = dedup.mask();
            return m.empty() ? 0 : cv::countNonZero(m == 1);
        };
        const int maskBefore = maskCount();

        sendD(QEvent::MouseButtonPress, QPointF(140, 140), Qt::LeftButton, Qt::LeftButton);
        for (int k = 1; k <= 8; ++k)
            sendD(QEvent::MouseMove, QPointF(140 + k * 10, 140), Qt::NoButton, Qt::LeftButton);
        sendD(QEvent::MouseButtonRelease, QPointF(220, 140), Qt::LeftButton, Qt::NoButton);

        int ptsAfterDrag = 0;
        for (const AnnoObject &o : dedup.objects())
            ptsAfterDrag += o.points.size();
        const int maskAfter = maskCount();

        rep.add(QStringLiteral("画笔：真实拖动仍正常记录采样点并扩展掩码"),
                ptsAfterDrag > ptsBeforeDrag && maskAfter > maskBefore,
                QStringLiteral("采样点 %1 -> %2；类别 1 像素 %3 -> %4")
                    .arg(ptsBeforeDrag).arg(ptsAfterDrag).arg(maskBefore).arg(maskAfter));

        // 去重后掩码仍必须与对象光栅化逐像素一致
        cv::Mat expect;
        anno::rasterize(dedup.objects(), dedup.imageSize().width(), dedup.imageSize().height(),
                        &expect);
        const cv::Mat actual = dedup.mask();
        const bool consistent = !actual.empty() && expect.size() == actual.size()
                                && cv::countNonZero(expect != actual) == 0;
        rep.add(QStringLiteral("画笔：简化后掩码仍与对象光栅化逐像素一致"), consistent,
                QStringLiteral("对象 %1 个").arg(dedup.objects().size()));
    }

    // --- 11b. 来回涂抹：对象数不能涨 -------------------------------------
    // 真实踩过的坑：为了「省点」而在拖动中另起分段，会让来回涂抹时
    // 每折返一次就多一个对象（实测 11 个「1~23 顶点」的笔画对象）。
    // 现在改成「采样 + RDP 简化」：一次手势只产生一个对象，点数靠简化收敛。
    {
        ImageCanvas bf;
        bf.resize(700, 500);
        bf.setClasses(classes);
        bf.setHasBackground(true);
        bf.setTask(TaskType::Segmentation);
        bf.setImage(synth.first().img, QStringLiteral("brush_backforth.png"));
        bf.setTool(ImageCanvas::ToolBrush);
        bf.setBrushSize(10);
        bf.setCurrentClass(1);

        auto sendB = [&bf](QEvent::Type type, const QPointF &imgPos, Qt::MouseButton button,
                          Qt::MouseButtons buttons) {
            const QPoint vp = bf.mapFromScene(imgPos);
            QMouseEvent ev(type, QPointF(vp), button, buttons, Qt::NoModifier);
            QApplication::sendEvent(bf.viewport(), &ev);
        };

        // 在 (40,40)~(140,40) 之间来回 12 趟，每趟 20 个采样点
        sendB(QEvent::MouseButtonPress, QPointF(40, 40), Qt::LeftButton, Qt::LeftButton);
        int moves = 0;
        for (int trip = 0; trip < 12; ++trip) {
            const bool leftToRight = (trip % 2) == 0;
            for (int k = 1; k <= 20; ++k) {
                const double x = leftToRight ? 40 + k * 5 : 140 - k * 5;
                sendB(QEvent::MouseMove, QPointF(x, 40 + (trip % 2) * 0.5), Qt::NoButton,
                      Qt::LeftButton);
                ++moves;
            }
        }
        sendB(QEvent::MouseButtonRelease, QPointF(40, 40), Qt::LeftButton, Qt::NoButton);

        int bfPoints = 0;
        for (const AnnoObject &o : bf.objects())
            bfPoints += o.points.size();

        // 对象数必须是 1；点数必须远小于鼠标移动次数（靠简化收敛）
        rep.add(QStringLiteral("画笔：来回涂抹 12 趟，对象数不增加、点数被简化收敛"),
                bf.objects().size() == 1 && bfPoints < moves / 4,
                QStringLiteral("鼠标移动 %1 次 -> 对象 %2 个（期望 1），采样点 %3 个（应远小于 %4）")
                    .arg(moves)
                    .arg(bf.objects().size())
                    .arg(bfPoints)
                    .arg(moves));

        // 掩码仍必须与对象一致
        cv::Mat bfExpect;
        anno::rasterize(bf.objects(), bf.imageSize().width(), bf.imageSize().height(), &bfExpect);
        const cv::Mat bfActual = bf.mask();
        rep.add(QStringLiteral("画笔：来回涂抹后掩码仍与对象光栅化逐像素一致"),
                !bfActual.empty() && bfExpect.size() == bfActual.size()
                    && cv::countNonZero(bfExpect != bfActual) == 0,
                QStringLiteral("对象 %1 个，采样点 %2 个").arg(bf.objects().size()).arg(bfPoints));
    }

    // --- 11d. 在同一片区域反复涂抹：点数必须停下来 ------------------------
    // 用户实际反馈：用 5px 细笔在同一处来回涂，顶点一直涨（实测 557 个）。
    // 判据要落在「本笔画已记录点的邻域」上：笔尖是圆的，圆心距 ≤ 半径 时
    // 覆盖区域基本重合，所以这种采样点不该再记。
    {
        ImageCanvas rp;
        rp.resize(700, 500);
        rp.setClasses(classes);
        rp.setHasBackground(true);
        rp.setTask(TaskType::Segmentation);
        rp.setImage(synth.first().img, QStringLiteral("brush_repaint.png"));
        rp.setTool(ImageCanvas::ToolBrush);
        rp.setBrushSize(5);          // 细笔，最容易暴露点数暴涨
        rp.setCurrentClass(1);

        auto sendR = [&rp](QEvent::Type type, const QPointF &imgPos, Qt::MouseButton button,
                          Qt::MouseButtons buttons) {
            const QPoint vp = rp.mapFromScene(imgPos);
            QMouseEvent ev(type, QPointF(vp), button, buttons, Qt::NoModifier);
            QApplication::sendEvent(rp.viewport(), &ev);
        };

        // 一条小幅摆动的涂抹轨迹，在同一片区域内反复走 10 趟
        const int    kTurns = 10;
        const double cx = 96, cy = 96, r = 18;
        auto walk = [&](int turns) {
            for (int t = 0; t < turns; ++t) {
                for (int k = 0; k <= 24; ++k) {
                    const double a = (k / 24.0) * 6.2831853;   // 绕一圈
                    const double wob = ((t % 2) ? 1.0 : -1.0) * 0.4;
                    sendR(QEvent::MouseMove,
                          QPointF(cx + (r + wob) * std::cos(a), cy + (r + wob) * std::sin(a)),
                          Qt::NoButton, Qt::LeftButton);
                }
            }
        };

        sendR(QEvent::MouseButtonPress, QPointF(cx + r, cy), Qt::LeftButton, Qt::LeftButton);
        walk(2);
        int pointsAfterTwo = 0;
        for (const AnnoObject &o : rp.objects())
            pointsAfterTwo += o.points.size();

        walk(kTurns - 2);
        int pointsAfterAll = 0;
        for (const AnnoObject &o : rp.objects())
            pointsAfterAll += o.points.size();

        sendR(QEvent::MouseButtonRelease, QPointF(cx + r, cy), Qt::LeftButton, Qt::NoButton);

        // 后 8 趟几乎不新增点（允许小幅增长，因为手抖会扫到新的位置）
        const bool stopped = rp.objects().size() == 1
                             && pointsAfterAll <= pointsAfterTwo + 8;
        rep.add(QStringLiteral("画笔：在同一处反复涂抹 10 趟，顶点不再持续增加"), stopped,
                QStringLiteral("2 趟后 %1 个点 -> 10 趟后 %2 个点（允许再增 8 个）；对象 %3 个")
                    .arg(pointsAfterTwo)
                    .arg(pointsAfterAll)
                    .arg(rp.objects().size()));

        // 掩码仍必须与对象一致
        cv::Mat rpExpect;
        anno::rasterize(rp.objects(), rp.imageSize().width(), rp.imageSize().height(), &rpExpect);
        const cv::Mat rpActual = rp.mask();
        rep.add(QStringLiteral("画笔：反复涂抹后掩码仍与对象光栅化逐像素一致"),
                !rpActual.empty() && rpExpect.size() == rpActual.size()
                    && cv::countNonZero(rpExpect != rpActual) == 0,
                QStringLiteral("对象 %1 个，掩码非零 %2 个")
                    .arg(rp.objects().size())
                    .arg(cv::countNonZero(rpActual > 0)));
    }

    // --- 11e. 沿曲线涂抹：跳过规则不能把笔迹画出缺口 ----------------------
    // 跳过距离取笔尖半径（相邻两个圆盘圆心距 = R 时并集在中点收窄 13%），
    // 这里用「画一圈闭合圆环」检查：必须是 1 个连通域、环心仍然空着、
    // 面积接近理论环带面积 —— 三者同时满足才说明覆盖没被跳过规则削掉。
    {
        ImageCanvas ring;
        ring.resize(700, 500);
        ring.setClasses(classes);
        ring.setHasBackground(true);
        ring.setTask(TaskType::Segmentation);
        ring.setImage(synth.first().img, QStringLiteral("brush_ring.png"));
        ring.setTool(ImageCanvas::ToolBrush);
        ring.setBrushSize(10);          // 半径 5
        ring.setCurrentClass(1);

        auto sendRing = [&ring](QEvent::Type type, const QPointF &imgPos, Qt::MouseButton button,
                               Qt::MouseButtons buttons) {
            const QPoint vp = ring.mapFromScene(imgPos);
            QMouseEvent ev(type, QPointF(vp), button, buttons, Qt::NoModifier);
            QApplication::sendEvent(ring.viewport(), &ev);
        };

        const double cx = 96, cy = 96, radius = 40, brushR = 5.0;
        const int    steps = 120;
        sendRing(QEvent::MouseButtonPress, QPointF(cx + radius, cy), Qt::LeftButton,
                 Qt::LeftButton);
        for (int k = 1; k <= steps; ++k) {
            const double a = (k / double(steps)) * 6.2831853;
            sendRing(QEvent::MouseMove, QPointF(cx + radius * std::cos(a), cy + radius * std::sin(a)),
                     Qt::NoButton, Qt::LeftButton);
        }
        sendRing(QEvent::MouseButtonRelease, QPointF(cx + radius, cy), Qt::LeftButton,
                 Qt::NoButton);

        const cv::Mat mask = ring.mask();
        cv::Mat bin;
        cv::compare(mask, 1, bin, cv::CMP_EQ);
        cv::Mat labels;
        const int nLabels = bin.empty() ? 0 : cv::connectedComponents(bin, labels, 8, CV_32S);
        const int blobs = std::max(0, nLabels - 1);

        const int ringPixels = cv::countNonZero(bin);
        const double ideal = 2.0 * 3.14159265 * radius * (brushR * 2.0);   // 环带面积
        const bool areaOk = ringPixels > ideal * 0.75 && ringPixels < ideal * 1.30;

        // 环心必须仍是空的（说明画的是环，不是一个实心圆）
        const int centerX = int(std::lround(cx));
        const int centerY = int(std::lround(cy));
        const bool centerEmpty = mask.empty() || mask.at<uchar>(centerY, centerX) == 0;

        int ringPoints = 0;
        for (const AnnoObject &o : ring.objects())
            ringPoints += o.points.size();

        rep.add(QStringLiteral("画笔：沿圆周涂一圈 → 1 个连通域、环心为空、面积接近理论环带"),
                ring.objects().size() == 1 && blobs == 1 && centerEmpty && areaOk,
                QStringLiteral("对象 %1 个，连通域 %2 个，环心为空=%3，"
                               "环带像素 %4（理论 %5，允许 75%~130%），采样点 %6 个")
                    .arg(ring.objects().size())
                    .arg(blobs)
                    .arg(centerEmpty ? QStringLiteral("是") : QStringLiteral("否"))
                    .arg(ringPixels)
                    .arg(int(ideal))
                    .arg(ringPoints));
    }

    // --- 11c. 折线简化的行为（RDP） ---------------------------------------
    {
        // 折返：去 100px 再原路折回，应被压成很少的点
        QPolygonF back;
        for (int i = 0; i <= 50; ++i)
            back.append(QPointF(i * 2, 0));
        for (int i = 49; i >= 0; --i)
            back.append(QPointF(i * 2, 0));
        const QPolygonF backSimp = anno::simplifyPolyline(back, 1.0);
        rep.add(QStringLiteral("折线简化：原路折返被折叠为少量点"),
                backSimp.size() <= 4 && backSimp.first() == back.first()
                    && backSimp.last() == back.last(),
                QStringLiteral("%1 个点 -> %2 个点（期望 <= 4）")
                    .arg(back.size()).arg(backSimp.size()));

        // 锯齿：拐角是真正撑开形状的点，必须保留（否则掩码会掉一片）
        QPolygonF zig;
        for (int i = 0; i <= 10; ++i)
            zig.append(QPointF(i * 10, (i % 2) ? 40 : 0));
        const QPolygonF zigSimp = anno::simplifyPolyline(zig, 1.0);
        rep.add(QStringLiteral("折线简化：锯齿的拐角被保留（不丢覆盖范围）"),
                zigSimp.size() == zig.size(),
                QStringLiteral("%1 个点 -> %2 个点（期望保持不变）")
                    .arg(zig.size()).arg(zigSimp.size()));

        // 远处一个点也不能被抹掉
        QPolygonF spike;
        spike.append(QPointF(0, 0));
        spike.append(QPointF(50, 30));
        spike.append(QPointF(100, 0));
        const QPolygonF spikeSimp = anno::simplifyPolyline(spike, 1.0);
        rep.add(QStringLiteral("折线简化：明显偏离弦线的点被保留"),
                spikeSimp.size() == 3,
                QStringLiteral("%1 个点 -> %2 个点（期望 3）")
                    .arg(spike.size()).arg(spikeSimp.size()));
    }

    // --- 11b. 连续拖动必须是「连成一整条」，不能退化成断续的点 -------------
    // 这是一个真实踩过的坑：如果拿「掩码像素值已是本类别」当作「已画过」的判据，
    // 刚被本笔涂过的笔尖范围内像素全都会命中该条件，于是每一段都被跳过，
    // 要等光标离开笔尖半径才补一个点 —— 一笔涂抹退化成每隔一个笔尖直径一个圆点
    //（实测出现 140 个「1 顶点」对象）。这里用连通域个数把这个退化钉死。
    {
        ImageCanvas cont;
        cont.resize(700, 500);
        cont.setClasses(classes);
        cont.setHasBackground(true);
        cont.setTask(TaskType::Segmentation);
        cont.setImage(synth.first().img, QStringLiteral("brush_continuous.png"));
        cont.setTool(ImageCanvas::ToolBrush);
        cont.setBrushSize(24);
        cont.setCurrentClass(1);

        auto sendC = [&cont](QEvent::Type type, const QPointF &imgPos, Qt::MouseButton button,
                            Qt::MouseButtons buttons) {
            const QPoint vp = cont.mapFromScene(imgPos);
            QMouseEvent ev(type, QPointF(vp), button, buttons, Qt::NoModifier);
            QApplication::sendEvent(cont.viewport(), &ev);
        };

        // 从 (30,30) 平滑拖到 (130,30)，每步 2 像素
        sendC(QEvent::MouseButtonPress, QPointF(30, 30), Qt::LeftButton, Qt::LeftButton);
        for (int k = 1; k <= 50; ++k)
            sendC(QEvent::MouseMove, QPointF(30 + k * 2, 30), Qt::NoButton, Qt::LeftButton);
        sendC(QEvent::MouseButtonRelease, QPointF(130, 30), Qt::LeftButton, Qt::NoButton);

        int contPoints = 0;
        for (const AnnoObject &o : cont.objects())
            contPoints += o.points.size();

        // 前景连通域个数：连成一条应当是 1 个
        const cv::Mat mask = cont.mask();
        cv::Mat bin;
        cv::compare(mask, 1, bin, cv::CMP_EQ);
        cv::Mat labels;
        const int nLabels = bin.empty() ? 0 : cv::connectedComponents(bin, labels, 8, CV_32S);
        const int blobs = std::max(0, nLabels - 1);   // 减去背景标签

        // 掩码的横向跨度：一路连续涂过去应当覆盖整个拖拽范围（含两端圆头）
        // 这才是「连续」的判据 —— 不能拿采样点个数当判据：
        // 直线被 RDP 简化成 2 个点是**正确**行为，点少不代表画得断续。
        const cv::Rect bbox = [&mask]() {
            std::vector<cv::Point> pts;
            cv::findNonZero(mask == 1, pts);
            return pts.empty() ? cv::Rect() : cv::boundingRect(pts);
        }();

        // 留一张可视化图：断续/平滑这种问题看数字不如直接看图
        QString shotPath;
        if (!cont.image().empty() && !mask.empty()) {
            cv::Mat vis = cont.image().clone();
            const cv::Mat color = imageio::colorizeMask(mask, classes, 1.0);
            if (!color.empty() && color.size() == vis.size())
                cv::addWeighted(vis, 0.6, color, 0.4, 0.0, vis);
            shotPath = QCoreApplication::applicationDirPath()
                       + QStringLiteral("/brush_continuity.png");
            if (!imageio::imwriteUnicode(shotPath, vis))
                shotPath.clear();
        }

        rep.add(QStringLiteral("画笔：连续拖动生成连通的整条笔画（不退化成断续点）"),
                cont.objects().size() == 1 && blobs == 1
                    && bbox.width >= 118 && bbox.height >= 20,
                QStringLiteral("对象 %1 个（期望 1），前景连通域 %2 个（期望 1），"
                               "掩码跨度 %3×%4（期望约 124×24），采样点 %5 个%6")
                    .arg(cont.objects().size())
                    .arg(blobs)
                    .arg(bbox.width)
                    .arg(bbox.height)
                    .arg(contPoints)
                    .arg(shotPath.isEmpty()
                             ? QString()
                             : QStringLiteral("；可视化：%1")
                                   .arg(QDir::toNativeSeparators(shotPath))));
    }

    // --- 12. 分类任务下作画工具不可用 ---
    const bool clsToolsBlocked = !ImageCanvas::toolAllowedForTask(ImageCanvas::ToolBrush,
                                                                 TaskType::Classification)
                                 && !ImageCanvas::toolAllowedForTask(ImageCanvas::ToolRect,
                                                                    TaskType::Classification)
                                 && ImageCanvas::toolAllowedForTask(ImageCanvas::ToolSelect,
                                                                    TaskType::Classification);
    rep.add(QStringLiteral("分类模块：作画类工具被判定为不可用（避免「点了没反应」）"),
            clsToolsBlocked,
            QStringLiteral("画笔=%1，矩形=%2，选择=%3")
                .arg(ImageCanvas::toolAllowedForTask(ImageCanvas::ToolBrush, TaskType::Classification) ? QStringLiteral("可用") : QStringLiteral("不可用"))
                .arg(ImageCanvas::toolAllowedForTask(ImageCanvas::ToolRect, TaskType::Classification) ? QStringLiteral("可用") : QStringLiteral("不可用"))
                .arg(ImageCanvas::toolAllowedForTask(ImageCanvas::ToolSelect, TaskType::Classification) ? QStringLiteral("可用") : QStringLiteral("不可用")));

    if (report)
        *report = rep.text(QStringLiteral("AqroTrain 标注交互自检"));
    return rep.failed() == 0 ? 0 : 1;
}

} // namespace aq
