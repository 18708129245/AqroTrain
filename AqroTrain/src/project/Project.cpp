// ============================================================================
//  Project.cpp —— 工程模型的序列化与路径约定
// ============================================================================
#include "project/Project.h"

#include <QDateTime>
#include <QDir>
#include <QJsonArray>

#include "core/JsonStore.h"

namespace aq {

namespace {
const char *kProjectFile = "project.json";
const char *kFormatTag   = "AqroTrain.Project";
const char *kImagesDir   = "images";
const char *kThumbsDir   = "thumbs";
const char *kModulesDir  = "modules";
} // namespace

QString Project::fileName()  { return QString::fromLatin1(kProjectFile); }
QString Project::formatTag() { return QString::fromLatin1(kFormatTag); }

// ---------------------------------------------------------------------------
//  模块引用
// ---------------------------------------------------------------------------
void Project::appendModuleRef(const ModuleRef &ref)
{
    m_modules.append(ref);
    renumberModuleRefs();
}

void Project::removeModuleRefsFrom(int index)
{
    if (index < 0 || index >= m_modules.size())
        return;
    m_modules.remove(index, m_modules.size() - index);
    renumberModuleRefs();
}

void Project::renumberModuleRefs()
{
    for (int i = 0; i < m_modules.size(); ++i)
        m_modules[i].order = i;
}

// ---------------------------------------------------------------------------
//  路径
// ---------------------------------------------------------------------------
QString Project::filePath() const
{
    return m_root + QLatin1Char('/') + fileName();
}

QString Project::imagesDir() const
{
    return m_root + QLatin1Char('/') + QLatin1String(kImagesDir);
}

QString Project::thumbsDir() const
{
    return m_root + QLatin1Char('/') + QLatin1String(kThumbsDir);
}

QString Project::modulesDir() const
{
    return m_root + QLatin1Char('/') + QLatin1String(kModulesDir);
}

QString Project::moduleDir(const QString &moduleId) const
{
    return modulesDir() + QLatin1Char('/') + moduleId;
}

// ---------------------------------------------------------------------------
//  序列化
// ---------------------------------------------------------------------------
QJsonObject Project::toJson() const
{
    QJsonObject o;
    o.insert(QStringLiteral("format"), formatTag());
    o.insert(QStringLiteral("version"), 1);
    o.insert(QStringLiteral("name"), m_name);
    o.insert(QStringLiteral("imageFormat"), imageFormatKey(m_imageFormat));
    o.insert(QStringLiteral("created"),
             m_created.isEmpty() ? QDateTime::currentDateTime().toString(Qt::ISODate) : m_created);
    o.insert(QStringLiteral("description"), m_description);

    QJsonArray arr;
    for (const ModuleRef &r : m_modules) {
        QJsonObject m;
        m.insert(QStringLiteral("id"), r.id);
        m.insert(QStringLiteral("name"), r.name);
        m.insert(QStringLiteral("type"), taskTypeKey(r.type));
        m.insert(QStringLiteral("order"), r.order);
        arr.append(m);
    }
    o.insert(QStringLiteral("modules"), arr);
    return o;
}

Project Project::fromJson(const QJsonObject &o, const QString &root)
{
    Project p;
    p.m_root        = root;
    p.m_name        = jsonio::str(o, QStringLiteral("name"), QDir(root).dirName());
    p.m_created     = jsonio::str(o, QStringLiteral("created"));
    p.m_description = jsonio::str(o, QStringLiteral("description"));

    bool ok = false;
    p.m_imageFormat = imageFormatFromKey(jsonio::str(o, QStringLiteral("imageFormat"), QStringLiteral("png")), &ok);
    if (!ok)
        p.m_imageFormat = ImageFormat::PNG;

    const QJsonArray arr = o.value(QStringLiteral("modules")).toArray();
    for (int i = 0; i < arr.size(); ++i) {
        const QJsonObject m = arr.at(i).toObject();
        ModuleRef r;
        r.id    = jsonio::str(m, QStringLiteral("id"));
        r.name  = jsonio::str(m, QStringLiteral("name"));
        r.type  = taskTypeFromKey(jsonio::str(m, QStringLiteral("type")), &ok);
        r.order = jsonio::integer(m, QStringLiteral("order"), i);
        if (r.id.isEmpty())
            continue;
        p.m_modules.append(r);
    }
    p.renumberModuleRefs();
    return p;
}

bool Project::save(QString *err) const
{
    return jsonio::writeObject(filePath(), toJson(), err);
}

bool Project::load(const QString &root, Project *out, QString *err)
{
    QJsonObject o;
    const QString path = root + QLatin1Char('/') + fileName();
    if (!jsonio::readObject(path, &o, err))
        return false;

    const QString fmt = jsonio::str(o, QStringLiteral("format"));
    if (fmt != formatTag()) {
        if (err) {
            *err = QStringLiteral("该目录不是 AqroTrain 工程：\n%1\n（format = \"%2\"，期望 \"%3\"）")
                       .arg(QDir::toNativeSeparators(path), fmt, formatTag());
        }
        return false;
    }

    if (out)
        *out = fromJson(o, root);
    return true;
}

} // namespace aq
