// ============================================================================
//  Module.cpp —— 模块模型的序列化与路径约定
// ============================================================================
#include "project/Module.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>

#include "core/JsonStore.h"

namespace aq {

namespace {
const char *kModuleFile  = "module.json";
const char *kSamplesFile = "samples.json";
const char *kLabelsDir   = "labels";
const char *kModelsDir   = "models";
const char *kResultsDir  = "results";
const char *kLogsDir     = "logs";
const char *kModelStem   = "model";
} // namespace

QString Module::fileName()        { return QString::fromLatin1(kModuleFile); }
QString Module::samplesFileName() { return QString::fromLatin1(kSamplesFile); }
QString Module::labelsDirName()   { return QString::fromLatin1(kLabelsDir); }
QString Module::modelsDirName()   { return QString::fromLatin1(kModelsDir); }
QString Module::resultsDirName()  { return QString::fromLatin1(kResultsDir); }
QString Module::logsDirName()     { return QString::fromLatin1(kLogsDir); }
QString Module::backgroundClassName() { return QStringLiteral("背景"); }

// ---------------------------------------------------------------------------
//  类别
// ---------------------------------------------------------------------------
int Module::targetClassCount() const
{
    if (m_classes.isEmpty())
        return 0;
    return hasBackground() ? m_classes.size() - 1 : m_classes.size();
}

bool Module::isTargetClass(int classId) const
{
    if (classId < 0 || classId >= m_classes.size())
        return false;
    if (hasBackground() && classId == 0)
        return false;
    return true;
}

QString Module::className(int classId) const
{
    if (classId < 0 || classId >= m_classes.size())
        return QString();
    return m_classes[classId].name;
}

QColor Module::classColor(int classId) const
{
    if (classId < 0 || classId >= m_classes.size())
        return defaultClassColor(classId < 0 ? 0 : classId);
    return m_classes[classId].color;
}

void Module::ensureDefaultClasses()
{
    if (hasBackground()) {
        if (m_classes.isEmpty()) {
            m_classes.append(ClassInfo(backgroundClassName(), QColor(QStringLiteral("#000000"))));
        } else if (m_classes.first().name != backgroundClassName()) {
            // 索引 0 必须是背景：保留其名字位置，强制写上背景
            m_classes[0].name  = backgroundClassName();
            m_classes[0].color = QColor(QStringLiteral("#000000"));
        }
        // 检测 / 分割至少要有一个目标类别才能标注
        if (m_classes.size() < 2)
            m_classes.append(ClassInfo(QStringLiteral("缺陷1"), defaultClassColor(0)));
    } else {
        if (m_classes.isEmpty())
            m_classes.append(ClassInfo(QStringLiteral("OK"), defaultClassColor(0)));
        if (m_classes.size() < 2)
            m_classes.append(ClassInfo(QStringLiteral("NG"), defaultClassColor(1)));
    }
}

// ---------------------------------------------------------------------------
//  路径
// ---------------------------------------------------------------------------
QString Module::filePath() const
{
    return m_root + QLatin1Char('/') + fileName();
}

QString Module::samplesFilePath() const
{
    return m_root + QLatin1Char('/') + samplesFileName();
}

QString Module::labelsDir() const  { return m_root + QLatin1Char('/') + QLatin1String(kLabelsDir); }
QString Module::modelsDir() const  { return m_root + QLatin1Char('/') + QLatin1String(kModelsDir); }
QString Module::resultsDir() const { return m_root + QLatin1Char('/') + QLatin1String(kResultsDir); }
QString Module::logsDir() const    { return m_root + QLatin1Char('/') + QLatin1String(kLogsDir); }

QString Module::modelPath() const
{
    return modelsDir() + QLatin1Char('/') + QLatin1String(kModelStem) + QLatin1Char('.')
           + modelFileSuffix(m_type);
}

QString Module::modelMetaPath() const
{
    return modelPath() + QStringLiteral(".json");
}

QString Module::metricsPath() const
{
    return resultsDir() + QLatin1Char('/') + QStringLiteral("metrics.json");
}

QString Module::labelPath(const QString &imageKey) const
{
    return labelsDir() + QLatin1Char('/') + QFileInfo(imageKey).completeBaseName() + QStringLiteral(".json");
}

QString Module::maskPath(const QString &imageKey) const
{
    return labelsDir() + QLatin1Char('/') + QFileInfo(imageKey).completeBaseName() + QStringLiteral(".png");
}

QString Module::resultPath(const QString &imageKey) const
{
    return resultsDir() + QLatin1Char('/') + QFileInfo(imageKey).completeBaseName() + QStringLiteral(".json");
}

// ---------------------------------------------------------------------------
//  序列化
// ---------------------------------------------------------------------------
QJsonObject Module::toJson() const
{
    QJsonObject o;
    o.insert(QStringLiteral("id"), m_id);
    o.insert(QStringLiteral("name"), m_name);
    o.insert(QStringLiteral("type"), taskTypeKey(m_type));
    o.insert(QStringLiteral("order"), m_order);
    o.insert(QStringLiteral("hasBackground"), hasBackground());
    o.insert(QStringLiteral("classes"), jsonio::classesToJson(m_classes));

    QJsonObject tr;
    tr.insert(QStringLiteral("backend"), m_train.backend);
    tr.insert(QStringLiteral("model"), m_train.model);
    tr.insert(QStringLiteral("epochs"), m_train.epochs);
    tr.insert(QStringLiteral("batchSize"), m_train.batchSize);
    tr.insert(QStringLiteral("learningRate"), m_train.learningRate);
    tr.insert(QStringLiteral("inputWidth"), m_train.inputWidth);
    tr.insert(QStringLiteral("inputHeight"), m_train.inputHeight);
    tr.insert(QStringLiteral("useGpu"), m_train.useGpu);
    tr.insert(QStringLiteral("gpuIndex"), m_train.gpuIndex);
    tr.insert(QStringLiteral("numWorkers"), m_train.numWorkers);
    if (!m_train.extra.isEmpty())
        tr.insert(QStringLiteral("extra"), QJsonObject::fromVariantMap(m_train.extra));
    o.insert(QStringLiteral("train"), tr);

    QJsonObject te;
    te.insert(QStringLiteral("scoreThreshold"), m_test.scoreThreshold);
    te.insert(QStringLiteral("nmsThreshold"), m_test.nmsThreshold);
    te.insert(QStringLiteral("iouThreshold"), m_test.iouThreshold);
    te.insert(QStringLiteral("minArea"), m_test.minArea);
    o.insert(QStringLiteral("test"), te);

    return o;
}

Module Module::fromJson(const QJsonObject &o, const QString &root)
{
    Module m;
    m.m_root  = root;
    m.m_id    = jsonio::str(o, QStringLiteral("id"), QFileInfo(root).fileName());
    m.m_name  = jsonio::str(o, QStringLiteral("name"), m.m_id);
    m.m_order = jsonio::integer(o, QStringLiteral("order"), 0);

    bool ok = false;
    m.m_type = taskTypeFromKey(jsonio::str(o, QStringLiteral("type")), &ok);

    if (o.contains(QStringLiteral("classes")))
        m.m_classes = jsonio::classesFromJson(o.value(QStringLiteral("classes")).toArray(), &ok);

    const QJsonObject tr = o.value(QStringLiteral("train")).toObject();
    m.m_train.task         = m.m_type;
    m.m_train.backend      = jsonio::str(tr, QStringLiteral("backend"), QStringLiteral("opencv"));
    m.m_train.model        = jsonio::str(tr, QStringLiteral("model"), QStringLiteral("resnet18"));
    m.m_train.epochs       = jsonio::integer(tr, QStringLiteral("epochs"), 50);
    m.m_train.batchSize    = jsonio::integer(tr, QStringLiteral("batchSize"), 8);
    m.m_train.learningRate = jsonio::num(tr, QStringLiteral("learningRate"), 0.001);
    m.m_train.inputWidth   = jsonio::integer(tr, QStringLiteral("inputWidth"), 224);
    m.m_train.inputHeight  = jsonio::integer(tr, QStringLiteral("inputHeight"), 224);
    m.m_train.useGpu       = jsonio::boolean(tr, QStringLiteral("useGpu"), false);
    m.m_train.gpuIndex     = jsonio::integer(tr, QStringLiteral("gpuIndex"), 0);
    m.m_train.numWorkers   = jsonio::integer(tr, QStringLiteral("numWorkers"), 2);
    m.m_train.extra        = tr.value(QStringLiteral("extra")).toObject().toVariantMap();
    m.m_train.datasetRoot  = root;
    m.m_train.outputDir    = m.modelsDir();

    const QJsonObject te = o.value(QStringLiteral("test")).toObject();
    m.m_test.scoreThreshold = jsonio::num(te, QStringLiteral("scoreThreshold"), 0.5);
    m.m_test.nmsThreshold   = jsonio::num(te, QStringLiteral("nmsThreshold"), 0.45);
    m.m_test.iouThreshold   = jsonio::num(te, QStringLiteral("iouThreshold"), 0.5);
    m.m_test.minArea        = jsonio::integer(te, QStringLiteral("minArea"), 16);

    m.m_train.numClasses = m.targetClassCount();
    return m;
}

bool Module::save(QString *err) const
{
    return jsonio::writeObject(filePath(), toJson(), err);
}

bool Module::load(const QString &moduleDir, Module *out, QString *err)
{
    QJsonObject o;
    const QString path = moduleDir + QLatin1Char('/') + fileName();
    if (!jsonio::readObject(path, &o, err))
        return false;
    if (out)
        *out = fromJson(o, moduleDir);
    return true;
}

} // namespace aq
