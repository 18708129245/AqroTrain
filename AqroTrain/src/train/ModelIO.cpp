// ============================================================================
//  ModelIO.cpp —— 模型文件与元数据的通用约定
//  ---------------------------------------------------------------------------
//  模型由两部分组成：
//    · 主模型文件：<模块>/models/model.<clsmodel|detmodel|segmodel>
//      - OpenCV 后端：OpenCV FileStorage（内存模式导出，由 Qt 写盘）
//      - LibTorch 后端：torch::save 序列化的模块
//    · 元数据文件：<主模型文件>.json
//      记录 task / backend / classes / 输入尺寸 / 指标摘要，
//      供「模型测试」「智能标注」在不加载网络的情况下识别任务与类别。
// ============================================================================
#include "train/ITrainer.h"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>

#include "core/JsonStore.h"

namespace aq {
namespace modelio {

QString defaultModelFileName(TaskType task)
{
    return QStringLiteral("model.") + modelFileSuffix(task);
}

QString defaultModelPath(const QString &outputDir, TaskType task)
{
    if (outputDir.trimmed().isEmpty())
        return defaultModelFileName(task);
    return QDir(outputDir).absoluteFilePath(defaultModelFileName(task));
}

QString metaPath(const QString &modelPath)
{
    return modelPath + QStringLiteral(".json");
}

bool writeModelMeta(const QString &modelPath, const ModelMeta &meta, QString *err)
{
    QJsonObject o;
    o.insert(QStringLiteral("format"), QStringLiteral("AqroTrain.Model"));
    o.insert(QStringLiteral("version"), 1);
    o.insert(QStringLiteral("task"), taskTypeKey(meta.task));
    o.insert(QStringLiteral("backend"), meta.backend);
    o.insert(QStringLiteral("model"), meta.model);
    o.insert(QStringLiteral("inputWidth"), meta.inputWidth);
    o.insert(QStringLiteral("inputHeight"), meta.inputHeight);
    o.insert(QStringLiteral("classes"), jsonio::classesToJson(meta.classes));
    o.insert(QStringLiteral("trainedAt"),
             meta.trainedAt.isEmpty() ? QDateTime::currentDateTime().toString(Qt::ISODate)
                                      : meta.trainedAt);
    o.insert(QStringLiteral("metricsSummary"), meta.metricsSummary);
    return jsonio::writeObject(metaPath(modelPath), o, err);
}

bool readModelMeta(const QString &modelPath, ModelMeta *meta, QString *err)
{
    if (meta)
        *meta = ModelMeta();

    const QString mp = metaPath(modelPath);
    QJsonObject o;
    if (!jsonio::readObject(mp, &o, err))
        return false;

    ModelMeta m;
    bool ok = false;
    m.task            = taskTypeFromKey(jsonio::str(o, QStringLiteral("task")), &ok);
    m.backend         = jsonio::str(o, QStringLiteral("backend"));
    m.model           = jsonio::str(o, QStringLiteral("model"));
    m.inputWidth      = jsonio::integer(o, QStringLiteral("inputWidth"), 224);
    m.inputHeight     = jsonio::integer(o, QStringLiteral("inputHeight"), 224);
    m.trainedAt       = jsonio::str(o, QStringLiteral("trainedAt"));
    m.metricsSummary  = jsonio::str(o, QStringLiteral("metricsSummary"));
    m.classes         = jsonio::classesFromJson(o.value(QStringLiteral("classes")).toArray(), &ok);
    m.valid           = true;

    if (meta)
        *meta = m;
    return true;
}

bool readModelTask(const QString &modelPath, TaskType *task, QString *err)
{
    ModelMeta meta;
    if (!readModelMeta(modelPath, &meta, err))
        return false;
    if (task)
        *task = meta.task;
    return true;
}

bool exportModel(const QString &modelPath, const QString &destDir,
                 QString *outMainPath, QString *err)
{
    const QFileInfo src(modelPath);
    if (!src.exists()) {
        if (err) *err = QStringLiteral("模型文件不存在：%1").arg(src.fileName());
        return false;
    }
    if (!QDir().mkpath(destDir)) {
        if (err) *err = QStringLiteral("无法创建导出目录：%1").arg(QDir::toNativeSeparators(destDir));
        return false;
    }

    const QString destMain = destDir + QLatin1Char('/') + src.fileName();
    if (QFileInfo::exists(destMain) && !QFile::remove(destMain)) {
        if (err) *err = QStringLiteral("目标位置已存在同名文件且无法覆盖：%1").arg(src.fileName());
        return false;
    }
    if (!QFile::copy(modelPath, destMain)) {
        if (err) *err = QStringLiteral("复制模型文件失败：%1").arg(src.fileName());
        return false;
    }

    // 元数据一并导出，否则部署端无法识别任务与类别
    const QString srcMeta = metaPath(modelPath);
    if (QFileInfo::exists(srcMeta)) {
        const QString destMeta = metaPath(destMain);
        if (QFileInfo::exists(destMeta))
            QFile::remove(destMeta);
        if (!QFile::copy(srcMeta, destMeta)) {
            if (err) *err = QStringLiteral("复制模型元数据失败：%1").arg(QFileInfo(srcMeta).fileName());
            return false;
        }
    }

    if (outMainPath)
        *outMainPath = destMain;
    return true;
}

} // namespace modelio
} // namespace aq
