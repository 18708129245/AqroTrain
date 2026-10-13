// ============================================================================
//  Types.cpp —— 基础类型辅助实现
// ============================================================================
#include "core/Types.h"

#include <QStringList>

namespace aq {

// ---------------------------------------------------------------------------
//  模块类型
// ---------------------------------------------------------------------------
QString taskTypeName(TaskType t)
{
    switch (t) {
    case TaskType::Classification: return QStringLiteral("图像分类");
    case TaskType::Detection:      return QStringLiteral("目标检测");
    case TaskType::Segmentation:   return QStringLiteral("图像分割");
    }
    return QStringLiteral("未知任务");
}

QString taskTypeKey(TaskType t)
{
    switch (t) {
    case TaskType::Classification: return QStringLiteral("classification");
    case TaskType::Detection:      return QStringLiteral("detection");
    case TaskType::Segmentation:   return QStringLiteral("segmentation");
    }
    return QStringLiteral("classification");
}

TaskType taskTypeFromKey(const QString &key, bool *ok)
{
    const QString k = key.trimmed().toLower();
    if (k == QLatin1String("classification") || k == QLatin1String("classify")) {
        if (ok) *ok = true;
        return TaskType::Classification;
    }
    if (k == QLatin1String("detection") || k == QLatin1String("detect")) {
        if (ok) *ok = true;
        return TaskType::Detection;
    }
    if (k == QLatin1String("segmentation") || k == QLatin1String("segment")) {
        if (ok) *ok = true;
        return TaskType::Segmentation;
    }
    if (ok) *ok = false;
    return TaskType::Classification;
}

QVector<TaskType> allTaskTypes()
{
    return { TaskType::Classification, TaskType::Detection, TaskType::Segmentation };
}

QString modelFileSuffix(TaskType t)
{
    switch (t) {
    case TaskType::Classification: return QStringLiteral("clsmodel");
    case TaskType::Detection:      return QStringLiteral("detmodel");
    case TaskType::Segmentation:   return QStringLiteral("segmodel");
    }
    return QStringLiteral("clsmodel");
}

// ---------------------------------------------------------------------------
//  数据划分
// ---------------------------------------------------------------------------
QString splitKey(Split s)
{
    switch (s) {
    case Split::None:  return QStringLiteral("none");
    case Split::Train: return QStringLiteral("train");
    case Split::Test:  return QStringLiteral("test");
    }
    return QStringLiteral("none");
}

QString splitName(Split s)
{
    switch (s) {
    case Split::None:  return QStringLiteral("未划分");
    case Split::Train: return QStringLiteral("训练集");
    case Split::Test:  return QStringLiteral("测试集");
    }
    return QStringLiteral("未划分");
}

Split splitFromKey(const QString &key, bool *ok)
{
    const QString k = key.trimmed().toLower();
    if (k == QLatin1String("train")) { if (ok) *ok = true; return Split::Train; }
    if (k == QLatin1String("test"))  { if (ok) *ok = true; return Split::Test; }
    if (k == QLatin1String("none") || k.isEmpty()) { if (ok) *ok = true; return Split::None; }
    if (ok) *ok = false;
    return Split::None;
}

// ---------------------------------------------------------------------------
//  标注状态
// ---------------------------------------------------------------------------
QString annoStateKey(AnnoState s)
{
    switch (s) {
    case AnnoState::None: return QStringLiteral("none");
    case AnnoState::Pre:  return QStringLiteral("pre");
    case AnnoState::Done: return QStringLiteral("done");
    }
    return QStringLiteral("none");
}

QString annoStateName(AnnoState s)
{
    switch (s) {
    case AnnoState::None: return QStringLiteral("未标注");
    case AnnoState::Pre:  return QStringLiteral("预标注");
    case AnnoState::Done: return QStringLiteral("已完成");
    }
    return QStringLiteral("未标注");
}

AnnoState annoStateFromKey(const QString &key, bool *ok)
{
    const QString k = key.trimmed().toLower();
    if (k == QLatin1String("pre"))  { if (ok) *ok = true; return AnnoState::Pre; }
    if (k == QLatin1String("done")) { if (ok) *ok = true; return AnnoState::Done; }
    if (k == QLatin1String("none") || k.isEmpty()) { if (ok) *ok = true; return AnnoState::None; }
    if (ok) *ok = false;
    return AnnoState::None;
}

// ---------------------------------------------------------------------------
//  工程图片格式
// ---------------------------------------------------------------------------
QString imageFormatKey(ImageFormat f) { return imageFormatExt(f); }

QString imageFormatExt(ImageFormat f)
{
    switch (f) {
    case ImageFormat::PNG: return QStringLiteral("png");
    case ImageFormat::BMP: return QStringLiteral("bmp");
    case ImageFormat::TIF: return QStringLiteral("tif");
    case ImageFormat::JPG: return QStringLiteral("jpg");
    }
    return QStringLiteral("png");
}

QString imageFormatName(ImageFormat f)
{
    switch (f) {
    case ImageFormat::PNG: return QStringLiteral("PNG");
    case ImageFormat::BMP: return QStringLiteral("BMP");
    case ImageFormat::TIF: return QStringLiteral("TIFF");
    case ImageFormat::JPG: return QStringLiteral("JPEG");
    }
    return QStringLiteral("PNG");
}

ImageFormat imageFormatFromKey(const QString &key, bool *ok)
{
    const QString k = key.trimmed().toLower();
    if (ok) *ok = true;
    if (k == QLatin1String("bmp")) return ImageFormat::BMP;
    if (k == QLatin1String("tif") || k == QLatin1String("tiff")) return ImageFormat::TIF;
    if (k == QLatin1String("jpg") || k == QLatin1String("jpeg")) return ImageFormat::JPG;
    if (k == QLatin1String("png")) return ImageFormat::PNG;
    if (ok) *ok = false;
    return ImageFormat::PNG;
}

QVector<ImageFormat> allImageFormats()
{
    return { ImageFormat::PNG, ImageFormat::BMP, ImageFormat::TIF, ImageFormat::JPG };
}

// ---------------------------------------------------------------------------
//  类别
// ---------------------------------------------------------------------------
QColor defaultClassColor(int index)
{
    static const char *kPalette[] = {
        "#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4",
        "#46F0F0", "#F032E6", "#BCF60C", "#FABEBE", "#008080", "#E6BEFF",
        "#9A6324", "#FFFAC8", "#800000", "#AAFFC3"
    };
    const int n = int(sizeof(kPalette) / sizeof(kPalette[0]));
    if (index < 0) index = 0;
    return QColor(QString::fromLatin1(kPalette[index % n]));
}

// ---------------------------------------------------------------------------
//  参数摘要
// ---------------------------------------------------------------------------
QString TrainConfig::summary() const
{
    const QString gpu = useGpu ? QStringLiteral("GPU#%1").arg(gpuIndex) : QStringLiteral("CPU");
    return QStringLiteral(
               "模块=%1 | 后端=%2 | 模型=%3 | Epoch=%4 | Batch=%5 | LR=%6 | 输入=%7x%8 | 设备=%9 | 类别数=%10")
        .arg(taskTypeName(task))
        .arg(backend)
        .arg(model)
        .arg(epochs)
        .arg(batchSize)
        .arg(learningRate)
        .arg(inputWidth)
        .arg(inputHeight)
        .arg(gpu)
        .arg(numClasses);
}

// ---------------------------------------------------------------------------
//  指标摘要
// ---------------------------------------------------------------------------
QString Metrics::summary() const
{
    if (!valid)
        return QStringLiteral("尚无评价结果（请先训练或执行测试）");

    return QStringLiteral(
               "%1 | 精确率 %2% | 召回率 %3% | F1 %4 | 漏检率 %5% (%6/%7) | 过检率 %8% (%9/%10) | 测试样本 %11")
        .arg(taskTypeName(task))
        .arg(precision * 100.0, 0, 'f', 2)
        .arg(recall * 100.0, 0, 'f', 2)
        .arg(f1, 0, 'f', 4)
        .arg(missRate * 100.0, 0, 'f', 2)
        .arg(ngMissed)
        .arg(ngTotal)
        .arg(overkillRate * 100.0, 0, 'f', 2)
        .arg(okOverkill)
        .arg(okTotal)
        .arg(testSamples);
}

} // namespace aq
