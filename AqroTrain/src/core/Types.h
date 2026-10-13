// ============================================================================
//  Types.h —— 全局基础类型与契约定义
//  ---------------------------------------------------------------------------
//  AqroTrain：对标阿丘科技 AIDI 的视觉标注训练平台。
//  本文件被工程层、标注层、训练层、界面层共同依赖，刻意不引入重依赖。
//
//  与 AIDI 概念的对应关系：
//    · TaskType  —— 模块类型（分类模块 / 检测模块 / 分割模块）
//    · Split     —— 数据划分（未标注 / 训练集 / 测试集）
//    · SampleState.ok —— OK 图（无缺陷的正例样本）
//    · AnnoState —— 标注状态（未标注 / 预标注待复核 / 已完成）
//  ============================================================================
#pragma once

#include <QColor>
#include <QMetaType>
#include <QString>
#include <QVariantMap>
#include <QRectF>
#include <QVector>

#include <opencv2/core.hpp>

#include <vector>

namespace aq {

// ---------------------------------------------------------------------------
//  模块类型（对应 AIDI 的「模块」）
// ---------------------------------------------------------------------------
enum class TaskType {
    Classification = 0,   ///< 分类模块：整图一个类别
    Detection      = 1,   ///< 检测模块：多个矩形 / 多边形目标
    Segmentation   = 2    ///< 分割模块：逐像素类别
};

QString           taskTypeName(TaskType t);            ///< 「图像分类」
QString           taskTypeKey(TaskType t);             ///< "classification"
TaskType          taskTypeFromKey(const QString &key, bool *ok = nullptr);
QVector<TaskType> allTaskTypes();

/// 模型文件扩展名（不含点）：分类 clsmodel / 检测 detmodel / 分割 segmodel
QString modelFileSuffix(TaskType t);

// ---------------------------------------------------------------------------
//  数据划分（对应 AIDI 的训练集 / 测试集 / 未标注）
// ---------------------------------------------------------------------------
enum class Split {
    None  = 0,   ///< 未标注（不参与训练，可用于人工评价）
    Train = 1,   ///< 训练集
    Test  = 2    ///< 测试集
};

QString splitKey(Split s);
QString splitName(Split s);
Split   splitFromKey(const QString &key, bool *ok = nullptr);

// ---------------------------------------------------------------------------
//  标注状态
// ---------------------------------------------------------------------------
enum class AnnoState {
    None = 0,   ///< 未标注
    Pre  = 1,   ///< 智能标注产生，待人工复核
    Done = 2    ///< 已完成（人工确认）
};

QString   annoStateKey(AnnoState s);
QString   annoStateName(AnnoState s);
AnnoState annoStateFromKey(const QString &key, bool *ok = nullptr);

// ---------------------------------------------------------------------------
//  工程图片格式（AIDI 新建工程时需选择，导入时统一转码）
// ---------------------------------------------------------------------------
enum class ImageFormat { PNG = 0, BMP = 1, TIF = 2, JPG = 3 };

QString              imageFormatKey(ImageFormat f);
QString              imageFormatExt(ImageFormat f);    ///< "png"
QString              imageFormatName(ImageFormat f);   ///< "PNG"
ImageFormat          imageFormatFromKey(const QString &key, bool *ok = nullptr);
QVector<ImageFormat> allImageFormats();

// ---------------------------------------------------------------------------
//  标签类别
// ---------------------------------------------------------------------------
struct ClassInfo {
    QString name;
    QColor  color;

    ClassInfo() = default;
    ClassInfo(const QString &n, const QColor &c) : name(n), color(c) {}
};

/// 默认调色板（高对比、循环使用）
QColor defaultClassColor(int index);

// ---------------------------------------------------------------------------
//  样本状态（每张图在每个模块下各有一份）
// ---------------------------------------------------------------------------
struct SampleState {
    Split     split = Split::None;
    bool      ok    = false;   ///< OK 图（无缺陷正例）
    AnnoState state = AnnoState::None;
};

// ---------------------------------------------------------------------------
//  训练 / 测试参数
// ---------------------------------------------------------------------------
struct TrainConfig {
    TaskType task = TaskType::Classification;

    QString backend     = QStringLiteral("opencv");   ///< "opencv" | "libtorch"
    QString model       = QStringLiteral("resnet18"); ///< 模型结构标识
    int     epochs      = 50;
    int     batchSize   = 8;
    double  learningRate = 0.001;
    int     inputWidth  = 224;
    int     inputHeight = 224;
    bool    useGpu      = false;
    int     gpuIndex    = 0;
    int     numClasses  = 0;
    int     numWorkers  = 2;

    QString datasetRoot;   ///< 训练数据根目录（模块的 images/labels）
    QString outputDir;     ///< 模型输出目录

    QVariantMap extra;     ///< 各任务特有参数

    QString summary() const;
};

struct TestConfig {
    double scoreThreshold = 0.5;    ///< 置信度阈值
    double nmsThreshold   = 0.45;   ///< NMS 阈值
    double iouThreshold   = 0.5;    ///< 评价时的 IoU 匹配阈值
    int    minArea        = 16;     ///< 分割连通域最小面积（像素）
};

// ---------------------------------------------------------------------------
//  训练样本（训练后端与工程层之间的传递契约）
//  ---------------------------------------------------------------------------
struct TrainSample {
    QString imageKey;      ///< 图像池中的文件名
    QString imagePath;     ///< 图像绝对路径
    QString labelPath;     ///< 标注真源（.json）
    QString maskPath;      ///< 分割掩码（.png），非分割任务为空
    int     classId = -1;  ///< 分类任务的类别索引

    bool valid() const { return !imagePath.isEmpty(); }
};

// ---------------------------------------------------------------------------
//  推理结果
// ---------------------------------------------------------------------------
struct DetectionBox {
    QRectF rect;
    int    classId = 0;
    float  score   = 0.f;
};

struct PredictionResult {
    TaskType task = TaskType::Classification;

    // --- 分类 ---
    int                classId = -1;
    float              score   = 0.f;
    std::vector<float> scores;

    // --- 检测 ---
    std::vector<DetectionBox> boxes;

    // --- 分割 ---
    cv::Mat mask;   ///< CV_8UC1，像素值 = 类别索引

    bool valid() const { return classId >= 0 || !boxes.empty() || !mask.empty(); }
};

// ---------------------------------------------------------------------------
//  评价指标
// ---------------------------------------------------------------------------
struct ClassMetric {
    int     classId = 0;
    QString name;
    long    tp = 0;
    long    fp = 0;
    long    fn = 0;
    double  precision = 0.0;
    double  recall    = 0.0;
    double  f1        = 0.0;
    double  iou       = 0.0;   ///< 分割任务：该类的 IoU
};

struct Metrics {
    TaskType task = TaskType::Classification;
    bool     valid = false;

    QVector<ClassMetric> perClass;

    double precision = 0.0;   ///< 宏平均
    double recall    = 0.0;
    double f1        = 0.0;
    double mIoU      = 0.0;   ///< 分割：平均 IoU

    // AIDI 侧重的两个工业指标
    double missRate     = 0.0;   ///< 漏检率 = 未检出 NG 数 / NG 总数
    double overkillRate = 0.0;   ///< 过检率 = OK 判为 NG 数 / OK 总数

    int ngTotal     = 0;
    int ngMissed    = 0;
    int okTotal     = 0;
    int okOverkill  = 0;

    int testSamples = 0;   ///< 参与评价的样本数

    QString summary() const;
};

// ---------------------------------------------------------------------------
//  数据集统计
// ---------------------------------------------------------------------------
struct DatasetStats {
    int total     = 0;
    int annotated = 0;
    int train     = 0;
    int test      = 0;
    int okImages  = 0;

    int unannotated() const { return total - annotated; }
};

} // namespace aq

Q_DECLARE_METATYPE(aq::TaskType)
Q_DECLARE_METATYPE(aq::Split)
Q_DECLARE_METATYPE(aq::AnnoState)
Q_DECLARE_METATYPE(aq::PredictionResult)
Q_DECLARE_METATYPE(aq::ClassMetric)
Q_DECLARE_METATYPE(aq::Metrics)
