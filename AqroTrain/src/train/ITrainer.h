// ============================================================================
//  ITrainer.h —— 训练后端统一接口
//  ---------------------------------------------------------------------------
//  界面层只依赖本接口。更换训练后端（OpenCV 轻量训练 / LibTorch 深度学习）
//  不需要改动任何 UI 代码。
//
//  与 VisionTrain 的差异：训练数据不再由后端自行扫描目录，
//  而是由工程层通过 TrainSample 列表显式传入。这样后端与
//  「工程 / 模块 / 数据划分」的目录布局完全解耦，也能精确控制
//  「只训练划分到训练集的样本」。
//
//  线程模型：
//    · initialize() / start() / requestStop() / isRunning() / lastModelPath()
//      由主线程调用；
//    · 训练在内部工作线程执行，进度通过信号以队列方式投递回主线程；
//    · 信号 logMessage / progressChanged / epochFinished / trainingFinished
//      保证在主线程被接收。
// ============================================================================
#pragma once

#include <QObject>
#include <QString>
#include <QVector>

#include "core/Types.h"

namespace aq {

class ITrainer : public QObject
{
    Q_OBJECT

public:
    enum LogLevel {
        Info    = 0,
        Success = 1,
        Warning = 2,
        Error   = 3
    };
    Q_ENUM(LogLevel)

    explicit ITrainer(QObject *parent = nullptr);
    ~ITrainer() override;

    // --- 后端信息 ---------------------------------------------------------
    /// 后端显示名，例如 "OpenCV 轻量训练后端"
    virtual QString backendName() const = 0;

    /// 后端当前是否可用（依赖是否满足）。不可用时通过 reason 返回原因。
    virtual bool isAvailable(QString *reason = nullptr) const = 0;

    // --- 训练流程 ---------------------------------------------------------
    /// 校验参数并准备训练（主线程调用）。失败返回 false 并写 err。
    /// samples 只包含「数据划分 = 训练集、且已有标注」的样本。
    virtual bool initialize(const TrainConfig &cfg,
                            const QVector<TrainSample> &samples,
                            const QVector<ClassInfo> &classes,
                            QString *err) = 0;

    /// 异步启动训练。会立即返回，进度通过信号上报。
    virtual void start() = 0;

    /// 请求中止训练；实际停止是异步的，完成后仍会发出 trainingFinished(false,...)。
    virtual void requestStop() = 0;

    virtual bool isRunning() const = 0;

    /// 训练成功结束后的模型文件路径；未训练完成时为空。
    virtual QString lastModelPath() const = 0;

    // --- 推理 -------------------------------------------------------------
    /// 载入模型文件（由 lastModelPath 产生，可能还有同名附属文件）。
    virtual bool loadModel(const QString &modelPath, QString *err) = 0;

    /**
     * 设置推理后处理参数（置信度 / NMS / IoU / 最小连通域面积）。
     *
     * 训练时这些参数随 TrainConfig::extra 传入；但「评价」与「智能标注」只
     * 加载模型、不执行 initialize，因此必须由调用方显式把模块的
     * **测试参数** 交给后端，否则推理会一直用后端内部默认值，
     * 表现为「界面上改了置信度阈值却对结果没有任何影响」。
     */
    virtual void setTestConfig(const TestConfig &cfg) = 0;

    /// 当前生效的推理后处理参数
    virtual TestConfig testConfig() const = 0;

    /// 对单张 BGR 图像推理。out 的内容随任务类型而定（见 PredictionResult）。
    virtual bool predict(const cv::Mat &bgr, PredictionResult *out, QString *err) = 0;

    /// 载入后模型所对应的任务类型（未载入时为 Classification）
    virtual TaskType modelTask() const = 0;

    /// 载入后模型所认识的类别（未载入时为空）
    virtual QVector<ClassInfo> modelClasses() const = 0;

signals:
    /// level 取值见 LogLevel
    void logMessage(const QString &text, int level);

    /// 0..100
    void progressChanged(int percent);

    /// 每个阶段 / epoch 结束
    void epochFinished(int epoch, int totalEpochs,
                       double loss, double metric,
                       double valLoss, double valMetric);

    /// 训练结束（无论成功、失败还是被中止）
    void trainingFinished(bool ok, const QString &modelPath, const QString &message);
};

// ---------------------------------------------------------------------------
//  模型文件约定
//  ---------------------------------------------------------------------------
namespace modelio {

/// 依据任务类型给出模型主文件名（分类 .clsmodel / 检测 .detmodel / 分割 .segmodel）
QString defaultModelFileName(TaskType task);

/// 模型目录下主文件的完整路径
QString defaultModelPath(const QString &outputDir, TaskType task);

/// 与主模型同名的元数据文件路径（= modelPath + ".json"）
QString metaPath(const QString &modelPath);

/// 模型元数据的内容（供界面识别任务、类别与后处理参数）
struct ModelMeta {
    TaskType           task = TaskType::Classification;
    QString            backend;
    QString            model;
    int                inputWidth = 224;
    int                inputHeight = 224;
    QVector<ClassInfo> classes;
    QString            trainedAt;
    QString            metricsSummary;
    bool               valid = false;
};

/// 读取模型元数据
bool readModelMeta(const QString &modelPath, ModelMeta *meta, QString *err = nullptr);

/// 只读取任务类型（内部调用 readModelMeta）
bool readModelTask(const QString &modelPath, TaskType *task, QString *err = nullptr);

/// 写入模型元数据
bool writeModelMeta(const QString &modelPath, const ModelMeta &meta, QString *err = nullptr);

/// 把主模型文件与元数据一起导出到目标目录（文件名保持不变）
bool exportModel(const QString &modelPath, const QString &destDir,
                 QString *outMainPath = nullptr, QString *err = nullptr);

} // namespace modelio

} // namespace aq
