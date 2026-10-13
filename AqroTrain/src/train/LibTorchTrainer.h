// ============================================================================
//  LibTorchTrainer.h —— LibTorch（PyTorch C++）深度学习训练后端
//  ---------------------------------------------------------------------------
//  设计要点：
//    · 本头文件「不依赖 LibTorch」：既不 include 任何 torch/... 头文件，
//      也不出现任何 torch:: 类型，所以未安装 LibTorch 的机器上照样能编译本
//      头文件（MOC 解析也安全）。
//    · 所有 LibTorch 相关的类型、模型定义与实现细节都藏在 .cpp 的
//      #ifdef AQROTRAIN_ENABLE_LIBTORCH 分支内，通过下面的 Impl 前置声明
//      指针（pimpl）访问。
//    · 当宏未定义时，.cpp 的 #else 分支提供完全自洽的降级实现：所有方法
//      返回失败/空值，isAvailable() 返回 false 并给出中文原因
//      （提示设置环境变量 LIBTORCH_DIR 或 /p:LibTorch_DIR=... 后重新编译），
//      构造函数/析构函数/全部虚函数都有定义，保证链接不报错。
//
//  与 VisionTrain 的实质性差异（移植时已改造）：
//    · 编译开关宏由 VISIONTRAIN_ENABLE_LIBTORCH 改为 AQROTRAIN_ENABLE_LIBTORCH；
//    · 训练数据不再由后端扫描目录，也不再读取 classes.json：
//      工程层通过 initialize() 显式传入 TrainSample 列表与 ClassInfo 列表，
//      样本的标注真源是 labels/*.json（对象列表），分割另有派生掩码 .png；
//    · 元数据统一由 modelio::writeModelMeta() 落盘，本类不再自行拼 JSON；
//    · 训练跑在内部 std::thread 上，通过 QMetaObject 队列调用把
//      logMessage / progressChanged / epochFinished / trainingFinished
//      投递回主线程（与 OpenCvTrainer 一致的线程模型）。
//
//  类别索引语义：
//    · 分类：classes 只含真实类别，索引即标签，从 0 开始，无背景类；
//    · 检测：索引 0 固定为「背景」，目标框 classId 从 1 开始；
//    · 分割：掩码像素值 = 类别索引，0 为背景，目标从 1 开始；
//    · 三者都满足 numClasses == classes.size()。
// ============================================================================
#pragma once

#include <QObject>
#include <QString>
#include <QVector>

#include "core/Types.h"
#include "train/ITrainer.h"

namespace aq {

// ---------------------------------------------------------------------------
//  Impl 前置声明
// ---------------------------------------------------------------------------
/// 实现细节载体。定义在 LibTorchTrainer.cpp 内部，包含 LibTorch 的模型对象、
/// 优化器与数据管线等；头文件保持完全干净，不泄漏任何第三方类型。
struct Impl;

// ---------------------------------------------------------------------------
//  LibTorchTrainer
// ---------------------------------------------------------------------------
/// 基于 LibTorch 的深度学习训练后端，支持分类 / 检测 / 分割三种任务。
///
/// 线程模型（与 ITrainer 的契约一致）：
///   · initialize() / start() / requestStop() / isRunning() / loadModel() /
///     predict() 由主线程调用；
///   · start() 立即返回，真正的训练跑在内部 std::thread 工作线程里；
///   · 工作线程通过信号（队列连接）把日志、进度、epoch 结果与结束通知
///     投递回主线程，因此 logMessage/progressChanged/epochFinished/
///     trainingFinished 保证在主线程被接收。
class LibTorchTrainer : public ITrainer
{
    Q_OBJECT

public:
    explicit LibTorchTrainer(QObject *parent = nullptr);
    ~LibTorchTrainer() override;

    // --- 后端信息 ---------------------------------------------------------
    /// 后端显示名："LibTorch 深度学习后端"
    QString backendName() const override;

    /// 是否可用。未以 AQROTRAIN_ENABLE_LIBTORCH 编译时返回 false，
    /// 并通过 reason 说明如何启用（设置 LIBTORCH_DIR 后重新编译）。
    bool isAvailable(QString *reason = nullptr) const override;

    // --- 训练流程 ---------------------------------------------------------
    /// 校验样本与配置（主线程同步调用，不启动训练）。
    /// samples 为「训练集且已标注」的样本，classes 为模块类别。
    bool initialize(const TrainConfig &cfg,
                    const QVector<TrainSample> &samples,
                    const QVector<ClassInfo> &classes,
                    QString *err) override;

    /// 异步启动训练（工作线程）。会立即返回。
    void start() override;

    /// 请求中止训练；实际停止是异步的，完成后仍会发出 trainingFinished(false,...)。
    void requestStop() override;

    bool isRunning() const override;

    /// 训练成功结束后的模型文件路径；未训练完成时为空。
    QString lastModelPath() const override;

    // --- 推理 -------------------------------------------------------------
    /// 载入 .clsmodel/.detmodel/.segmodel（并读取同名 .json 元数据）。
    bool loadModel(const QString &modelPath, QString *err) override;

    /// 单张 BGR 图像推理，按任务类型填充 PredictionResult。
    bool predict(const cv::Mat &bgr, PredictionResult *out, QString *err) override;

    /// 载入后模型对应的任务类型（未载入时为 Classification）
    TaskType modelTask() const override;

    /// 载入后模型所认识的类别（未载入时为空）
    QVector<ClassInfo> modelClasses() const override;

    /// 设置推理后处理参数（评价 / 智能标注等「只加载模型」的场景必须调用）
    void setTestConfig(const TestConfig &cfg) override;
    TestConfig testConfig() const override;

private:
    // -----------------------------------------------------------------------
    //  以下成员仅由 .cpp 内部的 Impl 使用，主线程访问需自行加锁。
    // -----------------------------------------------------------------------
    Impl *m_impl = nullptr;   ///< LibTorch 实现细节（含全部 torch 类型）

    /// 未启用 LibTorch 时仍要记住推理参数（界面可回读；启用分支存在 Impl::test）
    TestConfig m_test;
};

} // namespace aq
