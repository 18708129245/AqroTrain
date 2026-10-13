// ============================================================================
//  OpenCvTrainer.h —— 基于 OpenCV 机器学习模块的轻量训练后端
//  ---------------------------------------------------------------------------
//  定位：LibTorch 不可用时的「开箱即用」后端。三种任务都做**真实训练**，
//        但算法是传统机器学习而非深度网络，精度有限，适合小数据集验证流程。
//
//    · 图像分类：HSV 直方图 + 降采样灰度图 作为特征 -> 随机森林
//    · 图像分割：逐像素 (B,G,R,x/W,y/H) 特征 -> 随机森林（逐像素分类）
//    · 目标检测：滑窗 + 图像块特征 -> 随机森林，配合 NMS 输出框
//
//  与 VisionTrain 的实质性差异（移植时已改造）：
//    · 训练数据不再由本后端扫描目录，也不再读取 classes.json；
//      工程层通过 initialize() 显式传入 TrainSample 列表与 ClassInfo 列表。
//    · 元数据统一由 modelio::writeModelMeta() 落盘，本类不再自行拼 JSON。
//    · 模型文件用 OpenCV FileStorage 的「内存模式」+ Qt 写盘，支持中文路径。
//
//  注意：该后端不使用「epoch / 学习率」这类梯度下降概念，
//        它把训练过程拆成若干阶段上报，因此训练监控里看到的是阶段进度。
//        需要真正的深度学习训练请安装 LibTorch（见 props\LibTorch.props）。
// ============================================================================
#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

#include <atomic>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

#include "core/Types.h"
#include "train/ITrainer.h"

namespace aq {

class OpenCvTrainer : public ITrainer
{
    Q_OBJECT

public:
    explicit OpenCvTrainer(QObject *parent = nullptr);
    ~OpenCvTrainer() override;

    QString backendName() const override;
    bool    isAvailable(QString *reason = nullptr) const override;

    /// 校验参数并准备训练。samples 为「训练集且已标注」的样本，
    /// classes 为模块类别（分类任务无背景类；检测/分割索引 0 为背景）。
    bool initialize(const TrainConfig &cfg,
                    const QVector<TrainSample> &samples,
                    const QVector<ClassInfo> &classes,
                    QString *err) override;

    void start() override;
    void requestStop() override;
    bool isRunning() const override { return m_running.load(); }
    QString lastModelPath() const override { return m_modelPath; }

    bool loadModel(const QString &modelPath, QString *err) override;
    bool predict(const cv::Mat &bgr, PredictionResult *out, QString *err) override;

    /// 评价 / 智能标注等「只加载模型」的场景由调用方显式传入模块的测试参数
    void       setTestConfig(const TestConfig &cfg) override { m_test = cfg; }
    TestConfig testConfig() const override { return m_test; }

    TaskType           modelTask() const override { return m_modelTask; }
    QVector<ClassInfo> modelClasses() const override { return m_modelClasses; }

private:
    // --- 跨线程发信号（工作线程 -> 主线程，队列连接）---
    void emitLog(const QString &text, int level);
    void emitProgress(int percent);
    void emitEpoch(int epoch, int total, double loss, double metric, double valLoss, double valMetric);
    void emitFinished(bool ok, const QString &modelPath, const QString &message);

    // --- 训练主流程（工作线程内执行）---
    void runTraining();
    /// 校验 initialize() 收到的样本列表（不再扫描目录、不再读 classes.json）
    bool validateSamples(QString *err);
    bool trainClassification(QString *err);
    bool trainSegmentation(QString *err);
    bool trainDetection(QString *err);

    // --- 模型持久化（支持中文路径）---
    bool saveModelTo(const QString &path, QString *err);
    bool loadModelFrom(const QString &path, QString *err);
    /// 通过 modelio::writeModelMeta 落盘元数据
    bool writeMeta(const QString &modelPath, QString *err);

    // --- 特征提取 ---
    static cv::Mat classifyFeatures(const cv::Mat &bgr, const cv::Size &inputSize);
    static cv::Mat segmentFeatures(const cv::Mat &bgr);

    /**
     * 滑窗图块的特征：把图块缩放为 patch×patch 灰度图后拉平。
     *
     * 这沿用了原工程的基线实现。**已知局限**：拉平后的像素向量对窗口位置敏感，
     * 而检测能拿到的正样本只有「标注框本身」这一种对齐方式，因此同一个目标
     * 在窗口里偏移几十像素就可能被判成背景。实测该轻量检测器的精确率/召回率
     * 都不高（具体数值见 README 第八节），**它只用于跑通流程**；
     * 需要可用的检测精度请启用 LibTorch 后端。
     *
     * （曾尝试改为「灰度直方图 + 色相直方图」这类平移不变描述子，
     *  实测在本工程的合成数据上反而更差：召回率 37.5% → 12.5%、
     *  精确率 25% → 0.9%，因此保留基线实现。）
     */
    static cv::Mat patchFeature(const cv::Mat &bgr, const cv::Rect &roi, int patch);

    // --- 推理实现 ---
    bool predictClassification(const cv::Mat &bgr, PredictionResult *out, QString *err);
    bool predictSegmentation(const cv::Mat &bgr, PredictionResult *out, QString *err);
    bool predictDetection(const cv::Mat &bgr, PredictionResult *out, QString *err);

    // --- 状态 ---
    TrainConfig   m_cfg;
    TestConfig    m_test;              ///< 推理后处理参数（跟随模块配置）
    QVector<ClassInfo> m_classes;

    // 由 TrainSample 列表拆解而来的内部表示（下标一一对应）
    QStringList   m_imagePaths;        ///< 图像绝对路径
    QStringList   m_labelPaths;        ///< 标注真源 .json
    QStringList   m_maskPaths;         ///< 分割掩码 .png（非分割任务为空）
    QVector<int>  m_classIds;          ///< 分类任务的类别索引

    cv::Ptr<cv::ml::StatModel> m_model;
    TaskType      m_modelTask = TaskType::Classification;
    cv::Size      m_modelInput { 224, 224 };
    QVector<ClassInfo> m_modelClasses;
    QString       m_modelPath;         ///< 训练完成后的模型路径 / 载入的模型路径
    QString       m_metricsSummary;    ///< 训练阶段验证指标摘要，写入元数据

    std::atomic<bool> m_stop { false };
    std::atomic<bool> m_running { false };
    std::thread       m_worker;
};

} // namespace aq
