// ============================================================================
//  LibTorchTrainer.cpp —— LibTorch（PyTorch C++）深度学习训练后端实现
//  ---------------------------------------------------------------------------
//  移植自 VisionTrain 的 vt::LibTorchTrainer，实质性改造点：
//    · 编译开关宏 VISIONTRAIN_ENABLE_LIBTORCH -> AQROTRAIN_ENABLE_LIBTORCH；
//    · 训练数据改由工程层通过 initialize(cfg, samples, classes, err) 传入
//      （TrainSample + ClassInfo），本后端不再扫描目录、不再读 classes.json；
//    · 元数据改用 modelio::writeModelMeta() 落盘；
//    · 训练线程由内部 QThread 改为内部 std::thread（与 OpenCvTrainer 一致），
//      仍通过 QMetaObject 队列调用把信号投递回主线程。
//
//  编译开关：
//    · 定义 AQROTRAIN_ENABLE_LIBTORCH 时（见 props/LibTorch.props），
//      编译下面的完整实现：真实的分类 / 分割 / 检测训练 + 推理；
//    · 未定义时走 #else 降级分支：不依赖任何 LibTorch 头文件与库，
//      所有接口返回失败/空值，isAvailable() 返回 false 并说明启用方式
//      （提示设置环境变量 LIBTORCH_DIR 或 /p:LibTorch_DIR=... 后重新编译），
//      保证工程在没装 LibTorch 的机器上依然可以正常编译链接运行。
//
//  类别索引语义（与工程层约定一致）：
//    · 分类：classes 只含真实类别，索引即标签，从 0 开始，无背景类；
//            真值直接取 TrainSample.classId；
//    · 检测：索引 0 固定为「背景」，目标框 classId 从 1 开始；
//            labels/*.json 是对象列表，取每个对象的包围盒并归一化成
//            "classId cx cy w h"（0..1 的 YOLO 格式）供训练使用；
//    · 分割：优先用 TrainSample.maskPath 的派生掩码（CV_8UC1，像素值 =
//            类别索引，0 为背景，目标从 1 开始），缺失时由对象列表光栅化；
//    · 三种任务的 numClasses 都等于 classes.size()（即包含背景类）。
// ============================================================================

#define NOMINMAX   // 必须位于任何可能间接引入 windows.h 的头文件之前

// ---------------------------------------------------------------------------
//  公共头文件（#ifdef / #else 两个分支共用）
// ---------------------------------------------------------------------------
#include "train/LibTorchTrainer.h"

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>
#include <QJsonValue>
#include <QMetaObject>
#include <QMutex>
#include <QMutexLocker>
#include <QRegularExpression>
#include <QSet>
#include <QStringList>
#include <QVariant>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "annotate/AnnoObject.h"
#include "core/ImageIO.h"

// ===========================================================================
//  匿名命名空间：UTF-8 / 中文路径安全读写 + 通用小工具
//  （两个编译分支共用，避免符号冲突）
// ===========================================================================
namespace {

/// 未启用 LibTorch 时的统一原因文案。
const char *const kDisableReason =
    "未启用 LibTorch，请设置环境变量 LIBTORCH_DIR 或 /p:LibTorch_DIR=... 后重新编译";

// ---------------------------------------------------------------------------
//  文件读写（中文路径安全）
// ---------------------------------------------------------------------------
/// 以二进制方式读取文件全部字节。QFile 内部按 UTF-16 调用 Win32 API，
/// 因此中文路径可以正常打开；失败返回空 QByteArray。
QByteArray readAllBytes(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return QByteArray();
    return f.readAll();
}

/// 读取文本文件并按 UTF-8 解码；失败返回空串。
QString readTextFile(const QString &path)
{
    const QByteArray raw = readAllBytes(path);
    if (raw.isEmpty())
        return QString();
    return QString::fromUtf8(raw);
}

/// 中文路径安全读取图像：直接复用工程统一的 imageio::imreadUnicode()
/// （内部为「Qt 读字节 -> cv::imdecode」，不能用 cv::imread）。
inline cv::Mat readImageUnicode(const QString &path, int flags)
{
    return aq::imageio::imreadUnicode(path, flags);
}

/// QString -> UTF-8 std::string（LibTorch 文件接口只接受窄字符路径）。
inline std::string toStdString(const QString &s)
{
    const QByteArray b = s.toUtf8();
    return std::string(b.constData(), static_cast<size_t>(b.size()));
}

// ---------------------------------------------------------------------------
//  小工具
// ---------------------------------------------------------------------------
inline int clampInt(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/// 输入尺寸是否可被 8 整除（U-Net 三次下采样要求）。
inline bool divisibleBy8(int v)
{
    return v > 0 && (v % 8) == 0;
}

} // namespace

// ###########################################################################
//  ######################  未启用 LibTorch 的降级实现  #######################
//  ###########################################################################
#ifndef AQROTRAIN_ENABLE_LIBTORCH

namespace aq {

/// 降级实现里的 Impl 也要有定义，保证 pimpl 指针可以构造/析构。
struct Impl {
    TrainConfig cfg;
    QString     modelPath;
    QVector<ClassInfo> modelClasses;
};

LibTorchTrainer::LibTorchTrainer(QObject *parent)
    : ITrainer(parent)
    , m_impl(new Impl())
{
}

LibTorchTrainer::~LibTorchTrainer()
{
    delete m_impl;
    m_impl = nullptr;
}

QString LibTorchTrainer::backendName() const
{
    return QStringLiteral("LibTorch 深度学习后端");
}

bool LibTorchTrainer::isAvailable(QString *reason) const
{
    // 未启用编译开关：本后端不可用，并把启用方式回传给调用方（界面层提示用）。
    if (reason)
        *reason = QString::fromUtf8(kDisableReason);
    return false;
}

bool LibTorchTrainer::initialize(const TrainConfig &cfg,
                                 const QVector<TrainSample> &samples,
                                 const QVector<ClassInfo> &classes,
                                 QString *err)
{
    Q_UNUSED(cfg);
    Q_UNUSED(samples);
    Q_UNUSED(classes);
    if (err)
        *err = QString::fromUtf8(kDisableReason);
    return false;
}

void LibTorchTrainer::start()
{
    // 没有工作线程；仍立即发出结束信号，避免上层一直等待。
    emit trainingFinished(false, QString(), QString::fromUtf8(kDisableReason));
}

void LibTorchTrainer::requestStop()
{
    // 无后台线程，无需处理。
}

bool LibTorchTrainer::isRunning() const
{
    return false;
}

QString LibTorchTrainer::lastModelPath() const
{
    return QString();
}

bool LibTorchTrainer::loadModel(const QString &modelPath, QString *err)
{
    Q_UNUSED(modelPath);
    if (err)
        *err = QString::fromUtf8(kDisableReason);
    return false;
}

bool LibTorchTrainer::predict(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    Q_UNUSED(bgr);
    if (out)
        *out = PredictionResult();
    if (err)
        *err = QString::fromUtf8(kDisableReason);
    return false;
}

TaskType LibTorchTrainer::modelTask() const
{
    return TaskType::Classification;
}

QVector<ClassInfo> LibTorchTrainer::modelClasses() const
{
    return QVector<ClassInfo>();
}

void LibTorchTrainer::setTestConfig(const TestConfig &cfg)
{
    // 未启用 LibTorch 时没有推理能力，这里只记住参数，便于界面回读；
    // 真正的使用者在「训练后端信息」里会看到不可用原因。
    m_test = cfg;
}

TestConfig LibTorchTrainer::testConfig() const
{
    return m_test;
}

} // namespace aq

// ###########################################################################
//  ########################  启用 LibTorch 的实现  ###########################
//  ###########################################################################
#else // AQROTRAIN_ENABLE_LIBTORCH

// LibTorch 头文件必须在 Qt / OpenCV 之后包含，避免 MSVC 下的宏与
// 预编译头冲突；NOMINMAX 已在文件顶部定义。
#include <torch/torch.h>
#include <torch/serialize.h>

namespace aq {

// ===========================================================================
//  训练超参数
// ===========================================================================
namespace {

constexpr double kDetBoxLossWeight  = 5.0;    ///< 检测框回归损失权重
constexpr double kDetBgWeight       = 0.05;   ///< 检测背景类损失权重（缓解类别不平衡）
constexpr float  kDetScoreThreshold = 0.30f;  ///< 检测推理置信度阈值
constexpr float  kDetNmsIoU         = 0.45f;  ///< 检测 NMS 的 IoU 阈值
constexpr float  kMapIouThreshold   = 0.5f;   ///< mAP@0.5
constexpr int    kSeed              = 20240101; ///< 划分/打乱的固定种子（可复现）

/// 图像标准化参数（ImageNet 统计量，三种任务统一使用）。
const double kMean[3] = { 0.485, 0.456, 0.406 };
const double kStd[3]  = { 0.229, 0.224, 0.225 };

// ---------------------------------------------------------------------------
//  张量 / 图像互转
// ---------------------------------------------------------------------------
/// OpenCV BGR 图 -> 已归一化的 NCHW 张量（RGB、[0,1]、再按 ImageNet 标准化）。
torch::Tensor imageToTensor(const cv::Mat &bgr, int w, int h)
{
    cv::Mat rgb;
    if (bgr.channels() == 1)
        cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
    else if (bgr.channels() == 4)
        cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
    else
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(w, h), 0.0, 0.0, cv::INTER_LINEAR);

    // OpenCV 是 HWC/uint8，LibTorch 需要 CHW/float
    torch::Tensor t = torch::from_blob(resized.data, { h, w, 3 }, torch::kUInt8)
                          .to(torch::kFloat32)
                          .div(255.0f)
                          .permute({ 2, 0, 1 })
                          .contiguous();
    for (int c = 0; c < 3; ++c)
        t[c] = (t[c] - kMean[c]) / kStd[c];
    return t;   // [3,H,W]
}

/// 标准化逆变换 + CHW -> BGR（训练过程抽查可视化用）。
cv::Mat tensorToBgr(const torch::Tensor &t)
{
    torch::Tensor x = t.detach().to(torch::kCPU).clone();
    for (int c = 0; c < 3; ++c)
        x[c] = x[c] * kStd[c] + kMean[c];
    x = x.clamp(0.0, 1.0).mul(255.0).to(torch::kUInt8).permute({ 1, 2, 0 }).contiguous();
    cv::Mat rgb(static_cast<int>(x.size(0)), static_cast<int>(x.size(1)), CV_8UC3,
                x.data_ptr());
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

// ---------------------------------------------------------------------------
//  设备选择
// ---------------------------------------------------------------------------
/// 依据配置选择设备；useGpu 但无 CUDA 时回退 CPU，并通过 note 说明。
torch::Device makeDevice(const TrainConfig &cfg, QString *note)
{
    if (cfg.useGpu && torch::cuda::is_available()) {
        const int count = static_cast<int>(torch::cuda::device_count());
        const int idx = clampInt(cfg.gpuIndex, 0, count > 0 ? count - 1 : 0);
        if (note)
            *note = QStringLiteral("CUDA:%1").arg(idx);
        return torch::Device(torch::kCUDA, idx);
    }
    if (note) {
        *note = cfg.useGpu ? QStringLiteral("CPU（本机无可用 CUDA，已回退）")
                           : QStringLiteral("CPU");
    }
    return torch::Device(torch::kCPU);
}

// ===========================================================================
//  模型定义
//  ===========================================================================
//  ===========================================================================
// 说明：所有模型都用 TORCH_MODULE 包装（SmallCNN / ResNet18 / UNet / Detector），
// 持有者是 shared_ptr 语义，可直接喂给 torch::save / torch::load。
// 每个 Impl 都必须重写 forward，避免 MSVC 把继承来的虚函数当成隐藏重载。
// ===========================================================================

// ---------------------------------------------------------------------------
//  分类模型 A：内置小型 CNN（默认回退方案）
// ---------------------------------------------------------------------------
struct SmallCNNImpl : torch::nn::Module
{
    SmallCNNImpl(int numClasses, int inW, int inH)
    {
        (void)inW;   // 结构上对输入尺寸无约束，保留参数只为接口一致
        (void)inH;

        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1)));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
        conv3 = register_module("conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(128));
        pool = register_module("pool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(2).stride(2)));
        gap = register_module("gap", torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions(1)));
        fc = register_module("fc", torch::nn::Linear(128, numClasses));

        // Kaiming 初始化，保证小数据集上也能稳定起步
        torch::nn::init::kaiming_normal_(conv1->weight, 0.0,
                                         torch::nn::init::FanMode::FanOut,
                                         torch::nn::init::Nonlinearity::ReLU);
        torch::nn::init::kaiming_normal_(conv2->weight, 0.0,
                                         torch::nn::init::FanMode::FanOut,
                                         torch::nn::init::Nonlinearity::ReLU);
        torch::nn::init::kaiming_normal_(conv3->weight, 0.0,
                                         torch::nn::init::FanMode::FanOut,
                                         torch::nn::init::Nonlinearity::ReLU);
        torch::nn::init::constant_(fc->bias, 0.0);
    }

    torch::Tensor forward(torch::Tensor x) override
    {
        x = pool->forward(torch::relu(bn1->forward(conv1->forward(x))));
        x = pool->forward(torch::relu(bn2->forward(conv2->forward(x))));
        x = pool->forward(torch::relu(bn3->forward(conv3->forward(x))));
        x = gap->forward(x);      // [N,128,1,1]
        x = x.flatten(1);         // [N,128]
        return fc->forward(x);    // [N,numClasses]
    }

    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr }, conv3{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr }, bn3{ nullptr };
    torch::nn::MaxPool2d pool{ nullptr };
    torch::nn::AdaptiveAvgPool2d gap{ nullptr };
    torch::nn::Linear fc{ nullptr };
};
TORCH_MODULE(SmallCNN);

// ---------------------------------------------------------------------------
//  分类模型 B：ResNet18 风格骨干（cfg.model 含 "resnet" 时使用）
// ---------------------------------------------------------------------------
/// 经典残差块：两个 3x3 卷积 + 短路连接（尺寸/通道变化时用 1x1 卷积对齐）。
struct ResBasicBlockImpl : torch::nn::Module
{
    ResBasicBlockImpl(int inCh, int outCh, int stride)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(inCh, outCh, 3).stride(stride).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(outCh));
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(outCh, outCh, 3).stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(outCh));
        if (stride != 1 || inCh != outCh) {
            down = register_module("down", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(inCh, outCh, 1)
                                      .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(outCh)));
        }
    }

    torch::Tensor forward(torch::Tensor x) override
    {
        torch::Tensor id = x;
        if (down)
            id = down->forward(id);
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        return torch::relu(x + id);
    }

    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr };
    torch::nn::Sequential down{ nullptr };
};
TORCH_MODULE(ResBasicBlock);

/// 简化版 ResNet18：保留 4 个 stage，每个 stage 1 个残差块（参数量更小，
/// 小数据集上更容易收敛；如需标准 18 层结构，把每 stage 补齐为 [2,2,2,2] 即可）。
struct ResNet18Impl : torch::nn::Module
{
    explicit ResNet18Impl(int numClasses)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        maxpool = register_module("maxpool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

        layer1 = register_module("layer1", ResBasicBlock(64, 64, 1));
        layer2 = register_module("layer2", ResBasicBlock(64, 128, 2));
        layer3 = register_module("layer3", ResBasicBlock(128, 256, 2));
        layer4 = register_module("layer4", ResBasicBlock(256, 512, 2));

        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions(1)));
        fc = register_module("fc", torch::nn::Linear(512, numClasses));
        torch::nn::init::constant_(fc->bias, 0.0);
    }

    torch::Tensor forward(torch::Tensor x) override
    {
        x = maxpool->forward(torch::relu(bn1->forward(conv1->forward(x))));
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avgpool->forward(x).flatten(1);
        return fc->forward(x);
    }

    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::MaxPool2d maxpool{ nullptr };
    ResBasicBlock layer1{ nullptr }, layer2{ nullptr }, layer3{ nullptr }, layer4{ nullptr };
    torch::nn::AdaptiveAvgPool2d avgpool{ nullptr };
    torch::nn::Linear fc{ nullptr };
};
TORCH_MODULE(ResNet18);

// ---------------------------------------------------------------------------
//  分割模型：小型 U-Net（编码器 3 次下采样 + 解码器 3 次上采样 + 跳跃连接）
//  输入 [N,3,H,W]（H/W 需能被 8 整除）-> 输出 [N,numClasses,H,W] 的 logits。
// ---------------------------------------------------------------------------
struct UNetImpl : torch::nn::Module
{
    explicit UNetImpl(int numClasses)
    {
        enc1 = register_module("enc1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        enc2 = register_module("enc2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        enc3 = register_module("enc3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        // 瓶颈
        mid = register_module("mid", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        // 解码：拼接跳跃特征后做两次卷积融合
        dec3 = register_module("dec3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256 + 128, 128, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        dec2 = register_module("dec2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128 + 64, 64, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        dec1 = register_module("dec1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64 + 32, 32, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));

        pool = register_module("pool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(2).stride(2)));
        // 双线性插值上采样：比转置卷积更不容易产生棋盘伪影
        up = register_module("up", torch::nn::Upsample(
            torch::nn::UpsampleOptions()
                .scale_factor(std::vector<double>{ 2.0, 2.0 })
                .mode(torch::kBilinear)
                .align_corners(false)));
        outConv = register_module("outConv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(32, numClasses, 1)));
    }

    torch::Tensor forward(torch::Tensor x) override
    {
        torch::Tensor e1 = enc1->forward(x);
        torch::Tensor e2 = enc2->forward(pool->forward(e1));
        torch::Tensor e3 = enc3->forward(pool->forward(e2));
        torch::Tensor m  = mid->forward(pool->forward(e3));

        // 解码；上采样后裁到跳跃特征尺寸，避免奇数尺寸对不齐
        torch::Tensor d3 = up->forward(m);
        d3 = d3.slice(2, 0, e3.size(2)).slice(3, 0, e3.size(3));
        d3 = dec3->forward(torch::cat({ d3, e3 }, 1));

        torch::Tensor d2 = up->forward(d3);
        d2 = d2.slice(2, 0, e2.size(2)).slice(3, 0, e2.size(3));
        d2 = dec2->forward(torch::cat({ d2, e2 }, 1));

        torch::Tensor d1 = up->forward(d2);
        d1 = d1.slice(2, 0, e1.size(2)).slice(3, 0, e1.size(3));
        d1 = dec1->forward(torch::cat({ d1, e1 }, 1));

        return outConv->forward(d1);
    }

    torch::nn::Sequential enc1{ nullptr }, enc2{ nullptr }, enc3{ nullptr }, mid{ nullptr };
    torch::nn::Sequential dec3{ nullptr }, dec2{ nullptr }, dec1{ nullptr };
    torch::nn::MaxPool2d pool{ nullptr };
    torch::nn::Upsample up{ nullptr };
    torch::nn::Conv2d outConv{ nullptr };
};
TORCH_MODULE(UNet);

// ---------------------------------------------------------------------------
//  检测模型：简化版单阶段 anchor-free 检测头
//  ---------------------------------------------------------------------------
//  【重要】这是「简化实现」，不是完整 YOLO：
//    · 骨干只有 3 层卷积 + 2 次 stride=2 下采样，输出 SxS 特征图（S=输入/4）；
//    · 每个网格位置直接回归「中心点相对网格左上角的偏移 + 宽高」，不用 anchor；
//    · 分类头输出 numClasses 个通道，通道 0 固定为背景（与 classes.json 一致），
//      目标类别 id 从 1 开始；
//    · loss = 逐网格多分类交叉熵（背景降权）+ 5 倍框回归 MSE；
//    · 没有 FPN 多尺度、anchor 聚类、标签分配（SimOTA 等）、Mosaic 增强，
//      因此大小目标混杂时精度会明显低于完整 YOLO。
//  替换成完整检测器的位置：
//    1) 用 YOLO/RetinaNet 风格的骨干 + 检测头替换 DetectorImpl；
//    2) 修改 buildModel() 里 TaskType::Detection 分支的构造参数（grid 等）；
//    3) 替换 detectionLoss() 的标签编码与损失（如 CIoU + 分类 BCE + 正样本分配）；
//    4) 替换 decodeDetections() 的解码（多尺度 / anchor 解码）；
//    5) 推理端 computeDetectionMetric() 的 mAP@0.5 统计保持通用，可继续复用。
// ---------------------------------------------------------------------------
struct DetectorImpl : torch::nn::Module
{
    DetectorImpl(int numClasses, int gridSize)
        : grid(gridSize)
        , numCls(numClasses)
    {
        stem = register_module("stem", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).stride(2).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        head = register_module("head", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions(true))));
        clsHead = register_module("clsHead", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, numClasses, 1)));
        boxHead = register_module("boxHead", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(64, 4, 1)));
    }

    /// 返回 { cls [N,S*S,numClasses], box [N,S*S,4] }；
    /// box 是原始回归值，解码时对其做 sigmoid 得到偏移/宽高。
    std::vector<torch::Tensor> forward(torch::Tensor x) override
    {
        torch::Tensor f = head->forward(stem->forward(x));   // [N,64,gh,gw]
        const int64_t gh = f.size(2);
        const int64_t gw = f.size(3);
        const int64_t n  = f.size(0);

        torch::Tensor cls = clsHead->forward(f)
                                .permute({ 0, 2, 3, 1 })
                                .contiguous()
                                .view({ n, gh * gw, numCls });
        torch::Tensor box = boxHead->forward(f)
                                .permute({ 0, 2, 3, 1 })
                                .contiguous()
                                .view({ n, gh * gw, 4 });
        return { cls, box };
    }

    int64_t grid = 0;
    int64_t numCls = 0;
    torch::nn::Sequential stem{ nullptr }, head{ nullptr };
    torch::nn::Conv2d clsHead{ nullptr }, boxHead{ nullptr };
};
TORCH_MODULE(Detector);

// ===========================================================================
//  数据集
// ===========================================================================
/// 检测标注（归一化 cxcywh）。
struct DetTarget {
    int   classId = 0;   ///< 1 起，0 表示背景
    float cx = 0.f;
    float cy = 0.f;
    float w  = 0.f;
    float h  = 0.f;
};

/// 一个样本：由工程层传入的 TrainSample 展开而来。
struct SampleItem {
    QString image;
    QString label;          ///< 标注真源 .json（对象列表）
    QString mask;           ///< 分割派生掩码 .png（非分割任务为空）
    int     classId = -1;   ///< 分类任务的真值类别索引
};

/// 数据集类型（与 TaskType 一一对应）。
enum class DatasetKind : int {
    Classify = 0,
    Segment  = 1,
    Detect   = 2
};

/// 简易 Dataset：get() 时按需从磁盘读一张样本并预处理。
/// 返回张量形状：
///   · 分类：{ [3,H,W], 标量类别索引 }
///   · 分割：{ [3,H,W], [H,W] 类别掩码 }
///   · 检测：{ [3,H,W], [numObj,5]（classId,cx,cy,w,h） }
class SampleDataset : public torch::data::datasets::Dataset<SampleDataset>
{
public:
    SampleDataset(std::vector<SampleItem> items, DatasetKind kind, int inputW, int inputH)
        : m_items(std::move(items))
        , m_kind(kind)
        , m_w(inputW)
        , m_h(inputH)
    {
    }

    torch::data::Example<> get(size_t index) override
    {
        const SampleItem &s = m_items[index];

        cv::Mat bgr = readImageUnicode(s.image, cv::IMREAD_COLOR);
        if (bgr.empty())
            bgr = cv::Mat::zeros(m_h, m_w, CV_8UC3);   // 坏图用黑图兜底，避免训练崩溃
        torch::Tensor img = imageToTensor(bgr, m_w, m_h);

        switch (m_kind) {
        case DatasetKind::Classify:
            return { img, loadClassLabel(s) };
        case DatasetKind::Segment:
            return { img, loadMaskLabel(s) };
        case DatasetKind::Detect:
        default:
            return { img, loadDetLabel(s) };
        }
    }

    torch::optional<size_t> size() const override
    {
        return m_items.size();
    }

private:
    /// 分类标签：直接取工程层给出的 TrainSample.classId（0 起）。
    torch::Tensor loadClassLabel(const SampleItem &s) const
    {
        const int id = s.classId >= 0 ? s.classId : 0;
        return torch::tensor(id, torch::kLong);
    }

    /// 分割标签：优先用派生掩码 .png（CV_8UC1，像素值 = 类别索引，0 为背景）；
    /// 掩码缺失时由标注对象列表光栅化得到。
    torch::Tensor loadMaskLabel(const SampleItem &s) const
    {
        cv::Mat mask;
        if (!s.mask.isEmpty())
            mask = readImageUnicode(s.mask, cv::IMREAD_GRAYSCALE);

        if (mask.empty()) {
            QVector<AnnoObject> objects;
            anno::readLabel(s.label, s.mask, TaskType::Segmentation,
                            &objects, nullptr, nullptr, 16, nullptr);
            if (!objects.isEmpty())
                anno::rasterize(objects, m_w, m_h, &mask);
        }

        if (mask.empty())
            return torch::zeros({ m_h, m_w }, torch::kLong);
        if (mask.type() != CV_8UC1)
            mask.convertTo(mask, CV_8UC1);
        // 最近邻缩放，避免插值产生不存在的类别索引
        cv::Mat resized;
        cv::resize(mask, resized, cv::Size(m_w, m_h), 0.0, 0.0, cv::INTER_NEAREST);
        return torch::from_blob(resized.data, { m_h, m_w }, torch::kUInt8)
            .to(torch::kLong)
            .clone();
    }

    /// 检测标签：由标注对象列表得到 [numObj,5] 的 "classId cx cy w h"（归一化 0..1）。
    /// 若文件不是对象列表 JSON（老数据 / 手工文件），退回按文本行解析。
    torch::Tensor loadDetLabel(const SampleItem &s) const
    {
        std::vector<float> vals;

        QVector<AnnoObject> objects;
        if (anno::readLabel(s.label, s.mask, TaskType::Detection,
                            &objects, nullptr, nullptr, 16, nullptr)
            && !objects.isEmpty()) {
            for (const AnnoObject &o : objects) {
                if (!o.isValid() || o.classId <= 0)
                    continue;
                const QRectF r = o.boundingRect();
                if (r.width() <= 0.0 || r.height() <= 0.0)
                    continue;
                vals.push_back(static_cast<float>(o.classId));
                vals.push_back(static_cast<float>((r.x() + r.width() * 0.5) / m_w));
                vals.push_back(static_cast<float>((r.y() + r.height() * 0.5) / m_h));
                vals.push_back(static_cast<float>(r.width() / m_w));
                vals.push_back(static_cast<float>(r.height() / m_h));
            }
        }

        if (vals.empty()) {
            const QString txt = readTextFile(s.label);
            const QStringList lines = txt.split(
                QRegularExpression(QStringLiteral("[\\r\\n]+")), Qt::SkipEmptyParts);
            for (const QString &line : lines) {
                const QStringList parts = line.trimmed().split(
                    QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts);
                if (parts.size() < 5)
                    continue;
                for (int i = 0; i < 5; ++i)
                    vals.push_back(parts.at(i).toFloat());
            }
        }

        if (vals.empty())
            return torch::zeros({ 0, 5 }, torch::kFloat32);
        const int64_t nObj = static_cast<int64_t>(vals.size() / 5);
        return torch::from_blob(vals.data(), { nObj, 5 }, torch::kFloat32).clone();
    }

    std::vector<SampleItem> m_items;
    DatasetKind             m_kind = DatasetKind::Classify;
    int                     m_w = 224;
    int                     m_h = 224;
};

// ===========================================================================
//  指标计算
// ===========================================================================
/// 分类准确率：遍历整个数据集统计 top-1 正确率。
double computeAccuracy(SampleDataset &ds, torch::nn::Module &model,
                       const torch::Device &dev,
                       const std::atomic<bool> &stop, bool *aborted)
{
    torch::NoGradGuard noGrad;
    model.eval();

    auto loader = torch::data::make_data_loader(
        ds, torch::data::DataLoaderOptions().batch_size(1).workers(0));

    int64_t correct = 0;
    int64_t total = 0;
    for (auto &batch : *loader) {
        if (stop.load()) {
            if (aborted)
                *aborted = true;
            break;
        }
        torch::Tensor img = batch.data.to(dev);
        torch::Tensor lbl = batch.target.to(dev).flatten();
        torch::Tensor pred = model.forward(img).argmax(1);
        correct += pred.eq(lbl).sum().item<int64_t>();
        total += lbl.size(0);
    }
    model.train();
    return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
}

/// 分割：把一个 batch 的预测/真值累加进混淆矩阵（行=真实类别，列=预测类别）。
torch::Tensor updateConfusion(const torch::Tensor &logits, const torch::Tensor &target,
                              int numClasses)
{
    torch::NoGradGuard noGrad;
    torch::Tensor pred = logits.detach().argmax(1).to(torch::kCPU).flatten().to(torch::kLong);
    torch::Tensor gt = target.detach().to(torch::kCPU).flatten().to(torch::kLong);

    const int64_t n = qMin(pred.numel(), gt.numel());
    if (n <= 0)
        return torch::zeros({ numClasses, numClasses }, torch::kFloat64);
    pred = pred.narrow(0, 0, n);
    gt = gt.narrow(0, 0, n);

    const torch::Tensor idx = gt * numClasses + pred;
    return torch::bincount(idx, torch::Tensor(), numClasses * numClasses)
        .to(torch::kFloat64)
        .reshape({ numClasses, numClasses });
}

/// 由混淆矩阵计算 mIoU（忽略真实与预测中都没出现的类别）。
double miouFromConfusion(const torch::Tensor &conf, int numClasses)
{
    const torch::Tensor c = conf.to(torch::kCPU).to(torch::kFloat64);
    const double eps = 1e-9;
    double sum = 0.0;
    int cnt = 0;
    for (int k = 0; k < numClasses; ++k) {
        const double tp = c[k][k].item<double>();
        const double fp = c.select(0, k).sum().item<double>() - tp;   // 列和 - tp
        const double fn = c.select(1, k).sum().item<double>() - tp;   // 行和 - tp
        const double denom = tp + fp + fn;
        if (denom <= eps)
            continue;
        sum += tp / denom;
        ++cnt;
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

/// 检测框（归一化坐标），用于 NMS 与 mAP。
struct DetPred {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float score = 0.f;
    int   classId = 0;   ///< 1 起
};

/// 计算两个归一化框的 IoU。
inline float boxIoU(const DetPred &a, const DetPred &b)
{
    const float xx1 = std::max(a.x1, b.x1);
    const float yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2);
    const float yy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.f, xx2 - xx1);
    const float ih = std::max(0.f, yy2 - yy1);
    const float inter = iw * ih;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

/// 贪心 NMS：同一类别的框按分数从高到低保留，IoU 超阈值的抑制掉。
void nmsDetections(std::vector<DetPred> &preds, float iouThreshold)
{
    std::sort(preds.begin(), preds.end(),
              [](const DetPred &a, const DetPred &b) { return a.score > b.score; });

    std::vector<DetPred> kept;
    std::vector<bool> suppressed(preds.size(), false);
    for (size_t i = 0; i < preds.size(); ++i) {
        if (suppressed[i])
            continue;
        kept.push_back(preds[i]);
        for (size_t j = i + 1; j < preds.size(); ++j) {
            if (suppressed[j] || preds[j].classId != preds[i].classId)
                continue;
            if (boxIoU(preds[i], preds[j]) > iouThreshold)
                suppressed[j] = true;
        }
    }
    preds.swap(kept);
}

/// 单图解码结果：类别 id（1 起） -> 该类的候选框。
using DetPredMap = std::map<int, std::vector<DetPred>>;

/// 把网络输出解码成归一化坐标的检测框（含置信度阈值过滤；NMS 由调用方决定何时做）。
DetPredMap decodeDetections(const torch::Tensor &cls, const torch::Tensor &box,
                            int numClasses, int grid, float scoreThreshold)
{
    DetPredMap result;
    const int64_t cells = cls.size(1);
    const int64_t stride = static_cast<int64_t>(grid);
    if (cells <= 0 || stride <= 0 || numClasses <= 1)
        return result;

    const torch::Tensor prob = torch::sigmoid(cls.detach()).to(torch::kCPU).contiguous();
    const torch::Tensor reg = torch::sigmoid(box.detach()).to(torch::kCPU).contiguous();
    const float *pProb = prob.data_ptr<float>();
    const float *pReg = reg.data_ptr<float>();

    for (int64_t c = 0; c < cells; ++c) {
        const int gy = static_cast<int>(c / stride);
        const int gx = static_cast<int>(c % stride);
        const float *baseP = pProb + c * numClasses;
        const float *baseR = pReg + c * 4;

        // 从 1 开始取最大类别（跳过背景类 0）
        int best = 0;
        float bestScore = scoreThreshold;
        for (int k = 1; k < numClasses; ++k) {
            if (baseP[k] > bestScore) {
                bestScore = baseP[k];
                best = k;
            }
        }
        if (best == 0)
            continue;

        const float w = std::min(std::max(baseR[2], 1e-4f), 1.f);
        const float h = std::min(std::max(baseR[3], 1e-4f), 1.f);
        const float cx = (static_cast<float>(gx) + baseR[0]) / static_cast<float>(stride);
        const float cy = (static_cast<float>(gy) + baseR[1]) / static_cast<float>(stride);

        DetPred d;
        d.classId = best;
        d.score = bestScore;
        d.x1 = std::max(0.f, cx - w * 0.5f);
        d.y1 = std::max(0.f, cy - h * 0.5f);
        d.x2 = std::min(1.f, cx + w * 0.5f);
        d.y2 = std::min(1.f, cy + h * 0.5f);
        result[best].push_back(d);
    }
    return result;
}

/// mAP@0.5（简化实现）：
///   · 每张图先按类别做 NMS，再与真实框按 IoU>=0.5 贪心匹配（不重复匹配）；
///   · 逐类别累计 TP/FP，用 precision-recall 曲线积分求 AP，最后取均值。
double computeDetectionMetric(SampleDataset &ds, torch::nn::Module &model,
                              const torch::Device &dev, int numClasses, int grid,
                              float scoreThreshold, float nmsIoU,
                              const std::atomic<bool> &stop, bool *aborted)
{
    torch::NoGradGuard noGrad;
    model.eval();

    auto loader = torch::data::make_data_loader(
        ds, torch::data::DataLoaderOptions().batch_size(1).workers(0));

    // 每类的全部预测：{score, 是否 TP}
    std::vector<std::vector<std::pair<float, bool>>> perClass(
        static_cast<size_t>(qMax(1, numClasses)));
    std::vector<int64_t> gtCount(static_cast<size_t>(qMax(1, numClasses)), 0);

    for (auto &batch : *loader) {
        if (stop.load()) {
            if (aborted)
                *aborted = true;
            break;
        }
        torch::Tensor img = batch.data.to(dev);
        torch::Tensor target = batch.target.to(torch::kCPU);

        std::vector<torch::Tensor> outs = model.forward(img);
        DetPredMap preds = decodeDetections(outs[0][0], outs[1][0], numClasses, grid,
                                            scoreThreshold);

        // 整理真实框（仅保留 id 合法的）
        std::vector<DetTarget> gts;
        if (target.dim() == 2 && target.size(0) > 0 && target.size(1) >= 5) {
            const int64_t nObj = target.size(0);
            for (int64_t i = 0; i < nObj; ++i) {
                DetTarget g;
                g.classId = static_cast<int>(target[i][0].item<float>());
                g.cx = target[i][1].item<float>();
                g.cy = target[i][2].item<float>();
                g.w = target[i][3].item<float>();
                g.h = target[i][4].item<float>();
                if (g.classId >= 1 && g.classId < numClasses)
                    gts.push_back(g);
            }
        }
        for (const DetTarget &g : gts)
            gtCount[static_cast<size_t>(g.classId)] += 1;

        for (auto &kv : preds) {
            if (kv.first < 1 || kv.first >= numClasses)
                continue;
            std::vector<DetPred> list = kv.second;
            nmsDetections(list, nmsIoU);

            // 该类别的真实框转成 xyxy，并带一个「是否已被匹配」标记
            std::vector<std::pair<DetPred, bool>> gtsOfClass;
            for (const DetTarget &g : gts) {
                if (g.classId != kv.first)
                    continue;
                DetPred gd;
                gd.x1 = g.cx - g.w * 0.5f;
                gd.y1 = g.cy - g.h * 0.5f;
                gd.x2 = g.cx + g.w * 0.5f;
                gd.y2 = g.cy + g.h * 0.5f;
                gtsOfClass.emplace_back(gd, false);
            }

            for (const DetPred &p : list) {
                int bestIdx = -1;
                float bestIou = 0.f;
                for (size_t gi = 0; gi < gtsOfClass.size(); ++gi) {
                    if (gtsOfClass[gi].second)
                        continue;
                    const float iou = boxIoU(p, gtsOfClass[gi].first);
                    if (iou > bestIou) {
                        bestIou = iou;
                        bestIdx = static_cast<int>(gi);
                    }
                }
                bool isTp = false;
                if (bestIdx >= 0 && bestIou >= kMapIouThreshold) {
                    gtsOfClass[static_cast<size_t>(bestIdx)].second = true;
                    isTp = true;
                }
                perClass[static_cast<size_t>(kv.first)].emplace_back(p.score, isTp);
            }
        }
    }

    // 逐类别算 AP
    double sumAp = 0.0;
    int validClasses = 0;
    for (int c = 1; c < numClasses; ++c) {
        auto &list = perClass[static_cast<size_t>(c)];
        const int64_t nGt = gtCount[static_cast<size_t>(c)];
        if (nGt == 0 && list.empty())
            continue;                       // 该类别在验证集里完全没出现，跳过
        ++validClasses;
        if (list.empty())
            continue;                       // 有 GT 但没预测出来 -> AP 记 0

        std::sort(list.begin(), list.end(),
                  [](const std::pair<float, bool> &a, const std::pair<float, bool> &b) {
                      return a.first > b.first;
                  });

        std::vector<double> prec(list.size(), 0.0), rec(list.size(), 0.0);
        double tp = 0.0, fp = 0.0;
        for (size_t i = 0; i < list.size(); ++i) {
            if (list[i].second)
                tp += 1.0;
            else
                fp += 1.0;
            prec[i] = tp / (tp + fp);
            rec[i] = nGt > 0 ? tp / static_cast<double>(nGt) : 0.0;
        }
        // 单调化 precision，再按 recall 增量积分
        for (size_t i = prec.size(); i-- > 1;) {
            if (prec[i - 1] < prec[i])
                prec[i - 1] = prec[i];
        }
        double ap = 0.0;
        double prevRec = 0.0;
        for (size_t i = 0; i < prec.size(); ++i) {
            ap += (rec[i] - prevRec) * prec[i];
            prevRec = rec[i];
        }
        sumAp += ap;
    }

    model.train();
    return validClasses > 0 ? sumAp / validClasses : 0.0;
}

// ===========================================================================
//  检测损失（简化实现）
// ===========================================================================
/// 网格级「多分类交叉熵 + 框回归 MSE」：
///   编码方式：目标中心落在哪个网格，该网格即为正样本，
///     tx = cx*S - gx（相对网格左上角的偏移），ty 同理，tw = w，th = h；
///   分类：每个网格位置都参与交叉熵，背景类（0）降权以缓解正负样本失衡。
torch::Tensor detectionLoss(const torch::Tensor &outCls, const torch::Tensor &outBox,
                            const torch::Tensor &target, int grid, int numClasses)
{
    const int64_t cells = outCls.size(1);
    const int64_t n = outCls.size(0);

    std::vector<torch::Tensor> clsTargets;
    std::vector<torch::Tensor> boxTargets;
    std::vector<torch::Tensor> regMasks;
    clsTargets.reserve(static_cast<size_t>(n));
    boxTargets.reserve(static_cast<size_t>(n));
    regMasks.reserve(static_cast<size_t>(n));

    for (int64_t b = 0; b < n; ++b) {
        torch::Tensor ct = torch::zeros({ cells }, torch::kLong);          // 全背景
        torch::Tensor bt = torch::zeros({ cells, 4 }, torch::kFloat32);
        torch::Tensor rm = torch::zeros({ cells }, torch::kFloat32);

        if (b < target.size(0) && target.dim() == 2 && target.size(1) >= 5 && cells > 0) {
            const torch::Tensor row = target[b];
            const int64_t nObj = row.size(0);
            for (int64_t i = 0; i < nObj; ++i) {
                const int   cid = static_cast<int>(row[i][0].item<float>());
                const float cx = row[i][1].item<float>();
                const float cy = row[i][2].item<float>();
                const float w  = row[i][3].item<float>();
                const float h  = row[i][4].item<float>();

                if (cid < 1 || cid >= numClasses)   // 0 是背景，越界忽略
                    continue;

                int gx = static_cast<int>(std::floor(cx * static_cast<float>(grid)));
                int gy = static_cast<int>(std::floor(cy * static_cast<float>(grid)));
                gx = clampInt(gx, 0, grid - 1);
                gy = clampInt(gy, 0, grid - 1);
                const int64_t cell = static_cast<int64_t>(gy) * grid + gx;
                if (cell < 0 || cell >= cells)
                    continue;

                ct[cell] = cid;
                bt[cell][0] = cx * static_cast<float>(grid) - static_cast<float>(gx);
                bt[cell][1] = cy * static_cast<float>(grid) - static_cast<float>(gy);
                bt[cell][2] = w;
                bt[cell][3] = h;
                rm[cell] = 1.0f;
            }
        }
        clsTargets.push_back(ct);
        boxTargets.push_back(bt);
        regMasks.push_back(rm);
    }

    const torch::Tensor clsT = torch::stack(clsTargets, 0);              // [N,cells]
    const torch::Tensor boxT = torch::stack(boxTargets, 0);              // [N,cells,4]
    const torch::Tensor mask = torch::stack(regMasks, 0).unsqueeze(2);   // [N,cells,1]

    // 分类损失：背景类权重降为 kDetBgWeight，其余为 1，缓解正负样本严重失衡
    torch::Tensor clsWeight = torch::full({ numClasses }, 1.0, torch::kFloat32);
    clsWeight[0] = static_cast<float>(kDetBgWeight);

    torch::nn::functional::CrossEntropyFuncOptions lossOpts;
    lossOpts.weight(clsWeight).reduction(torch::Reduction::Mean).ignore_index(-100);

    const torch::Tensor clsLoss = torch::nn::functional::cross_entropy(
        outCls.reshape({ -1, numClasses }), clsT.reshape({ -1 }), lossOpts);

    // 框回归损失：仅统计正样本网格
    const double posCount = static_cast<double>(mask.sum().item<float>());
    torch::Tensor boxLoss;
    if (posCount > 0.0) {
        const torch::Tensor diff = (torch::sigmoid(outBox) - boxT) * mask;
        boxLoss = diff.pow(2).sum() / std::max(1.0, posCount);
    } else {
        boxLoss = outBox.sum() * 0.0;   // 无正样本时保持计算图连通
    }
    return clsLoss + kDetBoxLossWeight * boxLoss;
}

// ---------------------------------------------------------------------------
//  DataLoader 选项
// ---------------------------------------------------------------------------
// 说明：检测任务的目标数量随图变化，无法固定 collate 后的形状，因此统一按
// batch_size = 1 迭代，再由各训练循环手工累积梯度凑成配置的 batchSize。
// workers 固定为 0（LibTorch 自带的多线程加载在 Windows 上易与 Qt 事件循环
// 争抢资源，这里保持单线程，训练耗时用进度信号向界面反馈即可）。
inline torch::data::DataLoaderOptions oneShotOptions()
{
    return torch::data::DataLoaderOptions().batch_size(1).workers(0);
}

// ---------------------------------------------------------------------------
//  验证结果
// ---------------------------------------------------------------------------
/// 验证集评估结果。
struct EvalResult {
    double  valLoss = 0.0;      ///< 验证损失
    double  valMetric = 0.0;    ///< 任务指标（acc / mIoU / mAP@0.5）
    QString metricName;         ///< 指标显示名
    bool    aborted = false;    ///< 评估过程是否被 requestStop 打断
};

} // namespace

// ===========================================================================
//  Impl：全部 LibTorch 相关状态与训练流程
//  ===========================================================================
struct Impl {
    // --- 配置与数据 ---
    TrainConfig             cfg;
    QVector<ClassInfo>      classInfos;      ///< 工程层传入的类别（下标即类别 id）
    QStringList             classNames;      ///< 仅名称，供日志与元数据使用
    std::vector<SampleItem> trainItems;
    std::vector<SampleItem> valItems;

    // --- 运行时 ---
    QString       deviceNote;
    torch::Device device{ torch::kCPU };
    bool          hasModel = false;    ///< 模型是否已构建/载入
    bool          loaded   = false;    ///< 是否已完成 loadModel（用于推理）
    QString       modelPath;
    QString       metaPath;
    QVector<ClassInfo> modelClasses;   ///< 载入后的模型类别（供 modelClasses() 返回）
    QString       metricsSummary;      ///< 训练阶段验证指标摘要，写入元数据

    // 三类模型（按任务只用一个，其余保持空）
    SmallCNN clsNet{ nullptr };
    ResNet18 resNet{ nullptr };
    UNet     segNet{ nullptr };
    Detector detNet{ nullptr };

    // --- 线程（内部 std::thread，与 OpenCvTrainer 保持一致的线程模型）---
    std::shared_ptr<std::atomic<bool>> stopFlag;
    std::atomic<bool>  running{ false };
    std::thread        worker;
    mutable QMutex     stopMutex;      ///< 保护 stopFlag 指针的读取

    // --- 跨线程上报回调（由 LibTorchTrainer 在主线程构造，内部走队列连接）---
    std::function<void(const QString &, int)> logFn;
    std::function<void(int)>                  progressFn;
    std::function<void(int, int, double, double, double, double)> epochFn;
    std::function<void(bool, const QString &, const QString &)>   finishedFn;

    // --- epoch 内累计统计（每轮开始清零） ---
    double          sumTrainLoss = 0.0;   ///< 训练 loss 累加
    int64_t         nTrainSeen = 0;       ///< 已参与训练的样本数
    torch::Tensor   confusion;            ///< 分割：本轮累计混淆矩阵
    int64_t         gridSize = 0;         ///< 检测：特征图边长

    // --- 训练超参数（从 cfg 派生） ---
    int    batchSize = 8;
    int    inputW = 224;
    int    inputH = 224;
    double lr = 0.001;
    TestConfig test;                     ///< 推理后处理参数（阈值等）
    float  detScoreThreshold = kDetScoreThreshold;
    float  detNmsIoU = kDetNmsIoU;

    // --- 方法 ---
    /// 校验工程层传入的样本列表（不再扫描目录、不再读 classes.json）
    bool loadSamples(const QVector<TrainSample> &samples, QString *err);
    bool splitTrainVal(QString *err);
    void buildModel();
    void setTrainMode(bool training);
    void clearEpochStats();
    bool trainImpl(QString *err);
    bool trainClassification(torch::optim::Optimizer &opt, QString *err);
    bool trainSegmentation(torch::optim::Optimizer &opt, QString *err);
    bool trainDetection(torch::optim::Optimizer &opt, QString *err);
    EvalResult evaluate();
    bool saveModel(const QString &path, QString *err);
    bool loadModelImpl(const QString &path, QString *err);
    void predictImpl(const cv::Mat &bgr, PredictionResult *out, QString *err);
    void requestStop();
};

// ---------------------------------------------------------------------------
//  样本装载与校验
// ---------------------------------------------------------------------------
//  数据来源换成工程层传入的 TrainSample 列表：
//    · 分类：真值取 sample.classId（<0 表示该图无标注，跳过）；
//    · 检测：真值取 sample.labelPath 的标注对象列表；
//    · 分割：真值优先 sample.maskPath 的派生掩码，缺失时由对象列表光栅化。
//  这里只做「列表校验」，不访问 classes.json，也不扫描目录。
// ---------------------------------------------------------------------------
bool Impl::loadSamples(const QVector<TrainSample> &samples, QString *err)
{
    trainItems.clear();
    valItems.clear();

    int skippedNoImage = 0;
    int skippedNoLabel = 0;

    for (const TrainSample &s : samples) {
        if (!s.valid() || !QFileInfo::exists(s.imagePath)) {
            ++skippedNoImage;
            continue;
        }

        SampleItem item;
        item.image   = s.imagePath;
        item.label   = s.labelPath;
        item.mask    = s.maskPath;
        item.classId = s.classId;

        if (cfg.task == TaskType::Classification) {
            // 分类：没有有效类别标注的图无法参与训练
            if (s.classId < 0 || s.classId >= classInfos.size()) {
                ++skippedNoLabel;
                continue;
            }
        } else {
            // 检测 / 分割：必须有标注真源（对象列表或派生掩码）
            const bool hasLabel = !s.labelPath.trimmed().isEmpty()
                                  && QFileInfo::exists(s.labelPath);
            const bool hasMask = !s.maskPath.trimmed().isEmpty()
                                 && QFileInfo::exists(s.maskPath);
            if (!hasLabel && !hasMask) {
                ++skippedNoLabel;
                continue;
            }
        }

        trainItems.push_back(item);
    }

    if (trainItems.empty()) {
        if (err) {
            *err = QStringLiteral("没有可用的训练样本（图像缺失 %1 个，标注缺失 %2 个）")
                       .arg(skippedNoImage)
                       .arg(skippedNoLabel);
        }
        return false;
    }
    return true;
}

/// 按 valSplit（cfg.extra，缺省 0.2）划分训练 / 验证集，固定种子保证可复现
bool Impl::splitTrainVal(QString *err)
{
    Q_UNUSED(err);

    const int usable = static_cast<int>(trainItems.size());
    if (usable < 2) {
        valItems.clear();      // 只有一个样本：全部用于训练
        return true;
    }

    double split = 0.2;
    if (cfg.extra.contains(QStringLiteral("valSplit"))) {
        bool ok = false;
        const double v = cfg.extra.value(QStringLiteral("valSplit")).toDouble(&ok);
        if (ok)
            split = v;
    }
    if (!(split > 0.0 && split < 0.5))
        split = 0.0;

    std::mt19937 rng(static_cast<unsigned>(kSeed));
    std::shuffle(trainItems.begin(), trainItems.end(), rng);

    size_t nVal = static_cast<size_t>(split * static_cast<double>(usable));
    if (nVal >= static_cast<size_t>(usable))
        nVal = static_cast<size_t>(usable) - 1;      // 至少留一个训练样本
    if (nVal == 0 && split > 0.0 && usable >= 2)
        nVal = 1;                                    // 小数据集也给一个验证样本

    if (nVal > 0) {
        valItems.assign(trainItems.end() - static_cast<std::ptrdiff_t>(nVal),
                        trainItems.end());
        trainItems.resize(trainItems.size() - nVal);
    } else {
        valItems.clear();
    }
    return true;
}

// ---------------------------------------------------------------------------
//  模型构建 / 模式切换 / 统计清零
// ---------------------------------------------------------------------------
void Impl::buildModel()
{
    hasModel = false;
    const int numClasses = qMax(1, cfg.numClasses);
    const QString tag = cfg.model.trimmed().toLower();

    switch (cfg.task) {
    case TaskType::Classification:
        if (tag.contains(QStringLiteral("resnet"))) {
            // cfg.model 命中 resnet -> 使用 ResNet18 风格骨干
            resNet = ResNet18(numClasses);
            resNet->to(device);
            clsNet = nullptr;
        } else {
            // 其余情况（含模型标识无法识别）回退到内置小型 CNN
            clsNet = SmallCNN(numClasses, inputW, inputH);
            clsNet->to(device);
            resNet = nullptr;
        }
        break;
    case TaskType::Segmentation:
        segNet = UNet(numClasses);
        segNet->to(device);
        break;
    case TaskType::Detection:
    default:
        gridSize = qMax(1, inputW / 4);   // 检测骨干共下采样 4 倍
        detNet = Detector(numClasses, static_cast<int>(gridSize));
        detNet->to(device);
        break;
    }
    hasModel = true;
}

void Impl::setTrainMode(bool training)
{
    switch (cfg.task) {
    case TaskType::Classification:
        if (resNet)
            resNet->train(training);
        if (clsNet)
            clsNet->train(training);
        break;
    case TaskType::Segmentation:
        if (segNet)
            segNet->train(training);
        break;
    case TaskType::Detection:
    default:
        if (detNet)
            detNet->train(training);
        break;
    }
}

void Impl::clearEpochStats()
{
    sumTrainLoss = 0.0;
    nTrainSeen = 0;
    confusion = torch::Tensor();
}

// ---------------------------------------------------------------------------
//  分类训练
// ---------------------------------------------------------------------------
bool Impl::trainClassification(torch::optim::Optimizer &opt, QString *err)
{
    Q_UNUSED(err);
    SampleDataset ds(trainItems, DatasetKind::Classify, inputW, inputH);
    auto loader = torch::data::make_data_loader(ds, oneShotOptions());

    torch::nn::Module &model = resNet ? static_cast<torch::nn::Module &>(*resNet)
                                      : static_cast<torch::nn::Module &>(*clsNet);

    // 分类任务：classes.json 只有真实类别，标签索引从 0 开始，直接用交叉熵。
    // 这里用 nn::functional 接口（全部参数走 Options 对象），避免 torch::cross_entropy
    // 的位置参数/命名参数重载在 MSVC 下产生二义性。
    torch::nn::functional::CrossEntropyFuncOptions lossOpts;
    lossOpts.reduction(torch::Reduction::Mean);

    int accum = 0;
    opt.zero_grad();
    for (auto &batch : *loader) {
        if (stopFlag && stopFlag->load())
            break;
        torch::Tensor img = batch.data.to(device);
        torch::Tensor lbl = batch.target.to(device).flatten();

        torch::Tensor logits = model.forward(img);
        torch::Tensor loss = torch::nn::functional::cross_entropy(logits, lbl, lossOpts);

        loss.backward();
        sumTrainLoss += loss.item<double>();
        ++nTrainSeen;
        if (++accum >= batchSize) {
            opt.step();
            opt.zero_grad();
            accum = 0;
        }
    }
    if (accum > 0) {   // 收尾：最后一组不足一个 batch 也要更新
        opt.step();
        opt.zero_grad();
    }
    return true;
}

// ---------------------------------------------------------------------------
//  分割训练
// ---------------------------------------------------------------------------
bool Impl::trainSegmentation(torch::optimizer::Optimizer &opt, QString *err)
{
    Q_UNUSED(err);
    SampleDataset ds(trainItems, DatasetKind::Segment, inputW, inputH);
    auto loader = torch::data::make_data_loader(ds, oneShotOptions());

    const int numClasses = qMax(1, cfg.numClasses);
    if (!confusion.defined())
        confusion = torch::zeros({ numClasses, numClasses }, torch::kFloat64);

    // 255 = 忽略（未标注）像素
    torch::nn::functional::CrossEntropyFuncOptions lossOpts;
    lossOpts.ignore_index(255).reduction(torch::Reduction::Mean);

    int accum = 0;
    opt.zero_grad();
    for (auto &batch : *loader) {
        if (stopFlag && stopFlag->load())
            break;
        torch::Tensor img = batch.data.to(device);
        torch::Tensor mask = batch.target.to(device);
        if (mask.dim() == 1)
            mask = mask.view({ 1, inputH, inputW });

        torch::Tensor logits = segNet->forward(img);   // [N,numClasses,H,W]
        torch::Tensor loss = torch::nn::functional::cross_entropy(logits, mask, lossOpts);

        loss.backward();
        sumTrainLoss += loss.item<double>();
        ++nTrainSeen;
        if (++accum >= batchSize) {
            opt.step();
            opt.zero_grad();
            accum = 0;
        }
        confusion += updateConfusion(logits, mask, numClasses);
    }
    if (accum > 0) {
        opt.step();
        opt.zero_grad();
    }
    return true;
}

// ---------------------------------------------------------------------------
//  检测训练（简化实现，见 DetectorImpl 处说明）
// ---------------------------------------------------------------------------
bool Impl::trainDetection(torch::optimizer::Optimizer &opt, QString *err)
{
    Q_UNUSED(err);
    const int numClasses = qMax(1, cfg.numClasses);
    SampleDataset ds(trainItems, DatasetKind::Detect, inputW, inputH);
    auto loader = torch::data::make_data_loader(ds, oneShotOptions());

    // 目标数量随图变化 -> 一个 mini-batch 内的目标手动拼接成 [B,maxObj,5]，
    // 短的行用 0 补齐（0 视为背景、无正样本，不影响损失）。
    std::vector<torch::Tensor> imgBuf;
    std::vector<torch::Tensor> tgtBuf;

    // 把收集到的样本导成「已补齐的 batch 张量」
    auto flush = [&](bool force) -> bool {
        if (imgBuf.empty())
            return false;
        if (!force && static_cast<int>(imgBuf.size()) < batchSize)
            return false;

        const int64_t b = static_cast<int64_t>(imgBuf.size());
        torch::Tensor imgs = torch::cat(imgBuf, 0);                 // [B,3,H,W]
        int64_t maxObj = 0;
        for (const torch::Tensor &t : tgtBuf)
            maxObj = std::max(maxObj, t.size(0));
        if (maxObj <= 0)
            maxObj = 1;

        torch::Tensor tgt = torch::zeros({ b, maxObj, 5 }, torch::kFloat32);
        for (int64_t i = 0; i < b; ++i) {
            const torch::Tensor &t = tgtBuf[static_cast<size_t>(i)];
            if (t.size(0) > 0)
                tgt[i].narrow(0, 0, t.size(0)).copy_(t);
        }

        std::vector<torch::Tensor> outs = detNet->forward(imgs.to(device));
        torch::Tensor loss = detectionLoss(outs[0], outs[1], tgt.to(device),
                                           static_cast<int>(gridSize), numClasses);
        loss.backward();
        opt.step();
        opt.zero_grad();
        sumTrainLoss += loss.item<double>();
        nTrainSeen += b;

        imgBuf.clear();
        tgtBuf.clear();
        return true;
    };

    for (auto &batch : *loader) {
        if (stopFlag && stopFlag->load())
            break;
        imgBuf.push_back(batch.data);        // [1,3,H,W]
        tgtBuf.push_back(batch.target);      // [numObj,5]
        if (static_cast<int>(imgBuf.size()) >= batchSize)
            flush(true);
    }
    flush(true);   // 收尾
    return true;
}

// ---------------------------------------------------------------------------
//  验证集评估：算出 valLoss 与任务指标
// ---------------------------------------------------------------------------
EvalResult Impl::evaluate()
{
    EvalResult r;
    r.metricName = (cfg.task == TaskType::Classification)
                       ? QStringLiteral("acc")
                   : (cfg.task == TaskType::Segmentation) ? QStringLiteral("mIoU")
                                                          : QStringLiteral("mAP@0.5");

    if (valItems.empty()) {
        // 没有验证集：调用方会用训练集指标兜底
        r.valLoss = -1.0;
        r.valMetric = -1.0;
        return r;
    }

    const int numClasses = qMax(1, cfg.numClasses);
    torch::NoGradGuard noGrad;
    setTrainMode(false);

    SampleDataset ds(valItems,
                     cfg.task == TaskType::Classification ? DatasetKind::Classify
                     : cfg.task == TaskType::Segmentation ? DatasetKind::Segment
                                                          : DatasetKind::Detect,
                     inputW, inputH);
    auto loader = torch::data::make_data_loader(ds, oneShotOptions());

    switch (cfg.task) {
    // ---- 分类：验证损失 + 准确率 ----
    case TaskType::Classification: {
        torch::nn::functional::CrossEntropyFuncOptions lossOpts;
        lossOpts.reduction(torch::Reduction::Mean);
        int64_t correct = 0;
        int64_t total = 0;
        double sum = 0.0;
        for (auto &batch : *loader) {
            if (stopFlag && stopFlag->load()) {
                r.aborted = true;
                break;
            }
            torch::Tensor img = batch.data.to(device);
            torch::Tensor lbl = batch.target.to(device).flatten();
            torch::Tensor logits = resNet ? resNet->forward(img) : clsNet->forward(img);
            torch::Tensor loss = torch::nn::functional::cross_entropy(logits, lbl, lossOpts);
            sum += loss.item<double>();
            correct += logits.argmax(1).eq(lbl).sum().item<int64_t>();
            total += lbl.size(0);
        }
        r.valLoss = total > 0 ? sum / static_cast<double>(total) : 0.0;
        r.valMetric = total > 0
                          ? static_cast<double>(correct) / static_cast<double>(total)
                          : 0.0;
        break;
    }
    // ---- 分割：验证损失 + mIoU ----
    case TaskType::Segmentation: {
        torch::nn::functional::CrossEntropyFuncOptions lossOpts;
        lossOpts.ignore_index(255).reduction(torch::Reduction::Mean);
        torch::Tensor conf = torch::zeros({ numClasses, numClasses }, torch::kFloat64);
        double sum = 0.0;
        int64_t total = 0;
        for (auto &batch : *loader) {
            if (stopFlag && stopFlag->load()) {
                r.aborted = true;
                break;
            }
            torch::Tensor img = batch.data.to(device);
            torch::Tensor mask = batch.target.to(device);
            if (mask.dim() == 1)
                mask = mask.view({ 1, inputH, inputW });
            torch::Tensor logits = segNet->forward(img);
            torch::Tensor loss = torch::nn::functional::cross_entropy(logits, mask, lossOpts);
            sum += loss.item<double>();
            ++total;
            conf += updateConfusion(logits, mask, numClasses);
        }
        r.valLoss = total > 0 ? sum / static_cast<double>(total) : 0.0;
        r.valMetric = miouFromConfusion(conf, numClasses);
        break;
    }
    // ---- 检测：验证损失 + mAP@0.5 ----
    case TaskType::Detection:
    default: {
        std::vector<torch::Tensor> imgBuf;
        std::vector<torch::Tensor> tgtBuf;
        double lossSum = 0.0;
        int64_t batches = 0;
        auto flush = [&]() {
            if (imgBuf.empty())
                return;
            const int64_t b = static_cast<int64_t>(imgBuf.size());
            torch::Tensor imgs = torch::cat(imgBuf, 0);
            int64_t maxObj = 0;
            for (const torch::Tensor &t : tgtBuf)
                maxObj = std::max(maxObj, t.size(0));
            if (maxObj <= 0)
                maxObj = 1;
            torch::Tensor tgt = torch::zeros({ b, maxObj, 5 }, torch::kFloat32);
            for (int64_t i = 0; i < b; ++i) {
                const torch::Tensor &t = tgtBuf[static_cast<size_t>(i)];
                if (t.size(0) > 0)
                    tgt[i].narrow(0, 0, t.size(0)).copy_(t);
            }
            std::vector<torch::Tensor> outs = detNet->forward(imgs.to(device));
            torch::Tensor loss = detectionLoss(outs[0], outs[1], tgt.to(device),
                                               static_cast<int>(gridSize), numClasses);
            lossSum += loss.item<double>();
            ++batches;
            imgBuf.clear();
            tgtBuf.clear();
        };

        for (auto &batch : *loader) {
            if (stopFlag && stopFlag->load()) {
                r.aborted = true;
                break;
            }
            imgBuf.push_back(batch.data);
            tgtBuf.push_back(batch.target);
            if (static_cast<int>(imgBuf.size()) >= batchSize)
                flush();
        }
        flush();
        r.valLoss = batches > 0 ? lossSum / static_cast<double>(batches) : 0.0;
        r.valMetric = computeDetectionMetric(ds, *detNet, device, numClasses,
                                             static_cast<int>(gridSize),
                                             detScoreThreshold, detNmsIoU,
                                             *stopFlag, &r.aborted);
        break;
    }
    }

    setTrainMode(true);
    return r;
}

// ---------------------------------------------------------------------------
//  模型与元数据保存
// ---------------------------------------------------------------------------
bool Impl::saveModel(const QString &path, QString *err)
{
    // 输出目录不存在则创建
    const QDir dir = QFileInfo(path).absoluteDir();
    if (!dir.exists() && !dir.mkpath(QStringLiteral("."))) {
        if (err)
            *err = QStringLiteral("无法创建模型输出目录：%1").arg(dir.absolutePath());
        return false;
    }

    try {
        // LibTorch 的序列化接口在 Windows 上要求 UTF-8 窄字符路径
        switch (cfg.task) {
        case TaskType::Classification:
            if (resNet)
                torch::save(resNet, toStdString(path));
            else
                torch::save(clsNet, toStdString(path));
            break;
        case TaskType::Segmentation:
            torch::save(segNet, toStdString(path));
            break;
        case TaskType::Detection:
        default:
            torch::save(detNet, toStdString(path));
            break;
        }
    } catch (const std::exception &e) {
        if (err)
            *err = QStringLiteral("模型保存失败：%1").arg(QString::fromUtf8(e.what()));
        return false;
    }

    // 元数据统一由 modelio::writeModelMeta() 落盘（不再自行拼 JSON）
    modelio::ModelMeta meta;
    meta.task           = cfg.task;
    meta.backend        = QStringLiteral("libtorch");
    meta.model          = cfg.model;
    meta.inputWidth     = inputW;
    meta.inputHeight    = inputH;
    meta.classes        = classInfos;
    meta.trainedAt      = QDateTime::currentDateTime().toString(Qt::ISODate);
    meta.metricsSummary = metricsSummary;
    if (!modelio::writeModelMeta(path, meta, err))
        return false;

    modelPath = path;
    metaPath = modelio::metaPath(path);
    return true;
}

// ---------------------------------------------------------------------------
//  训练主流程（在工作线程里执行）
// ---------------------------------------------------------------------------
bool Impl::trainImpl(QString *err)
{
    // 训练过程统计（跨 epoch）
    double bestVal = -1.0;
    int    bestEpoch = 0;

    // 跨线程上报：工作线程通过下面的回调（内部用 QMetaObject 队列调用）把
    // 信号投递回主线程（trainer 自身属于主线程），与 OpenCvTrainer 做法一致。

    // 进度上报：只在百分比变化时发信号，避免刷屏
    int lastPercent = -1;
    auto emitProgress = [&](int percent) {
        const int p = clampInt(percent, 0, 100);
        if (p == lastPercent)
            return;
        lastPercent = p;
        if (progressFn)
            progressFn(p);
    };

    // 优化器：按 cfg.model 里的关键字选择 Adam；否则 SGD(动量 0.9)
    torch::nn::Module &model = (cfg.task == TaskType::Classification)
                                   ? (resNet ? static_cast<torch::nn::Module &>(*resNet)
                                             : static_cast<torch::nn::Module &>(*clsNet))
                               : (cfg.task == TaskType::Segmentation)
                                   ? static_cast<torch::nn::Module &>(*segNet)
                                   : static_cast<torch::nn::Module &>(*detNet);

    const QString tag = cfg.model.trimmed().toLower();

    // 优化器标签对应的日志名（真正的构造在下面的分派里）
    const bool useAdam = tag.contains(QStringLiteral("adam"));
    const QString optName = useAdam ? QStringLiteral("Adam")
                                    : QStringLiteral("SGD(momentum=0.9)");

    if (logFn) {
        logFn(QStringLiteral("优化器=%1 | 学习率=%2 | batch=%3 | 设备=%4")
                  .arg(optName)
                  .arg(lr, 0, 'g', 4)
                  .arg(batchSize)
                  .arg(deviceNote),
              ITrainer::Info);
        logFn(QStringLiteral("训练集 %1 张，验证集 %2 张，类别数 %3（%4 个 epoch）")
                  .arg(trainItems.size())
                  .arg(valItems.size())
                  .arg(qMax(1, cfg.numClasses))
                  .arg(cfg.epochs),
              ITrainer::Info);
        if (cfg.task == TaskType::Detection) {
            logFn(QStringLiteral("注意：检测为「简化版单阶段 anchor-free 检测头」"
                                 "（网格 %1x%1），非完整 YOLO，"
                                 "替换位置见 LibTorchTrainer.cpp 注释")
                      .arg(gridSize),
                  ITrainer::Warning);
        }
    }

    // 开启异常检测：出现 NaN/Inf 时直接抛出，避免静默训坏模型。
    // 用 RAII 保证无论正常结束还是异常退出都能恢复全局状态。
    struct AnomalyGuard {
        AnomalyGuard() { torch::autograd::AnomalyMode::set_enabled(true); }
        ~AnomalyGuard() { torch::autograd::AnomalyMode::set_enabled(false); }
    } anomalyGuard;
    (void)anomalyGuard;

    // 说明：在每次迭代前检查停止标志，保证 requestStop() 能在一张图之内生效。
    for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
        if (stopFlag && stopFlag->load())
            break;

        clearEpochStats();
        setTrainMode(true);
        emitProgress(epoch > 1 ? (epoch - 1) * 100 / qMax(1, cfg.epochs) : 0);

        // 用模板 lambda 统一两种优化器的调用路径：
        // torch::optim::Optimizer 的析构函数不是虚函数，不能直接放进
        // unique_ptr<Optimizer> 里持有派生类对象，所以在这里分派具体类型。
        auto runEpoch = [&](auto &optimizer) -> bool {
            try {
                if (cfg.task == TaskType::Classification)
                    return trainClassification(optimizer, err);
                if (cfg.task == TaskType::Segmentation)
                    return trainSegmentation(optimizer, err);
                return trainDetection(optimizer, err);
            } catch (const std::exception &e) {
                if (err)
                    *err = QStringLiteral("训练过程中发生异常（第 %1 轮）：%2")
                               .arg(epoch)
                               .arg(QString::fromUtf8(e.what()));
                return false;
            }
        };

        bool ok = false;
        try {
            if (useAdam) {
                torch::optim::Adam opt(model.parameters(), torch::optim::AdamOptions(lr));
                ok = runEpoch(opt);
            } else {
                torch::optim::SGD opt(model.parameters(),
                                      torch::optim::SGDOptions(lr)
                                          .momentum(0.9)
                                          .weight_decay(0.0));
                ok = runEpoch(opt);
            }
        } catch (const std::exception &e) {
            if (err)
                *err = QStringLiteral("训练过程中发生异常（第 %1 轮）：%2")
                           .arg(epoch)
                           .arg(QString::fromUtf8(e.what()));
            return false;
        }
        if (!ok)
            return false;

        const double trainLoss =
            nTrainSeen > 0 ? sumTrainLoss / static_cast<double>(nTrainSeen) : 0.0;

        // 训练集指标：分类用准确率，分割用本轮混淆矩阵算 mIoU；
        // 检测的训练集 mAP 开销较大，直接用验证阶段的 mAP 作为参考。
        double trainMetric = 0.0;
        if (cfg.task == TaskType::Classification) {
            bool accAborted = false;
            SampleDataset ds(trainItems, DatasetKind::Classify, inputW, inputH);
            torch::nn::Module &m = resNet ? static_cast<torch::nn::Module &>(*resNet)
                                          : static_cast<torch::nn::Module &>(*clsNet);
            trainMetric = computeAccuracy(ds, m, device, *stopFlag, &accAborted);
        } else if (cfg.task == TaskType::Segmentation) {
            trainMetric = miouFromConfusion(
                confusion.defined()
                    ? confusion
                    : torch::zeros({ qMax(1, cfg.numClasses), qMax(1, cfg.numClasses) },
                                   torch::kFloat64),
                qMax(1, cfg.numClasses));
        }

        // ---- 验证阶段 ----
        EvalResult ev = evaluate();
        if (ev.valLoss < 0.0) {          // 没有验证集：用训练集数值兜底
            ev.valLoss = trainLoss;
            ev.valMetric = trainMetric;
        }

        // 记录最佳轮次（以验证指标为准；没有验证集时上面已用训练指标兜底）
        if (bestEpoch == 0 || ev.valMetric > bestVal) {
            bestVal = ev.valMetric;
            bestEpoch = epoch;
        }

        // ---- 上报进度 / epoch 结果 / 日志（都在主线程被接收） ----
        emitProgress(epoch * 100 / qMax(1, cfg.epochs));
        if (epochFn) {
            epochFn(epoch, cfg.epochs, trainLoss, trainMetric, ev.valLoss, ev.valMetric);
        }
        if (logFn) {
            logFn(QStringLiteral("Epoch %1/%2 | 训练 loss=%3 %4=%5 | 验证 loss=%6 %4=%7")
                      .arg(epoch)
                      .arg(cfg.epochs)
                      .arg(trainLoss, 0, 'f', 4)
                      .arg(ev.metricName)
                      .arg(trainMetric, 0, 'f', 4)
                      .arg(ev.valLoss, 0, 'f', 4)
                      .arg(ev.valMetric, 0, 'f', 4),
                  ITrainer::Info);
        }

        // 记录最终指标摘要，写入模型元数据
        metricsSummary = QStringLiteral("训练 loss=%1 | 验证 loss=%2 | %3=%4")
                             .arg(trainLoss, 0, 'f', 4)
                             .arg(ev.valLoss, 0, 'f', 4)
                             .arg(ev.metricName)
                             .arg(ev.valMetric, 0, 'f', 4);

        if (ev.aborted)                  // 验证阶段被中止，直接退出
            break;
    }

    // 被中止：交给上层发 trainingFinished(false, ...)
    if (stopFlag && stopFlag->load())
        return false;

    if (bestEpoch > 0 && logFn) {
        logFn(QStringLiteral("最佳验证指标出现在第 %1 轮（%2 = %3）")
                  .arg(bestEpoch)
                  .arg(cfg.task == TaskType::Classification
                           ? QStringLiteral("acc")
                       : cfg.task == TaskType::Segmentation
                           ? QStringLiteral("mIoU")
                           : QStringLiteral("mAP@0.5"))
                  .arg(bestVal, 0, 'f', 4),
              ITrainer::Success);
    }

    // 保存模型 + 元数据
    const QString outPath = modelio::defaultModelPath(cfg.outputDir, cfg.task);
    if (outPath.isEmpty()) {
        if (err)
            *err = QStringLiteral("无法确定模型输出路径（outputDir=%1）").arg(cfg.outputDir);
        return false;
    }
    return saveModel(outPath, err);
}

void Impl::requestStop()
{
    QMutexLocker locker(&stopMutex);
    if (stopFlag)
        stopFlag->store(true);
}

// ---------------------------------------------------------------------------
//  推理
// ---------------------------------------------------------------------------
bool Impl::loadModelImpl(const QString &path, QString *err)
{
    if (!QFileInfo::exists(path)) {
        if (err)
            *err = QStringLiteral("模型文件不存在：%1").arg(path);
        return false;
    }

    // 读取同名 .json 元数据（task / classes / 输入尺寸 / numClasses）
    TrainConfig mcfg = cfg;
    QStringList names;
    QStringList metaModels;
    QVector<ClassInfo> metaClasses;
    int mw = 224, mh = 224, mnc = 0;
    bool haveMeta = false;

    // 元数据统一由 modelio::readModelMeta() 读取（task / classes / 输入尺寸）
    modelio::ModelMeta meta;
    if (modelio::readModelMeta(path, &meta, nullptr)) {
        mcfg.task = meta.task;
        metaClasses = meta.classes;
        for (const ClassInfo &c : meta.classes)
            names << c.name;
        mw = meta.inputWidth > 0 ? meta.inputWidth : mw;
        mh = meta.inputHeight > 0 ? meta.inputHeight : mh;
        mnc = meta.classes.size();
        metaModels << meta.model;
        haveMeta = true;
    }
    if (!haveMeta) {
        // 没有元数据时按当前配置猜测，并给出警告而不是直接失败
        if (logFn) {
            logFn(QStringLiteral("未找到模型元数据 %1，按当前配置推断结构")
                      .arg(modelio::metaPath(path)),
                  ITrainer::Warning);
        }
    }

    if (!names.isEmpty())
        classNames = names;
    if (!metaClasses.isEmpty())
        classInfos = metaClasses;
    if (mnc > 0)
        mcfg.numClasses = qMax(1, qMax(mnc, classInfos.size()));
    else if (mcfg.numClasses <= 0)
        mcfg.numClasses = qMax(1, qMax(classNames.size(), classInfos.size()));
    else if (classNames.size() > 0 && mcfg.numClasses != classNames.size())
        mcfg.numClasses = qMax(1, classNames.size());

    cfg = mcfg;
    cfg.model = metaModels.isEmpty() ? cfg.model : metaModels.first();
    inputW = mw;
    inputH = mh;

    try {
        switch (cfg.task) {
        case TaskType::Classification:
            if (cfg.model.trimmed().toLower().contains(QStringLiteral("resnet"))) {
                resNet = ResNet18(qMax(1, cfg.numClasses));
                resNet->to(device);
                torch::load(resNet, toStdString(path), device);
                clsNet = nullptr;
            } else {
                clsNet = SmallCNN(qMax(1, cfg.numClasses), inputW, inputH);
                clsNet->to(device);
                torch::load(clsNet, toStdString(path), device);
                resNet = nullptr;
            }
            break;
        case TaskType::Segmentation:
            segNet = UNet(qMax(1, cfg.numClasses));
            segNet->to(device);
            torch::load(segNet, toStdString(path), device);
            break;
        case TaskType::Detection:
        default:
            gridSize = qMax(1, inputW / 4);
            detNet = Detector(qMax(1, cfg.numClasses), static_cast<int>(gridSize));
            detNet->to(device);
            torch::load(detNet, toStdString(path), device);
            break;
        }
    } catch (const std::exception &e) {
        hasModel = false;
        loaded = false;
        if (err)
            *err = QStringLiteral("模型载入失败（结构与元数据可能不匹配）：%1")
                       .arg(QString::fromUtf8(e.what()));
        return false;
    }

    setTrainMode(false);
    hasModel = true;
    loaded = true;
    modelPath = path;
    metaPath = modelio::metaPath(path);
    modelClasses = classInfos;
    return true;
}

void Impl::predictImpl(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    if (!loaded || !hasModel) {
        if (err)
            *err = QStringLiteral("尚未载入模型，请先调用 loadModel()");
        return;
    }
    if (bgr.empty()) {
        if (err)
            *err = QStringLiteral("输入图像为空");
        return;
    }

    torch::NoGradGuard noGrad;
    const int origW = bgr.cols;
    const int origH = bgr.rows;
    const int numClasses = qMax(1, cfg.numClasses);

    out->task = cfg.task;
    *out = PredictionResult();
    out->task = cfg.task;

    torch::Tensor input = imageToTensor(bgr, inputW, inputH).unsqueeze(0).to(device);

    switch (cfg.task) {
    // ---- 分类：classId / score / scores ----
    case TaskType::Classification: {
        torch::Tensor logits = resNet ? resNet->forward(input) : clsNet->forward(input);
        torch::Tensor prob = torch::softmax(logits, 1).to(torch::kCPU).squeeze(0);
        const float *p = prob.data_ptr<float>();
        out->scores.assign(p, p + numClasses);
        int best = 0;
        for (int i = 1; i < numClasses; ++i) {
            if (out->scores[static_cast<size_t>(i)] > out->scores[static_cast<size_t>(best)])
                best = i;
        }
        out->classId = best;
        out->score = out->scores[static_cast<size_t>(best)];
        break;
    }
    // ---- 分割：mask（CV_8UC1，像素值 = 类别索引，尺寸还原到原图） ----
    case TaskType::Segmentation: {
        torch::Tensor logits = segNet->forward(input);        // [1,C,h,w]
        torch::Tensor pred = logits.argmax(1).to(torch::kCPU).squeeze(0).to(torch::kUInt8);
        const int h = static_cast<int>(pred.size(0));
        const int w = static_cast<int>(pred.size(1));
        cv::Mat small(h, w, CV_8UC1, pred.data_ptr());
        cv::Mat mask;
        cv::resize(small.clone(), mask, cv::Size(origW, origH), 0.0, 0.0,
                   cv::INTER_NEAREST);   // 最近邻，保证像素值仍是合法类别索引
        out->mask = mask.clone();
        break;
    }
    // ---- 检测：boxes（原图像素坐标 x,y,w,h）+ NMS ----
    case TaskType::Detection:
    default: {
        std::vector<torch::Tensor> outs = detNet->forward(input);
        DetPredMap preds = decodeDetections(outs[0][0], outs[1][0], numClasses,
                                            static_cast<int>(gridSize), detScoreThreshold);
        std::vector<DetPred> all;
        for (auto &kv : preds) {
            nmsDetections(kv.second, detNmsIoU);
            for (const DetPred &d : kv.second)
                all.push_back(d);
        }
        // 全类别再统一按分数截断，避免框太多
        std::sort(all.begin(), all.end(),
                  [](const DetPred &a, const DetPred &b) { return a.score > b.score; });
        if (static_cast<int>(all.size()) > 100)
            all.resize(100);

        out->boxes.clear();
        for (const DetPred &d : all) {
            DetectionBox box;
            const float x = d.x1 * static_cast<float>(origW);
            const float y = d.y1 * static_cast<float>(origH);
            const float w = (d.x2 - d.x1) * static_cast<float>(origW);
            const float h = (d.y2 - d.y1) * static_cast<float>(origH);
            box.rect = QRectF(static_cast<qreal>(x), static_cast<qreal>(y),
                              static_cast<qreal>(w), static_cast<qreal>(h));
            box.classId = d.classId;   // 1 起（0 为背景）
            box.score = d.score;
            out->boxes.push_back(box);
        }
        break;
    }
    }
}

// ===========================================================================
//  LibTorchTrainer 接口实现
// ===========================================================================
LibTorchTrainer::LibTorchTrainer(QObject *parent)
    : ITrainer(parent)
    , m_impl(new Impl())
{
}

LibTorchTrainer::~LibTorchTrainer()
{
    if (m_impl) {
        m_impl->requestStop();
        if (m_impl->worker.joinable())
            m_impl->worker.join();
        delete m_impl;
        m_impl = nullptr;
    }
}

QString LibTorchTrainer::backendName() const
{
    return QStringLiteral("LibTorch 深度学习后端");
}

bool LibTorchTrainer::isAvailable(QString *reason) const
{
    // 走到这里说明已用 AQROTRAIN_ENABLE_LIBTORCH 编译，后端可用
    if (reason)
        *reason = QStringLiteral("LibTorch %1").arg(QString::fromStdString(TORCH_VERSION));
    return true;
}

bool LibTorchTrainer::initialize(const TrainConfig &cfg,
                                 const QVector<TrainSample> &samples,
                                 const QVector<ClassInfo> &classes,
                                 QString *err)
{
    if (!m_impl) {
        if (err)
            *err = QStringLiteral("训练器未正确初始化");
        return false;
    }
    if (m_impl->running.load()) {
        if (err)
            *err = QStringLiteral("训练正在进行中，请先中止或等待结束");
        return false;
    }

    // ---- 基本参数校验 ----
    if (cfg.epochs <= 0) {
        if (err)
            *err = QStringLiteral("epochs 必须大于 0");
        return false;
    }
    if (cfg.batchSize <= 0) {
        if (err)
            *err = QStringLiteral("batchSize 必须大于 0");
        return false;
    }
    if (cfg.inputWidth < 16 || cfg.inputHeight < 16) {
        if (err)
            *err = QStringLiteral("输入尺寸过小（%1x%2），至少 16x16")
                       .arg(cfg.inputWidth).arg(cfg.inputHeight);
        return false;
    }
    if (cfg.task == TaskType::Segmentation
        && (!divisibleBy8(cfg.inputWidth) || !divisibleBy8(cfg.inputHeight))) {
        if (err)
            *err = QStringLiteral("分割任务要求输入尺寸能被 8 整除（当前 %1x%2）")
                       .arg(cfg.inputWidth).arg(cfg.inputHeight);
        return false;
    }
    if (classes.isEmpty()) {
        if (err)
            *err = QStringLiteral("模块尚未定义任何标签类别，请先在「标签类别」面板中添加。");
        return false;
    }

    m_impl->cfg = cfg;
    m_impl->inputW = cfg.inputWidth;
    m_impl->inputH = cfg.inputHeight;
    m_impl->batchSize = cfg.batchSize;
    m_impl->lr = cfg.learningRate > 0.0 ? cfg.learningRate : 0.001;
    m_impl->loaded = false;
    m_impl->hasModel = false;
    m_impl->modelPath.clear();
    m_impl->metricsSummary.clear();

    // 设备
    QString note;
    m_impl->device = makeDevice(cfg, &note);
    m_impl->deviceNote = note;

    // 类别：直接来自工程层传入的 ClassInfo 列表（不再读 classes.json）
    m_impl->classInfos = classes;
    m_impl->classNames.clear();
    for (const ClassInfo &c : classes)
        m_impl->classNames << c.name;
    m_impl->cfg.numClasses = qMax(1, classes.size());     // 含背景类

    // 推理后处理参数：优先取 cfg.extra 里的同名项，否则用后端默认值
    m_impl->test = TestConfig();
    if (cfg.extra.contains(QStringLiteral("scoreThreshold"))) {
        bool ok = false;
        const double v = cfg.extra.value(QStringLiteral("scoreThreshold")).toDouble(&ok);
        if (ok)
            m_impl->test.scoreThreshold = v;
    }
    if (cfg.extra.contains(QStringLiteral("nmsThreshold"))) {
        bool ok = false;
        const double v = cfg.extra.value(QStringLiteral("nmsThreshold")).toDouble(&ok);
        if (ok)
            m_impl->test.nmsThreshold = v;
    }
    {
        bool ok = false;
        const int ma = cfg.extra.value(QStringLiteral("minArea")).toInt(&ok);
        if (ok && ma > 0)
            m_impl->test.minArea = ma;
    }
    m_impl->detScoreThreshold =
        static_cast<float>(m_impl->test.scoreThreshold);
    m_impl->detNmsIoU = static_cast<float>(m_impl->test.nmsThreshold);

    // 输出目录：未显式给出时落在模块根目录的 models 下
    if (m_impl->cfg.outputDir.trimmed().isEmpty()
        && !m_impl->cfg.datasetRoot.trimmed().isEmpty()) {
        m_impl->cfg.outputDir =
            QDir(m_impl->cfg.datasetRoot).absoluteFilePath(QStringLiteral("models"));
    }

    // 样本校验与训练 / 验证划分（数据由工程层传入，不再扫描目录）
    if (!m_impl->loadSamples(samples, err))
        return false;
    if (!m_impl->splitTrainVal(err))
        return false;

    // 构建模型（模型标识不支持时回退到内置结构）
    m_impl->buildModel();
    return true;
}

void LibTorchTrainer::start()
{
    if (!m_impl || m_impl->running.load())
        return;
    if (m_impl->worker.joinable())
        m_impl->worker.join();

    // 每次训练使用全新的停止标志，避免上一轮的标志残留导致立刻退出
    {
        QMutexLocker locker(&m_impl->stopMutex);
        m_impl->stopFlag = std::make_shared<std::atomic<bool>>(false);
    }
    m_impl->running.store(true);

    // 跨线程上报回调：内部用 QMetaObject 队列调用，保证信号在主线程被接收
    // （trainer 自身属于主线程）。
    Impl *impl = m_impl;
    impl->logFn = [this](const QString &text, int level) {
        QMetaObject::invokeMethod(this, [this, text, level]() { emit logMessage(text, level); },
                                  Qt::QueuedConnection);
    };
    impl->progressFn = [this](int percent) {
        QMetaObject::invokeMethod(this, [this, percent]() { emit progressChanged(percent); },
                                  Qt::QueuedConnection);
    };
    impl->epochFn = [this](int epoch, int total, double loss, double metric,
                           double valLoss, double valMetric) {
        QMetaObject::invokeMethod(this, [this, epoch, total, loss, metric, valLoss, valMetric]() {
            emit epochFinished(epoch, total, loss, metric, valLoss, valMetric);
        }, Qt::QueuedConnection);
    };
    impl->finishedFn = [this](bool ok, const QString &path, const QString &msg) {
        QMetaObject::invokeMethod(this, [this, ok, path, msg]() {
            emit trainingFinished(ok, path, msg);
        }, Qt::QueuedConnection);
    };

    impl->worker = std::thread([impl]() {
        QString err;
        bool ok = false;
        try {
            ok = impl->trainImpl(&err);
        } catch (const std::exception &e) {
            ok = false;
            err = QStringLiteral("训练线程异常：%1").arg(QString::fromUtf8(e.what()));
        } catch (...) {
            ok = false;
            err = QStringLiteral("训练线程发生未知异常");
        }

        const bool aborted = (impl->stopFlag && impl->stopFlag->load());
        impl->running.store(false);

        if (ok) {
            if (impl->finishedFn)
                impl->finishedFn(true, impl->modelPath, QStringLiteral("训练完成"));
        } else if (aborted) {
            if (impl->finishedFn)
                impl->finishedFn(false, QString(), QStringLiteral("训练已被用户中止"));
        } else {
            if (impl->finishedFn)
                impl->finishedFn(false, QString(),
                                 err.isEmpty() ? QStringLiteral("训练失败") : err);
        }
    });

    emit logMessage(QStringLiteral("LibTorch 后端已启动，设备：%1").arg(m_impl->deviceNote),
                    static_cast<int>(ITrainer::Info));
}

void LibTorchTrainer::requestStop()
{
    if (!m_impl)
        return;
    m_impl->requestStop();

    // run 结束时仍会正常发出 trainingFinished(false,...)
    if (m_impl->running.load())
        emit logMessage(QStringLiteral("已请求中止训练，正在安全退出…"),
                        static_cast<int>(ITrainer::Warning));
}

bool LibTorchTrainer::isRunning() const
{
    return m_impl && m_impl->running.load();
}

QString LibTorchTrainer::lastModelPath() const
{
    return m_impl ? m_impl->modelPath : QString();
}

bool LibTorchTrainer::loadModel(const QString &modelPath, QString *err)
{
    if (!m_impl) {
        if (err)
            *err = QStringLiteral("训练器未正确初始化");
        return false;
    }
    if (m_impl->running.load()) {
        if (err)
            *err = QStringLiteral("训练正在进行中，无法载入模型");
        return false;
    }
    m_impl->cfg.outputDir = QFileInfo(modelPath).absolutePath();
    return m_impl->loadModelImpl(modelPath, err);
}

bool LibTorchTrainer::predict(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    if (!out) {
        if (err)
            *err = QStringLiteral("输出参数为空");
        return false;
    }
    if (!m_impl || !m_impl->loaded) {
        if (err)
            *err = QStringLiteral("尚未载入模型，请先调用 loadModel()");
        return false;
    }
    try {
        m_impl->predictImpl(bgr, out, err);
    } catch (const std::exception &e) {
        if (err)
            *err = QStringLiteral("推理失败：%1").arg(QString::fromUtf8(e.what()));
        return false;
    }
    return out->valid();
}

TaskType LibTorchTrainer::modelTask() const
{
    return m_impl ? m_impl->cfg.task : TaskType::Classification;
}

QVector<ClassInfo> LibTorchTrainer::modelClasses() const
{
    return m_impl ? m_impl->modelClasses : QVector<ClassInfo>();
}

void LibTorchTrainer::setTestConfig(const TestConfig &cfg)
{
    // 与其他访问器（modelTask / modelClasses）一致：这里不加锁。
    // 约定是「载入模型后、开始推理前」由主线程设置一次。
    if (m_impl)
        m_impl->test = cfg;
}

TestConfig LibTorchTrainer::testConfig() const
{
    return m_impl ? m_impl->test : TestConfig();
}

} // namespace aq

// 注意：Q_OBJECT 类声明在头文件里，moc 代码由工程自动生成
//（AqroTrain.vcxproj 中的 <QtMoc Include="src\train\LibTorchTrainer.h" /> 已覆盖本文件），
// 因此这里不需要也不能 #include "LibTorchTrainer.moc"。

#endif // AQROTRAIN_ENABLE_LIBTORCH
