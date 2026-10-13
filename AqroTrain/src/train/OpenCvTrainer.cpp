// ============================================================================
//  OpenCvTrainer.cpp —— OpenCV 轻量训练后端实现
//  ---------------------------------------------------------------------------
//  移植自 VisionTrain 的 vt::OpenCvTrainer，改造点见头文件顶部说明。
// ============================================================================
#include "train/OpenCvTrainer.h"

#include <QByteArray>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QMetaObject>
#include <QVariant>

#include <opencv2/imgproc.hpp>

#include "annotate/AnnoObject.h"
#include "core/ImageIO.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

namespace aq {

namespace {

/*
 * OpenCV 的 FileStorage 直接按文件名读写时有两个坑：
 *   1. 无法处理含中文的路径（内部用窄字符 fopen）；
 *   2. 依赖一个临时文件作为中转时，在受限环境下 %TEMP% 可能不可写。
 * 这里统一改用 FileStorage 的「内存模式」：
 *   保存 -> 序列化成字符串 -> 用 QFile 写盘（Qt 的路径处理是 Unicode 安全的）；
 *   读取 -> 用 QFile 读入字符串 -> 交给内存模式解析。
 * 全程不产生任何临时文件。
 * （故意不使用 cv::FileStorage 的临时文件中转，旧工程已在此踩坑。）
 */

bool statModelSave(const cv::Ptr<cv::ml::StatModel> &model, const QString &path, QString *err)
{
    if (!model) {
        if (err) *err = QStringLiteral("模型为空，无法保存。");
        return false;
    }

    std::string content;
    try {
        // 扩展名 .xml 用于告诉 FileStorage 采用 XML 格式
        cv::FileStorage fs(".xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
        if (!fs.isOpened()) {
            if (err) *err = QStringLiteral("无法创建内存 FileStorage。");
            return false;
        }
        // 与 cv::Algorithm::save 保持一致的顶层结构，便于用标准方式读回
        fs << model->getDefaultName() << "{";
        model->write(fs);
        fs << "}";
        content = fs.releaseAndGetString();
    } catch (const cv::Exception &e) {
        if (err) *err = QStringLiteral("序列化模型失败：%1").arg(QString::fromUtf8(e.what()));
        return false;
    }

    if (content.empty()) {
        if (err) *err = QStringLiteral("序列化模型得到空内容。");
        return false;
    }

    QDir().mkpath(QFileInfo(path).absolutePath());

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (err) *err = QStringLiteral("无法写入模型文件：%1").arg(path);
        return false;
    }
    const qint64 written = f.write(content.data(), static_cast<qint64>(content.size()));
    f.close();

    if (written != static_cast<qint64>(content.size())) {
        if (err) *err = QStringLiteral("模型文件写入不完整：%1").arg(path);
        return false;
    }
    return true;
}

cv::Ptr<cv::ml::StatModel> statModelLoad(const QString &path, QString *err)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        if (err) *err = QStringLiteral("模型文件不存在或无法读取：%1").arg(path);
        return nullptr;
    }
    const QByteArray bytes = f.readAll();
    f.close();

    if (bytes.isEmpty()) {
        if (err) *err = QStringLiteral("模型文件内容为空：%1").arg(path);
        return nullptr;
    }

    try {
        const std::string content(bytes.constData(), static_cast<size_t>(bytes.size()));
        cv::FileStorage fs(content, cv::FileStorage::READ | cv::FileStorage::MEMORY);
        if (!fs.isOpened()) {
            if (err) *err = QStringLiteral("模型文件格式无法识别：%1").arg(path);
            return nullptr;
        }

        const cv::FileNode fn = fs.getFirstTopLevelNode();
        if (fn.empty()) {
            if (err) *err = QStringLiteral("模型文件缺少顶层节点：%1").arg(path);
            return nullptr;
        }

        cv::Ptr<cv::ml::RTrees> rt = cv::ml::RTrees::create();
        rt->read(fn);
        if (rt->empty()) {
            if (err) *err = QStringLiteral("模型内容为空或已损坏：%1").arg(path);
            return nullptr;
        }
        return rt;
    } catch (const cv::Exception &e) {
        if (err) *err = QStringLiteral("模型解析失败：%1").arg(QString::fromUtf8(e.what()));
        return nullptr;
    }
}

/// 两个框的 IoU
double iou(const QRectF &a, const QRectF &b)
{
    const QRectF inter = a.intersected(b);
    const double ia = inter.width() * inter.height();
    if (ia <= 0)
        return 0.0;
    const double ua = a.width() * a.height() + b.width() * b.height() - ia;
    return ua > 0 ? ia / ua : 0.0;
}

/**
 * 从 RTrees::getVotes 的输出里取出「某个样本的各类票数」。
 *
 * 依据 OpenCV 源码 modules/ml/src/rtrees.cpp 的 getVotes()：
 *
 *     output.create(nsamples + 1, nclasses, CV_32S);
 *     第 0 行   = classLabels（各类别标签）
 *     第 i+1 行 = 第 i 个样本对每个类别的得票数
 *
 * 注意 flags 传 0 时走的是 PREDICT_MAX_VOTE 分支，即上面的布局；
 * 只有「传 RAW_OUTPUT 且类别数为 2」时才会退化成 (nsamples × ntrees) 的浮点矩阵。
 * 早期版本误把第 0 行当成票数，导致置信度恒为 0、检测候选被阈值全部滤掉。
 */
std::vector<double> votesOf(const cv::Mat &votes, int sampleIdx, int nSamples, int nClasses)
{
    std::vector<double> out(static_cast<size_t>(std::max(1, nClasses)), 0.0);

    if (nClasses <= 0 || sampleIdx < 0 || sampleIdx >= nSamples)
        return out;
    if (votes.empty() || votes.type() != CV_32S)
        return out;

    const int row = sampleIdx + 1;     // 第 0 行是类别标签，票数从第 1 行开始
    if (row >= votes.rows)
        return out;

    const int n = std::min(nClasses, votes.cols);
    for (int c = 0; c < n; ++c)
        out[c] = votes.at<int>(row, c);
    return out;
}

/**
 * 由标注对象列表得到检测真值框。
 * 新工程的标注真源是「对象列表」（AnnoObject），矩形 / 多边形 / 笔画都取其包围盒。
 */
QVector<DetectionBox> boxesFromObjects(const QVector<AnnoObject> &objects)
{
    QVector<DetectionBox> out;
    for (const AnnoObject &o : objects) {
        if (!o.isValid() || o.classId <= 0)
            continue;
        const QRectF r = o.boundingRect();
        if (r.width() <= 0.0 || r.height() <= 0.0)
            continue;
        DetectionBox b;
        b.rect = r;
        b.classId = o.classId;
        b.score = 1.f;
        out.append(b);
    }
    return out;
}

/// 读取一张图的检测真值框（标注文件读不到时返回空列表 = 无目标）
QVector<DetectionBox> readDetectionTruth(const QString &imagePath,
                                         const QString &labelPath,
                                         const QString &maskPath)
{
    QVector<AnnoObject> objects;
    anno::readLabel(labelPath, maskPath, TaskType::Detection, &objects, nullptr, nullptr, 16, nullptr);
    Q_UNUSED(imagePath);
    return boxesFromObjects(objects);
}

/// 读取一张图的分割真值掩码：优先用派生掩码 .png，缺失时由对象列表光栅化
cv::Mat readSegmentationTruth(const QString &maskPath, const QString &labelPath,
                              const cv::Size &imageSize)
{
    if (!maskPath.isEmpty()) {
        const cv::Mat m = imageio::imreadUnicode(maskPath, cv::IMREAD_GRAYSCALE);
        if (!m.empty())
            return m;
    }

    QVector<AnnoObject> objects;
    anno::readLabel(labelPath, maskPath, TaskType::Segmentation, &objects, nullptr, nullptr, 16, nullptr);

    if (objects.isEmpty() || imageSize.width <= 0 || imageSize.height <= 0)
        return cv::Mat();

    cv::Mat mask;
    anno::rasterize(objects, imageSize.width, imageSize.height, &mask);
    return mask;
}

/// 从 extra 里取一个 double 参数（缺省时用 def）
double extraDouble(const QVariantMap &extra, const QString &key, double def)
{
    if (!extra.contains(key))
        return def;
    bool ok = false;
    const double v = extra.value(key).toDouble(&ok);
    return ok ? v : def;
}

const char *kStageNote =
    "（OpenCV 轻量后端不使用 epoch/学习率，此处按处理阶段上报进度）";

} // namespace

// ---------------------------------------------------------------------------
OpenCvTrainer::OpenCvTrainer(QObject *parent) : ITrainer(parent) {}

OpenCvTrainer::~OpenCvTrainer()
{
    requestStop();
    if (m_worker.joinable())
        m_worker.join();
}

QString OpenCvTrainer::backendName() const
{
    return QStringLiteral("OpenCV 轻量训练后端");
}

bool OpenCvTrainer::isAvailable(QString *reason) const
{
    if (reason)
        reason->clear();
    return true;   // OpenCV 已随程序链接，始终可用
}

// ---------------------------------------------------------------------------
//  跨线程信号
// ---------------------------------------------------------------------------
void OpenCvTrainer::emitLog(const QString &text, int level)
{
    QMetaObject::invokeMethod(this, [this, text, level]() { emit logMessage(text, level); },
                              Qt::QueuedConnection);
}

void OpenCvTrainer::emitProgress(int percent)
{
    const int p = std::max(0, std::min(100, percent));
    QMetaObject::invokeMethod(this, [this, p]() { emit progressChanged(p); },
                              Qt::QueuedConnection);
}

void OpenCvTrainer::emitEpoch(int epoch, int total, double loss, double metric,
                              double valLoss, double valMetric)
{
    QMetaObject::invokeMethod(this, [epoch, total, loss, metric, valLoss, valMetric, this]() {
        emit epochFinished(epoch, total, loss, metric, valLoss, valMetric);
    }, Qt::QueuedConnection);
}

void OpenCvTrainer::emitFinished(bool ok, const QString &modelPath, const QString &message)
{
    QMetaObject::invokeMethod(this, [this, ok, modelPath, message]() {
        emit trainingFinished(ok, modelPath, message);
    }, Qt::QueuedConnection);
}

// ---------------------------------------------------------------------------
//  初始化：训练数据改为由工程层传入
// ---------------------------------------------------------------------------
bool OpenCvTrainer::initialize(const TrainConfig &cfg,
                               const QVector<TrainSample> &samples,
                               const QVector<ClassInfo> &classes,
                               QString *err)
{
    if (isRunning()) {
        if (err) *err = QStringLiteral("已有训练正在进行中。");
        return false;
    }

    m_cfg = cfg;
    m_modelPath.clear();
    m_model.release();
    m_metricsSummary.clear();

    // --- 类别来自工程层（不再读 classes.json）---
    m_classes = classes;
    if (m_classes.isEmpty()) {
        if (err) *err = QStringLiteral("模块尚未定义任何标签类别，请先在「标签类别」面板中添加。");
        return false;
    }

    // --- 把 TrainSample 列表拆成内部表示 ---
    m_imagePaths.clear();
    m_labelPaths.clear();
    m_maskPaths.clear();
    m_classIds.clear();
    m_imagePaths.reserve(samples.size());
    m_labelPaths.reserve(samples.size());
    m_maskPaths.reserve(samples.size());
    m_classIds.reserve(samples.size());

    for (const TrainSample &s : samples) {
        if (!s.valid())
            continue;
        m_imagePaths << s.imagePath;
        m_labelPaths << s.labelPath;
        m_maskPaths << s.maskPath;
        m_classIds << s.classId;
    }

    m_cfg.numClasses = m_classes.size();
    if (m_cfg.outputDir.trimmed().isEmpty() && !m_cfg.datasetRoot.trimmed().isEmpty())
        m_cfg.outputDir = QDir(m_cfg.datasetRoot).absoluteFilePath(QStringLiteral("models"));

    // --- 推理后处理参数（评价器按模块的 TestConfig 覆盖后调用 predict）---
    m_test = TestConfig();
    m_test.scoreThreshold = extraDouble(cfg.extra, QStringLiteral("scoreThreshold"), 0.5);
    m_test.nmsThreshold   = extraDouble(cfg.extra, QStringLiteral("nmsThreshold"), 0.45);
    m_test.iouThreshold   = extraDouble(cfg.extra, QStringLiteral("iouThreshold"), 0.5);
    {
        bool ok = false;
        const int ma = cfg.extra.value(QStringLiteral("minArea")).toInt(&ok);
        if (ok && ma > 0)
            m_test.minArea = ma;
    }

    return validateSamples(err);
}

void OpenCvTrainer::requestStop()
{
    if (m_running.load())
        m_stop = true;
}

// ---------------------------------------------------------------------------
//  样本校验（只检查传进来的列表，不访问 classes.json、不扫描目录）
// ---------------------------------------------------------------------------
bool OpenCvTrainer::validateSamples(QString *err)
{
    const int n = m_imagePaths.size();
    if (n < 2) {
        if (err)
            *err = QStringLiteral("有效训练样本不足（至少需要 2 张），当前为 %1 张。").arg(n);
        return false;
    }

    int missingImage = 0;
    int missingLabel = 0;
    int badClass = 0;

    for (int i = 0; i < n; ++i) {
        if (!QFileInfo::exists(m_imagePaths.at(i))) {
            ++missingImage;
            continue;
        }

        if (m_cfg.task == TaskType::Classification) {
            const int c = m_classIds.at(i);
            if (c < 0 || c >= m_classes.size())
                ++badClass;
            continue;
        }

        // 检测 / 分割：标注真源缺失则不可用于训练
        if (m_labelPaths.at(i).trimmed().isEmpty()
            || !QFileInfo::exists(m_labelPaths.at(i))) {
            ++missingLabel;
        }
    }

    if (missingImage >= n) {
        if (err)
            *err = QStringLiteral("训练集的 %1 张图像全部无法访问，请检查图像是否被移动。").arg(n);
        return false;
    }
    if (m_cfg.task == TaskType::Classification && badClass == n) {
        if (err)
            *err = QStringLiteral("没有任何样本带有有效的分类标注（classId 需在 0..%1 之间）。")
                       .arg(m_classes.size() - 1);
        return false;
    }
    if (m_cfg.task != TaskType::Classification && missingLabel == n) {
        if (err)
            *err = QStringLiteral("%1任务没有任何样本找到标注文件，请先完成标注。")
                       .arg(taskTypeName(m_cfg.task));
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
//  启动训练
// ---------------------------------------------------------------------------
void OpenCvTrainer::start()
{
    if (isRunning())
        return;
    if (m_worker.joinable())
        m_worker.join();

    m_stop = false;
    m_worker = std::thread([this]() { runTraining(); });
}

void OpenCvTrainer::runTraining()
{
    m_running = true;
    QString err;
    bool ok = false;

    try {
        emitLog(QStringLiteral("训练后端：%1").arg(backendName()), Info);
        emitLog(m_cfg.summary(), Info);
        emitLog(QString::fromUtf8(kStageNote), Info);
        emitLog(QStringLiteral("训练样本 %1 张，类别数 %2。")
                    .arg(m_imagePaths.size())
                    .arg(m_classes.size()), Info);

        if (m_stop.load()) {
            ok = false;
        } else {
            switch (m_cfg.task) {
            case TaskType::Classification: ok = trainClassification(&err); break;
            case TaskType::Segmentation:   ok = trainSegmentation(&err);   break;
            case TaskType::Detection:      ok = trainDetection(&err);      break;
            }
        }

        if (ok && !m_stop.load()) {
            QDir().mkpath(m_cfg.outputDir);
            m_modelPath = modelio::defaultModelPath(m_cfg.outputDir, m_cfg.task);
            emitLog(QStringLiteral("保存模型到：%1").arg(m_modelPath), Info);
            ok = saveModelTo(m_modelPath, &err) && writeMeta(m_modelPath, &err);
        }
    } catch (const cv::Exception &e) {
        ok = false;
        err = QStringLiteral("OpenCV 异常：%1").arg(QString::fromUtf8(e.what()));
    } catch (const std::exception &e) {
        ok = false;
        err = QStringLiteral("异常：%1").arg(QString::fromUtf8(e.what()));
    }

    QString message;
    if (m_stop.load()) {
        ok = false;
        m_modelPath.clear();
        message = QStringLiteral("训练已被用户中止");
    } else if (ok) {
        message = QStringLiteral("训练完成");
        emitProgress(100);
    } else {
        message = err.isEmpty() ? QStringLiteral("训练失败") : err;
        emitLog(message, Error);
    }

    m_running = false;
    emitFinished(ok, ok ? m_modelPath : QString(), message);
}

// ---------------------------------------------------------------------------
//  特征提取
// ---------------------------------------------------------------------------
cv::Mat OpenCvTrainer::classifyFeatures(const cv::Mat &bgr, const cv::Size &inputSize)
{
    cv::Mat img;
    cv::resize(bgr, img, inputSize, 0, 0, cv::INTER_AREA);

    // 1) HSV 三分量各 32 bin 一维直方图（96 维）
    cv::Mat hsv;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);

    const int bins = 32;
    const float hRangeArr[] = { 0.f, 180.f };
    const float sRangeArr[] = { 0.f, 256.f };
    const float vRangeArr[] = { 0.f, 256.f };
    const float *hRanges[] = { hRangeArr };
    const float *sRanges[] = { sRangeArr };
    const float *vRanges[] = { vRangeArr };
    const int ch0 = 0;

    cv::Mat h1, h2, h3;
    cv::calcHist(&ch[0], 1, &ch0, cv::Mat(), h1, 1, &bins, hRanges, true, false);
    cv::calcHist(&ch[1], 1, &ch0, cv::Mat(), h2, 1, &bins, sRanges, true, false);
    cv::calcHist(&ch[2], 1, &ch0, cv::Mat(), h3, 1, &bins, vRanges, true, false);
    cv::normalize(h1, h1, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(h2, h2, 1.0, 0.0, cv::NORM_L1);
    cv::normalize(h3, h3, 1.0, 0.0, cv::NORM_L1);

    // 2) 8x8 灰度缩略图（64 维）
    cv::Mat gray, small;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, small, cv::Size(8, 8), 0, 0, cv::INTER_AREA);
    cv::Mat grayF;
    small.convertTo(grayF, CV_32F, 1.0 / 255.0);

    cv::Mat feat;
    cv::Mat parts[4] = { h1.reshape(1, 1), h2.reshape(1, 1), h3.reshape(1, 1),
                         grayF.reshape(1, 1) };
    cv::hconcat(parts, 4, feat);
    return feat;   // 1 x 160, CV_32F
}

cv::Mat OpenCvTrainer::segmentFeatures(const cv::Mat &bgr)
{
    const int W = bgr.cols;
    const int H = bgr.rows;
    cv::Mat feats(H * W, 5, CV_32F);

    float *p = feats.ptr<float>(0);
    const float invW = W > 1 ? 1.f / (W - 1) : 0.f;
    const float invH = H > 1 ? 1.f / (H - 1) : 0.f;

    for (int y = 0; y < H; ++y) {
        const cv::Vec3b *row = bgr.ptr<cv::Vec3b>(y);
        const float ny = y * invH;
        for (int x = 0; x < W; ++x) {
            *p++ = row[x][0];
            *p++ = row[x][1];
            *p++ = row[x][2];
            *p++ = x * invW;
            *p++ = ny;
        }
    }
    return feats;
}

cv::Mat OpenCvTrainer::patchFeature(const cv::Mat &bgr, const cv::Rect &roi, int patch)
{
    const cv::Rect r = roi & cv::Rect(0, 0, bgr.cols, bgr.rows);
    cv::Mat feat = cv::Mat::zeros(1, patch * patch, CV_32F);
    if (r.area() <= 0)
        return feat;

    cv::Mat patchImg, gray, small;
    bgr(r).copyTo(patchImg);
    cv::cvtColor(patchImg, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, small, cv::Size(patch, patch), 0, 0, cv::INTER_AREA);
    cv::Mat f;
    small.convertTo(f, CV_32F, 1.0 / 255.0);
    return f.reshape(1, 1);
}

// ---------------------------------------------------------------------------
//  分类训练
// ---------------------------------------------------------------------------
bool OpenCvTrainer::trainClassification(QString *err)
{
    const int total = m_imagePaths.size();
    emitLog(QStringLiteral("提取图像特征（HSV 直方图 + 8x8 缩略图）…"), Info);

    std::vector<cv::Mat> feats;
    std::vector<int> labels;

    for (int i = 0; i < total; ++i) {
        if (m_stop.load())
            return false;

        // 真值类别直接来自 TrainSample.classId（不再解析标签文件）
        const int cls = m_classIds.at(i);
        if (cls < 0 || cls >= m_classes.size())
            continue;

        const cv::Mat img = imageio::imreadUnicode(m_imagePaths.at(i), cv::IMREAD_COLOR);
        if (img.empty())
            continue;

        feats.push_back(classifyFeatures(img, cv::Size(m_cfg.inputWidth, m_cfg.inputHeight)));
        labels.push_back(cls);

        if (total > 0)
            emitProgress(5 + 25 * (i + 1) / total);
    }

    if (feats.size() < 2) {
        if (err) *err = QStringLiteral("可用于分类训练的有效样本不足。");
        return false;
    }

    cv::Mat allFeats;
    cv::vconcat(feats, allFeats);
    const cv::Mat allLabels(labels);

    // 划分训练 / 验证集（固定种子，结果可复现）
    std::vector<int> order(feats.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(20240929));

    const int nVal = std::min(int(feats.size() * 0.2), int(feats.size()) - 1);
    std::vector<int> trIdx(order.begin() + nVal, order.end());
    std::vector<int> vaIdx(order.begin(), order.begin() + nVal);

    cv::Mat trF, trL, vaF, vaL;
    for (int i : trIdx) { trF.push_back(allFeats.row(i)); trL.push_back(allLabels.row(i)); }
    for (int i : vaIdx) { vaF.push_back(allFeats.row(i)); vaL.push_back(allLabels.row(i)); }

    emitLog(QStringLiteral("训练随机森林：训练 %1 张，验证 %2 张。")
                .arg(trF.rows).arg(vaF.rows), Info);

    auto rt = cv::ml::RTrees::create();
    rt->setMaxDepth(14);
    rt->setMinSampleCount(2);
    rt->setMaxCategories(32);
    rt->setCalculateVarImportance(false);
    rt->setActiveVarCount(0);
    rt->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 200, 1e-6));
    rt->train(trF, cv::ml::ROW_SAMPLE, trL);

    emitProgress(60);

    auto accuracy = [&rt](const cv::Mat &F, const cv::Mat &L) -> double {
        if (F.empty() || L.empty())
            return 0.0;
        cv::Mat res;
        rt->predict(F, res);
        int correct = 0;
        for (int i = 0; i < F.rows; ++i) {
            if (static_cast<int>(res.at<float>(i, 0)) == L.at<int>(i, 0))
                ++correct;
        }
        return double(correct) / F.rows;
    };

    const double trAcc = accuracy(trF, trL);
    const double vaAcc = accuracy(vaF, vaL);

    emitEpoch(1, 3, 0.0, 0.0, 0.0, 0.0);
    emitEpoch(2, 3, 1.0 - trAcc, trAcc, 0.0, 0.0);
    emitEpoch(3, 3, 0.0, 0.0, 1.0 - vaAcc, vaAcc);

    emitLog(QStringLiteral("训练集准确率：%1%").arg(trAcc * 100, 0, 'f', 2), Success);
    emitLog(QStringLiteral("验证集准确率：%1%").arg(vaAcc * 100, 0, 'f', 2), Success);

    m_model = rt;
    m_modelTask = TaskType::Classification;
    m_modelInput = cv::Size(m_cfg.inputWidth, m_cfg.inputHeight);
    m_modelClasses = m_classes;
    m_metricsSummary = QStringLiteral("训练集准确率 %1% / 验证集准确率 %2%")
                           .arg(trAcc * 100, 0, 'f', 2)
                           .arg(vaAcc * 100, 0, 'f', 2);

    emitProgress(85);
    return true;
}

// ---------------------------------------------------------------------------
//  分割训练
// ---------------------------------------------------------------------------
bool OpenCvTrainer::trainSegmentation(QString *err)
{
    const int maxTotalPixels = 300000;      ///< 总采样像素上限，控制训练时间
    const int maxPerImage   = 20000;        ///< 单图采样像素上限

    emitLog(QStringLiteral("逐像素采样 (B,G,R,x/W,y/H) 特征，上限 %1 个像素…")
                .arg(maxTotalPixels), Info);

    std::vector<float> data;
    std::vector<int> labels;
    data.reserve(static_cast<size_t>(maxTotalPixels) * 5);
    labels.reserve(maxTotalPixels);

    const int total = m_imagePaths.size();
    for (int i = 0; i < total && int(labels.size()) < maxTotalPixels; ++i) {
        if (m_stop.load())
            return false;

        const cv::Mat img = imageio::imreadUnicode(m_imagePaths.at(i), cv::IMREAD_COLOR);
        if (img.empty())
            continue;

        // 真值掩码：优先 .png 派生掩码，缺失时由标注对象列表光栅化
        const cv::Mat mask = readSegmentationTruth(m_maskPaths.at(i), m_labelPaths.at(i),
                                                   cv::Size(img.cols, img.rows));
        if (mask.empty() || img.size() != mask.size())
            continue;

        const int W = img.cols;
        const int H = img.rows;

        // 按图像大小自动决定采样步长
        const double ratio = double(W) * H / maxPerImage;
        const int stride = std::max(1, int(std::sqrt(std::max(1.0, ratio))));

        const float invW = W > 1 ? 1.f / (W - 1) : 0.f;
        const float invH = H > 1 ? 1.f / (H - 1) : 0.f;

        for (int y = 0; y < H; y += stride) {
            const cv::Vec3b *row = img.ptr<cv::Vec3b>(y);
            const uchar *mrow = mask.ptr<uchar>(y);
            for (int x = 0; x < W; x += stride) {
                if (int(labels.size()) >= maxTotalPixels)
                    break;
                data.push_back(float(row[x][0]));
                data.push_back(float(row[x][1]));
                data.push_back(float(row[x][2]));
                data.push_back(x * invW);
                data.push_back(y * invH);
                labels.push_back(int(mrow[x]));
            }
        }

        emitProgress(5 + 35 * (i + 1) / total);
    }

    if (labels.size() < 50) {
        if (err) *err = QStringLiteral("可用于分割训练的像素样本不足。");
        return false;
    }

    emitLog(QStringLiteral("共采集 %1 个像素样本，开始训练随机森林…").arg(labels.size()), Info);

    cv::Mat feats(int(labels.size()), 5, CV_32F, data.data());
    feats = feats.clone();                     // 与 data 解耦
    const cv::Mat responses(labels);

    std::vector<int> order(labels.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(20240929));

    const int nVal = int(labels.size() * 0.2);
    cv::Mat trF, trL, vaF, vaL;
    for (int i = nVal; i < int(order.size()); ++i) { trF.push_back(feats.row(order[i])); trL.push_back(responses.row(order[i])); }
    for (int i = 0; i < nVal; ++i) { vaF.push_back(feats.row(order[i])); vaL.push_back(responses.row(order[i])); }

    auto rt = cv::ml::RTrees::create();
    rt->setMaxDepth(16);
    rt->setMinSampleCount(4);
    rt->setMaxCategories(32);
    rt->setCalculateVarImportance(false);
    rt->setActiveVarCount(0);
    rt->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 150, 1e-6));
    rt->train(trF, cv::ml::ROW_SAMPLE, trL);

    emitProgress(70);

    // 在验证像素上计算像素准确率与 mIoU
    double pixelAcc = 0.0;
    double miou = 0.0;
    if (!vaF.empty()) {
        cv::Mat res;
        rt->predict(vaF, res);

        const int nCls = std::max(1, m_classes.size());
        std::vector<long long> inter(nCls, 0), uni(nCls, 0);
        long long correct = 0;

        for (int i = 0; i < vaF.rows; ++i) {
            const int gt = vaL.at<int>(i, 0);
            const int pr = static_cast<int>(res.at<float>(i, 0));
            if (gt == pr)
                ++correct;
            if (gt >= 0 && gt < nCls && pr >= 0 && pr < nCls) {
                if (gt == pr) { inter[gt]++; uni[gt]++; }
                else { uni[gt]++; uni[pr]++; }
            }
        }
        pixelAcc = double(correct) / vaF.rows;

        int used = 0;
        double sum = 0.0;
        for (int c = 0; c < nCls; ++c) {
            if (uni[c] == 0)
                continue;
            sum += double(inter[c]) / double(uni[c]);
            ++used;
        }
        miou = used > 0 ? sum / used : 0.0;
    }

    emitEpoch(1, 3, 0.0, 0.0, 0.0, 0.0);
    emitEpoch(2, 3, 1.0 - pixelAcc, pixelAcc, 0.0, 0.0);
    emitEpoch(3, 3, 0.0, 0.0, 1.0 - pixelAcc, miou);

    emitLog(QStringLiteral("验证集像素准确率：%1%").arg(pixelAcc * 100, 0, 'f', 2), Success);
    emitLog(QStringLiteral("验证集 mIoU：%1").arg(miou, 0, 'f', 4), Success);

    m_model = rt;
    m_modelTask = TaskType::Segmentation;
    m_modelInput = cv::Size(m_cfg.inputWidth, m_cfg.inputHeight);
    m_modelClasses = m_classes;
    m_metricsSummary = QStringLiteral("验证集像素准确率 %1% / mIoU %2")
                           .arg(pixelAcc * 100, 0, 'f', 2)
                           .arg(miou, 0, 'f', 4);

    emitProgress(85);
    return true;
}

// ---------------------------------------------------------------------------
//  检测训练
// ---------------------------------------------------------------------------
bool OpenCvTrainer::trainDetection(QString *err)
{
    const int patch = 16;                    ///< 图像块特征边长（16x16 = 256 维）
    const int maxPositives = 6000;           ///< 正样本上限
    const int maxNegPerImageCap = 40;        ///< 每图负样本数量上限（实际按正样本数动态收缩）

    emitLog(QStringLiteral("采集图像块样本（正样本=目标框，负样本=随机背景）…"), Info);

    std::vector<float> data;
    std::vector<int> labels;

    std::mt19937 rng(20240929);
    int positives = 0;

    const int total = m_imagePaths.size();
    for (int i = 0; i < total; ++i) {
        if (m_stop.load())
            return false;

        const cv::Mat img = imageio::imreadUnicode(m_imagePaths.at(i), cv::IMREAD_COLOR);
        if (img.empty())
            continue;

        // 真值框来自标注对象列表（矩形 / 多边形 / 笔画的包围盒）
        const QVector<DetectionBox> boxes =
            readDetectionTruth(m_imagePaths.at(i), m_labelPaths.at(i), m_maskPaths.at(i));

        // --- 正样本 ---
        // 关键：**对标注框做抖动增广**。
        // 如果只用「标注框本身」这一种对齐，模型就只认识训练时的那个对齐，
        // 检测阶段换一个偏移/尺度就判成背景 —— 实测召回率会掉到 25% 左右。
        // 这里对每个目标框生成若干个位置与尺度都略有差异的正样本，
        // 让模型在多个对齐下都见过「目标长什么样」。
        constexpr int kJitterPerBox = 5;
        std::uniform_real_distribution<double> jitterOff(-0.15, 0.15);
        std::uniform_real_distribution<double> jitterScale(0.85, 1.15);

        for (const DetectionBox &b : boxes) {
            for (int k = 0; k < kJitterPerBox; ++k) {
                if (positives >= maxPositives)
                    break;

                // k == 0 时保留原始标注框，其余分别抖动（固定种子，结果可复现）
                const double sx = (k == 0) ? 1.0 : jitterScale(rng);
                const double sy = (k == 0) ? 1.0 : jitterScale(rng);
                const double ox = (k == 0) ? 0.0 : jitterOff(rng) * b.rect.width();
                const double oy = (k == 0) ? 0.0 : jitterOff(rng) * b.rect.height();

                const double w = b.rect.width() * sx;
                const double h = b.rect.height() * sy;
                const cv::Rect roi(qRound(b.rect.x() + ox), qRound(b.rect.y() + oy),
                                   qRound(w), qRound(h));

                const cv::Rect clipped = roi & cv::Rect(0, 0, img.cols, img.rows);
                // 抖动后与真值的 IoU 不能太低，否则会把背景当成目标教会模型
                if (clipped.area() < 64
                    || iou(QRectF(clipped.x, clipped.y, clipped.width, clipped.height),
                           b.rect) < 0.5)
                    continue;

                const cv::Mat f = patchFeature(img, clipped, patch);
                const float *p = f.ptr<float>(0);
                data.insert(data.end(), p, p + patch * patch);
                labels.push_back(b.classId);
                ++positives;
            }
        }

        // --- 负样本：随机取与所有框 IoU < 0.3 的窗口 ---
        // 必须控制正负样本比例（约 1:3）。若按固定数量抽负样本，
        // 每图只有 1~2 个目标时会形成 20:1 的极端失衡，森林会严重偏向背景，
        // 推理阶段几乎所有滑窗都会被判成背景（表现为「检出 0 个目标」）。
        //
        // 负样本的尺寸范围要与推理阶段的滑窗尺寸范围对齐（32 ~ 131），
        // 否则模型见到的背景尺度与推理时不一致。
        // 正样本做了 kJitterPerBox 倍增广，负样本也要相应放大，才能维持约 1:3 的比例
        const int maxNegPerImage = std::max(2, std::min(maxNegPerImageCap * 2,
                                                       int(boxes.size()) * kJitterPerBox * 3 / 5));
        const int negMin = std::max(16, std::min(img.cols, img.rows) / 6);
        const int negMax = std::max(negMin + 1, std::min(img.cols, img.rows) * 2 / 3);
        std::uniform_int_distribution<int> dw(negMin, negMax);
        std::uniform_int_distribution<int> dh(negMin, negMax);

        int neg = 0;
        int tries = 0;
        while (neg < maxNegPerImage && tries < maxNegPerImage * 12) {
            ++tries;
            const int w = dw(rng);
            const int h = dh(rng);
            std::uniform_int_distribution<int> dx(0, std::max(1, img.cols - w));
            std::uniform_int_distribution<int> dy(0, std::max(1, img.rows - h));
            const cv::Rect roi(dx(rng), dy(rng), w, h);
            if ((roi & cv::Rect(0, 0, img.cols, img.rows)).area() < 16)
                continue;

            bool overlaps = false;
            for (const DetectionBox &b : boxes) {
                if (iou(QRectF(roi.x, roi.y, roi.width, roi.height), b.rect) > 0.3) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps)
                continue;

            const cv::Mat f = patchFeature(img, roi, patch);
            const float *p = f.ptr<float>(0);
            data.insert(data.end(), p, p + patch * patch);
            labels.push_back(0);            // 0 = 背景
            ++neg;
        }

        emitProgress(5 + 35 * (i + 1) / total);
    }

    if (positives < 4) {
        if (err) *err = QStringLiteral("目标框样本太少（至少 4 个），无法训练检测模型。");
        return false;
    }
    if (labels.size() < 20) {
        if (err) *err = QStringLiteral("图像块样本不足，无法训练检测模型。");
        return false;
    }

    emitLog(QStringLiteral("共采集 %1 个图像块（正样本 %2，负样本 %3），开始训练…")
                .arg(labels.size()).arg(positives).arg(int(labels.size()) - positives), Info);

    // 注意：维度必须与训练时写入 data 的维度一致（patch*patch）。
    // 用错维度会让 Mat 的步长与实际数据不符，读取越界直接崩溃。
    cv::Mat feats(int(labels.size()), patch * patch, CV_32F, data.data());
    feats = feats.clone();
    const cv::Mat responses(labels);

    std::vector<int> order(labels.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), std::mt19937(20240929));

    const int nVal = int(labels.size() * 0.2);
    cv::Mat trF, trL, vaF, vaL;
    for (int i = nVal; i < int(order.size()); ++i) { trF.push_back(feats.row(order[i])); trL.push_back(responses.row(order[i])); }
    for (int i = 0; i < nVal; ++i) { vaF.push_back(feats.row(order[i])); vaL.push_back(responses.row(order[i])); }

    auto rt = cv::ml::RTrees::create();
    rt->setMaxDepth(16);
    rt->setMinSampleCount(3);
    rt->setMaxCategories(32);
    rt->setCalculateVarImportance(false);
    rt->setActiveVarCount(0);
    rt->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 200, 1e-6));
    rt->train(trF, cv::ml::ROW_SAMPLE, trL);

    emitProgress(70);

    double vaAcc = 0.0;
    if (!vaF.empty()) {
        cv::Mat res;
        rt->predict(vaF, res);
        int correct = 0;
        for (int i = 0; i < vaF.rows; ++i)
            if (static_cast<int>(res.at<float>(i, 0)) == vaL.at<int>(i, 0))
                ++correct;
        vaAcc = double(correct) / vaF.rows;
    }

    emitEpoch(1, 3, 0.0, 0.0, 0.0, 0.0);
    emitEpoch(2, 3, 0.0, 0.0, 0.0, 0.0);
    emitEpoch(3, 3, 0.0, 0.0, 1.0 - vaAcc, vaAcc);

    emitLog(QStringLiteral("图像块分类验证准确率：%1%").arg(vaAcc * 100, 0, 'f', 2), Success);
    emitLog(QStringLiteral("说明：轻量后端的检测为「滑窗 + 块分类 + NMS」的简化实现，"
                           "该准确率是图像块级别指标，不等同于 mAP。"
                           "需要 mAP 与更高精度请使用 LibTorch 后端。"), Warning);

    m_model = rt;
    m_modelTask = TaskType::Detection;
    m_modelInput = cv::Size(patch, patch);
    m_modelClasses = m_classes;
    m_metricsSummary = QStringLiteral("图像块分类验证准确率 %1%")
                           .arg(vaAcc * 100, 0, 'f', 2);

    emitProgress(85);
    return true;
}

// ---------------------------------------------------------------------------
//  模型持久化
// ---------------------------------------------------------------------------
bool OpenCvTrainer::saveModelTo(const QString &path, QString *err)
{
    return statModelSave(m_model, path, err);
}

bool OpenCvTrainer::loadModelFrom(const QString &path, QString *err)
{
    cv::Ptr<cv::ml::StatModel> m = statModelLoad(path, err);
    if (!m)
        return false;
    m_model = m;
    return true;
}

/// 元数据不再自行拼 JSON，统一交给 modelio::writeModelMeta()
bool OpenCvTrainer::writeMeta(const QString &modelPath, QString *err)
{
    modelio::ModelMeta meta;
    meta.task           = m_modelTask;
    meta.backend        = QStringLiteral("opencv");
    meta.model          = m_cfg.model;
    meta.inputWidth     = m_modelInput.width;
    meta.inputHeight    = m_modelInput.height;
    meta.classes        = m_modelClasses;
    meta.trainedAt      = QDateTime::currentDateTime().toString(Qt::ISODate);
    meta.metricsSummary = m_metricsSummary;

    return modelio::writeModelMeta(modelPath, meta, err);
}

bool OpenCvTrainer::loadModel(const QString &modelPath, QString *err)
{
    // 元数据由 modelio::readModelMeta 读取（task / classes / 输入尺寸）
    modelio::ModelMeta meta;
    if (!modelio::readModelMeta(modelPath, &meta, err))
        return false;

    m_modelTask   = meta.task;
    m_modelInput  = cv::Size(meta.inputWidth > 0 ? meta.inputWidth : 224,
                             meta.inputHeight > 0 ? meta.inputHeight : 224);
    m_modelClasses = meta.classes;

    if (!loadModelFrom(modelPath, err))
        return false;

    m_cfg.task = m_modelTask;
    m_cfg.numClasses = m_modelClasses.size();
    m_modelPath = modelPath;
    return true;
}

// ---------------------------------------------------------------------------
//  推理
// ---------------------------------------------------------------------------
bool OpenCvTrainer::predict(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    if (!out) {
        if (err) *err = QStringLiteral("输出参数为空。");
        return false;
    }
    if (bgr.empty()) {
        if (err) *err = QStringLiteral("输入图像为空。");
        return false;
    }
    if (!m_model) {
        if (err) *err = QStringLiteral("尚未加载模型。");
        return false;
    }

    out->task = m_modelTask;
    out->classId = -1;
    out->score = 0.f;
    out->scores.clear();
    out->boxes.clear();
    out->mask = cv::Mat();

    switch (m_modelTask) {
    case TaskType::Classification: return predictClassification(bgr, out, err);
    case TaskType::Segmentation:   return predictSegmentation(bgr, out, err);
    case TaskType::Detection:      return predictDetection(bgr, out, err);
    }
    if (err) *err = QStringLiteral("未知的任务类型。");
    return false;
}

bool OpenCvTrainer::predictClassification(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    Q_UNUSED(err);
    const cv::Mat feat = classifyFeatures(bgr, m_modelInput);

    cv::Mat res;
    m_model->predict(feat, res);
    const int cls = static_cast<int>(res.at<float>(0, 0));

    const int nClasses = std::max(1, int(m_modelClasses.size()));

    float confidence = 1.0f;
    // 用森林投票估算置信度（票数从第 1 行开始，见 votesOf 注释）
    auto rt = m_model.dynamicCast<cv::ml::RTrees>();
    if (rt) {
        cv::Mat votes;
        try {
            rt->getVotes(feat, votes, 0);
            const std::vector<double> v = votesOf(votes, 0, 1, nClasses);
            double total = 0.0;
            for (double x : v)
                total += x;
            if (total > 0.0 && cls >= 0 && cls < int(v.size()))
                confidence = static_cast<float>(v[cls] / total);
        } catch (...) {
            // 投票失败不影响类别结果
        }
    }

    out->classId = cls;
    out->score = confidence;

    const int n = std::max(int(m_modelClasses.size()), cls + 1);
    out->scores.assign(n, 0.f);
    if (cls >= 0 && cls < n)
        out->scores[cls] = confidence;
    return true;
}

bool OpenCvTrainer::predictSegmentation(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    Q_UNUSED(err);
    // 大图先降采样再逐像素推理，最后把掩码放大回原尺寸（最近邻）
    const double kMaxPixels = 1.2e6;
    cv::Mat work = bgr;
    double shrink = 1.0;

    if (double(bgr.cols) * bgr.rows > kMaxPixels) {
        shrink = std::sqrt(kMaxPixels / (double(bgr.cols) * bgr.rows));
        cv::resize(bgr, work, cv::Size(), shrink, shrink, cv::INTER_AREA);
    }

    const cv::Mat feats = segmentFeatures(work);
    cv::Mat res;
    m_model->predict(feats, res);

    cv::Mat small(work.rows, work.cols, CV_8UC1);
    int idx = 0;
    for (int y = 0; y < work.rows; ++y) {
        uchar *row = small.ptr<uchar>(y);
        for (int x = 0; x < work.cols; ++x, ++idx) {
            int c = static_cast<int>(res.at<float>(idx, 0));
            if (c < 0) c = 0;
            if (c > 255) c = 255;
            row[x] = static_cast<uchar>(c);
        }
    }

    if (work.size() != bgr.size())
        cv::resize(small, out->mask, bgr.size(), 0, 0, cv::INTER_NEAREST);
    else
        out->mask = small;

    out->classId = 0;
    out->score = 1.0f;
    return true;
}

bool OpenCvTrainer::predictDetection(const cv::Mat &bgr, PredictionResult *out, QString *err)
{
    // 图块特征边长。默认取元数据里记录的输入尺寸；但**以模型自身的特征维度为准**
    // 再校验一次 —— 训练与推理的特征维度一旦不一致（例如元数据被外部改动、
    // 或用旧元数据配新模型），推理会静默给出垃圾结果甚至越界崩溃。
    int patch = m_modelInput.width > 0 ? m_modelInput.width : 16;
    if (m_model) {
        const int varCount = m_model->getVarCount();
        const int side = int(std::lround(std::sqrt(double(varCount > 0 ? varCount : 0))));
        if (side > 0 && side * side == varCount && side != patch) {
            emitLog(QStringLiteral("提示：模型的特征维度为 %1（%2×%3），"
                                  "与元数据记录的 %4 不一致，已按模型维度推理。")
                        .arg(varCount).arg(side).arg(side).arg(patch),
                    Warning);
            patch = side;
        }
    }

    // 多尺度 + 多长宽比滑窗
    std::vector<cv::Rect> windows;

    const int W = bgr.cols;
    const int H = bgr.rows;
    const int maxWindows = 20000;

    const double ratios[3][2] = { { 1.0, 1.0 }, { 2.0, 1.0 }, { 1.0, 2.0 } };

    for (double base = 32.0; base < std::min(W, H) * 0.8; base *= 1.6) {
        for (int r = 0; r < 3; ++r) {
            const int bw = std::max(16, int(base * ratios[r][0]));
            const int bh = std::max(16, int(base * ratios[r][1]));
            if (bw >= W || bh >= H)
                continue;

            const int strideX = std::max(8, bw / 4);
            const int strideY = std::max(8, bh / 4);

            for (int y = 0; y + bh <= H; y += strideY) {
                for (int x = 0; x + bw <= W; x += strideX) {
                    windows.emplace_back(x, y, bw, bh);
                    if (int(windows.size()) >= maxWindows)
                        goto done;
                }
            }
        }
    }
done:

    if (windows.empty()) {
        if (err) *err = QStringLiteral("图像过小，无法进行滑窗检测。");
        return false;
    }

    // 批量提取特征
    cv::Mat feats(int(windows.size()), patch * patch, CV_32F);
    for (size_t i = 0; i < windows.size(); ++i) {
        const cv::Mat f = patchFeature(bgr, windows[i], patch);
        f.copyTo(feats.row(int(i)));
    }

    cv::Mat res;
    m_model->predict(feats, res);

    // 置信度（森林投票，票数从第 1 行开始）
    cv::Mat votes;
    auto rt = m_model.dynamicCast<cv::ml::RTrees>();
    if (rt) {
        try { rt->getVotes(feats, votes, 0); } catch (...) { votes = cv::Mat(); }
    }

    const float kMinScore = static_cast<float>(m_test.scoreThreshold);
    const int nClasses = std::max(1, int(m_modelClasses.size()));

    // 先收集所有被判定为「目标类」的滑窗及其得分
    std::vector<DetectionBox> raw;
    for (size_t i = 0; i < windows.size(); ++i) {
        const int cls = static_cast<int>(res.at<float>(int(i), 0));
        if (cls <= 0)                        // 0 = 背景
            continue;

        float score = 1.0f;
        if (!votes.empty()) {
            const std::vector<double> v = votesOf(votes, int(i), int(windows.size()), nClasses);
            double total = 0.0;
            for (double x : v)
                total += x;
            if (total > 0.0 && cls >= 0 && cls < int(v.size()))
                score = static_cast<float>(v[cls] / total);
        }

        DetectionBox b;
        b.classId = cls;
        b.score = score;
        b.rect = QRectF(windows[i].x, windows[i].y, windows[i].width, windows[i].height);
        raw.push_back(b);
    }

    // 优先保留达到分数阈值的候选；若一个都没有（轻量后端在困难场景下常见），
    // 退化为「取得分最高的若干目标类滑窗」，而不是直接返回空结果，
    // 否则使用者会以为模型完全失效。
    std::vector<DetectionBox> candidates;
    for (const DetectionBox &b : raw) {
        if (b.score >= kMinScore)
            candidates.push_back(b);
    }

    const bool degraded = candidates.empty() && !raw.empty();
    if (degraded) {
        std::sort(raw.begin(), raw.end(),
                  [](const DetectionBox &a, const DetectionBox &b) { return a.score > b.score; });
        const size_t keep = std::min<size_t>(20, raw.size());
        candidates.assign(raw.begin(), raw.begin() + static_cast<long>(keep));
    }

    // 按类别做 NMS。
    // 多尺度滑窗会产生「大框套小框」的重复：两者 IoU 往往低于 nmsThreshold
    // （尺寸差得多时 IoU 上不去），单纯靠 IoU 抑制不掉，会留下一大堆重复框
    // 把精确率拖垮。因此这里在 IoU 抑制之外再加一条**包含抑制**：
    // 若低分框有 80% 以上面积落在某个高分框内部，就判为同一个目标的重复。
    std::sort(candidates.begin(), candidates.end(),
              [](const DetectionBox &a, const DetectionBox &b) { return a.score > b.score; });

    const double nmsIou = m_test.nmsThreshold > 0.0 ? m_test.nmsThreshold : 0.45;
    std::vector<bool> removed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (removed[i])
            continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (removed[j] || candidates[j].classId != candidates[i].classId)
                continue;
            if (iou(candidates[i].rect, candidates[j].rect) > nmsIou) {
                removed[j] = true;
                continue;
            }
            const QRectF inter = candidates[i].rect.intersected(candidates[j].rect);
            const double areaJ = candidates[j].rect.width() * candidates[j].rect.height();
            if (areaJ > 0.0 && inter.width() * inter.height() / areaJ >= 0.8)
                removed[j] = true;
        }
    }

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (!removed[i])
            out->boxes.push_back(candidates[i]);
    }

    emitLog(QStringLiteral("滑窗 %1 个，判定为目标类 %2 个，分数 ≥ %3 的候选 %4 个%5，NMS 后保留 %6 个。")
                .arg(windows.size())
                .arg(raw.size())
                .arg(double(kMinScore), 0, 'f', 2)
                .arg(degraded ? 0 : int(candidates.size()))
                .arg(degraded ? QStringLiteral("（已降级为取最高分候选）") : QString())
                .arg(out->boxes.size()), Info);

    if (!out->boxes.empty()) {
        out->classId = out->boxes.front().classId;
        out->score = out->boxes.front().score;
    }
    return true;
}

} // namespace aq
