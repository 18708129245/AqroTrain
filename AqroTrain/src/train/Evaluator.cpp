// ============================================================================
//  Evaluator.cpp —— 训练后的自动测试与指标计算（对应 AIDI 的「结果验证」）
//  ---------------------------------------------------------------------------
//  流程：按模块后端创建训练器 -> 载入模型 -> 在测试集上逐图推理
//        -> 与真值比对累加指标 -> finalize 汇总
//
//  真值来源：
//    · 分类：TrainSample.classId（<0 表示该图无标注，跳过 P/R 累加）
//    · 检测：TrainSample.labelPath 的标注对象列表（anno::readLabel）
//    · 分割：TrainSample.maskPath 的派生掩码；缺失时由对象列表光栅化
//
//  OK/NG 口径（检测 / 分割）：
//    真值含目标即 NG；预测存在「有效目标」即判为 NG。
//    漏检率 = 未检出 NG 数 / NG 总数，过检率 = OK 判为 NG / OK 总数。
// ============================================================================
#include "train/Evaluator.h"

#include <QFileInfo>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <limits>

#include "annotate/AnnoObject.h"
#include "core/ImageIO.h"
#include "core/Metrics.h"
#include "train/ITrainer.h"
#include "train/TrainerFactory.h"

namespace aq {

namespace {

/// 由标注对象列表得到检测真值框（矩形 / 多边形 / 笔画统一取其包围盒）
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

/// 预测框里是否存在「目标类且置信度达标」的框（用于 OK/NG 判定）
bool predHasTarget(const PredictionResult &pred, double scoreThreshold)
{
    for (const DetectionBox &b : pred.boxes) {
        if (b.classId >= 1 && b.score >= scoreThreshold)
            return true;
    }
    return false;
}

/// 预测掩码里是否存在面积 >= minArea 的非 0 连通域（用于 OK/NG 判定）
bool maskHasTarget(const cv::Mat &mask, int minArea)
{
    if (mask.empty())
        return false;

    cv::Mat bin;
    if (mask.type() != CV_8UC1)
        mask.convertTo(bin, CV_8UC1);
    else
        bin = mask;

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8);
    for (int i = 1; i < n; ++i) {   // 0 是背景
        if (stats.at<int>(i, cv::CC_STAT_AREA) >= minArea)
            return true;
    }
    return false;
}

/// 真值掩码里是否存在非 0 像素
bool maskHasForeground(const cv::Mat &mask)
{
    if (mask.empty())
        return false;
    return cv::countNonZero(mask) > 0;
}

} // namespace

// ---------------------------------------------------------------------------
Evaluator::Evaluator(QObject *parent) : QObject(parent) {}

Evaluator::~Evaluator() = default;

// ---------------------------------------------------------------------------
bool Evaluator::evaluate(const Module &module,
                         const QVector<TrainSample> &testSamples,
                         const QString &modelPath,
                         Metrics *out,
                         QVector<Item> *items,
                         QString *err)
{
    if (!out) {
        if (err) *err = QStringLiteral("输出指标参数为空。");
        return false;
    }

    const TaskType task = module.type();
    const QVector<ClassInfo> &classes = module.classes();

    // ---- 1. 选择后端 ----
    // 优先按模块配置的后端创建；不可用时退回 OpenCV 轻量后端并说明原因。
    TrainerFactory::Backend backend = TrainerFactory::fromKey(module.train().backend);
    ITrainer *trainer = TrainerFactory::create(nullptr, backend);

    if (!trainer && backend != TrainerFactory::OpenCV) {
        const QString wanted = module.train().backend.isEmpty()
                                   ? QStringLiteral("auto")
                                   : module.train().backend;
        trainer = TrainerFactory::create(nullptr, TrainerFactory::OpenCV);
        if (err) {
            *err = QStringLiteral("后端「%1」不可用，已退回 OpenCV 轻量后端。")
                       .arg(wanted);
        }
        emit logMessage(QStringLiteral("后端「%1」不可用，已退回 OpenCV 轻量后端进行评价。")
                            .arg(wanted),
                        static_cast<int>(ITrainer::Warning));
    }
    if (!trainer) {
        if (err) *err = QStringLiteral("无法创建任何可用的训练后端。");
        return false;
    }

    // 作用域守卫：任何返回路径都释放训练器
    struct TrainerGuard {
        ITrainer *t = nullptr;
        ~TrainerGuard() { delete t; }
    } guard { trainer };

    // ---- 2. 载入模型 ----
    if (modelPath.trimmed().isEmpty() || !QFileInfo::exists(modelPath)) {
        if (err) *err = QStringLiteral("模型文件不存在：%1").arg(modelPath);
        return false;
    }

    QString loadErr;
    if (!trainer->loadModel(modelPath, &loadErr)) {
        if (err) *err = QStringLiteral("载入模型失败：%1").arg(loadErr);
        emit logMessage(QStringLiteral("载入模型失败：%1").arg(loadErr),
                        static_cast<int>(ITrainer::Error));
        return false;
    }

    const TaskType modelTask = trainer->modelTask();
    Q_UNUSED(modelTask);

    // ---- 3. 初始化指标累加器 ----
    metrics::reset(out, task, classes);

    const int numClasses = classes.size();
    const TestConfig testCfg = module.test();

    // 评价只加载模型、不执行 initialize，因此必须显式把模块的测试参数交给后端。
    // 否则推理会一直用后端内部默认阈值，表现为「界面上改了置信度阈值却对结果没影响」。
    trainer->setTestConfig(testCfg);

    // ---- 4. 逐样本推理与累加 ----
    const int total = testSamples.size();
    int done = 0;
    int readFailed = 0;

    for (const TrainSample &sample : testSamples) {
        const QString key = sample.imageKey.isEmpty() ? sample.imagePath : sample.imageKey;
        ++done;

        // 单张图读取失败：记录并跳过，不影响整体评价
        if (!sample.valid() || !QFileInfo::exists(sample.imagePath)) {
            ++readFailed;
            emit logMessage(QStringLiteral("跳过无法读取的图像：%1").arg(key),
                            static_cast<int>(ITrainer::Warning));
            emit progress(done, total, key);
            continue;
        }

        const cv::Mat img = imageio::imreadUnicode(sample.imagePath, cv::IMREAD_COLOR);
        if (img.empty()) {
            ++readFailed;
            emit logMessage(QStringLiteral("图像解码失败，已跳过：%1").arg(key),
                            static_cast<int>(ITrainer::Warning));
            emit progress(done, total, key);
            continue;
        }

        PredictionResult pred;
        QString predErr;
        if (!trainer->predict(img, &pred, &predErr)) {
            ++readFailed;
            emit logMessage(QStringLiteral("推理失败，已跳过：%1（%2）").arg(key, predErr),
                            static_cast<int>(ITrainer::Warning));
            emit progress(done, total, key);
            continue;
        }

        pred.task = task;

        // ---- 按任务类型累加指标 ----
        switch (task) {
        // ================= 分类 =================
        case TaskType::Classification: {
            // 真值类别取 sample.classId；<0 表示该图无标注：
            // 跳过 P/R 累加，但样本仍计入 testSamples。
            const int truth = sample.classId;
            if (truth >= 0 && truth < numClasses)
                metrics::addClassificationSample(out, truth, pred.classId);
            else
                ++out->testSamples;
            // 分类任务不适用漏检率 / 过检率，保持 0。
            break;
        }

        // ================= 检测 =================
        case TaskType::Detection: {
            QVector<AnnoObject> truthObjects;
            QString truthErr;
            const bool truthOk = anno::readLabel(sample.labelPath, sample.maskPath,
                                                 TaskType::Detection,
                                                 &truthObjects, nullptr, nullptr,
                                                 testCfg.minArea, &truthErr);
            if (!truthOk) {
                emit logMessage(QStringLiteral("读不到标注真值，按 OK 处理：%1").arg(key),
                                static_cast<int>(ITrainer::Warning));
            }

            // 逐类 IoU 匹配 -> 每类 tp/fp/fn 增量
            const QVector<DetectionBox> truths = boxesFromObjects(truthObjects);
            QVector<DetectionBox> predBoxes;
            predBoxes.reserve(static_cast<int>(pred.boxes.size()));
            for (const DetectionBox &b : pred.boxes)
                predBoxes.append(b);

            QVector<long> tp, fp, fn;
            metrics::matchBoxes(truths, predBoxes, testCfg.iouThreshold,
                                numClasses, &tp, &fp, &fn);
            metrics::accumulatePerClass(out, tp, fp, fn);
            ++out->testSamples;

            // OK / NG
            const bool truthIsNg = truthOk
                ? anno::hasTargetObject(truthObjects, classes, true)
                : false;   // 读不到标注文件时视为 OK
            const bool predIsNg = predHasTarget(pred, testCfg.scoreThreshold);
            metrics::addOkNgSample(out, truthIsNg, predIsNg);
            break;
        }

        // ================= 分割 =================
        case TaskType::Segmentation:
        default: {
            // 真值掩码：优先 sample.maskPath，缺失时由对象列表光栅化
            cv::Mat gt;
            QVector<AnnoObject> truthObjects;
            if (!sample.maskPath.trimmed().isEmpty())
                gt = imageio::imreadUnicode(sample.maskPath, cv::IMREAD_GRAYSCALE);

            if (gt.empty()) {
                anno::readLabel(sample.labelPath, sample.maskPath, TaskType::Segmentation,
                                &truthObjects, nullptr, nullptr, testCfg.minArea, nullptr);
                if (!truthObjects.isEmpty())
                    anno::rasterize(truthObjects, img.cols, img.rows, &gt);
            }

            if (gt.empty()) {
                emit logMessage(QStringLiteral("读不到分割真值掩码，按 OK 处理：%1").arg(key),
                                static_cast<int>(ITrainer::Warning));
                metrics::addOkNgSample(out, false, maskHasTarget(pred.mask, testCfg.minArea));
                ++out->testSamples;
                break;
            }

            metrics::addSegmentationSample(out, gt, pred.mask, qMax(1, numClasses));

            // OK / NG
            const bool truthIsNg = maskHasForeground(gt);
            const bool predIsNg = maskHasTarget(pred.mask, testCfg.minArea);
            metrics::addOkNgSample(out, truthIsNg, predIsNg);
            break;
        }
        }

        // ---- 收集每图预测（供调用方落盘 / 叠加显示）----
        if (items) {
            Item it;
            it.sample = sample;
            it.pred = pred;
            items->append(it);
        }

        emit progress(done, total, key);
    }

    // ---- 5. 一张都没成功则整体失败 ----
    if (total <= 0) {
        if (err) *err = QStringLiteral("测试集为空，无法进行评价。");
        return false;
    }
    if (readFailed >= total) {
        if (err) {
            *err = QStringLiteral("测试集的 %1 张图像全部读取或推理失败，无法评价。")
                       .arg(total);
        }
        return false;
    }

    // ---- 6. 汇总 ----
    metrics::finalize(out);
    return true;
}

} // namespace aq
