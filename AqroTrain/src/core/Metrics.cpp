// ============================================================================
//  Metrics.cpp —— 评价指标计算实现
// ============================================================================
#include "core/Metrics.h"

#include <QJsonArray>

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace aq {
namespace metrics {

// ---------------------------------------------------------------------------
//  基础
// ---------------------------------------------------------------------------
void reset(Metrics *m, TaskType task, const QVector<ClassInfo> &classes)
{
    if (!m)
        return;

    m->task  = task;
    m->valid = false;

    m->perClass.clear();
    m->perClass.reserve(classes.size());
    for (int i = 0; i < classes.size(); ++i) {
        ClassMetric cm;
        cm.classId = i;
        cm.name    = classes[i].name;
        m->perClass.append(cm);
    }

    m->precision = m->recall = m->f1 = m->mIoU = 0.0;
    m->missRate = m->overkillRate = 0.0;
    m->ngTotal = m->ngMissed = m->okTotal = m->okOverkill = 0;
    m->testSamples = 0;
}

ClassMetric makeClassMetric(int classId, const QString &name,
                            long tp, long fp, long fn, double iou)
{
    ClassMetric cm;
    cm.classId = classId;
    cm.name    = name;
    cm.tp = tp;
    cm.fp = fp;
    cm.fn = fn;
    cm.iou = iou;

    const double denomP = double(tp + fp);
    const double denomR = double(tp + fn);
    cm.precision = denomP > 0.0 ? double(tp) / denomP : 0.0;
    cm.recall    = denomR > 0.0 ? double(tp) / denomR : 0.0;
    cm.f1        = (cm.precision + cm.recall) > 0.0
                       ? 2.0 * cm.precision * cm.recall / (cm.precision + cm.recall)
                       : 0.0;
    return cm;
}

void finalize(Metrics *m)
{
    if (!m)
        return;

    double sumP = 0.0, sumR = 0.0, sumF = 0.0, sumIoU = 0.0;
    int    nIoU = 0;
    int    counted = 0;

    for (ClassMetric &cm : m->perClass) {
        cm = makeClassMetric(cm.classId, cm.name, cm.tp, cm.fp, cm.fn, cm.iou);

        // 完全没有真值也没有预测的类别不参与平均，避免拉低指标造成误判
        if (cm.tp == 0 && cm.fp == 0 && cm.fn == 0)
            continue;

        sumP += cm.precision;
        sumR += cm.recall;
        sumF += cm.f1;
        if (m->task == TaskType::Segmentation) {
            sumIoU += cm.iou;
            ++nIoU;
        }
        ++counted;
    }

    if (counted > 0) {
        m->precision = sumP / counted;
        m->recall    = sumR / counted;
        m->f1        = sumF / counted;
    }
    m->mIoU = nIoU > 0 ? sumIoU / nIoU : 0.0;

    m->missRate     = m->ngTotal > 0 ? double(m->ngMissed) / double(m->ngTotal) : 0.0;
    m->overkillRate = m->okTotal > 0 ? double(m->okOverkill) / double(m->okTotal) : 0.0;

    m->valid = true;
}

// ---------------------------------------------------------------------------
//  分类
// ---------------------------------------------------------------------------
void addClassificationSample(Metrics *m, int truth, int pred)
{
    if (!m)
        return;
    if (truth < 0 || truth >= m->perClass.size())
        return;

    // 真值类命中算 TP；否则该真值类记 FN、预测类记 FP
    if (pred == truth) {
        ++m->perClass[truth].tp;
    } else {
        ++m->perClass[truth].fn;
        if (pred >= 0 && pred < m->perClass.size())
            ++m->perClass[pred].fp;
    }
    ++m->testSamples;
}

// ---------------------------------------------------------------------------
//  检测
// ---------------------------------------------------------------------------
double boxIoU(const QRectF &a, const QRectF &b)
{
    const QRectF inter = a.intersected(b);
    if (inter.width() <= 0.0 || inter.height() <= 0.0)
        return 0.0;

    const double ia = inter.width() * inter.height();
    const double ua = a.width() * a.height() + b.width() * b.height() - ia;
    return ua > 0.0 ? ia / ua : 0.0;
}

void matchBoxes(const QVector<DetectionBox> &truths, const QVector<DetectionBox> &preds,
                double iouThreshold, int numClasses,
                QVector<long> *tpDelta, QVector<long> *fpDelta, QVector<long> *fnDelta)
{
    if (numClasses <= 0)
        return;

    if (tpDelta) { tpDelta->fill(0, numClasses); }
    if (fpDelta) { fpDelta->fill(0, numClasses); }
    if (fnDelta) { fnDelta->fill(0, numClasses); }

    QVector<bool> predUsed(preds.size(), false);

    // 逐类独立匹配；类内按「IoU 降序」贪心，与常见检测评价口径一致
    for (int c = 0; c < numClasses; ++c) {
        struct Pair { int gt; int pr; double iou; };
        QVector<Pair> candidates;

        for (int g = 0; g < truths.size(); ++g) {
            if (truths[g].classId != c)
                continue;
            for (int p = 0; p < preds.size(); ++p) {
                if (preds[p].classId != c || predUsed[p])
                    continue;
                const double iou = boxIoU(truths[g].rect, preds[p].rect);
                if (iou >= iouThreshold)
                    candidates.append({ g, p, iou });
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Pair &x, const Pair &y) { return x.iou > y.iou; });

        QVector<bool> gtUsed(truths.size(), false);
        for (const Pair &pair : candidates) {
            if (gtUsed[pair.gt] || predUsed[pair.pr])
                continue;
            gtUsed[pair.gt]   = true;
            predUsed[pair.pr] = true;
            if (tpDelta)
                (*tpDelta)[c] += 1;
        }

        for (int g = 0; g < truths.size(); ++g) {
            if (truths[g].classId == c && !gtUsed[g] && fnDelta)
                (*fnDelta)[c] += 1;
        }
        for (int p = 0; p < preds.size(); ++p) {
            if (preds[p].classId == c && !predUsed[p] && fpDelta)
                (*fpDelta)[c] += 1;
        }
    }
}

void accumulatePerClass(Metrics *m, const QVector<long> &tp, const QVector<long> &fp,
                        const QVector<long> &fn)
{
    if (!m)
        return;
    const int n = qMin(m->perClass.size(), qMin(tp.size(), qMin(fp.size(), fn.size())));
    for (int i = 0; i < n; ++i) {
        m->perClass[i].tp += tp[i];
        m->perClass[i].fp += fp[i];
        m->perClass[i].fn += fn[i];
    }
}

// ---------------------------------------------------------------------------
//  分割
// ---------------------------------------------------------------------------
void addSegmentationSample(Metrics *m, const cv::Mat &gt, const cv::Mat &pred, int numClasses)
{
    if (!m || gt.empty() || pred.empty() || numClasses <= 0)
        return;

    cv::Mat g, p;
    if (gt.type() != CV_8UC1)
        gt.convertTo(g, CV_8UC1);
    else
        g = gt;
    if (pred.type() != CV_8UC1)
        pred.convertTo(p, CV_8UC1);
    else
        p = pred;

    if (p.size() != g.size())
        cv::resize(p, p, g.size(), 0, 0, cv::INTER_NEAREST);

    // 用 256x256 的计数矩阵统计混淆对，支持类别数 <= 256
    // 行 = 真值类别，列 = 预测类别
    cv::Mat confusion = cv::Mat::zeros(256, 256, CV_32S);
    for (int y = 0; y < g.rows; ++y) {
        const uchar *gr = g.ptr<uchar>(y);
        const uchar *pr = p.ptr<uchar>(y);
        for (int x = 0; x < g.cols; ++x) {
            const int gi = gr[x];
            const int pi = pr[x];
            confusion.at<int>(gi, pi) += 1;
        }
    }

    const int n = qMin(numClasses, qMin(m->perClass.size(), 256));
    for (int c = 0; c < n; ++c) {
        const long tp = confusion.at<int>(c, c);
        long fp = 0, fn = 0;
        for (int k = 0; k < 256; ++k) {
            if (k == c)
                continue;
            fp += confusion.at<int>(k, c);   // 预测为该类但真值非该类
            fn += confusion.at<int>(c, k);   // 真值为该类但预测非该类
        }
        m->perClass[c].tp += tp;
        m->perClass[c].fp += fp;
        m->perClass[c].fn += fn;

        const double denom = double(tp + fp + fn);
        m->perClass[c].iou = denom > 0.0 ? double(tp) / denom : 0.0;
    }
    ++m->testSamples;
}

// ---------------------------------------------------------------------------
//  OK / NG
// ---------------------------------------------------------------------------
void addOkNgSample(Metrics *m, bool truthIsNg, bool predIsNg)
{
    if (!m)
        return;
    if (truthIsNg) {
        ++m->ngTotal;
        if (!predIsNg)
            ++m->ngMissed;
    } else {
        ++m->okTotal;
        if (predIsNg)
            ++m->okOverkill;
    }
}

// ---------------------------------------------------------------------------
//  JSON 往返
// ---------------------------------------------------------------------------
QJsonObject metricsToJson(const Metrics &m)
{
    QJsonObject o;
    o.insert(QStringLiteral("task"), taskTypeKey(m.task));
    o.insert(QStringLiteral("valid"), m.valid);
    o.insert(QStringLiteral("precision"), m.precision);
    o.insert(QStringLiteral("recall"), m.recall);
    o.insert(QStringLiteral("f1"), m.f1);
    o.insert(QStringLiteral("mIoU"), m.mIoU);
    o.insert(QStringLiteral("missRate"), m.missRate);
    o.insert(QStringLiteral("overkillRate"), m.overkillRate);
    o.insert(QStringLiteral("ngTotal"), m.ngTotal);
    o.insert(QStringLiteral("ngMissed"), m.ngMissed);
    o.insert(QStringLiteral("okTotal"), m.okTotal);
    o.insert(QStringLiteral("okOverkill"), m.okOverkill);
    o.insert(QStringLiteral("testSamples"), m.testSamples);

    QJsonArray arr;
    for (const ClassMetric &cm : m.perClass) {
        QJsonObject c;
        c.insert(QStringLiteral("classId"), cm.classId);
        c.insert(QStringLiteral("name"), cm.name);
        c.insert(QStringLiteral("tp"), double(cm.tp));
        c.insert(QStringLiteral("fp"), double(cm.fp));
        c.insert(QStringLiteral("fn"), double(cm.fn));
        c.insert(QStringLiteral("precision"), cm.precision);
        c.insert(QStringLiteral("recall"), cm.recall);
        c.insert(QStringLiteral("f1"), cm.f1);
        c.insert(QStringLiteral("iou"), cm.iou);
        arr.append(c);
    }
    o.insert(QStringLiteral("perClass"), arr);
    return o;
}

Metrics metricsFromJson(const QJsonObject &o)
{
    Metrics m;
    bool ok = false;
    m.task  = taskTypeFromKey(o.value(QStringLiteral("task")).toString(), &ok);
    m.valid = o.value(QStringLiteral("valid")).toBool(false);

    m.precision    = o.value(QStringLiteral("precision")).toDouble(0.0);
    m.recall       = o.value(QStringLiteral("recall")).toDouble(0.0);
    m.f1           = o.value(QStringLiteral("f1")).toDouble(0.0);
    m.mIoU         = o.value(QStringLiteral("mIoU")).toDouble(0.0);
    m.missRate     = o.value(QStringLiteral("missRate")).toDouble(0.0);
    m.overkillRate = o.value(QStringLiteral("overkillRate")).toDouble(0.0);
    m.ngTotal      = o.value(QStringLiteral("ngTotal")).toInt(0);
    m.ngMissed     = o.value(QStringLiteral("ngMissed")).toInt(0);
    m.okTotal      = o.value(QStringLiteral("okTotal")).toInt(0);
    m.okOverkill   = o.value(QStringLiteral("okOverkill")).toInt(0);
    m.testSamples  = o.value(QStringLiteral("testSamples")).toInt(0);

    const QJsonArray arr = o.value(QStringLiteral("perClass")).toArray();
    for (int i = 0; i < arr.size(); ++i) {
        const QJsonObject c = arr.at(i).toObject();
        ClassMetric cm;
        cm.classId   = c.value(QStringLiteral("classId")).toInt(i);
        cm.name      = c.value(QStringLiteral("name")).toString();
        cm.tp        = long(c.value(QStringLiteral("tp")).toDouble(0.0));
        cm.fp        = long(c.value(QStringLiteral("fp")).toDouble(0.0));
        cm.fn        = long(c.value(QStringLiteral("fn")).toDouble(0.0));
        cm.precision = c.value(QStringLiteral("precision")).toDouble(0.0);
        cm.recall    = c.value(QStringLiteral("recall")).toDouble(0.0);
        cm.f1        = c.value(QStringLiteral("f1")).toDouble(0.0);
        cm.iou       = c.value(QStringLiteral("iou")).toDouble(0.0);
        m.perClass.append(cm);
    }
    return m;
}

} // namespace metrics
} // namespace aq
