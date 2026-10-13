// ============================================================================
//  Metrics.h —— 评价指标计算（对应 AIDI 的结果验证环节）
//  ---------------------------------------------------------------------------
//  本文件只做纯计算，不涉及界面与文件布局：
//    · 分类：单标签混淆矩阵 -> 逐类 P / R / F1
//    · 检测：IoU 匹配（贪心）-> 逐类 TP / FP / FN -> P / R
//    · 分割：逐像素混淆矩阵 -> 逐类 P / R / IoU -> mIoU
//    · OK/NG 判定 -> 漏检率（未检出 NG / NG 总数）、过检率（OK 判为 NG / OK 总数）
//
//  计数约定：Metrics::perClass 的下标 == 类别索引，
//  因此调用前应先按类别数把 perClass 初始化为全部 0 计数（见 reset()）。
// ============================================================================
#pragma once

#include <QJsonObject>
#include <QRectF>
#include <QString>
#include <QVector>

#include <opencv2/core.hpp>

#include "core/Types.h"

namespace aq {
namespace metrics {

/// 按类别数把 Metrics 重置为「全部 0 计数」的累加器状态
void reset(Metrics *m, TaskType task, const QVector<ClassInfo> &classes);

/// 二分类 P / R / F1 与 IoU 的通用计算
ClassMetric makeClassMetric(int classId, const QString &name,
                            long tp, long fp, long fn, double iou = 0.0);

/// 由 perClass 的计数聚合出整体指标（宏平均）
void finalize(Metrics *m);

// --- 分类 ------------------------------------------------------------------

/// 累加一张分类样本（truth / pred 为类别索引；无背景类，索引 0 即第一个真实类别）
void addClassificationSample(Metrics *m, int truth, int pred);

// --- 检测 ------------------------------------------------------------------

/// 两个矩形的 IoU
double boxIoU(const QRectF &a, const QRectF &b);

/**
 * 单图检测结果匹配：真值框与预测框按类别分组后，按 IoU 降序贪心匹配。
 * 每类的 TP / FP / FN 增量会累加到 tpDelta / fpDelta / fnDelta（下标为类别索引）。
 */
void matchBoxes(const QVector<DetectionBox> &truths, const QVector<DetectionBox> &preds,
                double iouThreshold, int numClasses,
                QVector<long> *tpDelta, QVector<long> *fpDelta, QVector<long> *fnDelta);

// --- 分割 ------------------------------------------------------------------

/**
 * 累加一张分割样本：gt / pred 均为 CV_8UC1 且像素值 = 类别索引。
 * 尺寸不一致时先按最近邻把 pred 缩放到 gt 尺寸。
 */
void addSegmentationSample(Metrics *m, const cv::Mat &gt, const cv::Mat &pred, int numClasses);

// --- OK / NG ---------------------------------------------------------------

/**
 * 累加一张样本的 OK/NG 判定：
 *   truthIsNg —— 该图真值是否含目标（即 NG）
 *   predIsNg  —— 模型是否判定该图含目标（即 NG）
 * 漏检 = 真值 NG 但未检出；过检 = 真值 OK 但判为 NG。
 */
void addOkNgSample(Metrics *m, bool truthIsNg, bool predIsNg);

// --- 直接累加计数（供匹配类任务使用）----------------------------------------

/// 把检测匹配得到的每类增量写入 perClass
void accumulatePerClass(Metrics *m, const QVector<long> &tp, const QVector<long> &fp,
                        const QVector<long> &fn);

// --- JSON 往返 -------------------------------------------------------------

QJsonObject metricsToJson(const Metrics &m);
Metrics     metricsFromJson(const QJsonObject &o);

} // namespace metrics
} // namespace aq
