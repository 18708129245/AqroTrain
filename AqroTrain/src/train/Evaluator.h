// ============================================================================
//  Evaluator.h —— 训练后的自动测试与指标计算（对应 AIDI 的「结果验证」）
//  ---------------------------------------------------------------------------
//  流程：载入模型 -> 在测试集上逐图推理 -> 与真值比对 -> 汇总指标
//  同时把每张图的预测结果返回给调用方，供「显示测试结果」叠加显示。
//
//  指标口径（与 AIDI 对齐）：
//    · 分类：逐类精确率 / 召回率 / F1（混淆矩阵）。
//      「漏检率 / 过检率」是缺陷检测语义，分类任务不适用，按 0 处理并在
//      报告里注明；请用 P/R 评价分类模块。
//    · 检测：按类别做 IoU 匹配（贪心，阈值 test.iouThreshold）得逐类 P/R。
//    · 分割：逐像素混淆矩阵得逐类 P/R 与 IoU、mIoU。
//    · OK/NG 判定（检测与分割）：真值含目标即 NG；
//      漏检率 = 未检出 NG 数 / NG 总数，过检率 = OK 判为 NG / OK 总数。
// ============================================================================
#pragma once

#include <QObject>
#include <QString>
#include <QVector>

#include "core/Types.h"
#include "project/Module.h"

namespace aq {

class Evaluator : public QObject
{
    Q_OBJECT

public:
    /// 单张图的评价结果（预测结果由调用方落盘，供后续叠加显示）
    struct Item {
        TrainSample      sample;
        PredictionResult pred;
    };

    explicit Evaluator(QObject *parent = nullptr);
    ~Evaluator() override;

    /**
     * 在测试集上评价模型。同步执行，期间通过 progress() 上报进度。
     * module        当前模块（提供任务类型、类别、测试参数）
     * testSamples   测试集样本（其 train.samples 的 labelPath 为标注真源）
     * modelPath     待评价的模型文件
     * out           汇总指标
     * items         每张图的预测结果（可为 nullptr）
     */
    bool evaluate(const Module &module,
                  const QVector<TrainSample> &testSamples,
                  const QString &modelPath,
                  Metrics *out,
                  QVector<Item> *items = nullptr,
                  QString *err = nullptr);

signals:
    void progress(int done, int total, const QString &imageKey);
    void logMessage(const QString &text, int level);
};

} // namespace aq
