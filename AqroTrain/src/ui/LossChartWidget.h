// ============================================================================
//  LossChartWidget.h —— 训练曲线
//  ---------------------------------------------------------------------------
//  左右双纵轴：左轴为损失，右轴为该任务的评价指标（0..1，例如准确率 / mIoU）。
//  同时画训练集（实线）与验证集（虚线）两条曲线。
//
//  注意：OpenCV 轻量后端不上报 epoch，它把训练拆成若干「阶段」，
//  因此横轴单位是可变的，由 setXAxisLabel 说明。
// ============================================================================
#pragma once

#include <QVector>
#include <QWidget>

namespace aq {

class LossChartWidget : public QWidget
{
    Q_OBJECT

public:
    explicit LossChartWidget(QWidget *parent = nullptr);

    void clear();
    void setXAxisLabel(const QString &label) { m_xLabel = label; update(); }
    void setMetricName(const QString &name) { m_metricName = name; update(); }

    /// 追加一个数据点（x 为横轴刻度；metric / valMetric 取 0..1）
    void appendPoint(double x, double loss, double metric, double valLoss, double valMetric);

    int pointCount() const { return m_points.size(); }

protected:
    void paintEvent(QPaintEvent *e) override;

private:
    struct Point {
        double x = 0.0;
        double loss = 0.0;
        double metric = 0.0;
        double valLoss = 0.0;
        double valMetric = 0.0;
    };

    QVector<Point> m_points;
    QString m_xLabel = QStringLiteral("迭代");
    QString m_metricName = QStringLiteral("指标");
};

} // namespace aq
