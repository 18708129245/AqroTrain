// ============================================================================
//  LossChartWidget.cpp
// ============================================================================
#include "ui/LossChartWidget.h"

#include <QPainter>
#include <QPainterPath>

#include <algorithm>
#include <cmath>

namespace aq {

LossChartWidget::LossChartWidget(QWidget *parent) : QWidget(parent)
{
    setMinimumHeight(180);
    setAutoFillBackground(true);
    QPalette pal = palette();
    pal.setColor(QPalette::Window, QColor(0x1B, 0x1E, 0x24));
    setPalette(pal);
}

void LossChartWidget::clear()
{
    m_points.clear();
    update();
}

void LossChartWidget::appendPoint(double x, double loss, double metric, double valLoss, double valMetric)
{
    m_points.append({ x, loss, metric, valLoss, valMetric });
    update();
}

void LossChartWidget::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    const QRectF full = rect().adjusted(0, 0, -1, -1);
    p.fillRect(full, QColor(0x1B, 0x1E, 0x24));

    // 边距：左侧留给损失刻度，右侧留给指标刻度
    const double left = 46.0, right = 46.0, top = 22.0, bottom = 26.0;
    const QRectF plot = full.adjusted(left, top, -right, -bottom);
    if (plot.width() < 20 || plot.height() < 20)
        return;

    // --- 网格 ---
    p.setPen(QPen(QColor(0x33, 0x38, 0x42), 1, Qt::DotLine));
    for (int i = 0; i <= 4; ++i) {
        const double y = plot.top() + plot.height() * i / 4.0;
        p.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
    }

    p.setPen(QPen(QColor(0x55, 0x5B, 0x66), 1));
    p.drawRect(plot);

    // --- 标题 / 说明 ---
    p.setPen(QColor(0xC8, 0xCE, 0xD8));
    QFont f = p.font();
    f.setPointSizeF(f.pointSizeF() - 0.5);
    p.setFont(f);
    p.drawText(QRectF(full.left() + 6, full.top() + 2, full.width() - 12, top - 4),
               Qt::AlignLeft | Qt::AlignVCenter,
               QStringLiteral("横轴：%1　左轴：损失（实线=训练，虚线=验证）　右轴：%2")
                   .arg(m_xLabel, m_metricName));

    if (m_points.isEmpty()) {
        p.setPen(QColor(0x7A, 0x83, 0x94));
        p.drawText(plot, Qt::AlignCenter, QStringLiteral("等待训练开始…"));
        return;
    }

    // --- 量程 ---
    double xMin = m_points.first().x, xMax = m_points.first().x;
    double lossMax = 1e-9;
    for (const Point &pt : m_points) {
        xMin = std::min(xMin, pt.x);
        xMax = std::max(xMax, pt.x);
        lossMax = std::max({ lossMax, pt.loss, pt.valLoss });
    }
    if (xMax - xMin < 1e-9)
        xMax = xMin + 1.0;
    lossMax *= 1.15;

    auto xOf = [&](double x) { return plot.left() + (x - xMin) / (xMax - xMin) * plot.width(); };
    auto yLoss = [&](double v) { return plot.bottom() - std::min(1.0, v / lossMax) * plot.height(); };
    auto yMetric = [&](double v) { return plot.bottom() - std::max(0.0, std::min(1.0, v)) * plot.height(); };

    // --- 刻度文字 ---
    p.setPen(QColor(0x9A, 0xA3, 0xB2));
    for (int i = 0; i <= 4; ++i) {
        const double y = plot.top() + plot.height() * i / 4.0;
        const double v = lossMax * (1.0 - i / 4.0);
        p.drawText(QRectF(0, y - 8, left - 4, 16), Qt::AlignRight | Qt::AlignVCenter,
                   QString::number(v, 'f', v < 1.0 ? 2 : 1));
        p.drawText(QRectF(plot.right() + 4, y - 8, right - 6, 16), Qt::AlignLeft | Qt::AlignVCenter,
                   QString::number(1.0 - i / 4.0, 'f', 2));
    }

    // --- 曲线 ---
    auto drawCurve = [&](auto valueFn, auto yFn, const QColor &color, bool dashed) {
        QPainterPath path;
        bool first = true;
        for (const Point &pt : m_points) {
            const QPointF q(xOf(pt.x), yFn(valueFn(pt)));
            if (first) {
                path.moveTo(q);
                first = false;
            } else {
                path.lineTo(q);
            }
        }
        QPen pen(color);
        pen.setWidthF(1.8);
        pen.setCosmetic(true);
        if (dashed)
            pen.setStyle(Qt::DashLine);
        p.setPen(pen);
        p.setBrush(Qt::NoBrush);
        p.drawPath(path);
    };

    drawCurve([](const Point &pt) { return pt.loss; },   yLoss,   QColor(0xFF, 0x9F, 0x1C), false);
    drawCurve([](const Point &pt) { return pt.valLoss; }, yLoss,   QColor(0xFF, 0xD1, 0x8A), true);
    drawCurve([](const Point &pt) { return pt.metric; },  yMetric, QColor(0x3C, 0xB4, 0x4B), false);
    drawCurve([](const Point &pt) { return pt.valMetric; }, yMetric, QColor(0x9C, 0xE0, 0xA6), true);

    // --- 横轴刻度 ---
    p.setPen(QColor(0x9A, 0xA3, 0xB2));
    p.drawText(QRectF(plot.left(), plot.bottom() + 4, 80, bottom - 6),
               Qt::AlignLeft | Qt::AlignTop, QString::number(xMin, 'g', 4));
    p.drawText(QRectF(plot.right() - 80, plot.bottom() + 4, 80, bottom - 6),
               Qt::AlignRight | Qt::AlignTop, QString::number(xMax, 'g', 4));
}

} // namespace aq
