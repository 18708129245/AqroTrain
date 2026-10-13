// ============================================================================
//  MetricsPanel.h —— 评价指标面板（对应 AIDI 右侧的结果验证区）
//  ---------------------------------------------------------------------------
//  展示：
//    · 整体：精确率 / 召回率 / F1 / mIoU
//    · 工业指标：漏检率（未检出 NG / NG 总数）、过检率（OK 判为 NG / OK 总数）
//    · 逐类明细：TP / FP / FN / 精确率 / 召回率 / IoU
// ============================================================================
#pragma once

#include <QWidget>

#include "core/Types.h"

class QLabel;
class QPushButton;
class QTableWidget;

namespace aq {

class MetricsPanel : public QWidget
{
    Q_OBJECT

public:
    explicit MetricsPanel(QWidget *parent = nullptr);

    void setMetrics(const Metrics &m);
    void clear();

signals:
    void evaluateRequested();      ///< 重新测试
    void exportModelRequested();   ///< 导出模型

private:
    QLabel       *m_lblHeadline = nullptr;
    QLabel       *m_lblMissOver = nullptr;
    QTableWidget *m_table = nullptr;
    QPushButton  *m_btnEvaluate = nullptr;
    QPushButton  *m_btnExport = nullptr;
};

} // namespace aq
