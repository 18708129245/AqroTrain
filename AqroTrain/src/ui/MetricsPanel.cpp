// ============================================================================
//  MetricsPanel.cpp
// ============================================================================
#include "ui/MetricsPanel.h"

#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QVBoxLayout>

namespace aq {

namespace {

QString pct(double v)
{
    return QStringLiteral("%1%").arg(v * 100.0, 0, 'f', 2);
}

} // namespace

MetricsPanel::MetricsPanel(QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    m_lblHeadline = new QLabel(QStringLiteral("尚无评价结果"), this);
    m_lblHeadline->setWordWrap(true);
    layout->addWidget(m_lblHeadline);

    m_lblMissOver = new QLabel(this);
    m_lblMissOver->setWordWrap(true);
    m_lblMissOver->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(m_lblMissOver);

    m_table = new QTableWidget(this);
    m_table->setColumnCount(6);
    m_table->setHorizontalHeaderLabels({ QStringLiteral("类别"), QStringLiteral("TP"),
                                         QStringLiteral("FP"), QStringLiteral("FN"),
                                         QStringLiteral("精确率"), QStringLiteral("召回率") });
    m_table->verticalHeader()->setVisible(false);
    m_table->horizontalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    m_table->horizontalHeader()->setStretchLastSection(true);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->setSelectionMode(QAbstractItemView::NoSelection);
    layout->addWidget(m_table, 1);

    auto *btns = new QHBoxLayout();
    btns->setContentsMargins(0, 0, 0, 0);
    m_btnEvaluate = new QPushButton(QStringLiteral("重新测试"), this);
    m_btnEvaluate->setToolTip(QStringLiteral("用当前模型在测试集上重新跑一次评价"));
    m_btnExport = new QPushButton(QStringLiteral("导出模型…"), this);
    m_btnExport->setToolTip(QStringLiteral("把模型与元数据导出到指定目录，便于部署"));
    btns->addWidget(m_btnEvaluate);
    btns->addWidget(m_btnExport);
    layout->addLayout(btns);

    connect(m_btnEvaluate, &QPushButton::clicked, this, &MetricsPanel::evaluateRequested);
    connect(m_btnExport, &QPushButton::clicked, this, &MetricsPanel::exportModelRequested);

    clear();
}

void MetricsPanel::clear()
{
    m_lblHeadline->setText(QStringLiteral("尚无评价结果。\n训练完成后会自动在测试集上评价。"));
    m_lblMissOver->setText(QStringLiteral("漏检率 / 过检率 针对检测与分割模块的"
                                          "「该图是否含目标」判定；分类模块请用精确率与召回率。"));
    m_table->setRowCount(0);
}

void MetricsPanel::setMetrics(const Metrics &m)
{
    if (!m.valid) {
        clear();
        return;
    }

    m_lblHeadline->setText(
        QStringLiteral("<b>%1</b>　精确率 %2　召回率 %3　F1 %4%5\n测试样本 %6 张")
            .arg(taskTypeName(m.task), pct(m.precision), pct(m.recall),
                 QString::number(m.f1, 'f', 4),
                 m.task == TaskType::Segmentation ? QStringLiteral("　mIoU %1").arg(m.mIoU, 0, 'f', 4)
                                                  : QString())
            .arg(m.testSamples));

    if (m.task == TaskType::Classification) {
        m_lblMissOver->setText(QStringLiteral(
            "分类模块的漏检率 / 过检率不适用（那是缺陷有无的判定语义）。"));
    } else {
        m_lblMissOver->setText(QStringLiteral(
                                   "漏检率 %1（%2 / %3 张 NG 未检出）　过检率 %4（%5 / %6 张 OK 判为 NG）")
                                   .arg(pct(m.missRate))
                                   .arg(m.ngMissed)
                                   .arg(m.ngTotal)
                                   .arg(pct(m.overkillRate))
                                   .arg(m.okOverkill)
                                   .arg(m.okTotal));
    }

    m_table->setRowCount(m.perClass.size());
    for (int r = 0; r < m.perClass.size(); ++r) {
        const ClassMetric &cm = m.perClass.at(r);
        const QString name = cm.name.isEmpty() ? QStringLiteral("类别 %1").arg(cm.classId) : cm.name;

        const QString vals[6] = {
            QStringLiteral("%1. %2").arg(cm.classId).arg(name),
            QString::number(cm.tp),
            QString::number(cm.fp),
            QString::number(cm.fn),
            cm.tp + cm.fp > 0 ? pct(cm.precision) : QStringLiteral("—"),
            cm.tp + cm.fn > 0 ? pct(cm.recall) : QStringLiteral("—"),
        };
        for (int c = 0; c < 6; ++c) {
            auto *item = new QTableWidgetItem(vals[c]);
            if (c > 0)
                item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            m_table->setItem(r, c, item);
        }
    }
}

} // namespace aq
