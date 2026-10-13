// ============================================================================
//  TrainMonitorDialog.cpp
// ============================================================================
#include "ui/TrainMonitorDialog.h"

#include <QCloseEvent>
#include <QDialogButtonBox>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollBar>
#include <QVBoxLayout>

#include "ui/LossChartWidget.h"

namespace aq {

TrainMonitorDialog::TrainMonitorDialog(const QString &title, const QString &subtitle, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(title);
    resize(760, 620);

    auto *layout = new QVBoxLayout(this);

    auto *sub = new QLabel(subtitle, this);
    sub->setWordWrap(true);
    sub->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(sub);

    m_bar = new QProgressBar(this);
    m_bar->setRange(0, 100);
    m_bar->setValue(0);
    layout->addWidget(m_bar);

    m_stage = new QLabel(QStringLiteral("准备中…"), this);
    layout->addWidget(m_stage);

    m_chart = new LossChartWidget(this);
    layout->addWidget(m_chart, 1);

    m_log = new QPlainTextEdit(this);
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(5000);
    m_log->setStyleSheet(QStringLiteral("font-family: Consolas, 'Microsoft YaHei';"));
    layout->addWidget(m_log, 1);

    m_summary = new QLabel(this);
    m_summary->setWordWrap(true);
    layout->addWidget(m_summary);

    auto *buttons = new QDialogButtonBox(this);
    m_btnStop = buttons->addButton(QStringLiteral("中止训练"), QDialogButtonBox::RejectRole);
    m_btnStop->setEnabled(true);
    layout->addWidget(buttons);

    connect(m_btnStop, &QPushButton::clicked, this, [this]() {
        m_stopRequested = true;
        m_btnStop->setEnabled(false);
        m_stage->setText(QStringLiteral("正在中止…（等待当前阶段结束）"));
        emit stopClicked();
    });
}

void TrainMonitorDialog::appendLog(const QString &text, int level)
{
    QString color;
    switch (level) {
    case 1: color = QStringLiteral("#3CB44B"); break;   // Success
    case 2: color = QStringLiteral("#FFB020"); break;   // Warning
    case 3: color = QStringLiteral("#FF6060"); break;   // Error
    default: color = QStringLiteral("#C8CED8"); break;  // Info
    }
    m_log->appendHtml(QStringLiteral("<span style=\"color:%1\">%2</span>")
                          .arg(color, text.toHtmlEscaped()));
    m_log->verticalScrollBar()->setValue(m_log->verticalScrollBar()->maximum());
}

void TrainMonitorDialog::setProgress(int percent)
{
    m_bar->setValue(qBound(0, percent, 100));
}

void TrainMonitorDialog::setStage(const QString &text)
{
    m_stage->setText(text);
}

void TrainMonitorDialog::setXAxisLabel(const QString &label)
{
    m_chart->setXAxisLabel(label);
}

void TrainMonitorDialog::setMetricName(const QString &name)
{
    m_chart->setMetricName(name);
}

void TrainMonitorDialog::addPoint(int epoch, int total, double loss, double metric,
                                 double valLoss, double valMetric)
{
    Q_UNUSED(total)
    m_chart->appendPoint(epoch, loss, metric, valLoss, valMetric);
}

void TrainMonitorDialog::setCanStop(bool on)
{
    m_btnStop->setEnabled(on);
}

void TrainMonitorDialog::setFinished(bool ok, const QString &summary)
{
    m_running = false;
    m_btnStop->setEnabled(false);
    m_summary->setText(summary);
    m_summary->setStyleSheet(ok ? QStringLiteral("color:#3CB44B;")
                                : QStringLiteral("color:#FF6060;"));
    if (ok)
        m_bar->setValue(100);
}

void TrainMonitorDialog::closeEvent(QCloseEvent *e)
{
    if (m_running && !m_closeableIfRunning) {
        // 训练中直接关窗容易让人以为训练被放弃了，这里拦住并提示
        m_stage->setText(QStringLiteral("训练进行中：请点「中止训练」结束，或等待完成后再关闭窗口。"));
        e->ignore();
        return;
    }
    emit closedByUser();
    QDialog::closeEvent(e);
}

void TrainMonitorDialog::reject()
{
    if (m_running && !m_closeableIfRunning) {
        m_stage->setText(QStringLiteral("训练进行中：请点「中止训练」结束，或等待完成后再关闭窗口。"));
        return;
    }
    emit closedByUser();
    QDialog::reject();
}

} // namespace aq
