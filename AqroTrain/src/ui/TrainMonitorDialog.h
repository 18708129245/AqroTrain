// ============================================================================
//  TrainMonitorDialog.h —— 训练监控窗口
//  ---------------------------------------------------------------------------
//  实时显示：进度条、训练曲线、彩色日志；随时可「中止训练」。
//  训练结束后自动在测试集上评价（由主窗口驱动 Evaluator），
//  评价进度也复用本窗口的状态行与日志区。
// ============================================================================
#pragma once

#include <QDialog>
#include <QString>

class QLabel;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;

namespace aq {

class LossChartWidget;

class TrainMonitorDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TrainMonitorDialog(const QString &title, const QString &subtitle,
                               QWidget *parent = nullptr);

    void appendLog(const QString &text, int level);
    void setProgress(int percent);
    void setStage(const QString &text);
    void setXAxisLabel(const QString &label);
    void setMetricName(const QString &name);
    void addPoint(int epoch, int total, double loss, double metric, double valLoss, double valMetric);

    /// 使用者是否点了「中止」
    bool stopRequested() const { return m_stopRequested; }
    void setFinished(bool ok, const QString &summary);
    void setCanStop(bool on);
    /// 关闭按钮是否表现为「关闭窗口」而不是「中止」
    void setCloseableIfRunning(bool on) { m_closeableIfRunning = on; }

signals:
    void stopClicked();
    void closedByUser();

protected:
    void closeEvent(QCloseEvent *e) override;
    void reject() override;

private:
    QProgressBar     *m_bar = nullptr;
    QLabel           *m_stage = nullptr;
    QLabel           *m_summary = nullptr;
    QPlainTextEdit   *m_log = nullptr;
    LossChartWidget  *m_chart = nullptr;
    QPushButton      *m_btnStop = nullptr;
    bool              m_stopRequested = false;
    bool              m_running = true;
    bool              m_closeableIfRunning = false;
};

} // namespace aq
