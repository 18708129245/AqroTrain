// ============================================================================
//  PreAnnotateDialog.h —— 智能标注（AI 预标注）
//  ---------------------------------------------------------------------------
//  对标 AIDI 的「智能标注」：用已训练模型对未标注图像自动产生标注，
//  人工只需要复核修正，从「从零画」变成「改一改」。
//
//  流程：主窗口创建本对话框 -> 使用者点「开始」-> 主窗口在 batch 循环中逐图
//  推理并写盘，期间调用 setProgress / appendLog 刷新界面；「停止」按钮
//  通过 stopRequested 让主窗口在下一张之前结束。
//
//  预标注结果写入标注文件并标记 state = pre，在图像池与对象列表里都显示
//  「预」字样，人工确认后才转为 done。
// ============================================================================
#pragma once

#include <QDialog>

class QLabel;
class QProgressBar;
class QPlainTextEdit;
class QPushButton;
class QRadioButton;

namespace aq {

class PreAnnotateDialog : public QDialog
{
    Q_OBJECT

public:
    enum Scope {
        CurrentImage   = 0,   ///< 只处理当前这一张
        SelectedImages = 1,   ///< 处理图像池中选中的若干张
        AllUnannotated = 2    ///< 处理全部未标注图像
    };
    Q_ENUM(Scope)

    PreAnnotateDialog(const QString &moduleName, const QString &modelPath,
                      int currentCount, int selectedCount, int unannotatedCount,
                      bool modelAvailable, QWidget *parent = nullptr);

    Scope scope() const;

    void setProgress(int done, int total, const QString &current);
    /// level 取值与 ITrainer::LogLevel 一致（0 信息 / 1 成功 / 2 警告 / 3 错误）
    void appendLog(const QString &line, int level = 0);
    void setBusy(bool busy);
    void setFinished(int okCount, int failCount);

signals:
    void startRequested(int scope);
    void stopRequested();

private slots:
    void onStart();
    void onStop();

private:
    QRadioButton *m_rbCurrent = nullptr;
    QRadioButton *m_rbSelected = nullptr;
    QRadioButton *m_rbAll = nullptr;
    QProgressBar *m_bar = nullptr;
    QLabel       *m_lblStatus = nullptr;
    QPlainTextEdit *m_log = nullptr;
    QPushButton  *m_btnStart = nullptr;
    QPushButton  *m_btnStop = nullptr;
    bool          m_modelAvailable = false;
    /// 各范围在「非忙碌」状态下是否可选（避免 setBusy 覆盖掉置灰状态）
    bool          m_currentSelectable = true;
    bool          m_selectedSelectable = true;
    bool          m_allSelectable = true;
    int           m_total = 0;
    int           m_done = 0;
};

} // namespace aq
