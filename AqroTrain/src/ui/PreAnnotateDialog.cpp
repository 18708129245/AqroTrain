// ============================================================================
//  PreAnnotateDialog.cpp
// ============================================================================
#include "ui/PreAnnotateDialog.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QLabel>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

namespace aq {

PreAnnotateDialog::PreAnnotateDialog(const QString &moduleName, const QString &modelPath,
                                     int currentCount, int selectedCount, int unannotatedCount,
                                     bool modelAvailable, QWidget *parent)
    : QDialog(parent), m_modelAvailable(modelAvailable)
{
    setWindowTitle(QStringLiteral("智能标注（预标注）"));
    resize(640, 460);

    auto *layout = new QVBoxLayout(this);

    auto *info = new QLabel(this);
    info->setWordWrap(true);
    info->setText(QStringLiteral(
                      "<b>模块：</b>%1<br><b>模型：</b>%2<br><br>"
                      "智能标注会用该模型自动生成标注，并标记为「预」。"
                      "请逐张复核修正后再确认，预标注不会自动当作正式标注使用。")
                      .arg(moduleName, QDir::toNativeSeparators(modelPath)));
    layout->addWidget(info);

    m_rbCurrent = new QRadioButton(QStringLiteral("仅当前图像（%1 张）").arg(currentCount), this);
    m_rbSelected = new QRadioButton(QStringLiteral("图像池中选中的图像（%1 张）").arg(selectedCount), this);
    m_rbAll = new QRadioButton(QStringLiteral("全部未标注图像（%1 张）").arg(unannotatedCount), this);
    m_rbAll->setChecked(true);
    m_currentSelectable  = (currentCount > 0);
    m_selectedSelectable = (selectedCount > 0);
    m_allSelectable      = (unannotatedCount > 0);
    if (selectedCount <= 0) {
        m_rbSelected->setEnabled(false);
        m_rbSelected->setToolTip(QStringLiteral("请在左侧图像池中先选中若干张图"));
    }
    if (currentCount <= 0) {
        m_rbCurrent->setEnabled(false);
        m_rbCurrent->setToolTip(QStringLiteral("当前没有打开的图像"));
    }
    if (unannotatedCount <= 0) {
        m_rbAll->setEnabled(false);
        m_rbAll->setToolTip(QStringLiteral("没有未标注的图像"));
    }
    layout->addWidget(m_rbCurrent);
    layout->addWidget(m_rbSelected);
    layout->addWidget(m_rbAll);

    m_bar = new QProgressBar(this);
    m_bar->setRange(0, 100);
    m_bar->setValue(0);
    layout->addWidget(m_bar);

    m_lblStatus = new QLabel(QStringLiteral("就绪。"), this);
    m_lblStatus->setWordWrap(true);
    layout->addWidget(m_lblStatus);

    m_log = new QPlainTextEdit(this);
    m_log->setReadOnly(true);
    m_log->setMaximumBlockCount(2000);
    layout->addWidget(m_log, 1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    buttons->button(QDialogButtonBox::Close)->setText(QStringLiteral("关闭"));
    m_btnStart = buttons->addButton(QStringLiteral("开始预标注"), QDialogButtonBox::AcceptRole);
    m_btnStop = buttons->addButton(QStringLiteral("停止"), QDialogButtonBox::RejectRole);
    m_btnStop->setEnabled(false);
    layout->addWidget(buttons);

    connect(m_btnStart, &QPushButton::clicked, this, &PreAnnotateDialog::onStart);
    connect(m_btnStop, &QPushButton::clicked, this, &PreAnnotateDialog::onStop);
    connect(buttons, &QDialogButtonBox::rejected, this, [this]() {
        if (m_btnStop->isEnabled())
            onStop();
        else
            reject();
    });

    if (!m_modelAvailable) {
        m_btnStart->setEnabled(false);
        m_lblStatus->setText(QStringLiteral("该模块还没有可用模型。请先训练，"
                                            "或用「文件 → 导入模型」把一个已训练好的模型放入 "
                                            "modules/<模块>/models/ 目录。"));
    }
}

PreAnnotateDialog::Scope PreAnnotateDialog::scope() const
{
    if (m_rbCurrent->isChecked())
        return CurrentImage;
    if (m_rbSelected->isChecked())
        return SelectedImages;
    return AllUnannotated;
}

void PreAnnotateDialog::setProgress(int done, int total, const QString &current)
{
    m_done  = done;
    m_total = total;
    m_bar->setRange(0, total > 0 ? total : 1);
    m_bar->setValue(done);
    m_lblStatus->setText(QStringLiteral("正在处理 %1 / %2：%3").arg(done).arg(total).arg(current));
}

void PreAnnotateDialog::appendLog(const QString &line, int level)
{
    QString color;
    switch (level) {
    case 1: color = QStringLiteral("#3CB44B"); break;   // 成功
    case 2: color = QStringLiteral("#FFB020"); break;   // 警告
    case 3: color = QStringLiteral("#FF6060"); break;   // 错误
    default: color = QStringLiteral("#C8CED8"); break;  // 信息
    }
    m_log->appendHtml(QStringLiteral("<span style=\"color:%1\">%2</span>")
                          .arg(color, line.toHtmlEscaped()));
}

void PreAnnotateDialog::setBusy(bool busy)
{
    m_btnStart->setEnabled(!busy && m_modelAvailable);
    m_btnStop->setEnabled(busy);
    // 忙碌时禁止切换范围，但不要破坏「本来就没有可处理项」时的置灰状态
    m_rbCurrent->setEnabled(!busy && m_currentSelectable);
    m_rbSelected->setEnabled(!busy && m_selectedSelectable);
    m_rbAll->setEnabled(!busy && m_allSelectable);
}

void PreAnnotateDialog::setFinished(int okCount, int failCount)
{
    m_lblStatus->setText(QStringLiteral("预标注完成：成功 %1 张，失败 %2 张。%3")
                             .arg(okCount)
                             .arg(failCount)
                             .arg(failCount > 0
                                      ? QStringLiteral("失败原因见下方日志。")
                                      : QStringLiteral("请在图像池中逐张复核后确认。")));
    setBusy(false);
}

void PreAnnotateDialog::onStart()
{
    setBusy(true);
    m_log->clear();
    emit startRequested(int(scope()));
}

void PreAnnotateDialog::onStop()
{
    m_btnStop->setEnabled(false);
    m_lblStatus->setText(QStringLiteral("正在停止…（当前这张处理完后结束）"));
    emit stopRequested();
}

} // namespace aq
