// ============================================================================
//  ExportModelDialog.cpp
// ============================================================================
#include "ui/ExportModelDialog.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSettings>
#include <QVBoxLayout>

#include "train/ITrainer.h"

namespace aq {

namespace {
const char *kLastExportDirKey = "AqroTrain/lastExportDir";
}

ExportModelDialog::ExportModelDialog(const Module &module, const Metrics &metrics, QWidget *parent)
    : QDialog(parent), m_module(module), m_metrics(metrics)
{
    setWindowTitle(QStringLiteral("导出模型"));
    setModal(true);
    resize(560, 300);

    auto *layout = new QVBoxLayout(this);

    auto *info = new QLabel(this);
    info->setWordWrap(true);
    const bool hasModel = QFileInfo::exists(m_module.modelPath());
    QString text = QStringLiteral(
                       "<b>模块：</b>%1（%2）<br>"
                       "<b>模型文件：</b>%3<br>"
                       "<b>类别数：</b>%4（目标类别 %5）<br>"
                       "<b>模型状态：</b>%6")
                       .arg(m_module.name(), taskTypeName(m_module.type()),
                            QDir::toNativeSeparators(m_module.modelPath()))
                       .arg(m_module.classCount())
                       .arg(m_module.targetClassCount())
                       .arg(hasModel ? QStringLiteral("已训练") : QStringLiteral("尚未训练"));
    if (m_metrics.valid) {
        text += QStringLiteral("<br><b>指标摘要：</b>%1").arg(m_metrics.summary());
    }
    info->setText(text);
    layout->addWidget(info);

    auto *row = new QWidget(this);
    auto *rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    m_edDir = new QLineEdit(row);
    auto *btnBrowse = new QPushButton(QStringLiteral("浏览…"), row);
    rowLayout->addWidget(m_edDir, 1);
    rowLayout->addWidget(btnBrowse);
    auto *form = new QFormLayout();
    form->addRow(QStringLiteral("导出到目录"), row);
    layout->addLayout(form);

    QSettings settings;
    m_edDir->setText(QDir::toNativeSeparators(
        settings.value(QString::fromLatin1(kLastExportDirKey),
                       QDir::homePath() + QStringLiteral("/Documents/AqroTrainModels")).toString()));

    m_lblResult = new QLabel(this);
    m_lblResult->setWordWrap(true);
    m_lblResult->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(m_lblResult);
    layout->addStretch(1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    buttons->button(QDialogButtonBox::Close)->setText(QStringLiteral("关闭"));
    m_btnExport = buttons->addButton(QStringLiteral("导出"), QDialogButtonBox::AcceptRole);
    m_btnExport->setEnabled(hasModel);
    layout->addWidget(buttons);

    connect(btnBrowse, &QPushButton::clicked, this, &ExportModelDialog::onBrowse);
    connect(m_btnExport, &QPushButton::clicked, this, &ExportModelDialog::onExport);
    connect(m_edDir, &QLineEdit::textChanged, this, &ExportModelDialog::validate);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

    if (!hasModel)
        m_lblResult->setText(QStringLiteral("该模块还没有训练出模型，请先执行「训练」。"));
    else
        validate();
}

QString ExportModelDialog::destDir() const
{
    return QDir::fromNativeSeparators(m_edDir->text().trimmed());
}

void ExportModelDialog::onBrowse()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择导出目录"), m_edDir->text());
    if (!dir.isEmpty())
        m_edDir->setText(QDir::toNativeSeparators(dir));
}

void ExportModelDialog::validate()
{
    if (m_btnExport)
        m_btnExport->setEnabled(QFileInfo::exists(m_module.modelPath()) && !destDir().isEmpty());
}

void ExportModelDialog::onExport()
{
    const QString dir = destDir();
    if (dir.isEmpty()) {
        m_lblResult->setText(QStringLiteral("请先选择导出目录。"));
        return;
    }

    QString mainPath;
    QString err;
    if (!modelio::exportModel(m_module.modelPath(), dir, &mainPath, &err)) {
        m_lblResult->setText(QStringLiteral("<span style='color:#ff6060'>导出失败：%1</span>").arg(err));
        return;
    }

    QSettings settings;
    settings.setValue(QString::fromLatin1(kLastExportDirKey), dir);

    m_lblResult->setText(QStringLiteral("导出成功：\n%1\n（同目录下还有同名 .json 元数据文件）")
                             .arg(QDir::toNativeSeparators(mainPath)));
}

} // namespace aq
