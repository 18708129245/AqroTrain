// ============================================================================
//  NewProjectDialog.cpp
// ============================================================================
#include "ui/NewProjectDialog.h"

#include <QComboBox>
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

namespace aq {

namespace {
const char *kLastProjectParentKey = "AqroTrain/lastProjectParent";
}

NewProjectDialog::NewProjectDialog(QWidget *parent) : QDialog(parent)
{
    setWindowTitle(QStringLiteral("新建工程"));
    setModal(true);

    auto *layout = new QVBoxLayout(this);

    auto *form = new QFormLayout();
    form->setSpacing(6);

    m_edName = new QLineEdit(this);
    m_edName->setPlaceholderText(QStringLiteral("例如：表面缺陷检测"));
    form->addRow(QStringLiteral("工程名称"), m_edName);

    auto *dirRow = new QWidget(this);
    auto *dirLayout = new QHBoxLayout(dirRow);
    dirLayout->setContentsMargins(0, 0, 0, 0);
    m_edDir = new QLineEdit(dirRow);
    auto *btnBrowse = new QPushButton(QStringLiteral("浏览…"), dirRow);
    dirLayout->addWidget(m_edDir, 1);
    dirLayout->addWidget(btnBrowse);
    form->addRow(QStringLiteral("保存位置"), dirRow);

    m_cmbFormat = new QComboBox(this);
    for (ImageFormat f : allImageFormats())
        m_cmbFormat->addItem(imageFormatName(f), imageFormatKey(f));
    m_cmbFormat->setToolTip(QStringLiteral(
        "导入图像时会统一转码为该格式后保存到工程的 images/ 目录。\n"
        "缺陷检测建议用 PNG（无损，不会引入压缩伪影）。"));
    form->addRow(QStringLiteral("工程图片格式"), m_cmbFormat);

    layout->addLayout(form);

    m_lblPreview = new QLabel(this);
    m_lblPreview->setWordWrap(true);
    m_lblPreview->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(m_lblPreview);

    m_lblWarn = new QLabel(this);
    m_lblWarn->setWordWrap(true);
    m_lblWarn->setObjectName(QStringLiteral("panelWarn"));
    layout->addWidget(m_lblWarn);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("创建"));
    buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("取消"));
    layout->addWidget(buttons);

    // 默认位置：上次使用的目录，其次回退到「文档\AqroTrainProjects」
    QSettings settings;
    QString defParent = settings.value(QString::fromLatin1(kLastProjectParentKey)).toString();
    if (defParent.isEmpty() || !QFileInfo(defParent).isDir())
        defParent = QDir(QDir::homePath() + QStringLiteral("/Documents"))
                        .absoluteFilePath(QStringLiteral("AqroTrainProjects"));
    m_edDir->setText(QDir::toNativeSeparators(defParent));

    connect(btnBrowse, &QPushButton::clicked, this, &NewProjectDialog::onBrowse);
    connect(buttons, &QDialogButtonBox::accepted, this, &NewProjectDialog::onAccept);
    connect(buttons, &QDialogButtonBox::rejected, this, &NewProjectDialog::reject);
    connect(m_edName, &QLineEdit::textChanged, this, &NewProjectDialog::validate);
    connect(m_edDir, &QLineEdit::textChanged, this, &NewProjectDialog::validate);

    validate();
}

QString NewProjectDialog::projectName() const
{
    return m_edName->text().trimmed();
}

QString NewProjectDialog::parentDir() const
{
    return QDir::fromNativeSeparators(m_edDir->text().trimmed());
}

QString NewProjectDialog::projectRoot() const
{
    return QDir::cleanPath(parentDir() + QLatin1Char('/') + projectName());
}

ImageFormat NewProjectDialog::imageFormat() const
{
    bool ok = false;
    return imageFormatFromKey(m_cmbFormat->currentData().toString(), &ok);
}

void NewProjectDialog::onBrowse()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择工程的保存位置"), m_edDir->text());
    if (!dir.isEmpty())
        m_edDir->setText(QDir::toNativeSeparators(dir));
}

void NewProjectDialog::validate()
{
    const QString root = projectRoot();
    const bool hasName = !projectName().isEmpty() && !parentDir().isEmpty();

    m_lblPreview->setText(hasName
                              ? QStringLiteral("工程将创建于：\n%1").arg(QDir::toNativeSeparators(root))
                              : QStringLiteral("请填写工程名称与保存位置。"));

    QString warn;
    if (hasName) {
        const QFileInfo fi(root);
        if (fi.exists() && fi.isDir()) {
            const QDir d(root);
            const int n = d.entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden).size();
            if (n > 0)
                warn = QStringLiteral("该目录已存在且非空，请换一个位置或名称。");
        }
        const QFileInfo parentInfo(parentDir());
        if (!warn.isEmpty()) {
            // 已有更重要的警告
        } else if (!parentInfo.exists()) {
            warn = QStringLiteral("保存位置尚不存在，创建工程时会一并创建。");
        }
    }
    m_lblWarn->setText(warn);
}

void NewProjectDialog::onAccept()
{
    if (projectName().isEmpty() || parentDir().isEmpty()) {
        m_lblWarn->setText(QStringLiteral("请填写工程名称与保存位置。"));
        return;
    }

    const QFileInfo fi(projectRoot());
    if (fi.exists() && fi.isDir()) {
        const QDir d(projectRoot());
        const int n = d.entryList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden).size();
        if (n > 0) {
            m_lblWarn->setText(QStringLiteral("该目录已存在且非空，请换一个位置或名称。"));
            return;
        }
    }

    QSettings settings;
    settings.setValue(QString::fromLatin1(kLastProjectParentKey), parentDir());

    accept();
}

} // namespace aq
