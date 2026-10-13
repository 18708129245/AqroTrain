// ============================================================================
//  AddModuleDialog.cpp
// ============================================================================
#include "ui/AddModuleDialog.h"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QVBoxLayout>

namespace aq {

namespace {

QString description(TaskType t)
{
    switch (t) {
    case TaskType::Classification:
        return QStringLiteral(
            "<b>图像分类</b>：整张图对应一个类别，点选类别即完成标定，不需要涂抹。\n"
            "类别索引 0..N-1，没有背景类。");
    case TaskType::Detection:
        return QStringLiteral(
            "<b>目标检测</b>：拖拽画矩形框（或点一下放置固定尺寸框）标出目标位置。\n"
            "类别索引 0 固定为「背景」，目标类别从 1 开始。");
    case TaskType::Segmentation:
        return QStringLiteral(
            "<b>图像分割</b>：画笔涂抹、多边形围区域、矩形填充，逐像素标出缺陷形状。\n"
            "类别索引 0 固定为「背景」，目标类别从 1 开始。掩码由标注对象光栅化得到。");
    }
    return QString();
}

} // namespace

AddModuleDialog::AddModuleDialog(QWidget *parent, TaskType defaultType) : QDialog(parent)
{
    setWindowTitle(QStringLiteral("添加模块"));
    setModal(true);
    resize(520, 320);

    auto *layout = new QVBoxLayout(this);

    auto *form = new QFormLayout();
    form->setSpacing(6);

    auto *typeBox = new QWidget(this);
    auto *typeLayout = new QVBoxLayout(typeBox);
    typeLayout->setContentsMargins(0, 0, 0, 0);
    m_rbCls = new QRadioButton(QStringLiteral("图像分类"), typeBox);
    m_rbDet = new QRadioButton(QStringLiteral("目标检测"), typeBox);
    m_rbSeg = new QRadioButton(QStringLiteral("图像分割"), typeBox);
    typeLayout->addWidget(m_rbCls);
    typeLayout->addWidget(m_rbDet);
    typeLayout->addWidget(m_rbSeg);
    form->addRow(QStringLiteral("模块类型"), typeBox);

    m_edName = new QLineEdit(this);
    form->addRow(QStringLiteral("模块名称"), m_edName);

    layout->addLayout(form);

    m_lblDesc = new QLabel(this);
    m_lblDesc->setWordWrap(true);
    m_lblDesc->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(m_lblDesc);
    layout->addStretch(1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
    buttons->button(QDialogButtonBox::Ok)->setText(QStringLiteral("添加"));
    buttons->button(QDialogButtonBox::Cancel)->setText(QStringLiteral("取消"));
    layout->addWidget(buttons);

    switch (defaultType) {
    case TaskType::Detection:    m_rbDet->setChecked(true); break;
    case TaskType::Segmentation: m_rbSeg->setChecked(true); break;
    default:                     m_rbCls->setChecked(true); break;
    }

    connect(m_rbCls, &QRadioButton::toggled, this, &AddModuleDialog::onTypeChanged);
    connect(m_rbDet, &QRadioButton::toggled, this, &AddModuleDialog::onTypeChanged);
    connect(m_rbSeg, &QRadioButton::toggled, this, &AddModuleDialog::onTypeChanged);
    connect(m_edName, &QLineEdit::textEdited, this, [this]() { m_nameEdited = true; });
    connect(buttons, &QDialogButtonBox::accepted, this, &AddModuleDialog::onAccept);
    connect(buttons, &QDialogButtonBox::rejected, this, &AddModuleDialog::reject);

    onTypeChanged();
}

TaskType AddModuleDialog::moduleType() const
{
    if (m_rbDet->isChecked())
        return TaskType::Detection;
    if (m_rbSeg->isChecked())
        return TaskType::Segmentation;
    return TaskType::Classification;
}

QString AddModuleDialog::moduleName() const
{
    return m_edName->text().trimmed();
}

void AddModuleDialog::onTypeChanged()
{
    const TaskType t = moduleType();
    m_lblDesc->setText(description(t));

    // 名称未手工改过时，跟随类型自动填一个建议名
    if (!m_nameEdited)
        m_edName->setText(taskTypeName(t));
}

void AddModuleDialog::onAccept()
{
    if (moduleName().isEmpty()) {
        m_edName->setText(taskTypeName(moduleType()));
        m_edName->setFocus();
        return;
    }
    accept();
}

} // namespace aq
