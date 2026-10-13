// ============================================================================
//  ClassPanel.cpp
// ============================================================================
#include "ui/ClassPanel.h"

#include <QColorDialog>
#include <QHBoxLayout>
#include <QIcon>
#include <QInputDialog>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QPushButton>
#include <QVBoxLayout>

namespace aq {

ClassPanel::ClassPanel(QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    m_list = new QListWidget(this);
    m_list->setSelectionMode(QAbstractItemView::SingleSelection);
    m_list->setUniformItemSizes(true);
    layout->addWidget(m_list, 1);

    auto *btns = new QHBoxLayout();
    btns->setContentsMargins(0, 0, 0, 0);
    btns->setSpacing(2);

    m_btnAdd = new QPushButton(QStringLiteral("添加"), this);
    m_btnAdd->setToolTip(QStringLiteral("添加一个新的目标类别"));
    m_btnRemove = new QPushButton(QStringLiteral("删除"), this);
    m_btnRemove->setToolTip(QStringLiteral("删除选中的类别（背景类不可删除）"));
    m_btnRename = new QPushButton(QStringLiteral("重命名"), this);
    m_btnRename->setToolTip(QStringLiteral("重命名选中的类别"));
    m_btnColor = new QPushButton(QStringLiteral("颜色"), this);
    m_btnColor->setToolTip(QStringLiteral("修改选中类别的显示颜色"));

    btns->addWidget(m_btnAdd);
    btns->addWidget(m_btnRemove);
    btns->addWidget(m_btnRename);
    btns->addWidget(m_btnColor);
    layout->addLayout(btns);

    auto *hint = new QLabel(
        QStringLiteral("检测 / 分割模块的索引 0 是「背景」，"
                       "用它作画不会产生有效标注。"),
        this);
    hint->setWordWrap(true);
    hint->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(hint);

    connect(m_btnAdd, &QPushButton::clicked, this, &ClassPanel::onAdd);
    connect(m_btnRemove, &QPushButton::clicked, this, &ClassPanel::onRemove);
    connect(m_btnRename, &QPushButton::clicked, this, &ClassPanel::onRename);
    connect(m_btnColor, &QPushButton::clicked, this, &ClassPanel::onRecolor);
    connect(m_list, &QListWidget::itemDoubleClicked, this, &ClassPanel::onItemActivated);
    connect(m_list, &QListWidget::currentRowChanged, this, &ClassPanel::onRowChanged);
}

QIcon ClassPanel::colorIcon(const QColor &c)
{
    QPixmap pm(16, 16);
    pm.fill(Qt::transparent);
    QPainter p(&pm);
    p.setRenderHint(QPainter::Antialiasing, true);
    p.setPen(QPen(QColor(0x20, 0x20, 0x20), 1));
    p.setBrush(c);
    p.drawRoundedRect(QRectF(1, 1, 13, 13), 2, 2);
    p.end();
    return QIcon(pm);
}

bool ClassPanel::isBackgroundSelected() const
{
    return hasBackground() && m_current == 0;
}

void ClassPanel::setClasses(const QVector<ClassInfo> &classes, TaskType task)
{
    m_classes = classes;
    m_task = task;
    if (m_current >= m_classes.size())
        m_current = m_classes.isEmpty() ? -1 : 0;
    rebuild();
}

void ClassPanel::setCurrentClass(int index)
{
    if (index < 0 || index >= m_classes.size())
        return;
    if (m_current == index) {
        // 即便值相同，也保证列表选中行与之一致
        if (m_list->currentRow() != index) {
            m_rebuilding = true;
            m_list->setCurrentRow(index);
            m_rebuilding = false;
        }
        return;
    }

    m_current = index;
    m_rebuilding = true;
    m_list->setCurrentRow(index);
    m_rebuilding = false;

    emit currentClassChanged(index);
}

int ClassPanel::selectedRow() const
{
    return m_list->currentRow();
}

void ClassPanel::rebuild()
{
    m_rebuilding = true;
    m_list->clear();

    for (int i = 0; i < m_classes.size(); ++i) {
        const ClassInfo &c = m_classes.at(i);
        const bool isBg = hasBackground() && i == 0;

        auto *item = new QListWidgetItem(colorIcon(c.color),
                                         QStringLiteral("%1. %2%3")
                                             .arg(i)
                                             .arg(c.name, isBg ? QStringLiteral("（背景）") : QString()));
        item->setToolTip(isBg
                             ? QStringLiteral("背景类：不能用于标注目标")
                             : QStringLiteral("类别索引 %1，可用于标注").arg(i));
        m_list->addItem(item);
    }

    if (m_current >= 0 && m_current < m_classes.size())
        m_list->setCurrentRow(m_current);

    m_rebuilding = false;

    m_btnRemove->setEnabled(!m_classes.isEmpty() && !isBackgroundSelected());
    m_btnRename->setEnabled(!m_classes.isEmpty());
    m_btnColor->setEnabled(!m_classes.isEmpty() && !isBackgroundSelected());

    // 刷新期间 currentRowChanged 被抑制，这里显式补发一次，
    // 否则画布的当前类别会停在初始值上（这个坑在 VisionTrain 上踩过）。
    if (m_current >= 0)
        emit currentClassChanged(m_current);
}

void ClassPanel::onRowChanged(int row)
{
    if (m_rebuilding || row < 0 || row >= m_classes.size())
        return;

    m_current = row;
    emit currentClassChanged(row);
    emit classPicked(row);   // 使用者主动点选
}

void ClassPanel::onItemActivated(QListWidgetItem *)
{
    onRename();
}

void ClassPanel::onAdd()
{
    bool ok = false;
    const QString name = QInputDialog::getText(this, QStringLiteral("添加类别"),
                                               QStringLiteral("类别名称："),
                                               QLineEdit::Normal,
                                               QStringLiteral("缺陷%1").arg(m_classes.size()),
                                               &ok);
    if (!ok || name.trimmed().isEmpty())
        return;

    ClassInfo c(name.trimmed(), defaultClassColor(m_classes.size()));
    m_classes.append(c);

    rebuild();
    emit classesModified(m_classes);

    // 新类别直接成为当前类别，便于立即标注
    setCurrentClass(m_classes.size() - 1);
}

void ClassPanel::onRemove()
{
    if (isBackgroundSelected()) {
        QMessageBox::information(this, QStringLiteral("无法删除"),
                                 QStringLiteral("「背景」类别是检测 / 分割模块的固定索引 0，不可删除。"));
        return;
    }
    const int row = selectedRow();
    if (row < 0 || row >= m_classes.size())
        return;

    if (m_classes.size() <= 1) {
        QMessageBox::information(this, QStringLiteral("无法删除"),
                                 QStringLiteral("至少需要保留一个类别。"));
        return;
    }

    const QString name = m_classes.at(row).name;
    if (QMessageBox::question(this, QStringLiteral("删除类别"),
                              QStringLiteral("确定删除类别「%1」吗？\n"
                                             "已用该类标注的对象不会被自动修改，"
                                             "但变成「未知类别」并在界面上以灰色显示。")
                                  .arg(name))
        != QMessageBox::Yes) {
        return;
    }

    m_classes.remove(row);
    if (m_current >= m_classes.size())
        m_current = m_classes.isEmpty() ? -1 : m_classes.size() - 1;

    rebuild();
    emit classesModified(m_classes);
}

void ClassPanel::onRename()
{
    const int row = selectedRow();
    if (row < 0 || row >= m_classes.size())
        return;

    bool ok = false;
    const QString name = QInputDialog::getText(this, QStringLiteral("重命名类别"),
                                               QStringLiteral("类别名称："),
                                               QLineEdit::Normal,
                                               m_classes.at(row).name, &ok);
    if (!ok || name.trimmed().isEmpty())
        return;

    m_classes[row].name = name.trimmed();
    rebuild();
    emit classesModified(m_classes);
}

void ClassPanel::onRecolor()
{
    const int row = selectedRow();
    if (row < 0 || row >= m_classes.size())
        return;
    if (hasBackground() && row == 0) {
        QMessageBox::information(this, QStringLiteral("无法修改"),
                                 QStringLiteral("背景类固定为黑色，用于区分「无目标」。"));
        return;
    }

    const QColor c = QColorDialog::getColor(m_classes.at(row).color, this,
                                            QStringLiteral("选择类别颜色"));
    if (!c.isValid())
        return;

    m_classes[row].color = c;
    rebuild();
    emit classesModified(m_classes);
}

} // namespace aq
