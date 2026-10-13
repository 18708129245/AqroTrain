// ============================================================================
//  ObjectPanel.cpp
// ============================================================================
#include "ui/ObjectPanel.h"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QPushButton>
#include <QVBoxLayout>

namespace aq {

namespace {

QString typeName(AnnoObject::Type t)
{
    switch (t) {
    case AnnoObject::Rect:    return QStringLiteral("矩形");
    case AnnoObject::Polygon: return QStringLiteral("多边形");
    case AnnoObject::Stroke:  return QStringLiteral("笔画");
    }
    return QStringLiteral("对象");
}

} // namespace

ObjectPanel::ObjectPanel(QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    m_lblInfo = new QLabel(QStringLiteral("本图标注对象：0 个"), this);
    layout->addWidget(m_lblInfo);

    m_list = new QListWidget(this);
    m_list->setSelectionMode(QAbstractItemView::SingleSelection);
    m_list->setUniformItemSizes(true);
    m_list->setContextMenuPolicy(Qt::CustomContextMenu);
    layout->addWidget(m_list, 1);

    auto *row = new QHBoxLayout();
    row->setContentsMargins(0, 0, 0, 0);
    row->addWidget(new QLabel(QStringLiteral("类别"), this));
    m_cmbClass = new QComboBox(this);
    m_cmbClass->setToolTip(QStringLiteral("修改选中对象的类别"));
    row->addWidget(m_cmbClass, 1);
    layout->addLayout(row);

    auto *btns = new QHBoxLayout();
    btns->setContentsMargins(0, 0, 0, 0);
    m_btnDelete = new QPushButton(QStringLiteral("删除"), this);
    m_btnCopy = new QPushButton(QStringLiteral("复制"), this);
    m_btnConfirm = new QPushButton(QStringLiteral("确认预标注"), this);
    m_btnConfirm->setToolTip(QStringLiteral("把所有预标注对象标记为已人工复核"));
    btns->addWidget(m_btnDelete);
    btns->addWidget(m_btnCopy);
    layout->addLayout(btns);
    layout->addWidget(m_btnConfirm);

    connect(m_list, &QListWidget::currentRowChanged, this, &ObjectPanel::onRowChanged);
    connect(m_list, &QListWidget::customContextMenuRequested, this, &ObjectPanel::onContextMenu);
    connect(m_cmbClass, SIGNAL(currentIndexChanged(int)), this, SLOT(onClassComboChanged(int)));
    connect(m_btnDelete, &QPushButton::clicked, this, [this]() {
        if (m_selected >= 0)
            emit deleteRequested(m_selected);
    });
    connect(m_btnCopy, &QPushButton::clicked, this, [this]() {
        if (m_selected >= 0)
            emit copyRequested(m_selected);
    });
    connect(m_btnConfirm, &QPushButton::clicked, this, &ObjectPanel::confirmPreRequested);
}

void ObjectPanel::setClassChoices(const QVector<ClassInfo> &classes, bool hasBackground)
{
    m_classes = classes;
    m_hasBackground = hasBackground;

    m_rebuilding = true;
    m_cmbClass->clear();
    for (int i = 0; i < classes.size(); ++i) {
        if (hasBackground && i == 0)
            continue;   // 背景不能作为目标类别
        m_cmbClass->addItem(QStringLiteral("%1. %2").arg(i).arg(classes.at(i).name), i);
    }
    m_rebuilding = false;
}

void ObjectPanel::setObjects(const QVector<AnnoObject> &objects, const QVector<ClassInfo> &classes,
                             bool hasBackground)
{
    m_objects = objects;
    setClassChoices(classes, hasBackground);
    rebuild();
}

void ObjectPanel::setSelected(int index)
{
    if (m_selected == index)
        return;
    m_selected = index;

    m_rebuilding = true;
    if (index >= 0 && index < m_list->count())
        m_list->setCurrentRow(index);
    else
        m_list->clearSelection();
    m_rebuilding = false;

    // 同步类别下拉框
    m_rebuilding = true;
    if (index >= 0 && index < m_objects.size()) {
        const int cid = m_objects.at(index).classId;
        const int pos = m_cmbClass->findData(cid);
        m_cmbClass->setCurrentIndex(pos >= 0 ? pos : -1);
        m_cmbClass->setEnabled(true);
    } else {
        m_cmbClass->setCurrentIndex(-1);
        m_cmbClass->setEnabled(false);
    }
    m_rebuilding = false;

    m_btnDelete->setEnabled(index >= 0);
    m_btnCopy->setEnabled(index >= 0);
}

void ObjectPanel::rebuild()
{
    m_rebuilding = true;
    m_list->clear();

    int preCount = 0;
    for (int i = 0; i < m_objects.size(); ++i) {
        const AnnoObject &o = m_objects.at(i);
        if (o.pre)
            ++preCount;

        QString clsName = QStringLiteral("未知类别");
        if (o.classId >= 0 && o.classId < m_classes.size())
            clsName = m_classes.at(o.classId).name;

        QString detail;
        if (o.type == AnnoObject::Rect)
            detail = QStringLiteral("%1×%2").arg(int(o.rect.width())).arg(int(o.rect.height()));
        else
            detail = QStringLiteral("%1 顶点").arg(o.points.size());

        auto *item = new QListWidgetItem(
            QStringLiteral("#%1  %2  %3  %4%5")
                .arg(i + 1)
                .arg(clsName, typeName(o.type), detail,
                     o.pre ? QStringLiteral("  [预]") : QString()));
        item->setToolTip(QStringLiteral("对象 %1：%2\n几何：%3\n%4")
                             .arg(i + 1)
                             .arg(clsName, detail,
                                  o.derived ? QStringLiteral("由掩码反推得到（可编辑）")
                                            : QStringLiteral("人工标注")));
        m_list->addItem(item);
    }

    if (m_selected >= 0 && m_selected < m_list->count())
        m_list->setCurrentRow(m_selected);
    else if (m_selected >= m_objects.size()) {
        m_selected = -1;
        m_list->clearSelection();
    }

    m_rebuilding = false;

    m_lblInfo->setText(preCount > 0
                           ? QStringLiteral("本图标注对象：%1 个（其中预标注 %2 个，待复核）")
                                 .arg(m_objects.size())
                                 .arg(preCount)
                           : QStringLiteral("本图标注对象：%1 个").arg(m_objects.size()));

    const bool anyPre = preCount > 0;
    m_btnConfirm->setEnabled(anyPre);
    m_btnConfirm->setVisible(anyPre);

    setSelected(m_selected);
}

void ObjectPanel::onRowChanged(int row)
{
    if (m_rebuilding)
        return;
    m_selected = row;
    setSelected(row);
    emit objectSelected(row);
}

void ObjectPanel::onClassComboChanged(int comboIndex)
{
    if (m_rebuilding || m_selected < 0 || m_selected >= m_objects.size())
        return;
    if (comboIndex < 0)
        return;
    const int classId = m_cmbClass->itemData(comboIndex).toInt();
    if (classId == m_objects.at(m_selected).classId)
        return;
    emit classChangeRequested(m_selected, classId);
}

void ObjectPanel::onContextMenu(const QPoint &pos)
{
    QListWidgetItem *item = m_list->itemAt(pos);
    if (item && !item->isSelected()) {
        m_list->clearSelection();
        item->setSelected(true);
        m_list->setCurrentItem(item);
    }
    const int idx = m_list->currentRow();
    if (idx < 0)
        return;

    QMenu menu(this);
    QAction *actCopy = menu.addAction(QStringLiteral("复制该对象"));
    QAction *actDelete = menu.addAction(QStringLiteral("删除该对象"));

    QAction *chosen = menu.exec(m_list->viewport()->mapToGlobal(pos));
    if (!chosen)
        return;
    if (chosen == actCopy)
        emit copyRequested(idx);
    else if (chosen == actDelete)
        emit deleteRequested(idx);
}

} // namespace aq
