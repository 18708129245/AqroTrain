// ============================================================================
//  ImageListPanel.cpp
// ============================================================================
#include "ui/ImageListPanel.h"

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMenu>
#include <QVBoxLayout>

namespace aq {

ImageListPanel::ImageListPanel(QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(4);

    auto *top = new QHBoxLayout();
    top->setContentsMargins(0, 0, 0, 0);
    m_lblSummary = new QLabel(QStringLiteral("图像池：0 张"), this);
    top->addWidget(m_lblSummary, 1);
    layout->addLayout(top);

    auto *filterRow = new QHBoxLayout();
    filterRow->setContentsMargins(0, 0, 0, 0);
    m_chkFilter = new QCheckBox(QStringLiteral("只看未标注"), this);
    m_chkFilter->setToolTip(QStringLiteral("只列出尚未标注的图像，便于逐张清标"));
    filterRow->addWidget(m_chkFilter);
    filterRow->addStretch(1);
    layout->addLayout(filterRow);

    m_list = new QListWidget(this);
    m_list->setSelectionMode(QAbstractItemView::ExtendedSelection);   // CTRL / SHIFT 多选
    m_list->setUniformItemSizes(true);
    m_list->setAlternatingRowColors(true);
    m_list->setContextMenuPolicy(Qt::CustomContextMenu);
    layout->addWidget(m_list, 1);

    connect(m_list, &QListWidget::currentRowChanged, this, &ImageListPanel::onCurrentRowChanged);
    connect(m_list, &QListWidget::customContextMenuRequested, this, &ImageListPanel::onContextMenu);
    connect(m_chkFilter, &QCheckBox::toggled, this, &ImageListPanel::onFilterToggled);
}

void ImageListPanel::setImages(const QStringList &keys)
{
    m_keys = keys;
    if (m_states.size() != m_keys.size())
        m_states.resize(m_keys.size());
    rebuild();
}

void ImageListPanel::setStates(const QVector<SampleState> &states)
{
    m_states = states;
    if (m_states.size() != m_keys.size())
        m_states.resize(m_keys.size());
    rebuild();
}

void ImageListPanel::setFilterUnannotated(bool on)
{
    if (m_chkFilter->isChecked() != on)
        m_chkFilter->setChecked(on);
    else {
        m_filterUnannotated = on;
        rebuild();
    }
}

bool ImageListPanel::filterUnannotated() const
{
    return m_filterUnannotated;
}

int ImageListPanel::currentIndex() const
{
    return m_current;
}

int ImageListPanel::poolIndexAtRow(int row) const
{
    if (row < 0 || row >= m_rows.size())
        return -1;
    return m_rows.at(row);
}

QVector<int> ImageListPanel::selectedPoolIndexes() const
{
    QVector<int> out;
    const QList<QListWidgetItem *> items = m_list->selectedItems();
    for (QListWidgetItem *it : items) {
        const int idx = poolIndexAtRow(m_list->row(it));
        if (idx >= 0)
            out.append(idx);
    }
    if (out.isEmpty() && m_current >= 0)
        out.append(m_current);   // 没有选中项时退化为「当前图」
    return out;
}

QVector<int> ImageListPanel::selectedIndexes() const
{
    return selectedPoolIndexes();
}

void ImageListPanel::rebuild()
{
    m_rebuilding = true;

    const QVector<int> keep = selectedPoolIndexes();

    m_list->clear();
    m_rows.clear();

    int annotated = 0;

    for (int i = 0; i < m_keys.size(); ++i) {
        const SampleState st = (i < m_states.size()) ? m_states.at(i) : SampleState();
        if (st.state != AnnoState::None)
            ++annotated;

        if (m_filterUnannotated && st.state != AnnoState::None)
            continue;

        QString splitTag;
        if (st.split == Split::Train)
            splitTag = QStringLiteral("[训练]");
        else if (st.split == Split::Test)
            splitTag = QStringLiteral("[测试]");
        else
            splitTag = QStringLiteral("[    ]");

        QString stateTag;
        switch (st.state) {
        case AnnoState::None: stateTag = QStringLiteral("·");  break;
        case AnnoState::Pre:  stateTag = QStringLiteral("预"); break;
        case AnnoState::Done: stateTag = QStringLiteral("●");  break;
        }

        const QString okTag = st.ok ? QStringLiteral(" OK") : QString();

        auto *item = new QListWidgetItem(
            QStringLiteral("%1 %2 %3%4").arg(splitTag, stateTag, m_keys.at(i), okTag));
        item->setToolTip(QStringLiteral("%1\n数据划分：%2\n标注状态：%3\nOK 图：%4")
                             .arg(m_keys.at(i), splitName(st.split), annoStateName(st.state),
                                  st.ok ? QStringLiteral("是") : QStringLiteral("否")));
        m_list->addItem(item);
        m_rows.append(i);
    }

    // 恢复选中
    for (int row = 0; row < m_rows.size(); ++row) {
        if (keep.contains(m_rows.at(row))) {
            m_list->item(row)->setSelected(true);
        }
    }
    if (m_current >= 0) {
        const int row = m_rows.indexOf(m_current);
        if (row >= 0)
            m_list->setCurrentRow(row);
    }

    m_lblSummary->setText(QStringLiteral("图像池：%1 张（已标注 %2）").arg(m_keys.size()).arg(annotated));

    m_rebuilding = false;
}

void ImageListPanel::setCurrentIndex(int index)
{
    m_current = index;
    const int row = m_rows.indexOf(index);
    if (row >= 0) {
        m_rebuilding = true;
        m_list->setCurrentRow(row);
        m_list->scrollToItem(m_list->item(row));
        m_rebuilding = false;
    }
}

void ImageListPanel::onCurrentRowChanged(int row)
{
    if (m_rebuilding)
        return;
    const int idx = poolIndexAtRow(row);
    if (idx < 0)
        return;
    m_current = idx;
    emit sampleActivated(idx);
}

void ImageListPanel::onFilterToggled(bool on)
{
    m_filterUnannotated = on;
    rebuild();
    emit filterChanged(on);
}

void ImageListPanel::onContextMenu(const QPoint &pos)
{
    QListWidgetItem *item = m_list->itemAt(pos);
    if (item && !item->isSelected()) {
        m_list->clearSelection();
        item->setSelected(true);
        m_list->setCurrentItem(item);
    }

    const QVector<int> sel = selectedPoolIndexes();

    QMenu menu(this);
    QAction *actImport = menu.addAction(QStringLiteral("导入图像…"));

    QAction *actTrain = nullptr;
    QAction *actTest = nullptr;
    QAction *actClearSplit = nullptr;
    QAction *actOk = nullptr;
    QAction *actNotOk = nullptr;
    QAction *actSmart = nullptr;
    QAction *actFolder = nullptr;
    QAction *actRemove = nullptr;

    if (!sel.isEmpty()) {
        menu.addSeparator();
        actTrain = menu.addAction(QStringLiteral("加入训练集"));
        actTest = menu.addAction(QStringLiteral("加入测试集"));
        actClearSplit = menu.addAction(QStringLiteral("移出划分（变为未标注）"));
        menu.addSeparator();
        actOk = menu.addAction(QStringLiteral("设为 OK 图（无缺陷正例）"));
        actNotOk = menu.addAction(QStringLiteral("取消 OK 图标记"));
        menu.addSeparator();
        actSmart = menu.addAction(QStringLiteral("智能标注选中图像…"));
        menu.addSeparator();
        actFolder = menu.addAction(QStringLiteral("打开所在文件夹"));
        actRemove = menu.addAction(QStringLiteral("从工程中删除图片"));
    }

    QAction *chosen = menu.exec(m_list->viewport()->mapToGlobal(pos));
    if (!chosen)
        return;

    if (chosen == actImport) {
        emit importRequested();
    } else if (chosen == actTrain) {
        emit splitRequested(sel, int(Split::Train));
    } else if (chosen == actTest) {
        emit splitRequested(sel, int(Split::Test));
    } else if (chosen == actClearSplit) {
        emit splitRequested(sel, int(Split::None));
    } else if (chosen == actOk) {
        emit okRequested(sel, true);
    } else if (chosen == actNotOk) {
        emit okRequested(sel, false);
    } else if (chosen == actSmart) {
        emit smartAnnotateRequested(sel);
    } else if (chosen == actFolder) {
        emit openFolderRequested(sel.first());
    } else if (chosen == actRemove) {
        emit removeRequested(sel);
    }
}

} // namespace aq
