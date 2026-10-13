// ============================================================================
//  ImageListPanel.h —— 图像池面板（对应 AIDI 左侧的图片列表）
//  ---------------------------------------------------------------------------
//  · 支持 CTRL / SHIFT / CTRL+A 多选；
//  · 每行显示：数据划分（训练/测试）、OK 标记、标注状态（空 / 预 / 完）；
//  · 右键菜单提供 AIDI 的常用操作：加入训练集 / 加入测试集 / 移出划分 /
//    设为 OK 图 / 智能标注选中 / 打开所在文件夹 / 删除图片；
//  · 可只看「未标注」的图，便于逐张清标。
// ============================================================================
#pragma once

#include <QVector>
#include <QWidget>

#include "core/Types.h"

class QCheckBox;
class QLabel;
class QListWidget;
class QListWidgetItem;

namespace aq {

class ImageListPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ImageListPanel(QWidget *parent = nullptr);

    /// keys 为图像池中的文件名（顺序即图像池顺序）
    void setImages(const QStringList &keys);
    /// states 与 keys 一一对应
    void setStates(const QVector<SampleState> &states);
    void setCurrentIndex(int index);
    int  currentIndex() const;
    QVector<int> selectedIndexes() const;
    int  count() const { return m_keys.size(); }

    void setFilterUnannotated(bool on);
    bool filterUnannotated() const;

signals:
    void sampleActivated(int index);                          ///< 单击 / 键盘切换
    void splitRequested(const QVector<int> &indexes, int split);   ///< split: 0=未划分 1=训练 2=测试
    void okRequested(const QVector<int> &indexes, bool ok);
    void removeRequested(const QVector<int> &indexes);
    void smartAnnotateRequested(const QVector<int> &indexes);
    void openFolderRequested(int index);
    void importRequested();
    void filterChanged(bool onlyUnannotated);

private slots:
    void onCurrentRowChanged(int row);
    void onContextMenu(const QPoint &pos);
    void onFilterToggled(bool on);

private:
    void rebuild();
    int  poolIndexAtRow(int row) const;
    QVector<int> selectedPoolIndexes() const;

    QListWidget *m_list  = nullptr;
    QCheckBox   *m_chkFilter = nullptr;
    QLabel      *m_lblSummary = nullptr;

    QStringList         m_keys;
    QVector<SampleState> m_states;
    QVector<int>        m_rows;      ///< 行号 -> 图像池下标
    bool                m_filterUnannotated = false;
    int                 m_current = -1;
    bool                m_rebuilding = false;
};

} // namespace aq
