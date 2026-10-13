// ============================================================================
//  ObjectPanel.h —— 标注对象列表
//  ---------------------------------------------------------------------------
//  把当前图上的每个标注对象列成一行，支持：
//    · 点选行 → 画布上高亮该对象（双向联动）；
//    · 改类别、删除、复制；
//    · 预标注对象显示「预」标记，提示尚未人工复核。
//
//  分割模块的掩码由对象列表光栅化而成，所以这个列表就是分割标注的「真源视图」。
// ============================================================================
#pragma once

#include <QVector>
#include <QWidget>

#include "annotate/AnnoObject.h"
#include "core/Types.h"

class QComboBox;
class QLabel;
class QListWidget;
class QPushButton;

namespace aq {

class ObjectPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ObjectPanel(QWidget *parent = nullptr);

    void setObjects(const QVector<AnnoObject> &objects, const QVector<ClassInfo> &classes,
                    bool hasBackground);
    void setSelected(int index);

    /// 刷新类别下拉框的候选项（类别被增删改后调用）
    void setClassChoices(const QVector<ClassInfo> &classes, bool hasBackground);

signals:
    void objectSelected(int index);
    void classChangeRequested(int index, int classId);
    void deleteRequested(int index);
    void copyRequested(int index);
    void confirmPreRequested();

private slots:
    void onRowChanged(int row);
    void onContextMenu(const QPoint &pos);
    void onClassComboChanged(int comboIndex);

private:
    void rebuild();

    QListWidget *m_list = nullptr;
    QComboBox   *m_cmbClass = nullptr;
    QPushButton *m_btnDelete = nullptr;
    QPushButton *m_btnCopy = nullptr;
    QPushButton *m_btnConfirm = nullptr;
    QLabel      *m_lblInfo = nullptr;

    QVector<AnnoObject> m_objects;
    QVector<ClassInfo>  m_classes;
    bool m_hasBackground = false;
    int  m_selected = -1;
    bool m_rebuilding = false;
};

} // namespace aq
