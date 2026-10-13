// ============================================================================
//  ClassPanel.h —— 标签类别面板
//  ---------------------------------------------------------------------------
//  · 增 / 删 / 重命名类别，自定义颜色；
//  · 检测与分割模块的索引 0 固定为「背景」，不可删除、不可用于标注；
//  · 区分两种信号：
//      classPicked          —— 只有**使用者主动点选**才发出（分类模块据此
//                              「点一下即完成标定」，程序化刷新不得触发）；
//      currentClassChanged  —— 当前类别变化（含程序化设置）。
// ============================================================================
#pragma once

#include <QVector>
#include <QWidget>

#include "core/Types.h"

class QListWidget;
class QListWidgetItem;
class QPushButton;

namespace aq {

class ClassPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ClassPanel(QWidget *parent = nullptr);

    void setClasses(const QVector<ClassInfo> &classes, TaskType task);
    QVector<ClassInfo> classes() const { return m_classes; }

    void setCurrentClass(int index);
    int  currentClass() const { return m_current; }

    /// 检测 / 分割模块中索引 0 为背景
    bool hasBackground() const { return m_task != TaskType::Classification; }

    /// 当前是否停在「不能用于标注」的背景类上
    bool isBackgroundSelected() const;

signals:
    /// 仅当使用者主动点选类别时发出
    void classPicked(int index);
    void currentClassChanged(int index);
    void classesModified(const QVector<ClassInfo> &classes);

private slots:
    void onAdd();
    void onRemove();
    void onRename();
    void onRecolor();
    void onItemActivated(QListWidgetItem *item);
    void onRowChanged(int row);

private:
    void rebuild();
    static QIcon colorIcon(const QColor &c);
    int  selectedRow() const;

    QListWidget *m_list = nullptr;
    QPushButton *m_btnAdd = nullptr;
    QPushButton *m_btnRemove = nullptr;
    QPushButton *m_btnRename = nullptr;
    QPushButton *m_btnColor = nullptr;

    QVector<ClassInfo> m_classes;
    TaskType m_task = TaskType::Classification;
    int      m_current = 0;
    bool     m_rebuilding = false;
};

} // namespace aq
