// ============================================================================
//  ModuleBar.h —— 模块链工具条（对应 AIDI 界面顶部的模块序列）
//  ---------------------------------------------------------------------------
//  一个工程由多个模块串联而成，模块条把它们画成一排「卡片」：
//      [ 分类-OK/NG ] [ 检测-划痕 ] [ 分割-污点 ]   [+]
//  选中某个模块即切换当前工作模块；「−」删除当前模块（连同其后所有模块）。
// ============================================================================
#pragma once

#include <QWidget>
#include <QVector>

#include "project/Project.h"

class QHBoxLayout;
class QToolButton;

namespace aq {

class ModuleBar : public QWidget
{
    Q_OBJECT

public:
    explicit ModuleBar(QWidget *parent = nullptr);

    void setModules(const QVector<Project::ModuleRef> &modules, int current);
    int  current() const { return m_current; }

signals:
    void moduleSelected(int index);
    void addRequested();
    void removeRequested(int index);
    void renameRequested(int index);

private:
    void rebuild();

    QWidget     *m_host   = nullptr;
    QHBoxLayout *m_layout = nullptr;
    QVector<QToolButton *> m_buttons;
    QVector<Project::ModuleRef> m_modules;
    QToolButton *m_btnAdd    = nullptr;
    QToolButton *m_btnRemove = nullptr;
    int          m_current   = -1;
};

} // namespace aq
