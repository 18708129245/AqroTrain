// ============================================================================
//  AddModuleDialog.h —— 添加模块（选择任务类型并命名）
//  ---------------------------------------------------------------------------
//  对标 AIDI「添加模块」：先选模块类型，再命名。
//  类型决定标注方式、类别索引约定与可用工具，因此这里给出每个类型的说明。
// ============================================================================
#pragma once

#include <QDialog>

#include "core/Types.h"

class QLabel;
class QLineEdit;
class QRadioButton;

namespace aq {

class AddModuleDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AddModuleDialog(QWidget *parent = nullptr, TaskType defaultType = TaskType::Classification);

    TaskType moduleType() const;
    QString  moduleName() const;

private slots:
    void onTypeChanged();
    void onAccept();

private:
    QRadioButton *m_rbCls = nullptr;
    QRadioButton *m_rbDet = nullptr;
    QRadioButton *m_rbSeg = nullptr;
    QLineEdit    *m_edName = nullptr;
    QLabel       *m_lblDesc = nullptr;
    bool          m_nameEdited = false;
};

} // namespace aq
