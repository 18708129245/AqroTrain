// ============================================================================
//  ExportModelDialog.h —— 导出模型（对标 AIDI「文件 → 导出模型」）
//  ---------------------------------------------------------------------------
//  把模块的模型主文件与元数据一起导出到指定目录，便于部署到其它设备。
//  导出后可用「测试 → 模型测试」重新载入该导出件推理，验证导出可用。
// ============================================================================
#pragma once

#include <QDialog>

#include "core/Types.h"
#include "project/Module.h"

class QLabel;
class QLineEdit;
class QPushButton;

namespace aq {

class ExportModelDialog : public QDialog
{
    Q_OBJECT

public:
    ExportModelDialog(const Module &module, const Metrics &metrics, QWidget *parent = nullptr);

    QString destDir() const;

private slots:
    void onBrowse();
    void onExport();
    void validate();

private:
    Module  m_module;
    Metrics m_metrics;

    QLineEdit   *m_edDir = nullptr;
    QLabel      *m_lblInfo = nullptr;
    QLabel      *m_lblResult = nullptr;
    QPushButton *m_btnExport = nullptr;
};

} // namespace aq
