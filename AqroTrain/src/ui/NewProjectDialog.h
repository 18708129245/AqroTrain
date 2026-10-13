// ============================================================================
//  NewProjectDialog.h —— 新建工程（名称 / 保存位置 / 工程图片格式）
//  ---------------------------------------------------------------------------
//  对标 AIDI「文件 → 新建工程」：选路径、填名称、选工程图片格式。
//  导入图像时会按该格式统一转码后落盘到 images/。
// ============================================================================
#pragma once

#include <QDialog>

#include "core/Types.h"

class QComboBox;
class QLabel;
class QLineEdit;

namespace aq {

class NewProjectDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NewProjectDialog(QWidget *parent = nullptr);

    QString     projectName() const;
    QString     parentDir() const;
    /// 工程根目录 = parentDir + "/" + projectName
    QString     projectRoot() const;

    ImageFormat imageFormat() const;

private slots:
    void onBrowse();
    void onAccept();
    void validate();

private:
    QLineEdit *m_edName = nullptr;
    QLineEdit *m_edDir = nullptr;
    QComboBox *m_cmbFormat = nullptr;
    QLabel    *m_lblPreview = nullptr;
    QLabel    *m_lblWarn = nullptr;
};

} // namespace aq
