// ============================================================================
//  TestModelDialog.h —— 模型测试（载入模型 + 任意图像，运行推理并可视化）
//  ---------------------------------------------------------------------------
//  · 模型来源可以是当前模块训练出的模型，也可以是从别处导入/导出的模型文件；
//  · 任务类型与类别从模型元数据（model.<ext>.json）读取，因此不需要人工指定；
//  · 结果可视化：分类显示 Top 得分，检测画框，分割叠加掩码占比。
// ============================================================================
#pragma once

#include <QDialog>
#include <QString>

#include "core/Types.h"

class QLabel;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;

namespace aq {

class TestModelDialog : public QDialog
{
    Q_OBJECT

public:
    explicit TestModelDialog(QWidget *parent = nullptr);

    void setDefaultModel(const QString &path);
    void setDefaultImage(const QString &path);

private slots:
    void onBrowseModel();
    void onBrowseImage();
    void onRun();

private:
    void showResult(const cv::Mat &vis, const QString &details);

    QLineEdit      *m_edModel = nullptr;
    QLineEdit      *m_edImage = nullptr;
    QLabel         *m_lblInfo = nullptr;
    QLabel         *m_lblImage = nullptr;
    QPlainTextEdit *m_txtDetail = nullptr;
    QPushButton    *m_btnRun = nullptr;
};

} // namespace aq
