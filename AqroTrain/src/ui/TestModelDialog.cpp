// ============================================================================
//  TestModelDialog.cpp
// ============================================================================
#include "ui/TestModelDialog.h"

#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPixmap>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QSplitter>
#include <QVBoxLayout>

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include "core/ImageIO.h"
#include "train/ITrainer.h"
#include "train/TrainerFactory.h"

namespace aq {

TestModelDialog::TestModelDialog(QWidget *parent) : QDialog(parent)
{
    setWindowTitle(QStringLiteral("模型测试"));
    resize(1040, 720);

    auto *layout = new QVBoxLayout(this);

    auto *form = new QFormLayout();

    auto *modelRow = new QWidget(this);
    auto *modelLayout = new QHBoxLayout(modelRow);
    modelLayout->setContentsMargins(0, 0, 0, 0);
    m_edModel = new QLineEdit(modelRow);
    auto *btnModel = new QPushButton(QStringLiteral("浏览…"), modelRow);
    modelLayout->addWidget(m_edModel, 1);
    modelLayout->addWidget(btnModel);
    form->addRow(QStringLiteral("模型文件"), modelRow);

    auto *imageRow = new QWidget(this);
    auto *imageLayout = new QHBoxLayout(imageRow);
    imageLayout->setContentsMargins(0, 0, 0, 0);
    m_edImage = new QLineEdit(imageRow);
    auto *btnImage = new QPushButton(QStringLiteral("浏览…"), imageRow);
    imageLayout->addWidget(m_edImage, 1);
    imageLayout->addWidget(btnImage);
    form->addRow(QStringLiteral("测试图像"), imageRow);

    layout->addLayout(form);

    m_lblInfo = new QLabel(this);
    m_lblInfo->setWordWrap(true);
    m_lblInfo->setObjectName(QStringLiteral("panelHint"));
    layout->addWidget(m_lblInfo);

    auto *splitter = new QSplitter(Qt::Horizontal, this);

    auto *imgBox = new QGroupBox(QStringLiteral("推理可视化"), splitter);
    auto *imgLayout = new QVBoxLayout(imgBox);
    auto *scroll = new QScrollArea(imgBox);
    scroll->setWidgetResizable(true);
    m_lblImage = new QLabel(imgBox);
    m_lblImage->setAlignment(Qt::AlignCenter);
    m_lblImage->setText(QStringLiteral("尚未运行推理"));
    scroll->setWidget(m_lblImage);
    imgLayout->addWidget(scroll);

    auto *detailBox = new QGroupBox(QStringLiteral("结果明细"), splitter);
    auto *detailLayout = new QVBoxLayout(detailBox);
    m_txtDetail = new QPlainTextEdit(detailBox);
    m_txtDetail->setReadOnly(true);
    detailLayout->addWidget(m_txtDetail);

    splitter->addWidget(imgBox);
    splitter->addWidget(detailBox);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 2);
    layout->addWidget(splitter, 1);

    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
    buttons->button(QDialogButtonBox::Close)->setText(QStringLiteral("关闭"));
    m_btnRun = buttons->addButton(QStringLiteral("开始推理"), QDialogButtonBox::AcceptRole);
    layout->addWidget(buttons);

    connect(btnModel, &QPushButton::clicked, this, &TestModelDialog::onBrowseModel);
    connect(btnImage, &QPushButton::clicked, this, &TestModelDialog::onBrowseImage);
    connect(m_btnRun, &QPushButton::clicked, this, &TestModelDialog::onRun);
    connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
}

void TestModelDialog::setDefaultModel(const QString &path)
{
    if (!path.isEmpty())
        m_edModel->setText(QDir::toNativeSeparators(path));
}

void TestModelDialog::setDefaultImage(const QString &path)
{
    if (!path.isEmpty())
        m_edImage->setText(QDir::toNativeSeparators(path));
}

void TestModelDialog::onBrowseModel()
{
    const QString f = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择模型文件"), m_edModel->text(),
        QStringLiteral("AqroTrain 模型 (*.clsmodel *.detmodel *.segmodel);;所有文件 (*.*)"));
    if (!f.isEmpty())
        m_edModel->setText(QDir::toNativeSeparators(f));
}

void TestModelDialog::onBrowseImage()
{
    const QString f = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择测试图像"), m_edImage->text(),
        QStringLiteral("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)"));
    if (!f.isEmpty())
        m_edImage->setText(QDir::toNativeSeparators(f));
}

void TestModelDialog::onRun()
{
    const QString modelPath = QDir::fromNativeSeparators(m_edModel->text().trimmed());
    const QString imagePath = QDir::fromNativeSeparators(m_edImage->text().trimmed());

    if (!QFileInfo::exists(modelPath)) {
        m_lblInfo->setText(QStringLiteral("模型文件不存在：%1").arg(QDir::toNativeSeparators(modelPath)));
        return;
    }
    if (!QFileInfo::exists(imagePath)) {
        m_lblInfo->setText(QStringLiteral("测试图像不存在：%1").arg(QDir::toNativeSeparators(imagePath)));
        return;
    }

    modelio::ModelMeta meta;
    QString err;
    if (!modelio::readModelMeta(modelPath, &meta, &err)) {
        m_lblInfo->setText(QStringLiteral("无法读取模型元数据：%1\n"
                                         "（元数据文件为模型同名的 .json，用来识别任务类型与类别）")
                               .arg(err));
        return;
    }

    // 元数据里记录了后端；若该后端当前不可用则退回 OpenCV 轻量后端
    TrainerFactory::Backend backend = TrainerFactory::fromKey(meta.backend);
    auto *trainer = TrainerFactory::create(this, backend);
    if (!trainer) {
        trainer = TrainerFactory::create(this, TrainerFactory::OpenCV);
        if (!trainer) {
            m_lblInfo->setText(QStringLiteral("没有可用的推理后端。"));
            return;
        }
        m_lblInfo->setText(QStringLiteral("模型记录的后端不可用，已退回 %1。").arg(trainer->backendName()));
    } else {
        m_lblInfo->setText(QStringLiteral("模型：%1（%2，后端 %3）")
                               .arg(QFileInfo(modelPath).fileName(), taskTypeName(meta.task),
                                    trainer->backendName()));
    }

    if (!trainer->loadModel(modelPath, &err)) {
        m_lblInfo->setText(QStringLiteral("载入模型失败：%1").arg(err));
        trainer->deleteLater();
        return;
    }

    const cv::Mat bgr = imageio::imreadUnicode(imagePath, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        m_lblInfo->setText(QStringLiteral("无法读取测试图像：%1").arg(QDir::toNativeSeparators(imagePath)));
        trainer->deleteLater();
        return;
    }

    PredictionResult pr;
    if (!trainer->predict(bgr, &pr, &err)) {
        m_lblInfo->setText(QStringLiteral("推理失败：%1").arg(err));
        trainer->deleteLater();
        return;
    }

    const TaskType task = trainer->modelTask();
    const QVector<ClassInfo> classes = trainer->modelClasses();

    cv::Mat vis = bgr.clone();
    QString details;

    auto className = [&](int id) {
        if (id >= 0 && id < classes.size())
            return classes.at(id).name;
        return QStringLiteral("类别%1").arg(id);
    };
    auto classColor = [&](int id) {
        const QColor c = (id >= 0 && id < classes.size()) ? classes.at(id).color : defaultClassColor(id);
        return cv::Scalar(c.blue(), c.green(), c.red());
    };

    if (task == TaskType::Classification) {
        details += QStringLiteral("任务：图像分类\n预测类别：%1（索引 %2）\n置信度：%3%\n\n--- 各类得分 ---\n")
                       .arg(className(pr.classId))
                       .arg(pr.classId)
                       .arg(pr.score * 100.0, 0, 'f', 2);
        for (int i = 0; i < int(pr.scores.size()); ++i) {
            details += QStringLiteral("%1. %2\n").arg(i).arg(className(i));
            details += QStringLiteral("    得分 %1%\n").arg(pr.scores[i] * 100.0, 0, 'f', 2);
        }
        cv::putText(vis,
                    QStringLiteral("%1 %2%").arg(className(pr.classId))
                        .arg(pr.score * 100.0, 0, 'f', 1)
                        .toStdString(),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, classColor(pr.classId), 2);
    } else if (task == TaskType::Detection) {
        details += QStringLiteral("任务：目标检测\n检出目标：%1 个\n\n").arg(pr.boxes.size());
        for (size_t i = 0; i < pr.boxes.size(); ++i) {
            const DetectionBox &b = pr.boxes[i];
            cv::rectangle(vis, cv::Rect(int(b.rect.x()), int(b.rect.y()),
                                        int(b.rect.width()), int(b.rect.height())),
                          classColor(b.classId), 2);
            const QString label = QStringLiteral("%1 %2%")
                                      .arg(className(b.classId))
                                      .arg(b.score * 100.0, 0, 'f', 1);
            cv::putText(vis, label.toStdString(),
                        cv::Point(int(b.rect.x()), std::max(12, int(b.rect.y()) - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, classColor(b.classId), 1);
            details += QStringLiteral("#%1  %2  置信度 %3%  位置 (%4, %5, %6×%7)\n")
                           .arg(i + 1)
                           .arg(className(b.classId))
                           .arg(b.score * 100.0, 0, 'f', 1)
                           .arg(int(b.rect.x())).arg(int(b.rect.y()))
                           .arg(int(b.rect.width())).arg(int(b.rect.height()));
        }
        if (pr.boxes.empty())
            details += QStringLiteral("（未检出任何目标，可尝试降低置信度阈值）\n");
    } else {
        if (!pr.mask.empty()) {
            const int total = pr.mask.rows * pr.mask.cols;
            int fg = 0;
            details += QStringLiteral("任务：图像分割\n掩码尺寸：%1×%2\n\n--- 各类像素占比 ---\n")
                           .arg(pr.mask.cols).arg(pr.mask.rows);
            for (int c = 1; c < std::min(256, classes.size()); ++c) {
                const int n = cv::countNonZero(pr.mask == c);
                fg += n;
                if (n > 0) {
                    details += QStringLiteral("%1. %2  像素 %3（%4%）\n")
                                   .arg(c)
                                   .arg(className(c))
                                   .arg(n)
                                   .arg(100.0 * n / std::max(1, total), 0, 'f', 2);
                }
            }
            details += QStringLiteral("\n前景合计占比：%1%")
                           .arg(100.0 * fg / std::max(1, total), 0, 'f', 2);

            cv::Mat color = imageio::colorizeMask(pr.mask, classes, 0.45);
            if (!color.empty() && color.size() == vis.size())
                cv::addWeighted(vis, 0.6, color, 0.4, 0.0, vis);
        } else {
            details += QStringLiteral("任务：图像分割\n未返回掩码。\n");
        }
    }

    showResult(vis, details);
    trainer->deleteLater();
}

void TestModelDialog::showResult(const cv::Mat &vis, const QString &details)
{
    const QImage img = imageio::matToQImage(vis);
    m_lblImage->setPixmap(QPixmap::fromImage(img));
    m_lblImage->resize(img.size());
    m_txtDetail->setPlainText(details);
}

} // namespace aq
