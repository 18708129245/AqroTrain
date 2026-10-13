// ============================================================================
//  ParamPanel.cpp
// ============================================================================
#include "ui/ParamPanel.h"

#include <QAbstractSpinBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QLabel>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>

#include "train/ITrainer.h"

namespace aq {

ParamPanel::ParamPanel(QWidget *parent) : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(4, 4, 4, 4);
    layout->setSpacing(6);

    layout->addWidget(buildTrainSection());
    layout->addWidget(buildTestSection());
    layout->addStretch(1);
}

QWidget *ParamPanel::makeCollapsible(QWidget *parent, const QString &title, QWidget *content,
                                     QToolButton **headerOut)
{
    auto *host = new QWidget(parent);
    auto *v = new QVBoxLayout(host);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(2);

    auto *head = new QToolButton(host);
    head->setText(title);
    head->setCheckable(true);
    head->setChecked(false);
    head->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    head->setArrowType(Qt::RightArrow);
    head->setObjectName(QStringLiteral("sectionHeader"));
    head->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    content->setVisible(false);
    v->addWidget(head);
    v->addWidget(content);

    if (headerOut)
        *headerOut = head;
    return host;
}

QWidget *ParamPanel::buildTrainSection()
{
    m_trainBody = new QWidget(this);
    auto *form = new QFormLayout(m_trainBody);
    form->setContentsMargins(8, 2, 2, 2);
    form->setSpacing(4);

    m_cmbBackend = new QComboBox(m_trainBody);
    m_cmbBackend->addItem(QStringLiteral("OpenCV 轻量后端（开箱即用）"), QStringLiteral("opencv"));
    m_cmbBackend->addItem(QStringLiteral("LibTorch 深度学习后端"), QStringLiteral("libtorch"));
    m_cmbBackend->addItem(QStringLiteral("自动（优先 LibTorch）"), QStringLiteral("auto"));
    form->addRow(QStringLiteral("训练后端"), m_cmbBackend);

    m_lblBackendHint = new QLabel(m_trainBody);
    m_lblBackendHint->setWordWrap(true);
    m_lblBackendHint->setObjectName(QStringLiteral("panelHint"));
    form->addRow(m_lblBackendHint);

    m_cmbModel = new QComboBox(m_trainBody);
    m_cmbModel->addItem(QStringLiteral("resnet18"), QStringLiteral("resnet18"));
    m_cmbModel->addItem(QStringLiteral("unet"), QStringLiteral("unet"));
    m_cmbModel->addItem(QStringLiteral("简化检测头"), QStringLiteral("detector"));
    form->addRow(QStringLiteral("模型结构"), m_cmbModel);

    m_spEpochs = new QSpinBox(m_trainBody);
    m_spEpochs->setRange(1, 100000);
    m_spEpochs->setValue(50);
    form->addRow(QStringLiteral("Epoch"), m_spEpochs);

    m_spBatch = new QSpinBox(m_trainBody);
    m_spBatch->setRange(1, 4096);
    m_spBatch->setValue(8);
    form->addRow(QStringLiteral("Batch Size"), m_spBatch);

    m_spLr = new QDoubleSpinBox(m_trainBody);
    m_spLr->setDecimals(6);
    m_spLr->setRange(0.000001, 10.0);
    m_spLr->setSingleStep(0.0001);
    m_spLr->setValue(0.001);
    form->addRow(QStringLiteral("学习率"), m_spLr);

    m_spInW = new QSpinBox(m_trainBody);
    m_spInW->setRange(32, 8192);
    m_spInW->setValue(224);
    form->addRow(QStringLiteral("输入宽度"), m_spInW);

    m_spInH = new QSpinBox(m_trainBody);
    m_spInH->setRange(32, 8192);
    m_spInH->setValue(224);
    form->addRow(QStringLiteral("输入高度"), m_spInH);

    m_cmbDevice = new QComboBox(m_trainBody);
    m_cmbDevice->addItem(QStringLiteral("CPU"), 0);
    m_cmbDevice->addItem(QStringLiteral("GPU"), 1);
    form->addRow(QStringLiteral("训练设备"), m_cmbDevice);

    m_spGpu = new QSpinBox(m_trainBody);
    m_spGpu->setRange(0, 16);
    m_spGpu->setValue(0);
    m_spGpu->setEnabled(false);
    form->addRow(QStringLiteral("GPU 序号"), m_spGpu);

    QWidget *host = makeCollapsible(this, QStringLiteral("训练参数"), m_trainBody, &m_headTrain);
    connect(m_headTrain, &QToolButton::toggled, this, &ParamPanel::onToggleTrain);
    connect(m_cmbBackend, SIGNAL(currentIndexChanged(int)), this, SLOT(onBackendChanged()));
    connect(m_cmbDevice, SIGNAL(currentIndexChanged(int)), this, SLOT(onBackendChanged()));

    // 任何参数变化都通知主窗口
    for (QObject *o : { static_cast<QObject *>(m_spEpochs), static_cast<QObject *>(m_spBatch),
                        static_cast<QObject *>(m_spLr), static_cast<QObject *>(m_spInW),
                        static_cast<QObject *>(m_spInH), static_cast<QObject *>(m_spGpu),
                        static_cast<QObject *>(m_cmbModel) }) {
        if (auto *sp = qobject_cast<QAbstractSpinBox *>(o))
            connect(sp, SIGNAL(editingFinished()), this, SIGNAL(changed()));
        else if (auto *cb = qobject_cast<QComboBox *>(o))
            connect(cb, SIGNAL(currentIndexChanged(int)), this, SIGNAL(changed()));
    }
    return host;
}

QWidget *ParamPanel::buildTestSection()
{
    m_testBody = new QWidget(this);
    auto *form = new QFormLayout(m_testBody);
    form->setContentsMargins(8, 2, 2, 2);
    form->setSpacing(4);

    m_spScore = new QDoubleSpinBox(m_testBody);
    m_spScore->setDecimals(3);
    m_spScore->setRange(0.0, 1.0);
    m_spScore->setSingleStep(0.05);
    m_spScore->setValue(0.5);
    form->addRow(QStringLiteral("置信度阈值"), m_spScore);

    m_spNms = new QDoubleSpinBox(m_testBody);
    m_spNms->setDecimals(3);
    m_spNms->setRange(0.0, 1.0);
    m_spNms->setSingleStep(0.05);
    m_spNms->setValue(0.45);
    form->addRow(QStringLiteral("NMS 阈值"), m_spNms);

    m_spIou = new QDoubleSpinBox(m_testBody);
    m_spIou->setDecimals(2);
    m_spIou->setRange(0.1, 1.0);
    m_spIou->setSingleStep(0.05);
    m_spIou->setValue(0.5);
    m_spIou->setToolTip(QStringLiteral("评价时判定「检出」所需的最小 IoU"));
    form->addRow(QStringLiteral("IoU 匹配阈值"), m_spIou);

    m_spMinArea = new QSpinBox(m_testBody);
    m_spMinArea->setRange(1, 10000000);
    m_spMinArea->setValue(16);
    m_spMinArea->setToolTip(QStringLiteral("分割结果中面积小于该值的连通域会被忽略"));
    form->addRow(QStringLiteral("最小连通域面积"), m_spMinArea);

    QWidget *host = makeCollapsible(this, QStringLiteral("测试参数"), m_testBody, &m_headTest);
    connect(m_headTest, &QToolButton::toggled, this, &ParamPanel::onToggleTest);

    for (QObject *o : { static_cast<QObject *>(m_spScore), static_cast<QObject *>(m_spNms),
                        static_cast<QObject *>(m_spIou), static_cast<QObject *>(m_spMinArea) }) {
        if (auto *sp = qobject_cast<QAbstractSpinBox *>(o))
            connect(sp, SIGNAL(editingFinished()), this, SIGNAL(changed()));
    }
    return host;
}

void ParamPanel::onToggleTrain(bool on)
{
    m_headTrain->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
    if (m_trainBody)
        m_trainBody->setVisible(on);
}

void ParamPanel::onToggleTest(bool on)
{
    m_headTest->setArrowType(on ? Qt::DownArrow : Qt::RightArrow);
    if (m_testBody)
        m_testBody->setVisible(on);
}

void ParamPanel::onBackendChanged()
{
    updateBackendDependentState();
    emit changed();
}

void ParamPanel::updateBackendDependentState()
{
    const QString backend = m_cmbBackend ? m_cmbBackend->currentData().toString() : QStringLiteral("opencv");
    const bool gradientBased = (backend != QLatin1String("opencv"));

    // OpenCV 轻量后端没有 epoch / 学习率 / Batch 的概念，置灰并说明
    m_spEpochs->setEnabled(m_editable && gradientBased);
    m_spBatch->setEnabled(m_editable && gradientBased);
    m_spLr->setEnabled(m_editable && gradientBased);

    const bool gpu = m_cmbDevice && m_cmbDevice->currentData().toInt() == 1;
    m_spGpu->setEnabled(m_editable && gpu && gradientBased);

    if (m_lblBackendHint) {
        if (!gradientBased) {
            m_lblBackendHint->setText(QStringLiteral(
                "OpenCV 轻量后端把训练拆成若干阶段，不使用 Epoch / 学习率 / Batch，"
                "因此这三项已置灰。需要梯度下降训练请安装 LibTorch。"));
        } else if (!m_libTorchAvailable) {
            m_lblBackendHint->setText(QStringLiteral(
                "LibTorch 当前不可用（未安装或未在编译期启用），训练会自动退回 OpenCV 轻量后端。"
                "设置环境变量 LIBTORCH_DIR 后重新编译即可启用。"));
        } else {
            m_lblBackendHint->setText(QStringLiteral("LibTorch 后端可用。"));
        }
    }
}

void ParamPanel::setConfig(const TrainConfig &train, const TestConfig &test, TaskType task,
                           int targetClasses, bool libTorchAvailable)
{
    m_task = task;
    m_libTorchAvailable = libTorchAvailable;

    const int bIdx = m_cmbBackend->findData(train.backend);
    m_cmbBackend->setCurrentIndex(bIdx >= 0 ? bIdx : 0);

    const int mIdx = m_cmbModel->findData(train.model);
    if (mIdx >= 0)
        m_cmbModel->setCurrentIndex(mIdx);

    m_spEpochs->setValue(qMax(1, train.epochs));
    m_spBatch->setValue(qMax(1, train.batchSize));
    m_spLr->setValue(train.learningRate > 0 ? train.learningRate : 0.001);
    m_spInW->setValue(qMax(32, train.inputWidth));
    m_spInH->setValue(qMax(32, train.inputHeight));
    m_cmbDevice->setCurrentIndex(train.useGpu ? 1 : 0);
    m_spGpu->setValue(qMax(0, train.gpuIndex));

    m_spScore->setValue(test.scoreThreshold);
    m_spNms->setValue(test.nmsThreshold);
    m_spIou->setValue(test.iouThreshold);
    m_spMinArea->setValue(qMax(1, test.minArea));

    // 检测 / 分割才有 NMS 与最小面积的概念
    const bool det = (task == TaskType::Detection);
    m_spNms->setEnabled(det);
    m_spMinArea->setEnabled(task == TaskType::Segmentation);
    m_spScore->setEnabled(det);

    updateBackendDependentState();
}

TrainConfig ParamPanel::trainConfig(const TrainConfig &base) const
{
    TrainConfig t = base;
    t.backend      = m_cmbBackend->currentData().toString();
    t.model        = m_cmbModel->currentData().toString();
    t.epochs       = m_spEpochs->value();
    t.batchSize    = m_spBatch->value();
    t.learningRate = m_spLr->value();
    t.inputWidth   = m_spInW->value();
    t.inputHeight  = m_spInH->value();
    t.useGpu       = m_cmbDevice->currentData().toInt() == 1;
    t.gpuIndex     = m_spGpu->value();
    t.task         = m_task;
    return t;
}

TestConfig ParamPanel::testConfig(const TestConfig &base) const
{
    TestConfig t = base;
    t.scoreThreshold = m_spScore->value();
    t.nmsThreshold   = m_spNms->value();
    t.iouThreshold   = m_spIou->value();
    t.minArea        = m_spMinArea->value();
    return t;
}

void ParamPanel::setEditable(bool on)
{
    m_editable = on;
    setEnabled(on);
    updateBackendDependentState();
}

} // namespace aq
