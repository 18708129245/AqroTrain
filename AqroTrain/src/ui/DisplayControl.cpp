// ============================================================================
//  DisplayControl.cpp
// ============================================================================
#include "ui/DisplayControl.h"

#include <QHBoxLayout>
#include <QLabel>
#include <QRadioButton>
#include <QSlider>

namespace aq {

DisplayControl::DisplayControl(QWidget *parent) : QWidget(parent)
{
    auto *row = new QHBoxLayout(this);
    row->setContentsMargins(6, 2, 6, 2);
    row->setSpacing(10);

    row->addWidget(new QLabel(QStringLiteral("显示类型："), this));

    m_rbImage = new QRadioButton(QStringLiteral("原图"), this);
    m_rbAnno  = new QRadioButton(QStringLiteral("标注"), this);
    m_rbTest  = new QRadioButton(QStringLiteral("测试结果"), this);
    m_rbAnno->setChecked(true);
    m_rbAnno->setToolTip(QStringLiteral("显示原图与人工标注"));
    m_rbTest->setToolTip(QStringLiteral("沿用最近一次测试的预测结果叠加显示"));

    row->addWidget(m_rbImage);
    row->addWidget(m_rbAnno);
    row->addWidget(m_rbTest);

    row->addSpacing(16);
    row->addWidget(new QLabel(QStringLiteral("标注透明度："), this));

    m_sldOpacity = new QSlider(Qt::Horizontal, this);
    m_sldOpacity->setRange(0, 255);
    m_sldOpacity->setValue(130);
    m_sldOpacity->setFixedWidth(150);
    m_sldOpacity->setToolTip(QStringLiteral("分割掩码叠加的不透明度"));
    row->addWidget(m_sldOpacity);

    m_lblOpacity = new QLabel(QStringLiteral("130"), this);
    m_lblOpacity->setFixedWidth(30);
    row->addWidget(m_lblOpacity);

    row->addStretch(1);

    connect(m_rbImage, &QRadioButton::toggled, this, [this](bool on) {
        if (on) emit modeChanged(int(ImageOnly));
    });
    connect(m_rbAnno, &QRadioButton::toggled, this, [this](bool on) {
        if (on) emit modeChanged(int(Annotation));
    });
    connect(m_rbTest, &QRadioButton::toggled, this, [this](bool on) {
        if (on) emit modeChanged(int(TestResult));
    });
    connect(m_sldOpacity, &QSlider::valueChanged, this, [this](int v) {
        m_lblOpacity->setText(QString::number(v));
        emit opacityChanged(v);
    });

    setTestResultAvailable(false);
}

DisplayControl::Mode DisplayControl::mode() const
{
    if (m_rbTest->isChecked())
        return TestResult;
    if (m_rbImage->isChecked())
        return ImageOnly;
    return Annotation;
}

void DisplayControl::setMode(Mode m)
{
    switch (m) {
    case ImageOnly:  m_rbImage->setChecked(true); break;
    case Annotation: m_rbAnno->setChecked(true);  break;
    case TestResult:
        if (m_testAvailable)
            m_rbTest->setChecked(true);
        else
            m_rbAnno->setChecked(true);
        break;
    }
}

int DisplayControl::opacity() const
{
    return m_sldOpacity->value();
}

void DisplayControl::setOpacity(int alpha)
{
    m_sldOpacity->setValue(qBound(0, alpha, 255));
}

void DisplayControl::setTestResultAvailable(bool on)
{
    m_testAvailable = on;
    m_rbTest->setEnabled(on);
    if (!on && m_rbTest->isChecked())
        m_rbAnno->setChecked(true);
    m_rbTest->setToolTip(on ? QStringLiteral("沿用最近一次测试的预测结果叠加显示")
                            : QStringLiteral("尚无测试结果：请先训练或执行「模型测试」"));
}

} // namespace aq
