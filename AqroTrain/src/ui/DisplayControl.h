// ============================================================================
//  DisplayControl.h —— 显示类型切换 + 掩码透明度
//  ---------------------------------------------------------------------------
//  对标 AIDI 图片显示控制区的三态切换：
//      原图  /  标注  /  测试结果
//  「测试结果」叠加 results/<图名>.json 记录的预测，用于结果验证环节。
// ============================================================================
#pragma once

#include <QWidget>

class QLabel;
class QRadioButton;
class QSlider;

namespace aq {

class DisplayControl : public QWidget
{
    Q_OBJECT

public:
    enum Mode {
        ImageOnly   = 0,   ///< 只显示原图
        Annotation  = 1,   ///< 原图 + 标注（掩码 / 框 / 多边形）
        TestResult  = 2    ///< 原图 + 模型预测结果
    };
    Q_ENUM(Mode)

    explicit DisplayControl(QWidget *parent = nullptr);

    Mode mode() const;
    void setMode(Mode m);

    int  opacity() const;
    void setOpacity(int alpha);

    /// 没有测试结果时禁用「测试结果」选项
    void setTestResultAvailable(bool on);

signals:
    void modeChanged(int mode);
    void opacityChanged(int alpha);

private:
    QRadioButton *m_rbImage = nullptr;
    QRadioButton *m_rbAnno  = nullptr;
    QRadioButton *m_rbTest  = nullptr;
    QSlider      *m_sldOpacity = nullptr;
    QLabel       *m_lblOpacity = nullptr;
    bool          m_testAvailable = false;
};

} // namespace aq
