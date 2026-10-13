// ============================================================================
//  ParamPanel.h —— 左侧可收起的「训练参数 / 测试参数」面板（对标 AIDI）
//  ---------------------------------------------------------------------------
//  AIDI 把参数放在左侧两个可展开收起的选项卡里。这里用两个可折叠区块实现：
//      ▸ 训练参数    后端 / 模型 / Epoch / Batch / 学习率 / 输入尺寸 / 设备
//      ▸ 测试参数    置信度阈值 / NMS 阈值 / IoU 匹配阈值 / 最小连通域面积
//
//  注意：OpenCV 轻量后端不使用 epoch / 学习率 / Batch 这类梯度下降概念，
//  选中它时相应控件会被置灰并给出说明，避免「填了参数却没生效」的误解。
// ============================================================================
#pragma once

#include <QWidget>

#include "core/Types.h"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QSpinBox;
class QToolButton;

namespace aq {

class ParamPanel : public QWidget
{
    Q_OBJECT

public:
    explicit ParamPanel(QWidget *parent = nullptr);

    /// 用配置填充界面。targetClasses 用于校验类别数。
    void setConfig(const TrainConfig &train, const TestConfig &test, TaskType task,
                   int targetClasses, bool libTorchAvailable);

    /// 读取界面上的训练参数（保持 datasetRoot / outputDir / numClasses 不变）
    TrainConfig  trainConfig(const TrainConfig &base) const;
    TestConfig   testConfig(const TestConfig &base) const;

    /// 尚未打开模块时整体置灰
    void setEditable(bool on);

signals:
    void changed();

private slots:
    void onBackendChanged();
    void onToggleTrain(bool on);
    void onToggleTest(bool on);

private:
    QWidget *buildTrainSection();
    QWidget *buildTestSection();
    void     updateBackendDependentState();
    static QWidget *makeCollapsible(QWidget *parent, const QString &title, QWidget *content,
                                    QToolButton **headerOut);

    // --- 训练参数控件 ---
    QToolButton    *m_headTrain = nullptr;
    QComboBox      *m_cmbBackend = nullptr;
    QComboBox      *m_cmbModel = nullptr;
    QSpinBox       *m_spEpochs = nullptr;
    QSpinBox       *m_spBatch = nullptr;
    QDoubleSpinBox *m_spLr = nullptr;
    QSpinBox       *m_spInW = nullptr;
    QSpinBox       *m_spInH = nullptr;
    QComboBox      *m_cmbDevice = nullptr;
    QSpinBox       *m_spGpu = nullptr;
    QLabel         *m_lblBackendHint = nullptr;
    QWidget        *m_trainBody = nullptr;

    // --- 测试参数控件 ---
    QToolButton    *m_headTest = nullptr;
    QDoubleSpinBox *m_spScore = nullptr;
    QDoubleSpinBox *m_spNms = nullptr;
    QDoubleSpinBox *m_spIou = nullptr;
    QSpinBox       *m_spMinArea = nullptr;
    QWidget        *m_testBody = nullptr;

    TaskType m_task = TaskType::Classification;
    bool     m_libTorchAvailable = false;
    bool     m_editable = false;
};

} // namespace aq
