// ============================================================================
//  MainWindow.h —— 应用主窗口
//  ---------------------------------------------------------------------------
//  把 AIDI 的工作流串成一条线：
//      新建/打开工程 -> 添加模块 -> 导入图像 -> 逐图标注 -> 数据划分
//        -> 训练 -> 自动测试与指标 -> 导出模型
//
//  界面骨架（对标 AIDI）：
//      菜单栏 / 主工具条 / 模块链工具条
//      左侧：图像池 ImageListPanel + 可收起的训练·测试参数面板 ParamPanel
//      中间：引导提示栏 + 标注工具条 + 标注画布 ImageCanvas
//      右侧：标签类别 ClassPanel + 标注对象 ObjectPanel + 指标 MetricsPanel
//      底部：显示类型切换 DisplayControl + 状态栏
// ============================================================================
#pragma once

#include <QMainWindow>
#include <QString>
#include <QVector>

#include "annotate/AnnoObject.h"
#include "core/Types.h"

class QAction;
class QActionGroup;
class QDockWidget;
class QLabel;
class QSlider;
class QSpinBox;
class QToolBar;
class QWidget;

namespace aq {

class ClassPanel;
class DisplayControl;
class ImageCanvas;
class ImageListPanel;
class ITrainer;
class MetricsPanel;
class ModuleBar;
class ObjectPanel;
class ParamPanel;
class ProjectManager;
class TrainMonitorDialog;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

    /// 供自检与自动化调用：直接打开指定工程（等价于「文件 → 打开工程」）
    bool openProjectAt(const QString &root);

protected:
    void closeEvent(QCloseEvent *e) override;

private slots:
    // --- 工程与模块 ---
    void onNewProject();
    void onOpenProject();
    void onCloseProject();
    void onAddModule();
    void onRemoveModule();
    void onModuleSelected(int index);
    void onRenameModule(int index);

    // --- 图像池 ---
    void onImportImages();
    void onImportFolder();
    void onSampleActivated(int index);
    void onPrevSample();
    void onNextSample();
    void onSplitRequested(const QVector<int> &indexes, int split);
    void onOkRequested(const QVector<int> &indexes, bool ok);
    void onRemoveImagesRequested(const QVector<int> &indexes);
    void onOpenImageFolder(int index);

    // --- 标注 ---
    void onSaveAnnotation();
    void onSaveAnnotationAndNext();
    void onClearAnnotation();
    void onDeleteAnnotationFile();
    void onSetCurrentImageOk();
    void onCopySelectedObject();
    void onPasteObjects();
    void onConfirmPreAnnotation();
    void onToolChanged(QAction *action);
    void onClassPicked(int classId);
    void onClassesModified();
    void onShowOverlayToggled(bool on);
    void onBrushSizeChanged(int px);
    void onFixedRectSizeChanged();
    void onOpacityChanged(int alpha);
    void onDisplayModeChanged(int mode);

    // --- 对象列表 ---
    void onObjectSelected(int index);
    void onObjectClassChange(int index, int classId);
    void onObjectDelete(int index);
    void onObjectCopy(int index);

    // --- 画布信号 ---
    void onCanvasModified(bool modified);
    void onCanvasZoomChanged(double scale);
    void onCanvasMousePos(const QPoint &imgPos, int gray);
    void onCanvasMouseLeft();
    void onCanvasClassChanged(int classId);
    void onCanvasMessage(const QString &text);
    void onCanvasSelectionChanged(int index);
    void onCanvasObjectsChanged();
    void onCanvasContextCommand(const QString &command);
    void onCanvasBackgroundBlocked(const QString &text);

    // --- 视图 ---
    void onZoomIn();
    void onZoomOut();
    void onZoomReset();
    void onZoomFit();

    // --- 训练 / 测试 ---
    void onShowParams();
    void onStartTraining();
    void onTrainFinished(bool ok, const QString &modelPath, const QString &message);
    void onEvaluate();
    void onTestModel();
    void onExportModel();
    void onImportModel();
    void onSmartAnnotate();

    // --- 帮助 ---
    void onBackendInfo();
    void onAbout();

private:
    // --- 构建 ---
    void createActions();
    void createMenus();
    void createToolBars();
    void createDocks();
    void createStatusBar();
    void createCentral();
    void wireProject();

    // --- 刷新 ---
    void updateHintBar();
    void updateActionStates();
    void updateStatsLabel();
    void updateWindowTitle();
    void refreshSampleList();
    void refreshModuleBar();
    void refreshClassPanel();
    void refreshObjectPanel();
    void refreshMetrics();
    void applyModuleToUi();
    void refreshDisplayMode();

    // --- 流程 ---
    bool openProject(const QString &root);
    bool ensureProjectOpen();
    bool ensureModule();
    bool ensureSample();
    bool maybeSaveAnnotation();
    void saveCurrentAnnotation();
    void loadSample(int index);
    void openResultOverlayFor(int index);

    /// 端到端自动评价；返回是否成功
    bool evaluateModel(const QString &modelPath);
    /// 批量预标注；scope 见 PreAnnotateDialog::Scope
    void runPreAnnotate(int scope, const QVector<int> &selectedIndexes);

    void warn(const QString &text, const QString &title = QStringLiteral("提示"));
    void info(const QString &text, const QString &title = QStringLiteral("提示"));

    // --- 数据 ---
    ProjectManager *m_project = nullptr;
    int             m_currentSample = -1;
    /// 正在载入样本：用于抑制载入过程中触发的自动保存等副作用
    bool            m_loadingSample = false;
    /// 已应用到界面的模块索引（-1 表示尚未应用）
    int             m_appliedModule = -1;

    QVector<AnnoObject> m_clipboard;
    bool                m_preStop = false;
    int                 m_displayMode = 1;   ///< 见 DisplayControl::Mode

    ITrainer           *m_trainer = nullptr;
    TrainMonitorDialog *m_monitor = nullptr;

    // --- 界面 ---
    ImageCanvas    *m_canvas      = nullptr;
    ModuleBar      *m_moduleBar   = nullptr;
    ImageListPanel *m_imageList   = nullptr;
    ClassPanel     *m_classPanel  = nullptr;
    ObjectPanel    *m_objectPanel = nullptr;
    MetricsPanel   *m_metrics     = nullptr;
    ParamPanel     *m_paramPanel  = nullptr;
    DisplayControl *m_display     = nullptr;

    QToolBar    *m_toolBarMain = nullptr;
    QToolBar    *m_toolBarAnno = nullptr;
    QDockWidget *m_dockImages  = nullptr;
    QDockWidget *m_dockParams  = nullptr;
    QDockWidget *m_dockClasses = nullptr;
    QDockWidget *m_dockObjects = nullptr;
    QDockWidget *m_dockMetrics = nullptr;

    QWidget *m_central = nullptr;

    // --- 动作：文件 ---
    QAction *m_actNewProject   = nullptr;
    QAction *m_actOpenProject  = nullptr;
    QAction *m_actCloseProject = nullptr;
    QAction *m_actAddModule    = nullptr;
    QAction *m_actRemoveModule = nullptr;
    QAction *m_actImportImages = nullptr;
    QAction *m_actImportFolder = nullptr;
    QAction *m_actImportModel  = nullptr;
    QAction *m_actSave         = nullptr;
    QAction *m_actExit         = nullptr;

    // --- 动作：标注 ---
    QActionGroup *m_toolGroup  = nullptr;
    QAction *m_actSelect    = nullptr;
    QAction *m_actBrush     = nullptr;
    QAction *m_actEraser    = nullptr;
    QAction *m_actPolygon   = nullptr;
    QAction *m_actRect      = nullptr;
    QAction *m_actFixedRect = nullptr;
    QAction *m_actPan       = nullptr;

    QAction *m_actUndo        = nullptr;
    QAction *m_actRedo        = nullptr;
    QAction *m_actClearAnno   = nullptr;
    QAction *m_actDeleteAnnoFile = nullptr;
    QAction *m_actShowOverlay = nullptr;
    QAction *m_actPrevSample  = nullptr;
    QAction *m_actNextSample  = nullptr;
    QAction *m_actSetOk       = nullptr;
    QAction *m_actCopyObject  = nullptr;
    QAction *m_actPasteObject = nullptr;
    QAction *m_actDeleteObject = nullptr;
    QAction *m_actConfirmPre  = nullptr;

    // --- 动作：视图 ---
    QAction *m_actZoomIn    = nullptr;
    QAction *m_actZoomOut   = nullptr;
    QAction *m_actZoomReset = nullptr;
    QAction *m_actZoomFit   = nullptr;

    // --- 动作：训练 / 测试 ---
    QAction *m_actShowParams = nullptr;
    QAction *m_actTrain      = nullptr;
    QAction *m_actEvaluate   = nullptr;
    QAction *m_actSmartAnno  = nullptr;
    QAction *m_actTest       = nullptr;
    QAction *m_actExport     = nullptr;

    // --- 动作：帮助 ---
    QAction *m_actBackend = nullptr;
    QAction *m_actAbout   = nullptr;

    // --- 画布上方引导提示栏 ---
    QLabel *m_hintBar = nullptr;

    // --- 标注工具参数 ---
    QSpinBox *m_spBrushSize  = nullptr;
    QSpinBox *m_spFixedW     = nullptr;
    QSpinBox *m_spFixedH     = nullptr;
    QAction  *m_actFixedSize = nullptr;
    QSlider  *m_sldOpacity   = nullptr;

    // --- 状态栏 ---
    QLabel *m_lblProject = nullptr;
    QLabel *m_lblModule  = nullptr;
    QLabel *m_lblStats   = nullptr;
    QLabel *m_lblPos     = nullptr;
    QLabel *m_lblZoom    = nullptr;
    QLabel *m_lblClass   = nullptr;
};

} // namespace aq
