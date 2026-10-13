// ============================================================================
//  MainWindow.cpp —— 主窗口：AIDI 式工作流接线
//  ---------------------------------------------------------------------------
//  界面骨架与 VisionTrain 的差异：
//    顶部多出「模块链」工具条（AIDI 的模块序列），
//    左侧是图像池 + 可收起的训练/测试参数面板，
//    右侧是标签类别 + 标注对象列表 + 评价指标。
//  数据流：ProjectManager 是唯一门面，界面层不直接碰文件。
// ============================================================================
#include "ui/MainWindow.h"

#include <QActionGroup>
#include <QApplication>
#include <QAbstractButton>
#include <QCloseEvent>
#include <QCursor>
#include <QDesktopServices>
#include <QDir>
#include <QDockWidget>
#include <QEventLoop>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSlider>
#include <QSpinBox>
#include <QStatusBar>
#include <QToolBar>
#include <QUrl>
#include <QVBoxLayout>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <functional>

#include "annotate/ImageCanvas.h"
#include "core/ImageIO.h"
#include "project/ProjectManager.h"
#include "train/Evaluator.h"
#include "train/ITrainer.h"
#include "train/TrainerFactory.h"
#include "ui/AddModuleDialog.h"
#include "ui/ClassPanel.h"
#include "ui/DisplayControl.h"
#include "ui/ExportModelDialog.h"
#include "ui/ImageListPanel.h"
#include "ui/MetricsPanel.h"
#include "ui/ModuleBar.h"
#include "ui/NewProjectDialog.h"
#include "ui/ObjectPanel.h"
#include "ui/ParamPanel.h"
#include "ui/PreAnnotateDialog.h"
#include "ui/TestModelDialog.h"
#include "ui/TrainMonitorDialog.h"

namespace aq {

namespace {

QAction *mkAction(QObject *parent, const QString &text, const QString &shortcut,
                  const QString &tip, bool checkable = false)
{
    auto *a = new QAction(text, parent);
    if (!shortcut.isEmpty())
        a->setShortcut(QKeySequence(shortcut));
    a->setStatusTip(tip);
    a->setToolTip(tip);
    a->setCheckable(checkable);
    return a;
}

} // namespace

// ===========================================================================
//  构造
// ===========================================================================
MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    setWindowTitle(QStringLiteral("AqroTrain —— 视觉标注训练平台"));
    resize(1560, 960);

    m_project = new ProjectManager(this);

    createActions();
    createMenus();
    createCentral();
    createDocks();
    createToolBars();
    createStatusBar();
    wireProject();

    applyModuleToUi();
    updateActionStates();
    updateStatsLabel();
    updateWindowTitle();
    updateHintBar();
}

MainWindow::~MainWindow() = default;

// ===========================================================================
//  动作
// ===========================================================================
void MainWindow::createActions()
{
    // --- 文件 ---
    m_actNewProject = mkAction(this, QStringLiteral("新建工程(&N)…"), QStringLiteral("Ctrl+N"),
                               QStringLiteral("新建工程：选择保存位置、工程名称与工程图片格式"));
    m_actOpenProject = mkAction(this, QStringLiteral("打开工程(&O)…"), QStringLiteral("Ctrl+O"),
                                QStringLiteral("打开一个已有的 AqroTrain 工程"));
    m_actCloseProject = mkAction(this, QStringLiteral("关闭工程(&W)"), QString(),
                                 QStringLiteral("关闭当前工程（会先保存）"));
    m_actAddModule = mkAction(this, QStringLiteral("添加模块(&M)…"), QStringLiteral("Ctrl+M"),
                              QStringLiteral("在当前模块之后添加分类 / 检测 / 分割模块"));
    m_actRemoveModule = mkAction(this, QStringLiteral("删除模块(&R)"), QString(),
                                 QStringLiteral("删除当前模块，并连同其后的所有模块一起删除"));
    m_actImportImages = mkAction(this, QStringLiteral("导入图像(&I)…"), QStringLiteral("Ctrl+I"),
                                 QStringLiteral("导入图像文件（会复制进工程并按工程格式转码）"));
    m_actImportFolder = mkAction(this, QStringLiteral("导入文件夹(&F)…"), QStringLiteral("Ctrl+Shift+I"),
                                 QStringLiteral("导入整个文件夹里的图像（不递归）"));
    m_actImportModel = mkAction(this, QStringLiteral("导入模型(&P)…"), QString(),
                                QStringLiteral("把一个已训练好的模型放进当前模块的 models/ 目录"));
    m_actSave = mkAction(this, QStringLiteral("保存标注(&S)"), QStringLiteral("Ctrl+S"),
                         QStringLiteral("保存当前图的标注"));
    m_actExit = mkAction(this, QStringLiteral("退出(&Q)"), QStringLiteral("Ctrl+Q"),
                         QStringLiteral("退出程序"));

    // --- 工具 ---
    m_toolGroup = new QActionGroup(this);
    m_toolGroup->setExclusive(true);

    struct ToolDef {
        QAction     **slot;
        const char   *icon;
        QString      text;
        QString      shortcut;
        QString      tip;
        ImageCanvas::Tool tool;
    };
    const ToolDef tools[] = {
        { &m_actSelect, "select.png", QStringLiteral("选择(&V)"), QStringLiteral("V"),
          QStringLiteral("选择 / 编辑已有对象：点选、拖动、拖角缩放、拖顶点"), ImageCanvas::ToolSelect },
        { &m_actBrush, "brush.png", QStringLiteral("画笔(&B)"), QStringLiteral("B"),
          QStringLiteral("画笔涂抹（仅图像分割）；A / D 调整笔刷大小"), ImageCanvas::ToolBrush },
        { &m_actEraser, "eraser.png", QStringLiteral("橡皮擦(&E)"), QStringLiteral("E"),
          QStringLiteral("擦除涂抹（仅图像分割）"), ImageCanvas::ToolEraser },
        { &m_actPolygon, "polygon.png", QStringLiteral("多边形(&G)"), QStringLiteral("G"),
          QStringLiteral("多边形围出区域（仅图像分割）；双击 / 右键 / 回车闭合，Esc 取消"),
          ImageCanvas::ToolPolygon },
        { &m_actRect, "rect.png", QStringLiteral("矩形(&R)"), QStringLiteral("R"),
          QStringLiteral("拖拽画矩形：检测模块用于画框，分割模块用于矩形填充"), ImageCanvas::ToolRect },
        { &m_actFixedRect, "plus.png", QStringLiteral("固定尺寸框"), QStringLiteral("F"),
          QStringLiteral("点一下即放置固定尺寸的框（AIDI「标准模式」，目标尺寸固定时更快）"),
          ImageCanvas::ToolFixedRect },
        { &m_actPan, "pan.png", QStringLiteral("平移(&H)"), QStringLiteral("H"),
          QStringLiteral("抓手平移画布（也可随时按住鼠标中键平移）"), ImageCanvas::ToolPan },
    };

    for (const ToolDef &d : tools) {
        QAction *a = mkAction(this, d.text, d.shortcut, d.tip, true);
        a->setIcon(QIcon(QStringLiteral(":/icons/") + QString::fromLatin1(d.icon)));
        a->setData(int(d.tool));
        m_toolGroup->addAction(a);
        *d.slot = a;
    }

    // --- 编辑 ---
    m_actUndo = mkAction(this, QStringLiteral("撤销(&U)"), QStringLiteral("Ctrl+Z"),
                         QStringLiteral("撤销上一次标注操作"));
    m_actRedo = mkAction(this, QStringLiteral("重做(&Y)"), QStringLiteral("Ctrl+Y"),
                         QStringLiteral("重做被撤销的操作"));
    m_actDeleteObject = mkAction(this, QStringLiteral("删除选中对象"), QStringLiteral("Delete"),
                                 QStringLiteral("删除当前选中的标注对象"));
    m_actCopyObject = mkAction(this, QStringLiteral("复制对象"), QStringLiteral("Ctrl+C"),
                               QStringLiteral("复制当前选中的标注对象"));
    m_actPasteObject = mkAction(this, QStringLiteral("粘贴对象"), QStringLiteral("Ctrl+V"),
                                QStringLiteral("把复制的对象粘贴到当前图（可跨图复用）"));
    m_actConfirmPre = mkAction(this, QStringLiteral("确认预标注"), QString(),
                               QStringLiteral("把当前图的预标注对象标记为已人工复核"));

    m_actClearAnno = mkAction(this, QStringLiteral("清空本图标注"), QString(),
                              QStringLiteral("清空当前图的全部标注对象（可用 Ctrl+Z 撤回）"));
    m_actDeleteAnnoFile = mkAction(this, QStringLiteral("删除本图标注文件"), QString(),
                                   QStringLiteral("从磁盘删除当前图的标注文件（含分割掩码）"));
    m_actSetOk = mkAction(this, QStringLiteral("设为 OK 图"), QString(),
                          QStringLiteral("把当前图标记为无缺陷正例（OK 图），用于训练与过检率统计"));
    m_actShowOverlay = mkAction(this, QStringLiteral("显示掩码叠加"), QString(),
                                QStringLiteral("是否叠加显示分割掩码"), true);
    m_actShowOverlay->setChecked(true);

    m_actPrevSample = mkAction(this, QStringLiteral("上一张"), QStringLiteral("PgUp"),
                               QStringLiteral("切换到上一张图像"));
    m_actNextSample = mkAction(this, QStringLiteral("下一张"), QStringLiteral("PgDown"),
                               QStringLiteral("切换到下一张图像"));

    // --- 视图 ---
    m_actZoomIn = mkAction(this, QStringLiteral("放大"), QStringLiteral("Ctrl+="),
                           QStringLiteral("放大画布"));
    m_actZoomOut = mkAction(this, QStringLiteral("缩小"), QStringLiteral("Ctrl+-"),
                            QStringLiteral("缩小画布"));
    m_actZoomReset = mkAction(this, QStringLiteral("1:1"), QStringLiteral("Ctrl+0"),
                              QStringLiteral("按原始像素显示"));
    m_actZoomFit = mkAction(this, QStringLiteral("适应窗口"), QStringLiteral("Ctrl+9"),
                            QStringLiteral("缩放到适应窗口"));

    // --- 训练 / 测试 ---
    m_actShowParams = mkAction(this, QStringLiteral("训练 / 测试参数(&P)"), QString(),
                               QStringLiteral("展开左侧的参数面板"));
    m_actTrain = mkAction(this, QStringLiteral("开始训练(&T)"), QStringLiteral("Ctrl+T"),
                          QStringLiteral("用训练集训练当前模块的模型，完成后自动在测试集上评价"));
    m_actEvaluate = mkAction(this, QStringLiteral("在测试集上评价"), QString(),
                             QStringLiteral("用当前模型在测试集上重新评价并刷新指标"));
    m_actSmartAnno = mkAction(this, QStringLiteral("智能标注（预标注）(&A)"),
                              QStringLiteral("Ctrl+Shift+A"),
                              QStringLiteral("用已训练模型对未标注图像自动预标注，人工只需复核"));
    m_actTest = mkAction(this, QStringLiteral("模型测试(&M)…"), QStringLiteral("Ctrl+Shift+T"),
                         QStringLiteral("载入模型与任意图像，运行推理并可视化结果"));
    m_actExport = mkAction(this, QStringLiteral("导出模型(&E)…"), QStringLiteral("Ctrl+E"),
                           QStringLiteral("把模型与元数据导出到指定目录，便于部署"));

    // --- 帮助 ---
    m_actBackend = mkAction(this, QStringLiteral("训练后端信息(&B)"), QString(),
                            QStringLiteral("查看当前可用的训练后端"));
    m_actAbout = mkAction(this, QStringLiteral("关于(&A)"), QString(),
                          QStringLiteral("关于 AqroTrain"));

    // --- 连接 ---
    connect(m_actNewProject, &QAction::triggered, this, &MainWindow::onNewProject);
    connect(m_actOpenProject, &QAction::triggered, this, &MainWindow::onOpenProject);
    connect(m_actCloseProject, &QAction::triggered, this, &MainWindow::onCloseProject);
    connect(m_actAddModule, &QAction::triggered, this, &MainWindow::onAddModule);
    connect(m_actRemoveModule, &QAction::triggered, this, &MainWindow::onRemoveModule);
    connect(m_actImportImages, &QAction::triggered, this, &MainWindow::onImportImages);
    connect(m_actImportFolder, &QAction::triggered, this, &MainWindow::onImportFolder);
    connect(m_actImportModel, &QAction::triggered, this, &MainWindow::onImportModel);
    connect(m_actSave, &QAction::triggered, this, &MainWindow::onSaveAnnotation);
    connect(m_actExit, &QAction::triggered, this, &MainWindow::close);

    connect(m_toolGroup, &QActionGroup::triggered, this, &MainWindow::onToolChanged);

    connect(m_actUndo, &QAction::triggered, this, [this]() { if (m_canvas) m_canvas->undo(); });
    connect(m_actRedo, &QAction::triggered, this, [this]() { if (m_canvas) m_canvas->redo(); });
    connect(m_actDeleteObject, &QAction::triggered, this, [this]() {
        if (m_canvas) m_canvas->removeSelectedObject();
    });
    connect(m_actCopyObject, &QAction::triggered, this, &MainWindow::onCopySelectedObject);
    connect(m_actPasteObject, &QAction::triggered, this, &MainWindow::onPasteObjects);
    connect(m_actConfirmPre, &QAction::triggered, this, &MainWindow::onConfirmPreAnnotation);
    connect(m_actClearAnno, &QAction::triggered, this, &MainWindow::onClearAnnotation);
    connect(m_actDeleteAnnoFile, &QAction::triggered, this, &MainWindow::onDeleteAnnotationFile);
    connect(m_actSetOk, &QAction::triggered, this, &MainWindow::onSetCurrentImageOk);
    connect(m_actShowOverlay, &QAction::toggled, this, &MainWindow::onShowOverlayToggled);
    connect(m_actPrevSample, &QAction::triggered, this, &MainWindow::onPrevSample);
    connect(m_actNextSample, &QAction::triggered, this, &MainWindow::onNextSample);

    connect(m_actZoomIn, &QAction::triggered, this, &MainWindow::onZoomIn);
    connect(m_actZoomOut, &QAction::triggered, this, &MainWindow::onZoomOut);
    connect(m_actZoomReset, &QAction::triggered, this, &MainWindow::onZoomReset);
    connect(m_actZoomFit, &QAction::triggered, this, &MainWindow::onZoomFit);

    connect(m_actShowParams, &QAction::triggered, this, &MainWindow::onShowParams);
    connect(m_actTrain, &QAction::triggered, this, &MainWindow::onStartTraining);
    connect(m_actEvaluate, &QAction::triggered, this, &MainWindow::onEvaluate);
    connect(m_actSmartAnno, &QAction::triggered, this, &MainWindow::onSmartAnnotate);
    connect(m_actTest, &QAction::triggered, this, &MainWindow::onTestModel);
    connect(m_actExport, &QAction::triggered, this, &MainWindow::onExportModel);

    connect(m_actBackend, &QAction::triggered, this, &MainWindow::onBackendInfo);
    connect(m_actAbout, &QAction::triggered, this, &MainWindow::onAbout);
}

void MainWindow::createMenus()
{
    QMenu *file = menuBar()->addMenu(QStringLiteral("文件(&F)"));
    file->addAction(m_actNewProject);
    file->addAction(m_actOpenProject);
    file->addAction(m_actCloseProject);
    file->addSeparator();
    file->addAction(m_actAddModule);
    file->addAction(m_actRemoveModule);
    file->addSeparator();
    file->addAction(m_actImportImages);
    file->addAction(m_actImportFolder);
    file->addSeparator();
    file->addAction(m_actImportModel);
    file->addAction(m_actExport);
    file->addSeparator();
    file->addAction(m_actSave);
    file->addSeparator();
    file->addAction(m_actExit);

    QMenu *edit = menuBar()->addMenu(QStringLiteral("编辑(&E)"));
    edit->addAction(m_actUndo);
    edit->addAction(m_actRedo);
    edit->addSeparator();
    edit->addAction(m_actCopyObject);
    edit->addAction(m_actPasteObject);
    edit->addAction(m_actDeleteObject);

    QMenu *anno = menuBar()->addMenu(QStringLiteral("标注(&A)"));
    anno->addAction(m_actSelect);
    anno->addAction(m_actBrush);
    anno->addAction(m_actEraser);
    anno->addAction(m_actPolygon);
    anno->addAction(m_actRect);
    anno->addAction(m_actFixedRect);
    anno->addAction(m_actPan);
    anno->addSeparator();
    anno->addAction(m_actConfirmPre);
    anno->addAction(m_actClearAnno);
    anno->addAction(m_actDeleteAnnoFile);
    anno->addAction(m_actSetOk);
    anno->addSeparator();
    anno->addAction(m_actPrevSample);
    anno->addAction(m_actNextSample);

    QMenu *train = menuBar()->addMenu(QStringLiteral("训练(&T)"));
    train->addAction(m_actShowParams);
    train->addAction(m_actTrain);
    train->addAction(m_actEvaluate);
    train->addAction(m_actSmartAnno);

    QMenu *test = menuBar()->addMenu(QStringLiteral("测试(&S)"));
    test->addAction(m_actTest);

    QMenu *view = menuBar()->addMenu(QStringLiteral("显示(&V)"));
    view->addAction(m_actShowOverlay);
    view->addSeparator();
    view->addAction(m_actZoomIn);
    view->addAction(m_actZoomOut);
    view->addAction(m_actZoomReset);
    view->addAction(m_actZoomFit);

    QMenu *help = menuBar()->addMenu(QStringLiteral("帮助(&H)"));
    help->addAction(m_actBackend);
    help->addAction(m_actAbout);
}

void MainWindow::createCentral()
{
    m_central = new QWidget(this);
    auto *v = new QVBoxLayout(m_central);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);

    m_moduleBar = new ModuleBar(m_central);
    v->addWidget(m_moduleBar);

    m_hintBar = new QLabel(m_central);
    m_hintBar->setWordWrap(true);
    m_hintBar->setObjectName(QStringLiteral("hintBar"));
    m_hintBar->setTextFormat(Qt::RichText);
    m_hintBar->setContentsMargins(8, 4, 8, 4);
    v->addWidget(m_hintBar);

    // 标注工具条（放在画布正上方，与 AIDI 的工具排布一致）
    m_toolBarAnno = new QToolBar(QStringLiteral("标注工具"), m_central);
    m_toolBarAnno->setMovable(false);
    m_toolBarAnno->setIconSize(QSize(20, 20));
    m_toolBarAnno->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    for (QAction *a : m_toolGroup->actions())
        m_toolBarAnno->addAction(a);
    m_toolBarAnno->addSeparator();

    m_toolBarAnno->addWidget(new QLabel(QStringLiteral(" 笔刷 "), m_toolBarAnno));
    m_spBrushSize = new QSpinBox(m_toolBarAnno);
    m_spBrushSize->setRange(1, 512);
    m_spBrushSize->setValue(24);
    m_spBrushSize->setSuffix(QStringLiteral(" px"));
    m_spBrushSize->setToolTip(QStringLiteral("画笔直径（也可用 A / D 键或 Ctrl+滚轮调整）"));
    m_toolBarAnno->addWidget(m_spBrushSize);

    m_toolBarAnno->addWidget(new QLabel(QStringLiteral("  固定框 "), m_toolBarAnno));
    m_spFixedW = new QSpinBox(m_toolBarAnno);
    m_spFixedW->setRange(2, 20000);
    m_spFixedW->setValue(64);
    m_spFixedW->setToolTip(QStringLiteral("固定尺寸框宽度"));
    m_toolBarAnno->addWidget(m_spFixedW);
    m_toolBarAnno->addWidget(new QLabel(QStringLiteral("×"), m_toolBarAnno));
    m_spFixedH = new QSpinBox(m_toolBarAnno);
    m_spFixedH->setRange(2, 20000);
    m_spFixedH->setValue(64);
    m_spFixedH->setToolTip(QStringLiteral("固定尺寸框高度"));
    m_toolBarAnno->addWidget(m_spFixedH);

    m_toolBarAnno->addSeparator();
    m_toolBarAnno->addAction(m_actUndo);
    m_toolBarAnno->addAction(m_actRedo);

    v->addWidget(m_toolBarAnno);

    m_canvas = new ImageCanvas(m_central);
    v->addWidget(m_canvas, 1);

    m_display = new DisplayControl(m_central);
    v->addWidget(m_display);

    setCentralWidget(m_central);

    // --- 画布信号 ---
    connect(m_canvas, &ImageCanvas::modifiedChanged, this, &MainWindow::onCanvasModified);
    connect(m_canvas, &ImageCanvas::zoomChanged, this, &MainWindow::onCanvasZoomChanged);
    connect(m_canvas, &ImageCanvas::mouseImagePos, this, &MainWindow::onCanvasMousePos);
    connect(m_canvas, &ImageCanvas::mouseLeft, this, &MainWindow::onCanvasMouseLeft);
    connect(m_canvas, &ImageCanvas::classIdChanged, this, &MainWindow::onCanvasClassChanged);
    connect(m_canvas, &ImageCanvas::classPicked, this, &MainWindow::onClassPicked);
    connect(m_canvas, &ImageCanvas::toolMessage, this, &MainWindow::onCanvasMessage);
    connect(m_canvas, &ImageCanvas::selectionChanged, this, &MainWindow::onCanvasSelectionChanged);
    connect(m_canvas, &ImageCanvas::objectsChanged, this, &MainWindow::onCanvasObjectsChanged);
    connect(m_canvas, &ImageCanvas::contextCommand, this, &MainWindow::onCanvasContextCommand);
    connect(m_canvas, &ImageCanvas::backgroundClassBlocked, this,
            &MainWindow::onCanvasBackgroundBlocked);
    connect(m_canvas, &ImageCanvas::requestSaveAndNext, this, &MainWindow::onSaveAnnotationAndNext);
    connect(m_canvas, &ImageCanvas::annoStateChanged, this, [this](int) { refreshSampleList(); });

    // --- 工具参数 ---
    connect(m_spBrushSize, SIGNAL(valueChanged(int)), this, SLOT(onBrushSizeChanged(int)));
    connect(m_spFixedW, SIGNAL(valueChanged(int)), this, SLOT(onFixedRectSizeChanged()));
    connect(m_spFixedH, SIGNAL(valueChanged(int)), this, SLOT(onFixedRectSizeChanged()));

    // --- 显示控制 ---
    connect(m_display, &DisplayControl::modeChanged, this, &MainWindow::onDisplayModeChanged);
    connect(m_display, &DisplayControl::opacityChanged, this, &MainWindow::onOpacityChanged);

    // --- 模块条 ---
    connect(m_moduleBar, &ModuleBar::moduleSelected, this, &MainWindow::onModuleSelected);
    connect(m_moduleBar, &ModuleBar::addRequested, this, &MainWindow::onAddModule);
    connect(m_moduleBar, &ModuleBar::removeRequested, this, &MainWindow::onRemoveModule);
    connect(m_moduleBar, &ModuleBar::renameRequested, this, &MainWindow::onRenameModule);

    m_actSelect->setChecked(true);
}

void MainWindow::createDocks()
{
    // --- 左侧：图像池 + 参数面板 ---
    m_imageList = new ImageListPanel(this);
    m_dockImages = new QDockWidget(QStringLiteral("图像池"), this);
    m_dockImages->setObjectName(QStringLiteral("dockImages"));
    m_dockImages->setWidget(m_imageList);
    addDockWidget(Qt::LeftDockWidgetArea, m_dockImages);

    m_paramPanel = new ParamPanel(this);
    m_dockParams = new QDockWidget(QStringLiteral("训练 / 测试参数"), this);
    m_dockParams->setObjectName(QStringLiteral("dockParams"));
    m_dockParams->setWidget(m_paramPanel);
    addDockWidget(Qt::LeftDockWidgetArea, m_dockParams);
    splitDockWidget(m_dockImages, m_dockParams, Qt::Vertical);
    resizeDocks({ m_dockImages, m_dockParams }, { 620, 380 }, Qt::Vertical);

    // --- 右侧：类别 + 对象 + 指标 ---
    m_classPanel = new ClassPanel(this);
    m_dockClasses = new QDockWidget(QStringLiteral("标签类别"), this);
    m_dockClasses->setObjectName(QStringLiteral("dockClasses"));
    m_dockClasses->setWidget(m_classPanel);
    addDockWidget(Qt::RightDockWidgetArea, m_dockClasses);

    m_objectPanel = new ObjectPanel(this);
    m_dockObjects = new QDockWidget(QStringLiteral("标注对象"), this);
    m_dockObjects->setObjectName(QStringLiteral("dockObjects"));
    m_dockObjects->setWidget(m_objectPanel);
    addDockWidget(Qt::RightDockWidgetArea, m_dockObjects);
    splitDockWidget(m_dockClasses, m_dockObjects, Qt::Vertical);

    m_metrics = new MetricsPanel(this);
    m_dockMetrics = new QDockWidget(QStringLiteral("评价指标"), this);
    m_dockMetrics->setObjectName(QStringLiteral("dockMetrics"));
    m_dockMetrics->setWidget(m_metrics);
    addDockWidget(Qt::RightDockWidgetArea, m_dockMetrics);
    splitDockWidget(m_dockObjects, m_dockMetrics, Qt::Vertical);
    resizeDocks({ m_dockClasses, m_dockObjects, m_dockMetrics }, { 280, 320, 300 }, Qt::Vertical);

    // --- 信号 ---
    connect(m_imageList, &ImageListPanel::sampleActivated, this, &MainWindow::onSampleActivated);
    connect(m_imageList, &ImageListPanel::splitRequested, this, &MainWindow::onSplitRequested);
    connect(m_imageList, &ImageListPanel::okRequested, this, &MainWindow::onOkRequested);
    connect(m_imageList, &ImageListPanel::removeRequested, this,
            &MainWindow::onRemoveImagesRequested);
    connect(m_imageList, &ImageListPanel::openFolderRequested, this,
            &MainWindow::onOpenImageFolder);
    connect(m_imageList, &ImageListPanel::importRequested, this, &MainWindow::onImportImages);
    connect(m_imageList, &ImageListPanel::smartAnnotateRequested, this,
            [this](const QVector<int> &sel) {
                runPreAnnotate(int(PreAnnotateDialog::SelectedImages), sel);
            });

    connect(m_classPanel, &ClassPanel::classPicked, this, &MainWindow::onClassPicked);
    connect(m_classPanel, &ClassPanel::classesModified, this, &MainWindow::onClassesModified);

    connect(m_objectPanel, &ObjectPanel::objectSelected, this, &MainWindow::onObjectSelected);
    connect(m_objectPanel, &ObjectPanel::classChangeRequested, this,
            &MainWindow::onObjectClassChange);
    connect(m_objectPanel, &ObjectPanel::deleteRequested, this, &MainWindow::onObjectDelete);
    connect(m_objectPanel, &ObjectPanel::copyRequested, this, &MainWindow::onObjectCopy);
    connect(m_objectPanel, &ObjectPanel::confirmPreRequested, this,
            &MainWindow::onConfirmPreAnnotation);

    connect(m_paramPanel, &ParamPanel::changed, this, [this]() {
        // 参数改动即时写回模块，避免忘记保存
        const Module *m = m_project ? m_project->currentModule() : nullptr;
        if (!m || !m_paramPanel)
            return;
        m_project->setCurrentConfig(m_paramPanel->trainConfig(m->train()),
                                   m_paramPanel->testConfig(m->test()), nullptr);
    });

    connect(m_metrics, &MetricsPanel::evaluateRequested, this, &MainWindow::onEvaluate);
    connect(m_metrics, &MetricsPanel::exportModelRequested, this, &MainWindow::onExportModel);
}

void MainWindow::createToolBars()
{
    m_toolBarMain = addToolBar(QStringLiteral("主工具条"));
    m_toolBarMain->setObjectName(QStringLiteral("toolBarMain"));
    m_toolBarMain->setMovable(false);
    m_toolBarMain->setIconSize(QSize(20, 20));
    m_toolBarMain->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);

    m_toolBarMain->addAction(m_actNewProject);
    m_toolBarMain->addAction(m_actOpenProject);
    m_toolBarMain->addAction(m_actImportImages);
    m_toolBarMain->addSeparator();
    m_toolBarMain->addAction(m_actSave);
    m_toolBarMain->addSeparator();
    m_toolBarMain->addAction(m_actSmartAnno);
    m_toolBarMain->addAction(m_actTrain);
    m_toolBarMain->addAction(m_actEvaluate);
    m_toolBarMain->addSeparator();
    m_toolBarMain->addAction(m_actTest);
    m_toolBarMain->addAction(m_actExport);
}

void MainWindow::createStatusBar()
{
    m_lblProject = new QLabel(QStringLiteral("未打开工程"), this);
    m_lblModule = new QLabel(this);
    m_lblStats = new QLabel(this);
    m_lblPos = new QLabel(this);
    m_lblZoom = new QLabel(this);
    m_lblClass = new QLabel(this);

    statusBar()->addWidget(m_lblProject, 1);
    statusBar()->addPermanentWidget(m_lblModule);
    statusBar()->addPermanentWidget(m_lblStats);
    statusBar()->addPermanentWidget(m_lblClass);
    statusBar()->addPermanentWidget(m_lblPos);
    statusBar()->addPermanentWidget(m_lblZoom);
}

void MainWindow::wireProject()
{
    connect(m_project, &ProjectManager::projectOpened, this, [this]() {
        refreshModuleBar();
        refreshSampleList();
        updateWindowTitle();
    });
    connect(m_project, &ProjectManager::projectClosed, this, [this]() {
        m_appliedModule = -1;
        m_currentSample = -1;
        m_canvas->setImage(cv::Mat(), QString());
        m_canvas->setObjects({}, AnnoState::None);
        m_canvas->clearResultOverlay();
        refreshModuleBar();
        refreshSampleList();
        refreshClassPanel();
        refreshObjectPanel();
        refreshMetrics();
        updateWindowTitle();
        updateStatsLabel();
        updateHintBar();
        updateActionStates();
    });
    connect(m_project, &ProjectManager::modulesChanged, this, [this]() { refreshModuleBar(); });
    connect(m_project, &ProjectManager::currentModuleChanged, this, [this](int) {
        if (m_appliedModule != m_project->currentModuleIndex())
            applyModuleToUi();
    });
    connect(m_project, &ProjectManager::samplesChanged, this, [this]() {
        refreshSampleList();
        updateStatsLabel();
    });
}

// ===========================================================================
//  刷新
// ===========================================================================
void MainWindow::refreshModuleBar()
{
    if (!m_moduleBar)
        return;
    if (!m_project || !m_project->isOpen()) {
        m_moduleBar->setModules({}, -1);
        return;
    }
    m_moduleBar->setModules(m_project->project().moduleRefs(), m_project->currentModuleIndex());
}

void MainWindow::refreshSampleList()
{
    if (!m_imageList || !m_project || !m_project->isOpen())
        return;

    QStringList keys;
    QVector<SampleState> states;
    for (int i = 0; i < m_project->imageCount(); ++i) {
        keys << m_project->imageKey(i);
        states << m_project->sampleState(i);
    }
    m_imageList->setImages(keys);
    m_imageList->setStates(states);
    if (m_currentSample >= 0)
        m_imageList->setCurrentIndex(m_currentSample);
}

void MainWindow::refreshClassPanel()
{
    if (!m_classPanel)
        return;
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (!m) {
        m_classPanel->setClasses({}, TaskType::Classification);
        return;
    }
    m_classPanel->setClasses(m->classes(), m->type());
    m_classPanel->setCurrentClass(m_canvas ? m_canvas->currentClass() : 0);
}

void MainWindow::refreshObjectPanel()
{
    if (!m_objectPanel)
        return;
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (!m || !m_canvas) {
        m_objectPanel->setObjects({}, {}, false);
        return;
    }
    m_objectPanel->setObjects(m_canvas->objects(), m->classes(), m->hasBackground());
    m_objectPanel->setSelected(m_canvas->selectedObject());
}

void MainWindow::refreshMetrics()
{
    if (!m_metrics)
        return;
    if (!m_project || !m_project->isOpen()) {
        m_metrics->clear();
        return;
    }
    m_metrics->setMetrics(m_project->loadMetrics());
}

void MainWindow::refreshDisplayMode()
{
    if (!m_display || !m_canvas)
        return;
    switch (m_display->mode()) {
    case DisplayControl::ImageOnly:
        m_canvas->setAnnotationVisible(false);
        m_canvas->clearResultOverlay();
        break;
    case DisplayControl::Annotation:
        m_canvas->setAnnotationVisible(true);
        m_canvas->clearResultOverlay();
        break;
    case DisplayControl::TestResult:
        m_canvas->setAnnotationVisible(false);
        openResultOverlayFor(m_currentSample);
        break;
    }
}

void MainWindow::updateWindowTitle()
{
    QString title = QStringLiteral("AqroTrain —— 视觉标注训练平台");
    if (m_project && m_project->isOpen()) {
        title = QStringLiteral("%1 [%2]").arg(m_project->name(),
                                             QDir::toNativeSeparators(m_project->root()));
        const Module *m = m_project->currentModule();
        if (m)
            title += QStringLiteral("  ·  %1（%2）").arg(m->name(), taskTypeName(m->type()));
        if (m_currentSample >= 0 && m_canvas && m_canvas->isModified())
            title += QStringLiteral("  *");
    }
    setWindowTitle(title);
}

void MainWindow::updateStatsLabel()
{
    if (!m_lblStats || !m_lblProject)
        return;

    if (!m_project || !m_project->isOpen()) {
        m_lblProject->setText(QStringLiteral("未打开工程"));
        m_lblStats->setText(QString());
        m_lblModule->setText(QString());
        return;
    }

    m_lblProject->setText(QStringLiteral("工程：%1  (%2)")
                              .arg(m_project->name(),
                                   QDir::toNativeSeparators(m_project->root())));

    const DatasetStats st = m_project->stats();
    m_lblStats->setText(QStringLiteral("图像 %1 张｜已标注 %2｜训练集 %3｜测试集 %4｜OK 图 %5")
                            .arg(st.total)
                            .arg(st.annotated)
                            .arg(st.train)
                            .arg(st.test)
                            .arg(st.okImages));

    const Module *m = m_project->currentModule();
    m_lblModule->setText(m ? QStringLiteral("模块：%1").arg(m->name()) : QString());
}

void MainWindow::updateActionStates()
{
    const bool open = m_project && m_project->isOpen();
    const Module *m = open ? m_project->currentModule() : nullptr;
    const bool hasModule = (m != nullptr);
    const bool hasImage = hasModule && m_canvas && m_canvas->hasImage();
    const bool hasModel = hasModule && m_project->hasModel();

    m_actCloseProject->setEnabled(open);
    m_actAddModule->setEnabled(open);
    m_actRemoveModule->setEnabled(hasModule);
    m_actImportImages->setEnabled(open);
    m_actImportFolder->setEnabled(open);
    m_actImportModel->setEnabled(hasModule);
    m_actSave->setEnabled(hasImage);

    m_actUndo->setEnabled(hasImage && m_canvas->canUndo());
    m_actRedo->setEnabled(hasImage && m_canvas->canRedo());
    m_actDeleteObject->setEnabled(hasImage && m_canvas->selectedObject() >= 0);
    m_actCopyObject->setEnabled(hasImage && m_canvas->selectedObject() >= 0);
    m_actPasteObject->setEnabled(hasImage && !m_clipboard.isEmpty());
    m_actConfirmPre->setEnabled(hasImage);
    m_actClearAnno->setEnabled(hasImage);
    m_actDeleteAnnoFile->setEnabled(hasImage);
    m_actSetOk->setEnabled(hasImage);
    m_actPrevSample->setEnabled(hasImage);
    m_actNextSample->setEnabled(hasImage);
    m_actShowOverlay->setEnabled(hasImage);

    m_actZoomIn->setEnabled(hasImage);
    m_actZoomOut->setEnabled(hasImage);
    m_actZoomReset->setEnabled(hasImage);
    m_actZoomFit->setEnabled(hasImage);

    m_actShowParams->setEnabled(hasModule);
    m_actTrain->setEnabled(hasModule);
    m_actEvaluate->setEnabled(hasModule && hasModel);
    m_actSmartAnno->setEnabled(hasModule);
    m_actTest->setEnabled(true);
    m_actExport->setEnabled(hasModule && hasModel);

    // 工具可用性随任务类型变化（并说明为什么不可用，避免「点了没反应」）
    if (m_canvas && hasModule) {
        const TaskType t = m_canvas->task();
        for (QAction *a : m_toolGroup->actions()) {
            const auto tool = static_cast<ImageCanvas::Tool>(a->data().toInt());
            const bool allowed = ImageCanvas::toolAllowedForTask(tool, t);
            a->setEnabled(hasImage && allowed);
            if (!allowed) {
                a->setToolTip(QStringLiteral("%1 —— 当前任务（%2）下不可用")
                                  .arg(a->text(), taskTypeName(t)));
            }
        }
    } else {
        for (QAction *a : m_toolGroup->actions())
            a->setEnabled(false);
    }

    if (m_paramPanel)
        m_paramPanel->setEditable(hasModule);
}

void MainWindow::updateHintBar()
{
    if (!m_hintBar)
        return;

    QString text;

    if (!m_project || !m_project->isOpen()) {
        text = QStringLiteral("<b>AqroTrain</b> 的工作流："
                              "新建工程 → 添加模块 → 导入图像 → 逐图标注 → "
                              "数据划分（加入训练集 / 测试集）→ 训练 → 结果验证 → 导出模型。"
                              "<br>请先「文件 → 新建工程」，或「文件 → 打开工程」。");
    } else if (!m_project->currentModule()) {
        text = QStringLiteral("<span style='color:#ffb020'><b>还差一步：</b></span>"
                              "这个工程还没有模块。点击「＋ 添加模块」建立一个"
                              "分类 / 检测 / 分割模块，然后导入图像。");
    } else {
        const Module *m = m_project->currentModule();
        const bool hasImages = m_project->imageCount() > 0;
        const bool hasClasses = m->targetClassCount() > 0;

        if (!hasImages) {
            text = QStringLiteral("<span style='color:#ffb020'><b>还差一步：</b></span>"
                                  "还没有图像。用「文件 → 导入图像 / 导入文件夹」把待标定的图"
                                  "加入工程（会复制进 images/ 并按工程格式转码）。");
        } else if (!hasClasses) {
            text = QStringLiteral("<span style='color:#ffb020'><b>还差一步：</b></span>"
                                  "还没有可标注的类别。请在右侧「标签类别」中"
                                  "添加至少一个<span style='color:#ffb020'>目标类别</span>。");
        } else if (m->type() == TaskType::Classification) {
            text = QStringLiteral("分类模块：<b>整张图对应一个类别，不需要涂抹</b>。"
                                  "在右侧点选类别即完成标定（点一下立即保存，"
                                  "并自动跳到下一张未标注的图）。"
                                  "若需要逐像素标注，请新建「图像分割」模块。");
        } else if (m->type() == TaskType::Detection) {
            text = QStringLiteral("检测模块：用<b>矩形</b>拖拽画框，或用<b>固定尺寸框</b>点一下放置；"
                                  "<b>选择</b>工具可点选、拖动、拖四角缩放，Delete 删除。"
                                  "按 <b>S</b> 保存并跳到下一张未标注的图。");
        } else {
            text = QStringLiteral("分割模块：<b>画笔</b>涂抹（A / D 调笔刷）、"
                                  "<b>多边形</b>围出区域、<b>矩形</b>填充；"
                                  "选中对象后可拖顶点微调，双击边插入顶点，右键顶点删除。"
                                  "按 <b>S</b> 保存并跳到下一张未标注的图。");
        }

        const DatasetStats st = m_project->stats();
        if (hasImages && st.annotated > 0 && st.train == 0) {
            text += QStringLiteral("<br><span style='color:#ffb020'>提示：</span>"
                                   "还没有样本加入<b>训练集</b>。在左侧图像池中多选图像"
                                   "（Ctrl / Shift / Ctrl+A），右键「加入训练集」。");
        }
    }

    m_hintBar->setText(text);
}

void MainWindow::applyModuleToUi()
{
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (!m) {
        m_appliedModule = -1;
        if (m_canvas) {
            m_canvas->setClasses({});
            m_canvas->setHasBackground(false);
        }
        refreshModuleBar();
        refreshClassPanel();
        refreshObjectPanel();
        refreshMetrics();
        updateActionStates();
        updateHintBar();
        return;
    }

    m_appliedModule = m_project->currentModuleIndex();

    if (m_canvas) {
        // 先把类别与背景设置好，再切任务：setTask 会发出一次任务说明，
        // 且工具可用性判定也依赖任务已生效
        m_canvas->setClasses(m->classes());
        m_canvas->setHasBackground(m->hasBackground());
        m_canvas->setTask(m->type());
        // 检测 / 分割默认选中第一个目标类别，避免停在「背景」上作画而没有效果
        const int first = m->firstTargetClass();
        if (m_canvas->currentClass() < first || m_canvas->currentClass() >= m->classCount())
            m_canvas->setCurrentClass(first);
    }

    refreshModuleBar();
    refreshClassPanel();
    refreshObjectPanel();
    refreshMetrics();

    const bool libTorch = TrainerFactory::libTorchAvailable(nullptr);
    if (m_paramPanel)
        m_paramPanel->setConfig(m->train(), m->test(), m->type(), m->targetClassCount(), libTorch);

    if (m_display)
        m_display->setTestResultAvailable(m_project->hasModel());

    // 当前工具在该任务下可能不可用，切到该任务的默认工具
    if (m_canvas && !ImageCanvas::toolAllowedForTask(m_canvas->tool(), m->type())) {
        if (m->type() == TaskType::Classification)
            m_actSelect->setChecked(true);
        else if (m->type() == TaskType::Detection)
            m_actRect->setChecked(true);
        else
            m_actBrush->setChecked(true);
        onToolChanged(nullptr);
    }

    updateActionStates();
    updateStatsLabel();
    updateWindowTitle();
    updateHintBar();
}

// ===========================================================================
//  工程流程
// ===========================================================================
bool MainWindow::openProjectAt(const QString &root)
{
    return openProject(root);
}

bool MainWindow::openProject(const QString &root)
{
    QString err;
    if (!m_project->openProject(root, &err)) {
        warn(err, QStringLiteral("无法打开工程"));
        return false;
    }

    m_currentSample = -1;
    m_appliedModule = -1;

    applyModuleToUi();
    refreshSampleList();

    if (m_project->imageCount() > 0)
        loadSample(0);

    refreshMetrics();
    updateStatsLabel();
    updateWindowTitle();
    updateActionStates();
    updateHintBar();
    return true;
}

void MainWindow::onNewProject()
{
    NewProjectDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted)
        return;

    if (m_project->isOpen()) {
        if (!maybeSaveAnnotation())
            return;
        m_project->closeProject();
    }

    // 建工程会先创建目录，而目录创建有可能被环境拒绝
    // （安全软件 / 组策略 / 权限）。这里不把使用者堵在一句报错上，
    // 而是给出三条能走下去的路：
    //   ① 直接用所选目录本身当工程目录（目录已存在，完全不需要新建）
    //   ② 换一个位置
    //   ③ 重试（比如刚在安全软件里点了允许）
    QString root = dlg.projectRoot();
    QString err;

    while (true) {
        if (m_project->createProject(root, dlg.projectName(), dlg.imageFormat(), &err))
            break;

        QMessageBox box(this);
        box.setWindowTitle(QStringLiteral("无法创建工程"));
        box.setIcon(QMessageBox::Warning);
        box.setText(QStringLiteral("无法在以下位置创建工程：\n%1")
                        .arg(QDir::toNativeSeparators(root)));
        box.setInformativeText(QStringLiteral("%1\n\n可以选择下面的方式继续：").arg(err));

        QPushButton *btnUseThis =
            box.addButton(QStringLiteral("直接用这个目录当工程目录"), QMessageBox::AcceptRole);
        QPushButton *btnPickOther =
            box.addButton(QStringLiteral("换一个文件夹…"), QMessageBox::ActionRole);
        QPushButton *btnMakeHere =
            box.addButton(QStringLiteral("建在上次的位置"), QMessageBox::ActionRole);
        QPushButton *btnRetry = box.addButton(QStringLiteral("重试"), QMessageBox::ActionRole);
        box.addButton(QStringLiteral("取消"), QMessageBox::RejectRole);
        // 用「目录已存在」的方式创建不需要任何写目录权限，
        // 因此在权限受限时它是最可能成功的一条路，设为默认按钮。
        box.setDefaultButton(btnUseThis);
        box.exec();

        QAbstractButton *clicked = box.clickedButton();
        if (clicked == btnUseThis) {
            // createProject 允许「已存在但为空的目录」直接作为工程根
            const QString cand = QDir::cleanPath(dlg.parentDir());
            if (cand.isEmpty() || cand == root) {
                warn(QStringLiteral("「直接用这个目录当工程目录」需要与刚才失败的位置不同，"
                                    "且该目录必须为空。\n\n请点「换一个文件夹…」另外选一个空目录。"),
                     QStringLiteral("无法使用该目录"));
                continue;
            }
            if (!QFileInfo(cand).isDir()) {
                warn(QStringLiteral("该位置当前不是一个已存在的目录：\n%1")
                         .arg(QDir::toNativeSeparators(cand)),
                     QStringLiteral("无法使用该目录"));
                continue;
            }
            root = cand;
            continue;
        }
        if (clicked == btnPickOther) {
            const QString dir = QFileDialog::getExistingDirectory(
                this, QStringLiteral("选择一个已存在的空目录作为工程目录"), dlg.parentDir());
            if (dir.isEmpty())
                return;
            root = QDir::fromNativeSeparators(dir);
            continue;
        }
        if (clicked == btnMakeHere) {
            QSettings settings;
            QString cand = settings.value(QStringLiteral("AqroTrain/lastProjectParent")).toString();
            if (cand.isEmpty())
                cand = QDir::homePath() + QStringLiteral("/Documents/AqroTrainProjects");
            root = QDir::cleanPath(cand + QLatin1Char('/') + dlg.projectName());
            continue;
        }
        if (clicked == btnRetry)
            continue;

        return;   // 取消
    }

    m_currentSample = -1;
    m_appliedModule = -1;
    applyModuleToUi();
    updateWindowTitle();

    info(QStringLiteral("工程已创建：\n%1\n\n接下来请点「＋ 添加模块」，再导入图像。")
             .arg(QDir::toNativeSeparators(root)),
         QStringLiteral("新建工程成功"));
    onAddModule();
}

void MainWindow::onOpenProject()
{
    const QString dir = QFileDialog::getExistingDirectory(
        this, QStringLiteral("选择工程目录（包含 project.json 的目录）"));
    if (dir.isEmpty())
        return;
    if (!maybeSaveAnnotation())
        return;

    m_project->closeProject();
    openProject(QDir::fromNativeSeparators(dir));
}

void MainWindow::onCloseProject()
{
    if (!maybeSaveAnnotation())
        return;
    m_project->closeProject();
    m_appliedModule = -1;
    m_currentSample = -1;
}

bool MainWindow::ensureProjectOpen()
{
    if (m_project->isOpen())
        return true;
    warn(QStringLiteral("请先新建或打开一个工程。"));
    return false;
}

bool MainWindow::ensureModule()
{
    if (!ensureProjectOpen())
        return false;
    if (m_project->currentModule())
        return true;
    warn(QStringLiteral("当前工程还没有模块。请点「＋ 添加模块」建立分类 / 检测 / 分割模块。"));
    return false;
}

bool MainWindow::ensureSample()
{
    if (!ensureModule())
        return false;
    if (m_currentSample >= 0 && m_currentSample < m_project->imageCount())
        return true;
    warn(QStringLiteral("请先在左侧图像池中选择一张图像。"));
    return false;
}

// ===========================================================================
//  模块
// ===========================================================================
void MainWindow::onAddModule()
{
    if (!ensureProjectOpen())
        return;
    if (!maybeSaveAnnotation())
        return;

    AddModuleDialog dlg(this);
    if (dlg.exec() != QDialog::Accepted)
        return;

    QString err;
    int index = -1;
    if (!m_project->addModule(dlg.moduleType(), dlg.moduleName(), &err, &index)) {
        warn(err, QStringLiteral("无法添加模块"));
        return;
    }

    QString se;
    m_project->setCurrentModule(index, &se);
    m_appliedModule = -1;
    applyModuleToUi();
    refreshSampleList();
    statusBar()->showMessage(QStringLiteral("已添加模块：%1").arg(dlg.moduleName()), 5000);
}

void MainWindow::onRemoveModule()
{
    if (!ensureModule())
        return;
    if (!maybeSaveAnnotation())
        return;

    const int idx = m_project->currentModuleIndex();
    const int total = m_project->moduleCount();

    QStringList willRemove;
    for (int i = idx; i < total; ++i) {
        willRemove << QStringLiteral("· %1（%2）")
                          .arg(m_project->moduleAt(i).name(),
                               taskTypeName(m_project->moduleAt(i).type()));
    }

    const QString msg = QStringLiteral(
                            "删除模块会连同它之后的所有模块一起删除，"
                            "包括它们的数据、模型与测试结果。\n\n"
                            "将从第 %1 个模块开始删除，共 %2 个：\n%3\n\n"
                            "该操作不可撤销。确定继续吗？")
                            .arg(idx + 1)
                            .arg(total - idx)
                            .arg(willRemove.join(QLatin1Char('\n')));

    if (QMessageBox::warning(this, QStringLiteral("删除模块"), msg,
                             QMessageBox::Yes | QMessageBox::No, QMessageBox::No)
        != QMessageBox::Yes) {
        return;
    }

    QString err;
    if (!m_project->removeModule(idx, &err)) {
        warn(err, QStringLiteral("删除模块失败"));
        return;
    }

    m_appliedModule = -1;
    m_currentSample = -1;
    applyModuleToUi();
    refreshSampleList();

    if (m_project->imageCount() > 0)
        loadSample(0);
}

void MainWindow::onModuleSelected(int index)
{
    if (!m_project || !m_project->isOpen())
        return;
    if (index == m_project->currentModuleIndex())
        return;
    if (!maybeSaveAnnotation())
        return;

    QString err;
    if (!m_project->setCurrentModule(index, &err)) {
        warn(err);
        return;
    }

    m_appliedModule = -1;
    applyModuleToUi();
    refreshSampleList();

    // 重新载入当前图在该模块下的标注
    if (m_currentSample >= 0)
        loadSample(m_currentSample);
}

void MainWindow::onRenameModule(int index)
{
    if (!m_project || !m_project->isOpen())
        return;
    if (index != m_project->currentModuleIndex())
        return;

    const Module *m = m_project->currentModule();
    if (!m)
        return;

    bool ok = false;
    const QString name = QInputDialog::getText(this, QStringLiteral("重命名模块"),
                                               QStringLiteral("模块名称："),
                                               QLineEdit::Normal, m->name(), &ok);
    if (!ok || name.trimmed().isEmpty())
        return;

    QString err;
    if (!m_project->renameCurrentModule(name, &err)) {
        warn(err, QStringLiteral("重命名失败"));
        return;
    }

    refreshModuleBar();
    updateWindowTitle();
    updateStatsLabel();
    statusBar()->showMessage(QStringLiteral("模块已重命名为：%1").arg(name.trimmed()), 5000);
}

// ===========================================================================
//  图像池
// ===========================================================================
void MainWindow::onImportImages()
{
    if (!ensureProjectOpen() || !ensureModule())
        return;

    const QStringList files = QFileDialog::getOpenFileNames(
        this, QStringLiteral("选择要导入的图像"), QString(),
        QStringLiteral("图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有文件 (*.*)"));
    if (files.isEmpty())
        return;

    int added = 0;
    QStringList failed;
    QString err;
    m_project->importImages(files, &err, &added, &failed);
    refreshSampleList();

    QString msg = QStringLiteral("成功导入 %1 张图像。").arg(added);
    if (!failed.isEmpty()) {
        msg += QStringLiteral("\n\n以下文件未能导入：\n%1")
                   .arg(failed.mid(0, 10).join(QLatin1Char('\n')));
        if (failed.size() > 10)
            msg += QStringLiteral("\n… 共 %1 个文件失败").arg(failed.size());
    }
    info(msg, QStringLiteral("导入图像"));

    if (added > 0 && m_currentSample < 0)
        loadSample(0);
    updateStatsLabel();
    updateHintBar();
}

void MainWindow::onImportFolder()
{
    if (!ensureProjectOpen() || !ensureModule())
        return;

    const QString dir = QFileDialog::getExistingDirectory(this,
                                                         QStringLiteral("选择包含图像的文件夹"));
    if (dir.isEmpty())
        return;

    int added = 0;
    QStringList failed;
    QString err;
    if (!m_project->importFolder(QDir::fromNativeSeparators(dir), &err, &added, &failed)) {
        warn(err, QStringLiteral("导入文件夹"));
        return;
    }

    refreshSampleList();
    QString msg = QStringLiteral("成功导入 %1 张图像。").arg(added);
    if (!failed.isEmpty())
        msg += QStringLiteral("\n\n有 %1 个文件未能导入。").arg(failed.size());
    info(msg, QStringLiteral("导入文件夹"));

    if (added > 0 && m_currentSample < 0)
        loadSample(0);
    updateStatsLabel();
    updateHintBar();
}

void MainWindow::onSampleActivated(int index)
{
    if (index == m_currentSample)
        return;
    loadSample(index);
}

void MainWindow::onPrevSample()
{
    if (!ensureSample())
        return;
    if (m_currentSample > 0) {
        if (maybeSaveAnnotation())
            loadSample(m_currentSample - 1);
    }
}

void MainWindow::onNextSample()
{
    if (!ensureSample())
        return;
    if (m_currentSample + 1 < m_project->imageCount()) {
        if (maybeSaveAnnotation())
            loadSample(m_currentSample + 1);
    }
}

void MainWindow::onSplitRequested(const QVector<int> &indexes, int split)
{
    if (!m_project || !m_project->isOpen() || indexes.isEmpty())
        return;
    m_project->setSplit(indexes, Split(split));
    refreshSampleList();
    updateStatsLabel();
    updateHintBar();
    statusBar()->showMessage(QStringLiteral("已把 %1 张图像设为「%2」")
                                 .arg(indexes.size())
                                 .arg(splitName(Split(split))),
                             5000);
}

void MainWindow::onOkRequested(const QVector<int> &indexes, bool ok)
{
    if (!m_project || !m_project->isOpen() || indexes.isEmpty())
        return;
    m_project->setOk(indexes, ok);
    refreshSampleList();
    updateStatsLabel();
    statusBar()->showMessage(QStringLiteral("已%1 %2 张图像的 OK 图标记")
                                 .arg(ok ? QStringLiteral("设置") : QStringLiteral("取消"))
                                 .arg(indexes.size()),
                             5000);
}

void MainWindow::onRemoveImagesRequested(const QVector<int> &indexes)
{
    if (!ensureProjectOpen() || indexes.isEmpty())
        return;

    if (QMessageBox::question(this, QStringLiteral("删除图像"),
                              QStringLiteral("确定从工程中删除选中的 %1 张图像吗？\n\n"
                                             "图像文件以及各模块下它们的标注与测试结果都会被删除。\n"
                                             "该操作不可撤销。")
                                  .arg(indexes.size()))
        != QMessageBox::Yes) {
        return;
    }

    // 从后往前删，避免下标错位
    QVector<int> sorted = indexes;
    std::sort(sorted.begin(), sorted.end(), std::greater<int>());

    QStringList errors;
    for (int idx : sorted) {
        QString err;
        if (!m_project->removeImage(idx, &err))
            errors << err;
    }

    if (m_currentSample >= m_project->imageCount())
        m_currentSample = m_project->imageCount() - 1;

    refreshSampleList();
    updateStatsLabel();

    if (!errors.isEmpty())
        warn(errors.join(QLatin1Char('\n')), QStringLiteral("部分图像删除失败"));

    if (m_project->imageCount() > 0) {
        loadSample(qMax(0, m_currentSample));
    } else {
        m_currentSample = -1;
        m_canvas->setImage(cv::Mat(), QString());
        m_canvas->setObjects({}, AnnoState::None);
        m_canvas->clearResultOverlay();
        refreshObjectPanel();
        updateHintBar();
        updateActionStates();
    }
}

void MainWindow::onOpenImageFolder(int)
{
    if (!m_project || !m_project->isOpen())
        return;
    QDesktopServices::openUrl(QUrl::fromLocalFile(m_project->imagesDir()));
}

// ===========================================================================
//  标注流程
// ===========================================================================
void MainWindow::loadSample(int index)
{
    if (!m_project || !m_project->isOpen())
        return;
    if (index < 0 || index >= m_project->imageCount())
        return;

    m_loadingSample = true;

    m_currentSample = index;
    const QString path = m_project->imagePaths().at(index);
    const QString key = m_project->imageKey(index);

    const cv::Mat img = imageio::imreadUnicode(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        m_loadingSample = false;
        warn(QStringLiteral("无法读取图像：\n%1").arg(QDir::toNativeSeparators(path)),
             QStringLiteral("读取失败"));
        return;
    }

    m_canvas->setImage(img, key);

    // 载入该图在当前模块下的标注
    QVector<AnnoObject> objects;
    int classId = -1;
    AnnoState state = AnnoState::None;
    m_project->loadAnnotation(index, &objects, &classId, &state, nullptr);

    m_canvas->setObjects(objects, state);

    // 分类模块：类别既是「标注」也是「当前类别」
    if (m_canvas->task() == TaskType::Classification) {
        const Module *m = m_project->currentModule();
        int want = classId;
        if (want < 0 || !m || want >= m->classCount())
            want = m ? m->firstTargetClass() : 0;
        m_canvas->setCurrentClass(want);
    }

    m_canvas->markSaved();
    m_canvas->clearHistory();

    m_loadingSample = false;

    m_imageList->setCurrentIndex(index);
    refreshObjectPanel();
    refreshClassPanel();
    refreshDisplayMode();
    updateActionStates();
    updateWindowTitle();
    updateHintBar();

    // 切图后把焦点交给画布，保证 V/B/E/G/R/F/H、A/D、数字键、S 立刻可用
    m_canvas->setFocus();
}

bool MainWindow::maybeSaveAnnotation()
{
    if (!m_project || !m_project->isOpen() || !m_canvas || !m_canvas->hasImage())
        return true;
    if (!m_canvas->isModified())
        return true;

    const auto answer = QMessageBox::question(
        this, QStringLiteral("未保存的标注"),
        QStringLiteral("「%1」的标注尚未保存。要先保存吗？").arg(m_canvas->imageName()),
        QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);

    if (answer == QMessageBox::Cancel)
        return false;
    if (answer == QMessageBox::Save) {
        saveCurrentAnnotation();
        return !m_canvas->isModified();
    }
    return true;
}

void MainWindow::saveCurrentAnnotation()
{
    if (!ensureSample())
        return;

    const Module *m = m_project->currentModule();
    if (!m)
        return;

    const bool isCls = (m->type() == TaskType::Classification);

    QString err;
    if (isCls) {
        if (m_canvas->classId() < 0) {
            warn(QStringLiteral("请先在右侧「标签类别」中点选一个类别，再保存分类标定。"));
            return;
        }
        if (!m_project->saveAnnotation(m_currentSample, {}, m_canvas->classId(),
                                       AnnoState::Done, &err)) {
            warn(err, QStringLiteral("保存失败"));
            return;
        }
    } else {
        if (!m_project->saveAnnotation(m_currentSample, m_canvas->objects(), -1,
                                       AnnoState::Done, &err)) {
            warn(err, QStringLiteral("保存失败"));
            return;
        }
    }

    m_canvas->markSaved();
    m_canvas->setAnnoState(AnnoState::Done);
    refreshSampleList();
    updateStatsLabel();
    updateWindowTitle();
    statusBar()->showMessage(QStringLiteral("已保存标注：%1").arg(m_canvas->imageName()), 4000);
}

void MainWindow::onSaveAnnotation()
{
    if (!ensureSample())
        return;
    saveCurrentAnnotation();
}

void MainWindow::onSaveAnnotationAndNext()
{
    if (!ensureModule())
        return;

    if (m_canvas->hasImage())
        saveCurrentAnnotation();

    // 跳到下一张「未标注」的图；没有就跳到下一张任意图
    const int count = m_project->imageCount();
    if (count <= 0)
        return;

    for (int step = 1; step <= count; ++step) {
        const int idx = (m_currentSample + step) % count;
        if (m_project->sampleState(idx).state == AnnoState::None) {
            loadSample(idx);
            statusBar()->showMessage(QStringLiteral("已跳到下一张未标注图像。"), 3000);
            return;
        }
    }

    if (m_currentSample + 1 < count)
        loadSample(m_currentSample + 1);
    else
        statusBar()->showMessage(QStringLiteral("本模块下已没有未标注的图像了。"), 4000);
}

void MainWindow::onClearAnnotation()
{
    if (!ensureSample())
        return;
    m_canvas->clearAllObjects();
    if (m_canvas->task() == TaskType::Classification)
        m_canvas->setAnnoState(AnnoState::None);
    refreshObjectPanel();
    statusBar()->showMessage(QStringLiteral("已清空本图标注（Ctrl+Z 可撤回）。"), 4000);
}

void MainWindow::onDeleteAnnotationFile()
{
    if (!ensureSample())
        return;

    if (QMessageBox::question(this, QStringLiteral("删除标注文件"),
                              QStringLiteral("从磁盘删除「%1」在本模块下的标注文件吗？"
                                             "（分割任务还会删除掩码文件）")
                                  .arg(m_canvas->imageName()))
        != QMessageBox::Yes) {
        return;
    }

    QString err;
    if (!m_project->removeAnnotationFile(m_currentSample, &err)) {
        warn(err, QStringLiteral("删除标注文件失败"));
        return;
    }

    m_canvas->setObjects({}, AnnoState::None);
    m_canvas->markSaved();
    m_canvas->clearHistory();
    refreshObjectPanel();
    refreshSampleList();
    updateStatsLabel();
    statusBar()->showMessage(QStringLiteral("已删除标注文件。"), 4000);
}

void MainWindow::onSetCurrentImageOk()
{
    if (!ensureSample())
        return;
    SampleState st = m_project->sampleState(m_currentSample);
    st.ok = !st.ok;
    m_project->setSampleState(m_currentSample, st);
    refreshSampleList();
    updateStatsLabel();
    statusBar()->showMessage(st.ok ? QStringLiteral("已设为 OK 图（无缺陷正例）。")
                                   : QStringLiteral("已取消 OK 图标记。"),
                             4000);
}

void MainWindow::onCopySelectedObject()
{
    if (!m_canvas)
        return;
    const int sel = m_canvas->selectedObject();
    if (sel < 0 || sel >= m_canvas->objects().size()) {
        statusBar()->showMessage(QStringLiteral("请先选中一个标注对象再复制。"), 4000);
        return;
    }
    m_clipboard.clear();
    m_clipboard.append(m_canvas->objects().at(sel));
    updateActionStates();
    statusBar()->showMessage(QStringLiteral("已复制 1 个标注对象（可切到别的图 Ctrl+V 粘贴）。"), 4000);
}

void MainWindow::onPasteObjects()
{
    if (!ensureSample() || m_clipboard.isEmpty())
        return;
    m_canvas->pasteObjects(m_clipboard);
    refreshObjectPanel();
}

void MainWindow::onConfirmPreAnnotation()
{
    if (!ensureSample())
        return;
    m_canvas->confirmPreAnnotation();
    saveCurrentAnnotation();
    refreshObjectPanel();
}

void MainWindow::onToolChanged(QAction *)
{
    if (!m_canvas)
        return;

    QAction *a = m_toolGroup->checkedAction();
    if (!a) {
        // 以画布当前工具为准（模块切换后会重置）
        for (QAction *c : m_toolGroup->actions()) {
            if (c->data().toInt() == int(m_canvas->tool())) {
                c->setChecked(true);
                a = c;
                break;
            }
        }
        if (!a)
            return;
    }

    m_canvas->setTool(static_cast<ImageCanvas::Tool>(a->data().toInt()));

    // 用「背景」类作画不会产生有效目标，这里提前给出一次明确提示
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (m && m->hasBackground() && m_canvas->currentClass() == 0
        && m_canvas->tool() != ImageCanvas::ToolSelect
        && m_canvas->tool() != ImageCanvas::ToolPan) {
        statusBar()->showMessage(QStringLiteral("提示：当前类别是「背景」，"
                                               "用它标注不会产生有效目标。请先选择目标类别。"),
                                 8000);
    }
}

void MainWindow::onClassPicked(int classId)
{
    if (m_loadingSample || !m_canvas)
        return;

    m_canvas->setCurrentClass(classId);

    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (!m)
        return;

    // 分类模块：点选类别即整图标定，立即保存并跳到下一张未标注的图
    if (m->type() == TaskType::Classification && m_canvas->hasImage()) {
        m_canvas->setAnnoState(AnnoState::Done);
        saveCurrentAnnotation();
        onSaveAnnotationAndNext();
    }
}

void MainWindow::onClassesModified()
{
    if (!m_project || !m_project->isOpen())
        return;

    if (!m_project->setCurrentClasses(m_classPanel->classes(), nullptr))
        return;

    const Module *m = m_project->currentModule();
    if (!m)
        return;

    if (m_canvas) {
        m_canvas->setClasses(m->classes());
        m_canvas->setHasBackground(m->hasBackground());
    }
    refreshClassPanel();
    refreshObjectPanel();
    updateHintBar();

    // 类别数变化会影响模型输出层，提示需要重新训练
    if (m_project->hasModel()) {
        statusBar()->showMessage(QStringLiteral("类别已修改：模型输出层与类别数不再匹配，"
                                                "请重新训练后再使用。"),
                                 10000);
    }
}

void MainWindow::onShowOverlayToggled(bool on)
{
    if (m_canvas)
        m_canvas->setShowOverlay(on);
}

void MainWindow::onBrushSizeChanged(int px)
{
    if (m_canvas)
        m_canvas->setBrushSize(px);
}

void MainWindow::onFixedRectSizeChanged()
{
    if (m_canvas)
        m_canvas->setFixedRectSize(QSize(m_spFixedW->value(), m_spFixedH->value()));
}

void MainWindow::onOpacityChanged(int alpha)
{
    if (m_canvas)
        m_canvas->setOverlayOpacity(alpha);
}

void MainWindow::onDisplayModeChanged(int)
{
    refreshDisplayMode();
}

// ===========================================================================
//  对象列表
// ===========================================================================
void MainWindow::onObjectSelected(int index)
{
    if (m_canvas)
        m_canvas->setSelectedObject(index);
}

void MainWindow::onObjectClassChange(int index, int classId)
{
    if (!m_canvas)
        return;
    m_canvas->setSelectedObject(index);
    m_canvas->setSelectedObjectClass(classId);
    refreshObjectPanel();
}

void MainWindow::onObjectDelete(int index)
{
    if (!m_canvas)
        return;
    m_canvas->setSelectedObject(index);
    m_canvas->removeSelectedObject();
    refreshObjectPanel();
}

void MainWindow::onObjectCopy(int index)
{
    if (!m_canvas)
        return;
    if (index < 0 || index >= m_canvas->objects().size())
        return;
    m_clipboard.clear();
    m_clipboard.append(m_canvas->objects().at(index));
    updateActionStates();
    statusBar()->showMessage(QStringLiteral("已复制对象，可 Ctrl+V 粘贴。"), 4000);
}

// ===========================================================================
//  画布信号
// ===========================================================================
void MainWindow::onCanvasModified(bool)
{
    updateWindowTitle();
    updateActionStates();
    refreshSampleList();
}

void MainWindow::onCanvasZoomChanged(double scale)
{
    if (m_lblZoom)
        m_lblZoom->setText(QStringLiteral("缩放 %1%").arg(int(scale * 100)));
}

void MainWindow::onCanvasMousePos(const QPoint &imgPos, int gray)
{
    if (!m_lblPos)
        return;
    if (gray < 0)
        m_lblPos->setText(QStringLiteral("(%1, %2) 图像外").arg(imgPos.x()).arg(imgPos.y()));
    else
        m_lblPos->setText(QStringLiteral("(%1, %2)  灰度 %3")
                              .arg(imgPos.x())
                              .arg(imgPos.y())
                              .arg(gray));
}

void MainWindow::onCanvasMouseLeft()
{
    if (m_lblPos)
        m_lblPos->setText(QString());
}

void MainWindow::onCanvasClassChanged(int classId)
{
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (m_classPanel && m && classId != m_classPanel->currentClass())
        m_classPanel->setCurrentClass(classId);

    if (m_lblClass) {
        const QString name = (m && classId >= 0 && classId < m->classCount())
                                 ? m->className(classId)
                                 : QStringLiteral("—");
        const bool bg = m && m->hasBackground() && classId == 0;
        m_lblClass->setText(bg ? QStringLiteral("当前类别：%1（背景，不可用于标注）").arg(name)
                               : QStringLiteral("当前类别：%1").arg(name));
    }
    if (m_canvas)
        m_canvas->setClassId(classId);
}

void MainWindow::onCanvasMessage(const QString &text)
{
    // 画布的引导语里带少量 HTML 标记（例如 <b>还差一步：</b>），
    // 而状态栏是纯文本 QLabel —— 直接塞进去会把标签原样显示出来（实测过）。
    // 这里剥掉标记再显示；带格式的引导语由上方引导提示栏（富文本）负责。
    QString plain = text;
    plain.remove(QRegularExpression(QStringLiteral("<[^>]*>")));
    plain.replace(QLatin1String("&nbsp;"), QLatin1String(" "));
    statusBar()->showMessage(plain, 6000);
}

void MainWindow::onCanvasSelectionChanged(int index)
{
    if (m_objectPanel)
        m_objectPanel->setSelected(index);
    updateActionStates();
}

void MainWindow::onCanvasObjectsChanged()
{
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (m_objectPanel && m)
        m_objectPanel->setObjects(m_canvas->objects(), m->classes(), m->hasBackground());
}

void MainWindow::onCanvasBackgroundBlocked(const QString &text)
{
    statusBar()->showMessage(text, 8000);
}

void MainWindow::onCanvasContextCommand(const QString &command)
{
    if (command != QLatin1String("showMenu"))
        return;

    QMenu menu(this);
    QAction *aCopy = (m_canvas->selectedObject() >= 0)
                         ? menu.addAction(QStringLiteral("复制选中对象"))
                         : nullptr;
    QAction *aPaste = !m_clipboard.isEmpty() ? menu.addAction(QStringLiteral("粘贴对象"))
                                             : nullptr;
    menu.addSeparator();
    QAction *aConfirm = menu.addAction(QStringLiteral("确认预标注"));
    QAction *aClear = menu.addAction(QStringLiteral("清空本图标注"));
    QAction *aDeleteFile = menu.addAction(QStringLiteral("删除本图标注文件"));
    QAction *aOk = menu.addAction(QStringLiteral("设为 / 取消 OK 图"));
    menu.addSeparator();
    QAction *aFit = menu.addAction(QStringLiteral("适应窗口"));
    QAction *aReset = menu.addAction(QStringLiteral("1:1"));

    QAction *chosen = menu.exec(QCursor::pos());
    if (!chosen)
        return;

    if (aCopy && chosen == aCopy)
        onCopySelectedObject();
    else if (aPaste && chosen == aPaste)
        onPasteObjects();
    else if (chosen == aConfirm)
        onConfirmPreAnnotation();
    else if (chosen == aClear)
        onClearAnnotation();
    else if (chosen == aDeleteFile)
        onDeleteAnnotationFile();
    else if (chosen == aOk)
        onSetCurrentImageOk();
    else if (chosen == aFit)
        onZoomFit();
    else if (chosen == aReset)
        onZoomReset();
}

// ===========================================================================
//  视图
// ===========================================================================
void MainWindow::onZoomIn()    { if (m_canvas) m_canvas->zoomIn(); }
void MainWindow::onZoomOut()   { if (m_canvas) m_canvas->zoomOut(); }
void MainWindow::onZoomReset() { if (m_canvas) m_canvas->zoomReset(); }
void MainWindow::onZoomFit()   { if (m_canvas) m_canvas->zoomFit(); }

// ===========================================================================
//  测试结果叠加
// ===========================================================================
void MainWindow::openResultOverlayFor(int index)
{
    if (!m_canvas || !m_project || index < 0)
        return;

    PredictionResult pr;
    if (!m_project->loadPrediction(index, &pr, nullptr)) {
        statusBar()->showMessage(QStringLiteral("该图没有测试结果，请先训练或执行「模型测试」。"), 6000);
        return;
    }

    cv::Mat vis = m_canvas->image();
    if (vis.empty())
        return;
    vis = vis.clone();

    const Module *m = m_project->currentModule();
    const QVector<ClassInfo> classes = m ? m->classes() : QVector<ClassInfo>();

    auto className = [&](int id) {
        if (id >= 0 && id < classes.size())
            return classes.at(id).name;
        return QStringLiteral("类别%1").arg(id);
    };
    auto classColor = [&](int id) {
        const QColor c = (id >= 0 && id < classes.size()) ? classes.at(id).color
                                                          : defaultClassColor(id);
        return cv::Scalar(c.blue(), c.green(), c.red());
    };

    if (pr.task == TaskType::Classification) {
        const QString text = QStringLiteral("%1 %2%")
                                 .arg(className(pr.classId))
                                 .arg(pr.score * 100.0, 0, 'f', 1);
        cv::putText(vis, text.toStdString(), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, classColor(pr.classId), 2);
    } else if (pr.task == TaskType::Detection) {
        for (const DetectionBox &b : pr.boxes) {
            cv::rectangle(vis,
                          cv::Rect(int(b.rect.x()), int(b.rect.y()),
                                   int(b.rect.width()), int(b.rect.height())),
                          classColor(b.classId), 2);
            const QString label = QStringLiteral("%1 %2%")
                                      .arg(className(b.classId))
                                      .arg(b.score * 100.0, 0, 'f', 1);
            cv::putText(vis, label.toStdString(),
                        cv::Point(int(b.rect.x()), std::max(12, int(b.rect.y()) - 4)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, classColor(b.classId), 1);
        }
    } else if (!pr.mask.empty()) {
        const cv::Mat color = imageio::colorizeMask(pr.mask, classes, 1.0);
        if (!color.empty() && color.size() == vis.size())
            cv::addWeighted(vis, 0.65, color, 0.35, 0.0, vis);
    }

    m_canvas->setResultOverlay(imageio::matToQImage(vis));
}

// ===========================================================================
//  训练与评价
// ===========================================================================
void MainWindow::onShowParams()
{
    if (m_dockParams) {
        m_dockParams->show();
        m_dockParams->raise();
    }
}

void MainWindow::onStartTraining()
{
    if (!ensureModule())
        return;

    const Module *m0 = m_project->currentModule();
    if (!m0)
        return;

    // 把界面上的参数写回模块后再校验
    if (m_paramPanel) {
        m_project->setCurrentConfig(m_paramPanel->trainConfig(m0->train()),
                                    m_paramPanel->testConfig(m0->test()), nullptr);
    }
    // ensureDefaultClasses 已由 setCurrentConfig → saveCurrentModule 之前的 setClasses 保证；
    // 这里再取一次最新的模块指针
    const Module *m = m_project->currentModule();
    if (!m)
        return;

    if (m->targetClassCount() < 1) {
        warn(QStringLiteral("当前模块还没有可训练的目标类别，请先在「标签类别」中添加。"));
        return;
    }

    const QVector<TrainSample> samples = m_project->trainSamples();

    // 样本量下限：不足时给出明确原因，而不是静默失败
    if (samples.size() < 2) {
        warn(QStringLiteral("%1至少需要 2 张「已标注」且「已加入训练集」的图像，当前只有 %2 张。\n\n"
                            "在左侧图像池中选中图像 → 右键「加入训练集」。\n"
                            "若已标注却显示为 0，请确认标注已保存（Ctrl+S），"
                            "以及标注状态不是「预标注」。")
                 .arg(taskTypeName(m->type()))
                 .arg(samples.size()),
             QStringLiteral("训练样本不足"));
        return;
    }

    TrainConfig cfg = m->train();
    cfg.task        = m->type();
    cfg.numClasses  = m->targetClassCount();
    cfg.datasetRoot = m->root();
    cfg.outputDir   = m->modelsDir();

    // 后处理阈值统一由模块的「测试参数」支配：
    // 训练后端从 TrainConfig::extra 里取这些值，训练与推理阶段才会一致。
    cfg.extra.insert(QStringLiteral("scoreThreshold"), m->test().scoreThreshold);
    cfg.extra.insert(QStringLiteral("nmsThreshold"), m->test().nmsThreshold);
    cfg.extra.insert(QStringLiteral("minArea"), m->test().minArea);
    cfg.extra.insert(QStringLiteral("iouThreshold"), m->test().iouThreshold);
    if (!cfg.extra.contains(QStringLiteral("valSplit")))
        cfg.extra.insert(QStringLiteral("valSplit"), 0.2);

    if (m_trainer) {
        m_trainer->deleteLater();
        m_trainer = nullptr;
    }

    m_trainer = TrainerFactory::create(this, TrainerFactory::fromKey(cfg.backend));
    if (!m_trainer) {
        warn(QStringLiteral("没有可用的训练后端。请查看「帮助 → 训练后端信息」。"));
        return;
    }

    QString err;
    if (!m_trainer->initialize(cfg, samples, m->classes(), &err)) {
        warn(err, QStringLiteral("无法开始训练"));
        m_trainer->deleteLater();
        m_trainer = nullptr;
        return;
    }

    if (m_monitor)
        m_monitor->deleteLater();

    m_monitor = new TrainMonitorDialog(
        QStringLiteral("训练监控 —— %1").arg(m->name()),
        QStringLiteral("后端：%1　训练样本：%2 张<br>%3")
            .arg(m_trainer->backendName())
            .arg(samples.size())
            .arg(cfg.summary()),
        this);

    if (m->type() == TaskType::Segmentation)
        m_monitor->setMetricName(QStringLiteral("像素准确率 / mIoU"));
    else if (m->type() == TaskType::Detection)
        m_monitor->setMetricName(QStringLiteral("验证准确率"));
    else
        m_monitor->setMetricName(QStringLiteral("准确率"));

    // OpenCV 轻量后端把训练拆成阶段上报，横轴不是 epoch
    m_monitor->setXAxisLabel(cfg.backend == QLatin1String("opencv") ? QStringLiteral("阶段")
                                                                   : QStringLiteral("Epoch"));

    connect(m_trainer, &ITrainer::logMessage, this,
            [this](const QString &text, int level) {
                if (m_monitor)
                    m_monitor->appendLog(text, level);
            });
    connect(m_trainer, &ITrainer::progressChanged, this, [this](int percent) {
        if (m_monitor) {
            m_monitor->setProgress(percent);
            m_monitor->setStage(QStringLiteral("训练中… %1%").arg(percent));
        }
    });
    connect(m_trainer, &ITrainer::epochFinished, this,
            [this](int epoch, int total, double loss, double metric, double valLoss,
                   double valMetric) {
                if (m_monitor)
                    m_monitor->addPoint(epoch, total, loss, metric, valLoss, valMetric);
            });
    connect(m_trainer, &ITrainer::trainingFinished, this, &MainWindow::onTrainFinished);
    connect(m_monitor, &TrainMonitorDialog::stopClicked, this, [this]() {
        if (m_trainer)
            m_trainer->requestStop();
    });

    m_monitor->show();
    m_trainer->start();
}

void MainWindow::onTrainFinished(bool ok, const QString &modelPath, const QString &message)
{
    if (m_monitor)
        m_monitor->setFinished(ok, message);

    if (!ok) {
        statusBar()->showMessage(QStringLiteral("训练未完成：%1").arg(message), 12000);
        if (m_trainer) {
            m_trainer->deleteLater();
            m_trainer = nullptr;
        }
        updateActionStates();
        return;
    }

    statusBar()->showMessage(QStringLiteral("训练完成：%1").arg(QDir::toNativeSeparators(modelPath)),
                             10000);

    if (m_display)
        m_display->setTestResultAvailable(true);

    // 训练完成后自动在测试集上评价（AIDI 的「训练完成后会自动进行一次测试」）
    evaluateModel(modelPath);

    if (m_trainer) {
        m_trainer->deleteLater();
        m_trainer = nullptr;
    }
    updateActionStates();
}

bool MainWindow::evaluateModel(const QString &modelPath)
{
    const Module *m = m_project ? m_project->currentModule() : nullptr;
    if (!m || !m_monitor)
        return false;

    const QVector<TrainSample> testSamples = m_project->testSamples();
    if (testSamples.isEmpty()) {
        m_monitor->appendLog(QStringLiteral("没有测试集样本，跳过自动评价。"
                                           "请在左侧图像池中多选图像 → 右键「加入测试集」。"),
                             ITrainer::Warning);
        m_monitor->setStage(QStringLiteral("训练完成（未做评价：测试集为空）"));
        return false;
    }

    m_monitor->appendLog(QStringLiteral("开始在测试集上评价（%1 张）…").arg(testSamples.size()),
                         ITrainer::Info);
    m_monitor->setStage(QStringLiteral("正在测试集上评价…"));

    Evaluator ev;
    connect(&ev, &Evaluator::progress, this, [this](int done, int total, const QString &key) {
        if (!m_monitor)
            return;
        m_monitor->setProgress(total > 0 ? int(100.0 * done / total) : 0);
        m_monitor->setStage(QStringLiteral("评价中 %1 / %2：%3").arg(done).arg(total).arg(key));
        // 评价是同步执行的，这里让界面有机会重绘
        QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    });
    connect(&ev, &Evaluator::logMessage, this, [this](const QString &text, int level) {
        if (m_monitor)
            m_monitor->appendLog(text, level);
    });

    Metrics mt;
    QVector<Evaluator::Item> items;
    QString err;
    if (!ev.evaluate(*m, testSamples, modelPath, &mt, &items, &err)) {
        m_monitor->appendLog(QStringLiteral("评价失败：%1").arg(err), ITrainer::Error);
        return false;
    }

    // 每张图的预测结果落盘，供「显示测试结果」叠加
    for (const Evaluator::Item &item : items) {
        const int idx = m_project->indexOfKey(item.sample.imageKey);
        if (idx >= 0)
            m_project->savePrediction(idx, item.pred, nullptr);
    }

    m_project->saveMetrics(mt, nullptr);
    refreshMetrics();

    if (m_display)
        m_display->setTestResultAvailable(true);

    if (m_monitor) {
        m_monitor->appendLog(mt.summary(), ITrainer::Success);
        for (const ClassMetric &cm : mt.perClass) {
            if (cm.tp == 0 && cm.fp == 0 && cm.fn == 0)
                continue;
            m_monitor->appendLog(
                QStringLiteral("  %1. %2：TP %3  FP %4  FN %5  精确率 %6%  召回率 %7%")
                    .arg(cm.classId)
                    .arg(cm.name)
                    .arg(cm.tp)
                    .arg(cm.fp)
                    .arg(cm.fn)
                    .arg(cm.precision * 100.0, 0, 'f', 2)
                    .arg(cm.recall * 100.0, 0, 'f', 2),
                ITrainer::Info);
        }
        m_monitor->setStage(QStringLiteral("评价完成。"));
        m_monitor->setProgress(100);
    }

    statusBar()->showMessage(QStringLiteral("评价完成：%1").arg(mt.summary()), 12000);
    return true;
}

void MainWindow::onEvaluate()
{
    if (!ensureModule())
        return;

    const Module *m = m_project->currentModule();
    if (!m)
        return;
    if (!QFileInfo::exists(m->modelPath())) {
        warn(QStringLiteral("当前模块还没有模型，请先训练。"), QStringLiteral("无法评价"));
        return;
    }

    if (!m_monitor) {
        m_monitor = new TrainMonitorDialog(QStringLiteral("评价结果 —— %1").arg(m->name()),
                                           QStringLiteral("在测试集上评价当前模型。"), this);
        m_monitor->setCanStop(false);
        m_monitor->setCloseableIfRunning(true);
    }
    m_monitor->show();
    m_monitor->raise();

    if (evaluateModel(m->modelPath())) {
        if (m_display) {
            m_display->setTestResultAvailable(true);
            m_display->setMode(DisplayControl::TestResult);
        }
        refreshDisplayMode();
    }
}

// ===========================================================================
//  智能标注
// ===========================================================================
void MainWindow::onSmartAnnotate()
{
    if (!ensureModule())
        return;
    runPreAnnotate(int(PreAnnotateDialog::AllUnannotated), m_imageList->selectedIndexes());
}

void MainWindow::runPreAnnotate(int scope, const QVector<int> &selectedIndexes)
{
    if (!ensureModule())
        return;

    const Module *m = m_project->currentModule();
    if (!m)
        return;

    if (!QFileInfo::exists(m->modelPath())) {
        warn(QStringLiteral("当前模块还没有可用模型，无法智能标注。\n\n"
                            "请先「训练 → 开始训练」，或用「文件 → 导入模型」"
                            "把一个已训练好的模型放进该模块的 models/ 目录。"),
             QStringLiteral("无法智能标注"));
        return;
    }

    // 计算待处理清单
    QVector<int> targets;
    switch (PreAnnotateDialog::Scope(scope)) {
    case PreAnnotateDialog::CurrentImage:
        if (m_currentSample >= 0)
            targets.append(m_currentSample);
        break;
    case PreAnnotateDialog::SelectedImages:
        targets = selectedIndexes;
        break;
    case PreAnnotateDialog::AllUnannotated:
    default:
        targets = m_project->unannotatedIndexes();
        break;
    }

    if (targets.isEmpty()) {
        warn(QStringLiteral("没有需要处理的图像。"), QStringLiteral("智能标注"));
        return;
    }

    auto *dlg = new PreAnnotateDialog(m->name(), m->modelPath(),
                                      m_currentSample >= 0 ? 1 : 0,
                                      selectedIndexes.size(),
                                      m_project->unannotatedIndexes().size(),
                                      true, this);
    dlg->setAttribute(Qt::WA_DeleteOnClose);

    connect(dlg, &PreAnnotateDialog::stopRequested, this, [this]() { m_preStop = true; });

    connect(dlg, &PreAnnotateDialog::startRequested, this, [this, dlg, targets](int) {
        const Module *mm = m_project->currentModule();
        if (!mm) {
            dlg->setFinished(0, targets.size());
            return;
        }

        ITrainer *inf = TrainerFactory::create(this, TrainerFactory::fromKey(mm->train().backend));
        if (!inf) {
            dlg->appendLog(QStringLiteral("没有可用的推理后端。"), ITrainer::Error);
            dlg->setFinished(0, targets.size());
            return;
        }

        QString err;
        if (!inf->loadModel(mm->modelPath(), &err)) {
            dlg->appendLog(QStringLiteral("载入模型失败：%1").arg(err), ITrainer::Error);
            dlg->setFinished(0, targets.size());
            inf->deleteLater();
            return;
        }

        // 预标注只加载模型、不训练，必须显式把模块的测试参数交给后端，
        // 否则「测试参数」里改的置信度 / 最小面积不会生效。
        inf->setTestConfig(mm->test());

        const bool isCls = (mm->type() == TaskType::Classification);

        m_preStop = false;
        dlg->setBusy(true);

        int okCount = 0, failCount = 0;
        for (int n = 0; n < targets.size(); ++n) {
            if (m_preStop) {
                dlg->appendLog(QStringLiteral("已被中止，剩余 %1 张未处理。")
                                   .arg(targets.size() - n),
                               ITrainer::Warning);
                break;
            }

            const int idx = targets.at(n);
            const QString key = m_project->imageKey(idx);
            dlg->setProgress(n, targets.size(), key);

            const cv::Mat img = imageio::imreadUnicode(m_project->imagePaths().at(idx),
                                                       cv::IMREAD_COLOR);
            if (img.empty()) {
                dlg->appendLog(QStringLiteral("[失败] %1：无法读取图像").arg(key), ITrainer::Error);
                ++failCount;
                QApplication::processEvents();
                continue;
            }

            PredictionResult pr;
            if (!inf->predict(img, &pr, &err)) {
                dlg->appendLog(QStringLiteral("[失败] %1：%2").arg(key, err), ITrainer::Error);
                ++failCount;
                QApplication::processEvents();
                continue;
            }

            QString saveErr;
            if (isCls) {
                // 分类模块的预标注就是整图的类别
                if (pr.classId >= 0
                    && m_project->saveAnnotation(idx, {}, pr.classId, AnnoState::Pre, &saveErr)) {
                    dlg->appendLog(QStringLiteral("[完成] %1 → %2（%3%）")
                                       .arg(key)
                                       .arg(mm->className(pr.classId))
                                       .arg(pr.score * 100.0, 0, 'f', 1),
                                   ITrainer::Success);
                    ++okCount;
                } else {
                    dlg->appendLog(QStringLiteral("[失败] %1：%2")
                                       .arg(key,
                                            saveErr.isEmpty() ? QStringLiteral("未得到有效类别")
                                                              : saveErr),
                                   ITrainer::Error);
                    ++failCount;
                }
            } else {
                QVector<AnnoObject> objects;
                if (mm->type() == TaskType::Detection) {
                    for (const DetectionBox &b : pr.boxes) {
                        if (b.score < mm->test().scoreThreshold)
                            continue;
                        AnnoObject o;
                        o.type    = AnnoObject::Rect;
                        o.classId = b.classId;
                        o.rect    = b.rect;
                        o.pre     = true;
                        objects.append(o);
                    }
                } else if (!pr.mask.empty()) {
                    objects = anno::objectsFromMask(pr.mask, mm->test().minArea);
                    for (AnnoObject &o : objects)
                        o.pre = true;
                }

                if (objects.isEmpty()) {
                    // 未检出也要登记为「已处理」，否则会被反复处理
                    dlg->appendLog(QStringLiteral("[跳过] %1：未检出目标").arg(key),
                                   ITrainer::Warning);
                    m_project->saveAnnotation(idx, {}, -1, AnnoState::Done, &saveErr);
                    ++okCount;
                } else if (m_project->saveAnnotation(idx, objects, -1, AnnoState::Pre, &saveErr)) {
                    dlg->appendLog(QStringLiteral("[完成] %1 → %2 个对象")
                                       .arg(key)
                                       .arg(objects.size()),
                                   ITrainer::Success);
                    ++okCount;
                } else {
                    dlg->appendLog(QStringLiteral("[失败] %1：%2").arg(key, saveErr),
                                   ITrainer::Error);
                    ++failCount;
                }
            }

            QApplication::processEvents();
        }

        inf->deleteLater();
        dlg->setProgress(targets.size(), targets.size(), QStringLiteral("完成"));
        dlg->setFinished(okCount, failCount);

        const bool currentTouched = targets.contains(m_currentSample);
        refreshSampleList();
        updateStatsLabel();
        updateHintBar();

        if (currentTouched && m_currentSample >= 0) {
            // 重新载入当前图，让预标注立刻可见
            m_loadingSample = true;
            QVector<AnnoObject> objs;
            int cid = -1;
            AnnoState st = AnnoState::None;
            m_project->loadAnnotation(m_currentSample, &objs, &cid, &st, nullptr);
            m_canvas->setObjects(objs, st);
            if (m_canvas->task() == TaskType::Classification && cid >= 0)
                m_canvas->setCurrentClass(cid);
            m_canvas->markSaved();
            m_loadingSample = false;
            refreshObjectPanel();
        }
    });

    dlg->show();
}

// ===========================================================================
//  测试与导出
// ===========================================================================
void MainWindow::onTestModel()
{
    TestModelDialog dlg(this);

    if (m_project && m_project->isOpen()) {
        const Module *m = m_project->currentModule();
        if (m && QFileInfo::exists(m->modelPath()))
            dlg.setDefaultModel(m->modelPath());
        if (m_currentSample >= 0)
            dlg.setDefaultImage(m_project->imagePaths().at(m_currentSample));
    }

    dlg.exec();
}

void MainWindow::onExportModel()
{
    if (!ensureModule())
        return;

    const Module *m = m_project->currentModule();
    if (!m)
        return;

    if (!QFileInfo::exists(m->modelPath())) {
        warn(QStringLiteral("当前模块还没有模型，请先训练。"), QStringLiteral("无法导出"));
        return;
    }

    ExportModelDialog dlg(*m, m_project->loadMetrics(), this);
    dlg.exec();
}

void MainWindow::onImportModel()
{
    if (!ensureModule())
        return;

    const QString f = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择模型文件"), QString(),
        QStringLiteral("AqroTrain 模型 (*.clsmodel *.detmodel *.segmodel);;所有文件 (*.*)"));
    if (f.isEmpty())
        return;

    const QFileInfo src(QDir::fromNativeSeparators(f));
    const Module *m = m_project->currentModule();
    if (!m)
        return;

    if (src.suffix().toLower() != modelFileSuffix(m->type())) {
        const auto answer = QMessageBox::question(
            this, QStringLiteral("模型类型不匹配"),
            QStringLiteral("该模型的后缀是 .%1，而当前模块是「%2」（期望 .%3）。\n\n"
                           "仍然导入吗？（导入后可能无法在该模块下正确推理）")
                .arg(src.suffix(), taskTypeName(m->type()), modelFileSuffix(m->type())));
        if (answer != QMessageBox::Yes)
            return;
    }

    if (!QDir().mkpath(m->modelsDir())) {
        warn(QStringLiteral("无法创建模型目录：%1").arg(QDir::toNativeSeparators(m->modelsDir())));
        return;
    }

    const QString dst = m->modelPath();
    if (QFileInfo::exists(dst) && !QFile::remove(dst)) {
        warn(QStringLiteral("目标位置已有同名模型且无法覆盖：%1")
                 .arg(QDir::toNativeSeparators(dst)));
        return;
    }
    if (!QFile::copy(src.absoluteFilePath(), dst)) {
        warn(QStringLiteral("复制模型文件失败。"));
        return;
    }

    // 元数据一并复制（推理时需要它来识别任务与类别）
    const QFileInfo meta(src.absoluteFilePath() + QStringLiteral(".json"));
    if (meta.exists())
        QFile::copy(meta.absoluteFilePath(), modelio::metaPath(dst));

    if (m_display)
        m_display->setTestResultAvailable(true);
    updateActionStates();

    info(QStringLiteral("模型已导入：\n%1\n\n现在可以「智能标注」或「在测试集上评价」了。")
             .arg(QDir::toNativeSeparators(dst)),
         QStringLiteral("导入模型"));
}

// ===========================================================================
//  帮助
// ===========================================================================
void MainWindow::onBackendInfo()
{
    QString status = TrainerFactory::backendStatus();
    status.replace(QLatin1Char('\n'), QStringLiteral("<br>"));
    QMessageBox::information(this, QStringLiteral("训练后端信息"),
                             QStringLiteral("<b>当前训练后端可用性</b><br><br>%1").arg(status));
}

void MainWindow::onAbout()
{
    QMessageBox::about(
        this, QStringLiteral("关于 AqroTrain"),
        QStringLiteral("<h3>AqroTrain —— 视觉标注训练平台</h3>"
                       "<p>对标阿丘科技 AIDI 的图像标注与训练软件。</p>"
                       "<p><b>工具链：</b>Visual Studio 2022 (v143) + Qt 5.12.10 + OpenCV 4.6.0</p>"
                       "<p><b>工作流：</b>新建工程 → 添加模块 → 导入图像 → 标注 → 数据划分 → "
                       "训练 → 结果验证 → 导出模型</p>"
                       "<p><b>模块类型：</b>图像分类 / 目标检测 / 图像分割</p>"
                       "<p>分类模块整图一个类别；检测与分割模块的索引 0 为「背景」，"
                       "目标类别从 1 开始。</p>"));
}

// ===========================================================================
//  关闭
// ===========================================================================
void MainWindow::closeEvent(QCloseEvent *e)
{
    if (m_trainer && m_trainer->isRunning()) {
        const auto answer = QMessageBox::question(
            this, QStringLiteral("训练进行中"),
            QStringLiteral("训练尚未结束。要中止训练并退出吗？"));
        if (answer != QMessageBox::Yes) {
            e->ignore();
            return;
        }
        m_trainer->requestStop();
    }

    if (!maybeSaveAnnotation()) {
        e->ignore();
        return;
    }

    if (m_project && m_project->isOpen())
        m_project->saveAll(nullptr);

    QMainWindow::closeEvent(e);
}

// ===========================================================================
//  小工具
// ===========================================================================
void MainWindow::warn(const QString &text, const QString &title)
{
    QMessageBox::warning(this, title, text);
}

void MainWindow::info(const QString &text, const QString &title)
{
    QMessageBox::information(this, title, text);
}

} // namespace aq
