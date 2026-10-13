// ============================================================================
//  ImageCanvas.h —— 标注画布（QGraphicsView + QGraphicsItem）
//  ---------------------------------------------------------------------------
//  场景坐标 == 图像像素坐标（原点在图像左上角），因此所有标注数据都直接用
//  像素坐标，缩放与平移完全交给视图变换，不需要手写坐标换算。
//
//  工具集（对标 AIDI）：
//    选择 / 画笔 / 橡皮擦 / 多边形 / 矩形 / 固定尺寸框 / 平移
//
//  编辑能力：
//    · 选中对象后可整体拖动；矩形可拖四角缩放；
//    · 多边形可拖顶点、双击边插入顶点、右键顶点删除顶点；
//    · 撤销/重做统一为「对象列表快照」，避免掩码与对象出现不一致的中间态。
//
//  掩码是**派生数据**：每次对象几何变化后按「受影响 ROI」增量重光栅化，
//  既保证与对象列表一致，又避免大图整幅重绘。
// ============================================================================
#pragma once

#include <QGraphicsView>
#include <QHash>
#include <QImage>
#include <QPointF>
#include <QPolygonF>
#include <QRectF>
#include <QVector>

#include <opencv2/core.hpp>

#include "annotate/AnnoObject.h"
#include "core/Types.h"

class QGraphicsPixmapItem;

namespace aq {

class ImageItem;
class MaskItem;
class ShapeItem;

class ImageCanvas : public QGraphicsView
{
    Q_OBJECT

public:
    enum Tool {
        ToolSelect    = 0,   ///< 选择 / 编辑对象
        ToolBrush     = 1,   ///< 画笔
        ToolEraser    = 2,   ///< 橡皮擦
        ToolPolygon   = 3,   ///< 多边形
        ToolRect      = 4,   ///< 矩形（检测=画框；分割=矩形填充）
        ToolFixedRect = 5,   ///< 固定尺寸框（AIDI「标准模式」：点一下就放置）
        ToolPan       = 6    ///< 抓手平移
    };
    Q_ENUM(Tool)

    explicit ImageCanvas(QWidget *parent = nullptr);
    ~ImageCanvas() override;

    // --- 任务与工具 -------------------------------------------------------
    void     setTask(TaskType t);
    TaskType task() const { return m_task; }

    void setTool(Tool t);
    Tool tool() const { return m_tool; }

    void setClasses(const QVector<ClassInfo> &classes);
    const QVector<ClassInfo> &classes() const { return m_classes; }

    void setCurrentClass(int index);
    int  currentClass() const { return m_classId; }

    /// 是否带背景类（检测 / 分割为 true）。用于拦截「用背景类作画」这类无效操作。
    void setHasBackground(bool on) { m_hasBackground = on; }
    bool hasBackground() const { return m_hasBackground; }

    void setBrushSize(int px);
    int  brushSize() const { return m_brushSize; }

    void  setFixedRectSize(const QSize &s);
    QSize fixedRectSize() const { return m_fixedSize; }

    void setOverlayOpacity(int alpha);    ///< 0..255
    int  overlayOpacity() const { return m_overlayOpacity; }

    void setShowOverlay(bool on);
    bool showOverlay() const { return m_showOverlay; }

    /// 该工具在当前任务下是否可用（供界面置灰与提示）
    static bool toolAllowedForTask(Tool t, TaskType task);

    // --- 图像与标注数据 ---------------------------------------------------
    void    setImage(const cv::Mat &bgr, const QString &name = QString());
    bool    hasImage() const { return !m_image.empty(); }
    cv::Mat image() const { return m_image; }
    QSize   imageSize() const;
    QString imageName() const { return m_imageName; }

    const QVector<AnnoObject> &objects() const { return m_objects; }
    void setObjects(const QVector<AnnoObject> &objects, AnnoState state = AnnoState::Done);

    /// 当前掩码（由对象列表光栅化得到；仅分割任务有意义）。
    /// 实现放在 .cpp：MaskItem 在此处是不完整类型。
    cv::Mat mask() const;

    int       classId() const { return m_classId; }          ///< 分类任务的类别
    void      setClassId(int id);
    AnnoState annoState() const { return m_state; }
    void      setAnnoState(AnnoState s);

    // --- 对象操作 ---------------------------------------------------------
    int  selectedObject() const { return m_selected; }
    void setSelectedObject(int index);
    void removeSelectedObject();
    void setSelectedObjectClass(int classId);
    /// 清空本图全部对象
    void clearAllObjects();
    /// 用智能标注的结果替换当前对象（已存在的会被覆盖）
    void applyPreAnnotation(const QVector<AnnoObject> &objects);
    /// 把全部 pre 对象标记为已复核
    void confirmPreAnnotation();

    bool isModified() const { return m_modified; }
    void markSaved();

    /// 是否显示人工标注（原图 / 标注 / 测试结果 三态切换中的「标注」）
    void setAnnotationVisible(bool on);
    bool annotationVisible() const { return m_annotationVisible; }

    /// 叠加一层「模型预测结果」可视化图（由主窗口用 OpenCV 画好）
    void setResultOverlay(const QImage &img);
    void clearResultOverlay();
    bool hasResultOverlay() const;

    /// 由对象列表重算整幅掩码（载入 / 反推 / 自检使用）
    void rebuildMask();

    // --- 视图控制 ---------------------------------------------------------
    void zoomIn();
    void zoomOut();
    void zoomReset();    ///< 1:1
    void zoomFit();      ///< 适应窗口
    /// 当前缩放系数。刻意不叫 scale()：QGraphicsView 已有 scale(qreal,qreal)，
    /// 同名重载会在派生类中把基类版本隐藏掉。
    double zoomLevel() const;

    // --- 撤销 -------------------------------------------------------------
    bool canUndo() const { return !m_undo.isEmpty(); }
    bool canRedo() const { return !m_redo.isEmpty(); }
    void undo();
    void redo();
    void clearHistory();

signals:
    void modifiedChanged(bool modified);
    void mouseImagePos(const QPoint &imgPos, int grayValue);
    void mouseLeft();
    void zoomChanged(double scale);
    void classIdChanged(int classId);
    /// 使用者主动选择了类别（数字键 1..9 或点选）——
    /// 分类模块据此「点一下即完成标定」；程序化设置不会发出本信号
    void classPicked(int classId);
    void objectsChanged();
    void selectionChanged(int index);
    void annoStateChanged(int state);
    void toolMessage(const QString &text);
    void polygonProgress(int vertexCount);
    /// S 键：保存并跳到下一张
    void requestSaveAndNext();
    /// 画布右键菜单命令（由主窗口执行：清空 / 删除标注文件 / 设为OK图 / 复制 / 粘贴）
    void contextCommand(const QString &command);
    /// 提示「当前是背景类，无法作画」
    void backgroundClassBlocked(const QString &action);

public slots:
    /// 粘贴外部对象（主窗口的粘贴命令）
    void pasteObjects(const QVector<AnnoObject> &objects);

protected:
    void drawBackground(QPainter *painter, const QRectF &rect) override;
    void drawForeground(QPainter *painter, const QRectF &rect) override;

    void mousePressEvent(QMouseEvent *e) override;
    void mouseMoveEvent(QMouseEvent *e) override;
    void mouseReleaseEvent(QMouseEvent *e) override;
    void mouseDoubleClickEvent(QMouseEvent *e) override;
    void wheelEvent(QWheelEvent *e) override;
    void keyPressEvent(QKeyEvent *e) override;
    void leaveEvent(QEvent *e) override;
    void resizeEvent(QResizeEvent *e) override;

private:
    // --- 场景同步 ---------------------------------------------------------
    void rebuildScene();
    void syncShapeItems();
    void updateViewScaleForItems();
    void refreshMaskAll();
    void refreshMaskRegion(const cv::Rect &roi);

    // --- 撤销 -------------------------------------------------------------
    struct Snapshot {
        QVector<AnnoObject> objects;
        int       classId = -1;
        AnnoState state   = AnnoState::None;
    };
    Snapshot snapshot() const;
    void     restore(const Snapshot &s);
    /// 记录「变更前」的状态。必须在真正修改数据之前调用。
    void pushUndo();
    void setModified(bool v);

    // --- 交互辅助 ---------------------------------------------------------
    QPointF toImage(const QPoint &viewPos) const;
    /// 命中对象：返回下标（-1 表示未命中）。vertexIndex / isCorner 输出顶点信息。
    int  hitObject(const QPointF &imgPos, int *vertexIndex, bool *isCorner) const;
    bool isBackgroundSelected() const;
    bool blockIfBackground(const QString &action);
    void updateCursorShape();

    cv::Rect objectRoi(const AnnoObject &obj) const;
    cv::Rect segmentRoi(const QPointF &a, const QPointF &b, double radius) const;
    cv::Rect clampRoi(const cv::Rect &r) const;

    void commitPolygon();
    void cancelPolygon();
    void beginStroke(bool erase);
    void extendStroke(const QPointF &imgPos);
    void finalizeCurrentStroke();

    // --- 笔画采样 ---------------------------------------------------------
    /// 该像素当前是否已经等于给定值 —— 即这一笔写下去不会改变任何像素。
    /// 画笔：像素已是本类别；橡皮擦：像素已是背景（没什么可擦）。
    /// **只用于「按下」那一次判断**，拖动过程中用它会把笔画打成断续的点。
    bool pixelAlreadyHasValue(const QPointF &imgPos, int value) const;
    /// 本次笔画的采样点空间索引：格子键 → 落在该格的点
    static qint64 strokeCellKey(qint64 cx, qint64 cy);
    /// 该采样点是否落在「本笔画已记录点」的邻域内。
    /// 笔尖是圆的：相邻两次落笔只要圆心距 ≤ 半径，覆盖区域就基本重合，
    /// 所以这个判据直接对应「这里已经画过了，不用再记一个点」。
    bool strokeAlreadyCovers(const QPointF &imgPos) const;
    /// 记录一个采样点（写入笔画对象 + 空间索引），并触发折线简化
    void appendStrokeSample(const QPointF &imgPos);
    /// 对当前笔画做一次 RDP 简化并同步掩码
    void simplifyCurrentStroke();

    void handleSelectPress(const QPointF &imgPos, Qt::MouseButton button);
    void handleSelectMove(const QPointF &imgPos);
    void handleSelectRelease();

    void updateCanvasCursorPositions();

    // --- 状态 -------------------------------------------------------------
    TaskType m_task = TaskType::Segmentation;
    Tool     m_tool = ToolBrush;

    cv::Mat            m_image;      ///< BGR
    QString            m_imageName;
    QVector<AnnoObject> m_objects;
    QVector<ClassInfo> m_classes;
    int                m_classId       = 0;
    bool               m_hasBackground = false;
    AnnoState          m_state         = AnnoState::None;
    int                m_selected      = -1;

    int  m_brushSize      = 24;
    QSize m_fixedSize     = QSize(64, 64);
    int  m_overlayOpacity = 130;
    bool m_showOverlay    = true;

    QGraphicsScene *m_scene     = nullptr;
    ImageItem      *m_imgItem   = nullptr;
    MaskItem       *m_maskItem  = nullptr;
    QGraphicsPixmapItem *m_resultItem = nullptr;
    QVector<ShapeItem *> m_shapeItems;
    bool m_annotationVisible = true;

    // --- 拖动状态 ---
    enum DragMode {
        DragNone = 0,
        DragPan,
        DragStroke,
        DragRect,
        DragMoveObject,
        DragResizeCorner,
        DragMoveVertex
    };
    DragMode m_dragMode   = DragNone;
    QPointF  m_dragStart;
    /// 上一个**真正落笔**的点（被跳过的采样点不会推进它）。
    /// 增量光栅化的 ROI 由它算起，因此它必须是「画布上确实画到的最后一点」。
    QPointF  m_dragLast;
    int      m_dragVertex = -1;   ///< 顶点下标或角点枚举
    bool     m_strokeErase = false;   ///< 本次手势是不是橡皮擦
    bool     m_dragMoved   = false;

    // --- 笔画采样状态（避免在同一处反复涂抹时点数无上限增长）---
    /// 本次手势是否已经真正写入过（决定是否记撤销、结束时要不要保留对象）
    bool m_strokeStarted = false;
    /// 本次手势的笔画对象下标（-1 表示本手势还没有对象）。
    /// 一次手势**只有一个**对象：对象数量不该由鼠标动线决定。
    int  m_strokeCurrentIndex = -1;
    /// 「已经画过这里」的判定距离：笔尖半径（下限 2 像素）
    double m_strokeSkipDist = 2.0;
    /// 本笔画已记录点的空间索引（格子键 -> 点表），用于 O(1) 邻域查询
    QHash<qint64, QVector<QPointF>> m_strokeGrid;
    /// 已进入索引的点数；超过上限后不再维护索引（改为只比上一个点），避免内存失控
    int  m_strokeGridCount = 0;
    /// 距离上次折线简化又累积了多少个采样点
    int  m_strokeSinceSimplify = 0;
    static const int kMaxStrokeGridPoints = 100000;

    bool   m_panning = false;
    QPoint m_panStart;
    int    m_panScrollH = 0;
    int    m_panScrollV = 0;

    // --- 多边形绘制中 ---
    QPolygonF m_polygon;
    bool      m_polygonActive = false;

    // --- 光标 ---
    QPointF m_cursorImg;
    bool    m_hasCursor = false;

    // --- 撤销 ---
    QVector<Snapshot> m_undo;
    QVector<Snapshot> m_redo;
    static const int kMaxUndo = 50;

    bool m_modified = false;
    /// 正在批量设置数据（载入样本），用于抑制副作用
    bool m_loading = false;
};

} // namespace aq
