// ============================================================================
//  ImageCanvas.cpp —— 标注画布实现
// ============================================================================
#include "annotate/ImageCanvas.h"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QScrollBar>
#include <QWheelEvent>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "annotate/GraphicsItems.h"
#include "core/ImageIO.h"

namespace aq {

namespace {

/// 光标容差（图像坐标）。取手感与精度之间的折中。
constexpr double kHitTolerance = 3.0;

/// 只按任务类型给出的操作说明。
/// 刻意**不**在这里判断「类别是否已建立」：类别状态由主窗口的引导提示栏统一负责，
/// 画布在 setTask 时拿到的类别可能是上一张图留下的、甚至是空的，
/// 据此给出「还差一步」会误导使用者（实测出现过「明明有类别却提示还没建类别」）。
QString taskHint(TaskType task)
{
    switch (task) {
    case TaskType::Classification:
        return QStringLiteral("分类模块：整张图对应一个类别，<b>不需要涂抹</b>。"
                              "在右侧点选类别即可完成标定（点一下立即保存）。"
                              "若需要逐像素标注，请新建「图像分割」模块。");
    case TaskType::Detection:
        return QStringLiteral("检测模块：用「矩形」拖拽画框，或用「固定尺寸框」点一下放置。"
                              "「选择」工具可点选框、拖动移动、拖四角缩放，<b>Delete</b> 删除。");
    case TaskType::Segmentation:
        return QStringLiteral("分割模块：用「画笔」涂抹、「橡皮擦」擦除、「多边形」围出区域、"
                              "「矩形」填充。选中对象后可拖顶点微调。");
    }
    return QString();
}

/// 带类别状态的完整引导语（供工具被拒绝这类「当下就要解释清楚」的场景使用）
QString annotationHint(TaskType task, bool hasClasses, bool hasBackground)
{
    Q_UNUSED(hasBackground)
    if (!hasClasses) {
        return QStringLiteral("<b>还差一步：</b>先在右侧「标签类别」中建立至少一个"
                              "<span style='color:#ffb020'>目标类别</span>，然后才能标注。");
    }
    return taskHint(task);
}

} // namespace

// ===========================================================================
//  构造 / 析构
// ===========================================================================
ImageCanvas::ImageCanvas(QWidget *parent) : QGraphicsView(parent)
{
    m_scene = new QGraphicsScene(this);
    setScene(m_scene);

    // 以鼠标为中心缩放由框架保证，不必自己算偏移
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorViewCenter);
    setDragMode(QGraphicsView::NoDrag);
    setMouseTracking(true);
    setRenderHint(QPainter::Antialiasing, true);
    setRenderHint(QPainter::SmoothPixmapTransform, false);
    setViewportUpdateMode(QGraphicsView::SmartViewportUpdate);
    setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setFocusPolicy(Qt::StrongFocus);
    setFrameShape(QFrame::NoFrame);

    m_imgItem = new ImageItem();
    m_maskItem = new MaskItem();
    m_resultItem = new QGraphicsPixmapItem();
    m_resultItem->setZValue(3.0);
    m_resultItem->setVisible(false);
    m_scene->addItem(m_imgItem);
    m_scene->addItem(m_maskItem);
    m_scene->addItem(m_resultItem);
    m_scene->setBackgroundBrush(QColor(0x1B, 0x1E, 0x24));

    m_maskItem->setVisible(m_showOverlay);
}

ImageCanvas::~ImageCanvas() = default;

// ===========================================================================
//  任务与工具
// ===========================================================================
bool ImageCanvas::toolAllowedForTask(Tool t, TaskType task)
{
    if (task == TaskType::Classification) {
        // 分类是整图一个类别，任何「作画」类工具都没有意义
        return t == ToolSelect || t == ToolPan;
    }
    if (task == TaskType::Detection) {
        return t == ToolSelect || t == ToolRect || t == ToolFixedRect || t == ToolPan;
    }
    // 分割：全部工具可用
    return true;
}

void ImageCanvas::setTask(TaskType t)
{
    if (m_task == t)
        return;

    m_task = t;

    // 切任务时清掉绘制中的中间状态
    cancelPolygon();
    m_dragMode = DragNone;

    // 换到当前任务可用的工具
    if (t == TaskType::Classification) {
        m_tool = ToolSelect;
    } else if (t == TaskType::Detection) {
        m_tool = (m_tool == ToolBrush || m_tool == ToolEraser || m_tool == ToolPolygon) ? ToolRect : m_tool;
    } else {
        m_tool = (m_tool == ToolSelect) ? ToolBrush : m_tool;
    }

    updateCursorShape();
    viewport()->update();
    emit toolMessage(taskHint(m_task));
    emit objectsChanged();
}

void ImageCanvas::setTool(Tool t)
{
    if (m_tool == t)
        return;

    if (!toolAllowedForTask(t, m_task)) {
        emit toolMessage(QStringLiteral("「%1」在当前任务下不可用。%2")
                             .arg(t == ToolSelect ? QStringLiteral("选择") : QStringLiteral("该工具"))
                             .arg(annotationHint(m_task, !m_classes.isEmpty(), m_hasBackground)));
        return;
    }

    cancelPolygon();
    m_tool = t;
    m_dragMode = DragNone;
    updateCursorShape();
    viewport()->update();
}

void ImageCanvas::setClasses(const QVector<ClassInfo> &classes)
{
    m_classes = classes;
    if (m_maskItem) {
        m_maskItem->setClasses(classes);
        // 类别颜色可能变了，掩码需要整体重着色
        if (!m_image.empty())
            refreshMaskAll();
    }
    syncShapeItems();
    viewport()->update();
}

void ImageCanvas::setCurrentClass(int index)
{
    if (index < 0 || index >= m_classes.size())
        return;
    if (m_classId == index)
        return;
    m_classId = index;
    updateCursorShape();
    viewport()->update();
    emit classIdChanged(index);
}

void ImageCanvas::setBrushSize(int px)
{
    m_brushSize = std::max(1, px);
    viewport()->update();
}

void ImageCanvas::setFixedRectSize(const QSize &s)
{
    m_fixedSize = QSize(std::max(2, s.width()), std::max(2, s.height()));
}

void ImageCanvas::setOverlayOpacity(int alpha)
{
    m_overlayOpacity = std::max(0, std::min(255, alpha));
    if (m_maskItem)
        m_maskItem->setAlpha(m_overlayOpacity);
}

void ImageCanvas::setShowOverlay(bool on)
{
    m_showOverlay = on;
    if (m_maskItem)
        m_maskItem->setVisible(on);
}

// ===========================================================================
//  图像与数据
// ===========================================================================
QSize ImageCanvas::imageSize() const
{
    if (m_image.empty())
        return QSize();
    return QSize(m_image.cols, m_image.rows);
}

cv::Mat ImageCanvas::mask() const
{
    return m_maskItem ? m_maskItem->mask() : cv::Mat();
}

void ImageCanvas::setImage(const cv::Mat &bgr, const QString &name)
{
    m_loading = true;

    m_image = bgr;
    m_imageName = name;

    m_objects.clear();
    m_selected = -1;
    m_state = AnnoState::None;
    m_undo.clear();
    m_redo.clear();
    cancelPolygon();
    m_dragMode = DragNone;

    m_imgItem->setImage(m_image);
    if (m_image.empty()) {
        m_maskItem->setSize(QSize(0, 0));
    } else {
        m_maskItem->setSize(QSize(m_image.cols, m_image.rows));
        m_maskItem->setClasses(m_classes);
        m_maskItem->setAlpha(m_overlayOpacity);
        m_maskItem->setVisible(m_showOverlay);
    }

    m_scene->setSceneRect(QRectF(0, 0, m_image.empty() ? 0 : m_image.cols,
                                 m_image.empty() ? 0 : m_image.rows));

    syncShapeItems();
    zoomFit();

    m_loading = false;
    setModified(false);

    emit objectsChanged();
    emit selectionChanged(-1);
    emit annoStateChanged(int(m_state));
    updateCursorShape();
}

void ImageCanvas::setObjects(const QVector<AnnoObject> &objects, AnnoState state)
{
    m_objects = objects;
    m_state = state;
    m_selected = -1;
    m_undo.clear();
    m_redo.clear();

    syncShapeItems();
    refreshMaskAll();

    emit objectsChanged();
    emit selectionChanged(-1);
    emit annoStateChanged(int(m_state));
}

void ImageCanvas::rebuildMask()
{
    refreshMaskAll();
}

void ImageCanvas::setClassId(int id)
{
    if (m_classId == id)
        return;
    m_classId = id;
    updateCursorShape();
    emit classIdChanged(id);
}

void ImageCanvas::setAnnoState(AnnoState s)
{
    if (m_state == s)
        return;
    m_state = s;
    emit annoStateChanged(int(s));
}

// ===========================================================================
//  对象操作
// ===========================================================================
void ImageCanvas::setSelectedObject(int index)
{
    const int clamped = (index >= 0 && index < m_objects.size()) ? index : -1;
    if (m_selected == clamped)
        return;
    m_selected = clamped;
    syncShapeItems();
    viewport()->update();
    emit selectionChanged(m_selected);
}

void ImageCanvas::removeSelectedObject()
{
    if (m_selected < 0 || m_selected >= m_objects.size())
        return;

    pushUndo();
    m_objects.remove(m_selected);
    m_selected = -1;

    syncShapeItems();
    refreshMaskAll();
    setModified(true);

    emit objectsChanged();
    emit selectionChanged(-1);
    emit toolMessage(QStringLiteral("已删除 1 个标注对象。"));
}

void ImageCanvas::setSelectedObjectClass(int classId)
{
    if (m_selected < 0 || m_selected >= m_objects.size())
        return;
    if (m_objects[m_selected].classId == classId)
        return;

    pushUndo();
    m_objects[m_selected].classId = classId;

    syncShapeItems();
    refreshMaskAll();
    setModified(true);
    emit objectsChanged();
    emit toolMessage(QStringLiteral("已将选中对象改为「%1」。").arg(m_classes.value(classId).name));
}

void ImageCanvas::clearAllObjects()
{
    if (m_objects.isEmpty())
        return;

    pushUndo();
    m_objects.clear();
    m_selected = -1;

    syncShapeItems();
    refreshMaskAll();
    setModified(true);

    emit objectsChanged();
    emit selectionChanged(-1);
    emit toolMessage(QStringLiteral("已清空本图标注。"));
}

void ImageCanvas::applyPreAnnotation(const QVector<AnnoObject> &objects)
{
    if (objects.isEmpty())
        return;

    pushUndo();
    m_objects = objects;
    for (AnnoObject &o : m_objects)
        o.pre = true;
    m_selected = -1;
    m_state = AnnoState::Pre;

    syncShapeItems();
    refreshMaskAll();
    setModified(true);

    emit objectsChanged();
    emit annoStateChanged(int(m_state));
    emit toolMessage(QStringLiteral("已载入 %1 个预标注对象，请人工复核后确认。").arg(objects.size()));
}

void ImageCanvas::confirmPreAnnotation()
{
    bool changed = false;
    for (AnnoObject &o : m_objects) {
        if (o.pre) {
            o.pre = false;
            changed = true;
        }
    }
    if (!changed)
        return;

    pushUndo();
    if (m_state == AnnoState::Pre)
        m_state = AnnoState::Done;

    syncShapeItems();
    setModified(true);
    emit objectsChanged();
    emit annoStateChanged(int(m_state));
    emit toolMessage(QStringLiteral("预标注已确认为正式标注。"));
}

void ImageCanvas::pasteObjects(const QVector<AnnoObject> &objects)
{
    if (objects.isEmpty())
        return;

    pushUndo();
    for (const AnnoObject &o : objects) {
        AnnoObject copy = o;
        copy.pre = false;
        // 稍微偏移，避免与源对象完全重叠而看不出来
        const QPointF d(6, 6);
        if (copy.type == AnnoObject::Rect)
            copy.rect.translate(d);
        else
            for (QPointF &p : copy.points)
                p += d;
        m_objects.append(copy);
    }

    syncShapeItems();
    refreshMaskAll();
    setModified(true);

    emit objectsChanged();
    emit toolMessage(QStringLiteral("已粘贴 %1 个标注对象。").arg(objects.size()));
}

void ImageCanvas::markSaved()
{
    setModified(false);
}

void ImageCanvas::setAnnotationVisible(bool on)
{
    m_annotationVisible = on;
    if (m_maskItem)
        m_maskItem->setVisible(on && m_showOverlay);
    for (ShapeItem *it : m_shapeItems)
        it->setVisible(on);
    viewport()->update();
}

void ImageCanvas::setResultOverlay(const QImage &img)
{
    if (!m_resultItem)
        return;
    if (img.isNull()) {
        clearResultOverlay();
        return;
    }
    m_resultItem->setPixmap(QPixmap::fromImage(img));
    m_resultItem->setVisible(true);
    viewport()->update();
}

void ImageCanvas::clearResultOverlay()
{
    if (!m_resultItem)
        return;
    m_resultItem->setPixmap(QPixmap());
    m_resultItem->setVisible(false);
    viewport()->update();
}

bool ImageCanvas::hasResultOverlay() const
{
    return m_resultItem && m_resultItem->isVisible() && !m_resultItem->pixmap().isNull();
}

// ===========================================================================
//  视图控制
// ===========================================================================
double ImageCanvas::zoomLevel() const
{
    return transform().m11();
}

void ImageCanvas::zoomIn()
{
    scale(1.25, 1.25);
    updateViewScaleForItems();
    emit zoomChanged(zoomLevel());
}

void ImageCanvas::zoomOut()
{
    scale(0.8, 0.8);
    updateViewScaleForItems();
    emit zoomChanged(zoomLevel());
}

void ImageCanvas::zoomReset()
{
    resetTransform();
    updateViewScaleForItems();
    emit zoomChanged(zoomLevel());
}

void ImageCanvas::zoomFit()
{
    if (m_image.empty())
        return;
    resetTransform();
    const QSizeF imgSize(m_image.cols, m_image.rows);
    const QSizeF viewSize = viewport()->size();
    if (imgSize.width() > 0 && imgSize.height() > 0 && viewSize.width() > 0 && viewSize.height() > 0) {
        const double sx = (viewSize.width() - 4) / imgSize.width();
        const double sy = (viewSize.height() - 4) / imgSize.height();
        const double s = std::min(sx, sy);
        if (s > 0.0001 && s < 1e6)
            scale(s, s);
    }
    updateViewScaleForItems();
    emit zoomChanged(zoomLevel());
}

// ===========================================================================
//  撤销
// ===========================================================================
ImageCanvas::Snapshot ImageCanvas::snapshot() const
{
    Snapshot s;
    s.objects = m_objects;
    s.classId = m_classId;
    s.state   = m_state;
    return s;
}

void ImageCanvas::restore(const Snapshot &s)
{
    m_objects = s.objects;
    if (s.classId != m_classId) {
        m_classId = s.classId;
        emit classIdChanged(m_classId);
    }
    m_state = s.state;

    if (m_selected >= m_objects.size())
        m_selected = -1;

    syncShapeItems();
    refreshMaskAll();
    updateCursorShape();

    emit objectsChanged();
    emit annoStateChanged(int(m_state));
}

void ImageCanvas::pushUndo()
{
    m_undo.append(snapshot());
    if (m_undo.size() > kMaxUndo)
        m_undo.remove(0);
    m_redo.clear();
}

void ImageCanvas::undo()
{
    if (m_undo.isEmpty())
        return;
    m_redo.append(snapshot());
    const Snapshot s = m_undo.takeLast();
    restore(s);
    setModified(true);
    emit toolMessage(QStringLiteral("已撤销。"));
}

void ImageCanvas::redo()
{
    if (m_redo.isEmpty())
        return;
    m_undo.append(snapshot());
    const Snapshot s = m_redo.takeLast();
    restore(s);
    setModified(true);
    emit toolMessage(QStringLiteral("已重做。"));
}

void ImageCanvas::clearHistory()
{
    m_undo.clear();
    m_redo.clear();
}

void ImageCanvas::setModified(bool v)
{
    if (m_loading)
        return;
    if (m_modified == v)
        return;
    m_modified = v;
    emit modifiedChanged(v);
}

// ===========================================================================
//  场景同步
// ===========================================================================
void ImageCanvas::rebuildScene()
{
    for (ShapeItem *it : m_shapeItems) {
        m_scene->removeItem(it);
        delete it;
    }
    m_shapeItems.clear();

    if (!m_imgItem)
        m_imgItem = new ImageItem();
    if (!m_maskItem)
        m_maskItem = new MaskItem();
    if (m_imgItem->scene() != m_scene)
        m_scene->addItem(m_imgItem);
    if (m_maskItem->scene() != m_scene)
        m_scene->addItem(m_maskItem);
}

void ImageCanvas::syncShapeItems()
{
    // 按需增删图元，避免每次重建带来的选中闪烁
    while (m_shapeItems.size() > m_objects.size()) {
        ShapeItem *it = m_shapeItems.takeLast();
        m_scene->removeItem(it);
        delete it;
    }
    while (m_shapeItems.size() < m_objects.size()) {
        auto *it = new ShapeItem();
        m_scene->addItem(it);
        m_shapeItems.append(it);
    }

    const double scaleNow = zoomLevel();
    for (int i = 0; i < m_shapeItems.size(); ++i) {
        m_shapeItems[i]->setViewScale(scaleNow);
        m_shapeItems[i]->setVisible(m_annotationVisible);
        const QColor color = (m_objects[i].classId >= 0 && m_objects[i].classId < m_classes.size())
                                 ? m_classes[m_objects[i].classId].color
                                 : defaultClassColor(m_objects[i].classId);
        m_shapeItems[i]->setGeometry(m_objects[i], color, i == m_selected, m_classes, m_hasBackground);
    }

    // 图元的 boundingRect 会随视图缩放变化，缩放后必须让场景重新索引，
    // 否则可能留下重绘残影。
    m_scene->invalidate();
}

void ImageCanvas::updateViewScaleForItems()
{
    const double s = zoomLevel();
    for (ShapeItem *it : m_shapeItems)
        it->setViewScale(s);
    m_scene->invalidate();
}

cv::Rect ImageCanvas::clampRoi(const cv::Rect &r) const
{
    if (m_image.empty())
        return cv::Rect();
    return r & cv::Rect(0, 0, m_image.cols, m_image.rows);
}

cv::Rect ImageCanvas::objectRoi(const AnnoObject &obj) const
{
    if (m_image.empty())
        return cv::Rect();
    const QRectF b = obj.boundingRect();
    if (!b.isValid())
        return cv::Rect();

    const int x0 = int(std::floor(b.left())) - 1;
    const int y0 = int(std::floor(b.top())) - 1;
    const int x1 = int(std::ceil(b.right())) + 1;
    const int y1 = int(std::ceil(b.bottom())) + 1;
    return clampRoi(cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)));
}

cv::Rect ImageCanvas::segmentRoi(const QPointF &a, const QPointF &b, double radius) const
{
    const double pad = std::max(1.0, radius) + 2.0;
    const int x0 = int(std::floor(std::min(a.x(), b.x()) - pad));
    const int y0 = int(std::floor(std::min(a.y(), b.y()) - pad));
    const int x1 = int(std::ceil(std::max(a.x(), b.x()) + pad));
    const int y1 = int(std::ceil(std::max(a.y(), b.y()) + pad));
    return clampRoi(cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1)));
}

void ImageCanvas::refreshMaskAll()
{
    if (!m_maskItem)
        return;
    if (m_image.empty()) {
        m_maskItem->setSize(QSize(0, 0));
        return;
    }
    // setSize 会重建掩码缓冲，因此只在尺寸不符时才调用
    if (m_maskItem->mask().size() != m_image.size())
        m_maskItem->setSize(QSize(m_image.cols, m_image.rows));
    else
        m_maskItem->maskRef().setTo(0);

    cv::Mat &buf = m_maskItem->maskRef();
    if (!m_objects.isEmpty()) {
        cv::Mat raster;
        anno::rasterize(m_objects, m_image.cols, m_image.rows, &raster);
        raster.copyTo(buf);
    }
    m_maskItem->setClasses(m_classes);
    m_maskItem->setAlpha(m_overlayOpacity);
    m_maskItem->setVisible(m_showOverlay);
    m_maskItem->refreshAll();
}

void ImageCanvas::refreshMaskRegion(const cv::Rect &roi)
{
    if (!m_maskItem || m_image.empty())
        return;
    const cv::Rect r = clampRoi(roi);
    if (r.width <= 0 || r.height <= 0)
        return;

    cv::Mat &buf = m_maskItem->maskRef();
    if (buf.size() != m_image.size())
        m_maskItem->setSize(QSize(m_image.cols, m_image.rows));

    cv::Mat &buf2 = m_maskItem->maskRef();
    anno::rasterizeRegion(m_objects, r, &buf2);
    m_maskItem->refreshRect(r);
}

// ===========================================================================
//  绘制
// ===========================================================================
void ImageCanvas::drawBackground(QPainter *painter, const QRectF &rect)
{
    // 图像之外的区域用棋盘格，便于判断有没有画到图像范围外
    const QImage cb = imageio::checkerboard(64, 64, 12);
    painter->fillRect(rect, QBrush(cb));
}

void ImageCanvas::drawForeground(QPainter *painter, const QRectF &rect)
{
    painter->save();
    painter->setRenderHint(QPainter::Antialiasing, true);

    // --- 正在绘制的多边形 ---
    if (m_polygonActive && !m_polygon.isEmpty()) {
        QPen pen(QColor(0xFF, 0xB0, 0x20));
        pen.setCosmetic(true);
        pen.setWidthF(1.8);
        pen.setStyle(Qt::DashLine);
        painter->setPen(pen);
        painter->setBrush(Qt::NoBrush);

        QPolygonF poly = m_polygon;
        if (m_hasCursor)
            poly.append(m_cursorImg);
        painter->drawPolyline(poly);

        pen.setStyle(Qt::SolidLine);
        pen.setColor(QColor(0xFF, 0xFF, 0xFF));
        painter->setPen(pen);
        const double hs = 7.0 / std::max(0.0001, zoomLevel());
        for (const QPointF &p : m_polygon)
            painter->drawRect(QRectF(p.x() - hs / 2, p.y() - hs / 2, hs, hs));
    }

    // --- 画笔圆环光标 ---
    if (m_hasCursor && (m_tool == ToolBrush || m_tool == ToolEraser)) {
        const double r = m_brushSize / 2.0;
        const bool blocked = isBackgroundSelected();
        QPen pen(blocked ? QColor(0xFF, 0x50, 0x50) : QColor(0xFF, 0xFF, 0xFF, 190));
        pen.setCosmetic(true);
        pen.setWidthF(1.2);
        if (blocked)
            pen.setStyle(Qt::DashLine);
        painter->setPen(pen);
        painter->setBrush(Qt::NoBrush);
        painter->drawEllipse(m_cursorImg, r, r);
    }

    painter->restore();
    QGraphicsView::drawForeground(painter, rect);
}

// ===========================================================================
//  交互辅助
// ===========================================================================
QPointF ImageCanvas::toImage(const QPoint &viewPos) const
{
    return mapToScene(viewPos);
}

bool ImageCanvas::isBackgroundSelected() const
{
    return m_hasBackground && m_classId == 0;
}

bool ImageCanvas::blockIfBackground(const QString &action)
{
    if (!isBackgroundSelected())
        return false;

    const QString msg = QStringLiteral("当前选中的是「背景」类别，用它%1不会产生有效标注。"
                                       "请先在右侧「标签类别」中选择一个目标类别。")
                            .arg(action);
    emit backgroundClassBlocked(msg);
    emit toolMessage(msg);
    return true;
}

int ImageCanvas::hitObject(const QPointF &imgPos, int *vertexIndex, bool *isCorner) const
{
    if (vertexIndex)
        *vertexIndex = -1;
    if (isCorner)
        *isCorner = false;

    // 先判选中对象的顶点手柄，再判其它对象本体（从上往下，后加的在上层）
    if (m_selected >= 0 && m_selected < m_shapeItems.size()) {
        const int v = m_shapeItems[m_selected]->hitVertex(imgPos, kHitTolerance, isCorner);
        if (v >= 0) {
            if (vertexIndex)
                *vertexIndex = v;
            return m_selected;
        }
    }

    for (int i = m_shapeItems.size() - 1; i >= 0; --i) {
        if (m_shapeItems[i]->hitTest(imgPos, kHitTolerance))
            return i;
    }
    return -1;
}

void ImageCanvas::updateCursorShape()
{
    switch (m_tool) {
    case ToolPan:
        viewport()->setCursor(Qt::OpenHandCursor);
        break;
    case ToolSelect:
        viewport()->setCursor(Qt::ArrowCursor);
        break;
    case ToolBrush:
    case ToolEraser:
    case ToolPolygon:
    case ToolRect:
    case ToolFixedRect:
        // 用自定义圆环光标（在 drawForeground 里画），系统光标设为十字
        viewport()->setCursor(Qt::BlankCursor);
        break;
    }
}

void ImageCanvas::updateCanvasCursorPositions()
{
    viewport()->update();
}

// ===========================================================================
//  鼠标
// ===========================================================================
void ImageCanvas::mousePressEvent(QMouseEvent *e)
{
    setFocus(Qt::MouseFocusReason);

    const QPointF imgPos = toImage(e->pos());
    m_cursorImg = imgPos;
    m_hasCursor = true;

    // 中键或抓手工具：平移
    if (e->button() == Qt::MiddleButton || m_tool == ToolPan) {
        m_dragMode   = DragPan;
        m_panning    = true;
        m_panStart   = e->pos();
        m_panScrollH = horizontalScrollBar()->value();
        m_panScrollV = verticalScrollBar()->value();
        viewport()->setCursor(Qt::ClosedHandCursor);
        e->accept();
        return;
    }

    if (m_image.empty()) {
        e->accept();
        return;
    }

    if (e->button() == Qt::RightButton) {
        // 多边形绘制中：右键闭合
        if (m_polygonActive) {
            commitPolygon();
            e->accept();
            return;
        }

        // 选中对象上右键：优先「删除顶点」（AIDI 的多边形编辑习惯）
        int vertex = -1;
        bool corner = false;
        const int hit = hitObject(imgPos, &vertex, &corner);
        if (hit >= 0 && hit == m_selected && vertex >= 0 && !corner
            && m_objects[hit].type == AnnoObject::Polygon) {
            if (m_objects[hit].points.size() <= 3) {
                emit toolMessage(QStringLiteral("多边形至少需要 3 个顶点，无法继续删除。"));
            } else {
                pushUndo();
                m_objects[hit].points.remove(vertex);
                syncShapeItems();
                refreshMaskAll();
                setModified(true);
                emit objectsChanged();
                emit toolMessage(QStringLiteral("已删除顶点 %1（剩余 %2 个）。")
                                     .arg(vertex + 1)
                                     .arg(m_objects[hit].points.size()));
            }
            e->accept();
            return;
        }

        emit contextCommand(QStringLiteral("showMenu"));
        e->accept();
        return;
    }

    if (e->button() != Qt::LeftButton) {
        e->accept();
        return;
    }

    // --- 各工具的按下行为 ---
    switch (m_tool) {
    case ToolBrush:
        if (blockIfBackground(QStringLiteral("涂抹"))) {
            e->accept();
            return;
        }
        beginStroke(false);
        e->accept();
        return;

    case ToolEraser:
        // 橡皮擦本身就是擦除，用背景类擦除是合理操作，不做拦截
        beginStroke(true);
        e->accept();
        return;

    case ToolPolygon: {
        if (blockIfBackground(QStringLiteral("绘制多边形"))) {
            e->accept();
            return;
        }
        m_polygonActive = true;
        m_polygon.append(imgPos);
        emit polygonProgress(m_polygon.size());
        emit toolMessage(QStringLiteral("多边形：左键加点，双击 / 右键 / 回车闭合，Esc 取消（%1 个顶点）")
                             .arg(m_polygon.size()));
        viewport()->update();
        e->accept();
        return;
    }

    case ToolFixedRect: {
        if (blockIfBackground(QStringLiteral("放置矩形框"))) {
            e->accept();
            return;
        }
        AnnoObject obj;
        obj.type    = AnnoObject::Rect;
        obj.classId = m_classId;
        obj.rect    = QRectF(imgPos.x() - m_fixedSize.width() / 2.0,
                             imgPos.y() - m_fixedSize.height() / 2.0,
                             m_fixedSize.width(), m_fixedSize.height());
        pushUndo();
        m_objects.append(obj);
        m_selected = m_objects.size() - 1;
        syncShapeItems();
        refreshMaskRegion(objectRoi(obj));
        setModified(true);
        emit objectsChanged();
        emit selectionChanged(m_selected);
        emit toolMessage(QStringLiteral("已放置 %1×%2 的固定尺寸框。")
                             .arg(m_fixedSize.width())
                             .arg(m_fixedSize.height()));
        e->accept();
        return;
    }

    case ToolRect: {
        if (blockIfBackground(QStringLiteral("画框"))) {
            e->accept();
            return;
        }
        m_dragMode  = DragRect;
        m_dragStart = imgPos;
        m_dragLast  = imgPos;
        m_dragMoved = false;
        e->accept();
        return;
    }

    case ToolSelect:
    case ToolPan:
    default:
        handleSelectPress(imgPos, e->button());
        e->accept();
        return;
    }
}

// ---------------------------------------------------------------------------
//  笔画：采样密度与折线简化
//  ---------------------------------------------------------------------------
//  鼠标每移动一个像素就会产生一个 move 事件，所以必须处理两件事：
//    · 采样太密（原地抖动、或在同一片区域反复涂抹）→ 点数无上限增长；
//    · 又不能让折线脱离真实笔迹 —— 否则表现为断续的圆点或凭空的弦线。
//
//  做法分两层：
//    ① **「这里已经画过了」就跳过**（strokeAlreadyCovers）：笔尖是圆的，
//       相邻两笔只要圆心距 ≤ 半径，覆盖区域就基本重合，所以直接按
//       「新采样点是否落在本笔画已记录点的半径邻域内」判断 ——
//       这正是「不停在相同地方画，顶点不该一直增加」的正面实现。
//       注意判据必须是**本笔画自己的采样轨迹**，不是掩码当前值：
//       笔尖范围内的像素刚被本笔自己涂过，用掩码值判断会让每一段都被跳过，
//       笔画退化成断续圆点（实测踩过）。
//    ② 采样到一定量后再做一次 **RDP 折线简化**，把锯齿里的冗余点也收掉。
//
//  跳过距离取笔尖半径是有依据的：两个半径 R、圆心距 R 的圆盘，
//  并集在中点处宽度是 2*sqrt(R²-(R/2)²) = 1.73R，比理想带宽窄 13%
//  —— 半径 12 像素时不到 1.6 像素，肉眼不可见；而点数直接降到 1/半径 的量级。
constexpr int kSimplifyEvery = 64;   ///< 每累积这么多个采样点就简化一次

/// 简化的容差：远小于笔刷半径，保证描边边界的变化肉眼不可见。
/// 同时用 1~3 像素封顶，避免粗笔时容差跟着放大到看得见。
double strokeSimplifyTolerance(double radius)
{
    return std::max(1.0, std::min(3.0, radius * 0.1));
}

qint64 ImageCanvas::strokeCellKey(qint64 cx, qint64 cy)
{
    // 双射编码（格子坐标 -> 整数），不引入哈希碰撞
    return (cx << 32) + (cy & 0xffffffffLL);
}

bool ImageCanvas::strokeAlreadyCovers(const QPointF &imgPos) const
{
    // 索引超出上限后只比上一个已记录点；这是内存失控时的降级路径
    if (m_strokeGridCount > kMaxStrokeGridPoints) {
        if (m_strokeCurrentIndex >= 0 && m_strokeCurrentIndex < m_objects.size()) {
            const AnnoObject &o = m_objects.at(m_strokeCurrentIndex);
            if (!o.points.isEmpty()) {
                const QPointF d = o.points.last() - imgPos;
                return (d.x() * d.x() + d.y() * d.y()) <= m_strokeSkipDist * m_strokeSkipDist;
            }
        }
        return false;
    }

    const double d  = m_strokeSkipDist;
    const double d2 = d * d;
    const qint64 cx = qint64(std::floor(imgPos.x() / d));
    const qint64 cy = qint64(std::floor(imgPos.y() / d));

    // 格子边长就是 d，所以相距 ≤ d 的点必然落在 3x3 邻域内；
    // 但邻域里的点未必真的够近，因此仍要逐个比真实距离。
    for (qint64 ix = cx - 1; ix <= cx + 1; ++ix) {
        for (qint64 iy = cy - 1; iy <= cy + 1; ++iy) {
            const auto it = m_strokeGrid.constFind(strokeCellKey(ix, iy));
            if (it == m_strokeGrid.constEnd())
                continue;
            for (const QPointF &q : it.value()) {
                const QPointF diff = q - imgPos;
                if (diff.x() * diff.x() + diff.y() * diff.y() <= d2)
                    return true;
            }
        }
    }
    return false;
}

bool ImageCanvas::pixelAlreadyHasValue(const QPointF &imgPos, int value) const
{
    if (!m_maskItem)
        return false;
    const cv::Mat mask = m_maskItem->mask();
    if (mask.empty())
        return false;

    const int x = int(std::lround(imgPos.x()));
    const int y = int(std::lround(imgPos.y()));
    if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows)
        return false;   // 画到图像外：不拦，由 ROI 裁剪去处理

    const int v = std::max(0, std::min(255, value));
    return mask.at<uchar>(y, x) == uchar(v);
}

void ImageCanvas::appendStrokeSample(const QPointF &imgPos)
{
    // 一次手势只记一次撤销：在「第一次真正写入」时才拍快照。
    // 这样「在已画区域空点一下」不会留下一个什么都没改的撤销步骤。
    if (!m_strokeStarted) {
        pushUndo();
        m_strokeStarted = true;
    }

    // 一次手势**只产生一个**笔画对象：对象数量不该由鼠标动线决定。
    if (m_strokeCurrentIndex < 0 || m_strokeCurrentIndex >= m_objects.size()) {
        AnnoObject o;
        o.type    = AnnoObject::Stroke;
        o.classId = m_strokeErase ? 0 : m_classId;
        o.radius  = m_brushSize / 2.0;
        m_objects.append(o);
        m_selected           = -1;
        m_strokeCurrentIndex = m_objects.size() - 1;
        syncShapeItems();
        // 新对象的第一个点：没有前置线段
        m_dragLast = imgPos;
    }

    AnnoObject &obj = m_objects[m_strokeCurrentIndex];
    obj.points.append(imgPos);

    // 记入空间索引，后续采样点就能 O(1) 判断「这里已经画过了」
    if (m_strokeGridCount <= kMaxStrokeGridPoints) {
        const double d = m_strokeSkipDist;
        const qint64 cx = qint64(std::floor(imgPos.x() / d));
        const qint64 cy = qint64(std::floor(imgPos.y() / d));
        m_strokeGrid[strokeCellKey(cx, cy)].append(imgPos);
        ++m_strokeGridCount;
    }

    // 只重画「上一落笔点 -> 本次采样点」这一段
    const cv::Rect roi = segmentRoi(m_dragLast, imgPos, obj.radius);
    m_dragLast = imgPos;
    refreshMaskRegion(roi);

    if (m_strokeCurrentIndex < m_shapeItems.size()) {
        const int cid = obj.classId;
        m_shapeItems[m_strokeCurrentIndex]->setGeometry(
            obj,
            (cid >= 0 && cid < m_classes.size()) ? m_classes[cid].color : defaultClassColor(cid),
            false, m_classes, m_hasBackground);
    }

    setModified(true);
    emit objectsChanged();

    // 累积到一定量就简化一次，避免长时间涂抹时内存与重算量一直涨
    if (++m_strokeSinceSimplify >= kSimplifyEvery)
        simplifyCurrentStroke();
}

void ImageCanvas::simplifyCurrentStroke()
{
    if (m_strokeCurrentIndex < 0 || m_strokeCurrentIndex >= m_objects.size())
        return;

    AnnoObject &obj = m_objects[m_strokeCurrentIndex];
    if (obj.type != AnnoObject::Stroke || obj.points.size() <= 2) {
        m_strokeSinceSimplify = 0;
        return;
    }

    const QPolygonF simplified =
        anno::simplifyPolyline(obj.points, strokeSimplifyTolerance(obj.radius));

    m_strokeSinceSimplify = 0;

    if (simplified.size() >= obj.points.size())
        return;   // 没有可合并的点

    obj.points = simplified;

    // 几何变了：按整个笔画的包围盒重算掩码，
    // rasterizeRegion 会把该 ROI 内所有对象重画一遍，
    // 因此被简化掉的像素会被正确抹掉，掩码与折线保持一致。
    refreshMaskRegion(objectRoi(obj));

    if (m_strokeCurrentIndex < m_shapeItems.size()) {
        const int cid = obj.classId;
        m_shapeItems[m_strokeCurrentIndex]->setGeometry(
            obj,
            (cid >= 0 && cid < m_classes.size()) ? m_classes[cid].color : defaultClassColor(cid),
            false, m_classes, m_hasBackground);
    }

    emit objectsChanged();
}

void ImageCanvas::beginStroke(bool erase)
{
    m_strokeErase         = erase;
    m_strokeStarted       = false;
    m_strokeCurrentIndex  = -1;
    m_strokeSinceSimplify = 0;
    m_strokeGrid.clear();
    m_strokeGridCount     = 0;
    m_dragMode            = DragStroke;
    m_dragLast            = m_cursorImg;
    // 「已经画过这里」的判定距离 = 笔尖半径（下限 2 像素，细笔也要有基本去重）
    m_strokeSkipDist      = std::max(2.0, m_brushSize / 2.0);

    const int classId = erase ? 0 : m_classId;

    // 落笔即画一个圆点（相当于点一下画个圆点）。
    // 但如果这个像素本来就已经是目标值 —— 「在已画区域重复落笔」或
    // 「橡皮擦划过空白处」—— 就先不创建对象：既不弄出一个什么都没改变的笔画，
    // 也不产生空的撤销步骤。若随后拖到需要真正写的地方，会在那里惰性创建。
    if (!pixelAlreadyHasValue(m_cursorImg, classId))
        appendStrokeSample(m_cursorImg);

    emit toolMessage(erase ? QStringLiteral("橡皮擦：拖动以擦除。")
                           : QStringLiteral("画笔：拖动以涂抹（A / D 调整笔刷大小）。"));
    updateCursorShape();
    viewport()->update();
}

void ImageCanvas::extendStroke(const QPointF &imgPos)
{
    if (m_dragMode != DragStroke)
        return;

    // 这里只有**一道**判据：新的采样点是否落在本笔画已记录点的邻域内。
    // 落进去就说明「这里已经画过了」，直接跳过 —— 不新增点、也不重算掩码。
    // 这就是「不停在相同地方画，顶点不该一直增加」的正面实现。
    //
    // 刻意不在这里做别的事：
    //  · 不用「掩码像素值已是本类别」判断（笔尖内像素刚被本笔涂过，会让笔画变断续圆点）；
    //  · 不为了省点而在拖动中另起新对象（对象数会随鼠标动线增长）。
    if (strokeAlreadyCovers(imgPos))
        return;

    appendStrokeSample(imgPos);
}

void ImageCanvas::handleSelectPress(const QPointF &imgPos, Qt::MouseButton)
{
    int vertex = -1;
    bool corner = false;
    const int hit = hitObject(imgPos, &vertex, &corner);

    if (hit < 0) {
        setSelectedObject(-1);
        return;
    }

    setSelectedObject(hit);

    if (vertex >= 0) {
        m_dragMode   = corner ? DragResizeCorner : DragMoveVertex;
        m_dragVertex = vertex;
    } else {
        m_dragMode   = DragMoveObject;
        m_dragVertex = -1;
    }

    m_dragStart = imgPos;
    m_dragLast  = imgPos;
    m_dragMoved = false;
    pushUndo();
}

void ImageCanvas::handleSelectMove(const QPointF &imgPos)
{
    if (m_selected < 0 || m_selected >= m_objects.size())
        return;

    AnnoObject &obj = m_objects[m_selected];
    const QPointF delta = imgPos - m_dragLast;
    if (delta.isNull())
        return;

    const cv::Rect before = objectRoi(obj);
    m_dragLast = imgPos;
    m_dragMoved = true;

    switch (m_dragMode) {
    case DragMoveObject: {
        if (obj.type == AnnoObject::Rect) {
            obj.rect.translate(delta);
        } else {
            for (QPointF &p : obj.points)
                p += delta;
        }
        break;
    }
    case DragMoveVertex: {
        if (m_dragVertex >= 0 && m_dragVertex < obj.points.size())
            obj.points[m_dragVertex] = imgPos;
        break;
    }
    case DragResizeCorner: {
        if (obj.type != AnnoObject::Rect)
            break;
        QRectF r = obj.rect.normalized();
        switch (m_dragVertex) {
        case ShapeItem::TopLeft:     r.setTopLeft(imgPos); break;
        case ShapeItem::TopRight:    r.setTopRight(imgPos); break;
        case ShapeItem::BottomRight: r.setBottomRight(imgPos); break;
        case ShapeItem::BottomLeft:  r.setBottomLeft(imgPos); break;
        default: break;
        }
        obj.rect = r.normalized();
        break;
    }
    default:
        return;
    }

    // 增量重光栅化：只刷新「旧包围盒 ∪ 新包围盒」的区域
    const cv::Rect after = objectRoi(obj);
    const cv::Rect both  = before.width > 0 && after.width > 0 ? (before | after)
                                                             : (before.width > 0 ? before : after);

    m_shapeItems[m_selected]->setGeometry(obj,
                                          (obj.classId >= 0 && obj.classId < m_classes.size())
                                              ? m_classes[obj.classId].color
                                              : defaultClassColor(obj.classId),
                                          true, m_classes, m_hasBackground);
    refreshMaskRegion(both);
    setModified(true);
    emit objectsChanged();
}

void ImageCanvas::handleSelectRelease()
{
    m_dragMode   = DragNone;
    m_dragVertex = -1;
    m_dragMoved  = false;
}

void ImageCanvas::mouseMoveEvent(QMouseEvent *e)
{
    const QPointF imgPos = toImage(e->pos());
    m_cursorImg = imgPos;
    m_hasCursor = true;

    // 状态栏坐标与灰度
    if (!m_image.empty()) {
        const int x = int(std::lround(imgPos.x()));
        const int y = int(std::lround(imgPos.y()));
        if (x >= 0 && y >= 0 && x < m_image.cols && y < m_image.rows) {
            const cv::Vec3b px = m_image.at<cv::Vec3b>(y, x);
            const int gray = int(std::lround(0.114 * px[0] + 0.587 * px[1] + 0.299 * px[2]));
            emit mouseImagePos(QPoint(x, y), gray);
        } else {
            emit mouseImagePos(QPoint(x, y), -1);
        }
    }

    switch (m_dragMode) {
    case DragPan: {
        const QPoint d = e->pos() - m_panStart;
        horizontalScrollBar()->setValue(m_panScrollH - d.x());
        verticalScrollBar()->setValue(m_panScrollV - d.y());
        return;
    }
    case DragStroke:
        extendStroke(imgPos);
        break;
    case DragMoveObject:
    case DragMoveVertex:
    case DragResizeCorner:
        handleSelectMove(imgPos);
        break;
    case DragRect:
        m_dragLast  = imgPos;
        m_dragMoved = true;
        break;
    case DragNone:
        // 选择工具下悬停到手柄/对象上时给出移动光标提示
        if (m_tool == ToolSelect) {
            int v = -1;
            bool corner = false;
            hitObject(imgPos, &v, &corner);
            viewport()->setCursor(v >= 0 ? (corner ? Qt::SizeFDiagCursor : Qt::CrossCursor)
                                         : Qt::ArrowCursor);
        }
        break;
    }

    if (m_tool == ToolBrush || m_tool == ToolEraser || m_polygonActive)
        viewport()->update();
}

void ImageCanvas::mouseReleaseEvent(QMouseEvent *e)
{
    if (m_dragMode == DragPan) {
        m_dragMode = DragNone;
        m_panning  = false;
        updateCursorShape();
        e->accept();
        return;
    }

    if (m_dragMode == DragStroke) {
        // 抬手位置要用在「尾部补齐」上，先同步光标位置
        m_cursorImg = toImage(e->pos());
        finalizeCurrentStroke();
        e->accept();
        return;
    }

    if (m_dragMode == DragRect) {
        const QPointF a = m_dragStart;
        const QPointF b = toImage(e->pos());
        const QRectF r = QRectF(a, b).normalized();

        // 太小的框视为误触，忽略
        if (r.width() >= 2.0 && r.height() >= 2.0) {
            AnnoObject obj;
            obj.type    = AnnoObject::Rect;
            obj.classId = m_classId;
            obj.rect    = r;
            pushUndo();
            m_objects.append(obj);
            m_selected = m_objects.size() - 1;
            syncShapeItems();
            refreshMaskRegion(objectRoi(obj));
            setModified(true);
            emit objectsChanged();
            emit selectionChanged(m_selected);
            emit toolMessage(QStringLiteral("已添加矩形：%1×%2（选择工具可调整）")
                                 .arg(int(r.width()))
                                 .arg(int(r.height())));
        }

        m_dragMode = DragNone;
        viewport()->update();
        e->accept();
        return;
    }

    if (m_dragMode == DragMoveObject || m_dragMode == DragMoveVertex || m_dragMode == DragResizeCorner) {
        handleSelectRelease();
        e->accept();
        return;
    }

    QGraphicsView::mouseReleaseEvent(e);
}

void ImageCanvas::finalizeCurrentStroke()
{
    m_dragMode = DragNone;

    // 本次手势什么都没写入（在已画区域空点、或橡皮擦划过空白）：
    // 不产生对象、不留下撤销点、也不把工程标记成已修改。
    if (!m_strokeStarted) {
        m_strokeCurrentIndex  = -1;
        m_strokeSinceSimplify = 0;
        m_strokeGrid.clear();
        m_strokeGridCount     = 0;
        updateCursorShape();
        viewport()->update();
        return;
    }

    // 尾部补齐：跳过规则下，最后一个采样点可能离抬手位置还差将近一个笔尖半径，
    // 直接收尾会让笔画「画不到手抬起来的地方」（实测 100px 拖拽跨度少 9px）。
    // 只补这一个点，代价可忽略，但尾巴才落在使用者意中的位置。
    if (m_strokeCurrentIndex >= 0 && m_strokeCurrentIndex < m_objects.size()) {
        AnnoObject &obj = m_objects[m_strokeCurrentIndex];
        if (!obj.points.isEmpty()) {
            const QPointF diff = m_cursorImg - obj.points.last();
            if (diff.x() * diff.x() + diff.y() * diff.y() > 0.25)
                appendStrokeSample(m_cursorImg);
        }
    }

    // 收尾再简化一次，把最后一次涂抹产生的重复点也折叠掉
    simplifyCurrentStroke();

    // 清理异常的空笔画（正常路径不会出现）
    if (m_strokeCurrentIndex >= 0 && m_strokeCurrentIndex < m_objects.size()
        && m_objects.at(m_strokeCurrentIndex).points.isEmpty()) {
        m_objects.removeAt(m_strokeCurrentIndex);
        syncShapeItems();
        refreshMaskAll();
    }

    m_strokeCurrentIndex  = -1;
    m_strokeSinceSimplify = 0;
    m_strokeGrid.clear();
    m_strokeGridCount     = 0;

    emit objectsChanged();
    setModified(true);
    updateCursorShape();
    viewport()->update();
}

void ImageCanvas::mouseDoubleClickEvent(QMouseEvent *e)
{
    const QPointF imgPos = toImage(e->pos());

    // 多边形绘制中：双击闭合
    if (m_polygonActive) {
        commitPolygon();
        e->accept();
        return;
    }

    // 选择工具下双击多边形的边：插入顶点
    if (m_tool == ToolSelect && m_selected >= 0 && m_selected < m_objects.size()
        && m_objects[m_selected].type == AnnoObject::Polygon) {
        AnnoObject &obj = m_objects[m_selected];
        int bestEdge = -1;
        double bestDist = kHitTolerance * 3.0;
        for (int i = 0; i < obj.points.size(); ++i) {
            const QPointF a = obj.points.at(i);
            const QPointF b = obj.points.at((i + 1) % obj.points.size());
            const QPointF ab = b - a;
            const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (len2 <= 1e-9)
                continue;
            double t = ((imgPos - a).x() * ab.x() + (imgPos - a).y() * ab.y()) / len2;
            t = std::max(0.0, std::min(1.0, t));
            const QPointF proj = a + ab * t;
            const double d = QLineF(proj, imgPos).length();
            if (d < bestDist) {
                bestDist = d;
                bestEdge = i + 1;
            }
        }
        if (bestEdge >= 0) {
            pushUndo();
            obj.points.insert(bestEdge % (obj.points.size() + 1), imgPos);
            syncShapeItems();
            refreshMaskAll();
            setModified(true);
            emit objectsChanged();
            emit toolMessage(QStringLiteral("已在边上插入顶点（共 %1 个）。").arg(obj.points.size()));
            e->accept();
            return;
        }
    }

    QGraphicsView::mouseDoubleClickEvent(e);
}

void ImageCanvas::wheelEvent(QWheelEvent *e)
{
    // Ctrl+滚轮：调整笔刷大小（与 AIDI 的手感一致）
    if (e->modifiers() & Qt::ControlModifier) {
        const int steps = e->angleDelta().y() / 120;
        if (steps != 0) {
            setBrushSize(m_brushSize + steps * 2);
            emit toolMessage(QStringLiteral("笔刷大小：%1").arg(m_brushSize));
        }
        e->accept();
        return;
    }

    // 普通滚轮：以鼠标为中心缩放（由 AnchorUnderMouse 保证）
    const int steps = e->angleDelta().y();
    if (steps > 0)
        scale(1.15, 1.15);
    else if (steps < 0)
        scale(1.0 / 1.15, 1.0 / 1.15);

    updateViewScaleForItems();
    emit zoomChanged(zoomLevel());
    e->accept();
}

void ImageCanvas::keyPressEvent(QKeyEvent *e)
{
    switch (e->key()) {
    case Qt::Key_Escape:
        if (m_polygonActive) {
            cancelPolygon();
            e->accept();
            return;
        }
        if (m_dragMode != DragNone) {
            // 取消本次拖动：还原到按下前的快照
            if (!m_undo.isEmpty())
                restore(m_undo.last());
            m_dragMode = DragNone;
            e->accept();
            return;
        }
        break;

    case Qt::Key_Return:
    case Qt::Key_Enter:
        if (m_polygonActive) {
            commitPolygon();
            e->accept();
            return;
        }
        break;

    case Qt::Key_S:
        // 主窗口也绑定了 S 快捷键；这里作为画布获得焦点时的兜底
        emit requestSaveAndNext();
        e->accept();
        return;

    case Qt::Key_Delete:
    case Qt::Key_Backspace:
        removeSelectedObject();
        e->accept();
        return;

    case Qt::Key_A:
        setBrushSize(m_brushSize - 2);
        emit toolMessage(QStringLiteral("笔刷大小：%1").arg(m_brushSize));
        e->accept();
        return;

    case Qt::Key_D:
        setBrushSize(m_brushSize + 2);
        emit toolMessage(QStringLiteral("笔刷大小：%1").arg(m_brushSize));
        e->accept();
        return;

    default:
        break;
    }

    // 数字键 1..9 快速切换类别。
    // 刻意只在这里处理而不是注册成 QAction 全局快捷键：
    // 全局数字快捷键会抢走参数输入框里的数字输入。
    if (e->key() >= Qt::Key_1 && e->key() <= Qt::Key_9) {
        const int wanted = e->key() - Qt::Key_1;
        if (wanted >= 0 && wanted < m_classes.size()) {
            setCurrentClass(wanted);
            emit classPicked(wanted);
            emit toolMessage(QStringLiteral("当前类别：%1").arg(m_classes.at(wanted).name));
        } else {
            emit toolMessage(QStringLiteral("类别索引 %1 不存在。").arg(wanted));
        }
        e->accept();
        return;
    }

    QGraphicsView::keyPressEvent(e);
}

void ImageCanvas::leaveEvent(QEvent *e)
{
    m_hasCursor = false;
    emit mouseLeft();
    viewport()->update();
    QGraphicsView::leaveEvent(e);
}

void ImageCanvas::resizeEvent(QResizeEvent *e)
{
    QGraphicsView::resizeEvent(e);
    updateViewScaleForItems();
}

// ===========================================================================
//  多边形
// ===========================================================================
void ImageCanvas::commitPolygon()
{
    if (m_polygon.size() >= 3) {
        AnnoObject obj;
        obj.type    = AnnoObject::Polygon;
        obj.classId = m_classId;
        obj.points  = m_polygon;

        pushUndo();
        m_objects.append(obj);
        m_selected = m_objects.size() - 1;
        syncShapeItems();
        refreshMaskRegion(objectRoi(obj));
        setModified(true);
        emit objectsChanged();
        emit selectionChanged(m_selected);
        emit toolMessage(QStringLiteral("已添加多边形（%1 个顶点）。").arg(m_polygon.size()));
    } else {
        emit toolMessage(QStringLiteral("多边形至少需要 3 个顶点，本次绘制已取消。"));
    }

    m_polygon.clear();
    m_polygonActive = false;
    emit polygonProgress(0);
    viewport()->update();
}

void ImageCanvas::cancelPolygon()
{
    if (!m_polygonActive && m_polygon.isEmpty())
        return;
    m_polygon.clear();
    m_polygonActive = false;
    emit polygonProgress(0);
    emit toolMessage(QStringLiteral("已取消本次多边形绘制。"));
    viewport()->update();
}

} // namespace aq
