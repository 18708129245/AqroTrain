// ============================================================================
//  GraphicsItems.cpp —— 画布图元实现
// ============================================================================
#include "annotate/GraphicsItems.h"

#include <QPainter>
#include <QPainterPath>
#include <QStyleOptionGraphicsItem>

#include <opencv2/imgproc.hpp>

#include "core/ImageIO.h"

#include <algorithm>
#include <cstring>

namespace aq {

namespace {

/// 由类别索引取颜色（越界用默认调色板）
QColor colorForClass(int classId, const QVector<ClassInfo> &classes)
{
    if (classId >= 0 && classId < classes.size())
        return classes[classId].color;
    return defaultClassColor(classId < 0 ? 0 : classId);
}

} // namespace

// ===========================================================================
//  ImageItem
// ===========================================================================
ImageItem::ImageItem(QGraphicsItem *parent) : QGraphicsItem(parent)
{
    setZValue(0.0);
}

void ImageItem::setImage(const cv::Mat &bgr)
{
    prepareGeometryChange();
    m_image = imageio::matToQImage(bgr);
    m_rect  = QRectF(0, 0, m_image.width(), m_image.height());
    update();
}

QRectF ImageItem::boundingRect() const
{
    return m_rect;
}

void ImageItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{
    if (m_image.isNull())
        return;
    // 缩小时开启平滑，放大时保持像素锐利，便于看清缺陷细节
    painter->setRenderHint(QPainter::SmoothPixmapTransform, painter->transform().m11() < 1.0);
    painter->drawImage(QPointF(0, 0), m_image);
}

// ===========================================================================
//  MaskItem
// ===========================================================================
MaskItem::MaskItem(QGraphicsItem *parent) : QGraphicsItem(parent)
{
    setZValue(1.0);
}

void MaskItem::setSize(const QSize &size)
{
    prepareGeometryChange();
    m_mask = cv::Mat::zeros(size.height(), size.width(), CV_8UC1);
    m_overlay = QImage(size, QImage::Format_RGBA8888);
    m_overlay.fill(Qt::transparent);
    m_rect = QRectF(0, 0, size.width(), size.height());
    update();
}

void MaskItem::setClasses(const QVector<ClassInfo> &classes)
{
    m_classes = classes;
    refreshAll();
}

void MaskItem::setAlpha(int alpha)
{
    m_alpha = std::max(0, std::min(255, alpha));
    refreshAll();
}

void MaskItem::refreshAll()
{
    if (m_mask.empty())
        return;
    rebuildOverlay(cv::Rect(0, 0, m_mask.cols, m_mask.rows));
}

void MaskItem::refreshRect(const cv::Rect &roiIn)
{
    if (m_mask.empty())
        return;
    const cv::Rect roi = roiIn & cv::Rect(0, 0, m_mask.cols, m_mask.rows);
    if (roi.width <= 0 || roi.height <= 0)
        return;
    rebuildOverlay(roi);
}

void MaskItem::rebuildOverlay(const cv::Rect &roi)
{
    if (m_mask.empty() || m_overlay.isNull())
        return;

    const int w = roi.width;
    const int h = roi.height;

    // 用与 VisionTrain 相同的做法着色：对 B/G/R 三个平面分别查表再合并。
    // 注意不能用单通道索引图配三通道 LUT（cv::LUT 会断言失败抛异常）。
    cv::Mat lutB(1, 256, CV_8UC1), lutG(1, 256, CV_8UC1), lutR(1, 256, CV_8UC1), lutA(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        // 索引 0 视为透明背景（不做叠加），其余按类别颜色
        const QColor c = colorForClass(i, m_classes);
        lutB.at<uchar>(0, i) = static_cast<uchar>(c.blue());
        lutG.at<uchar>(0, i) = static_cast<uchar>(c.green());
        lutR.at<uchar>(0, i) = static_cast<uchar>(c.red());
        lutA.at<uchar>(0, i) = (i == 0) ? 0 : static_cast<uchar>(m_alpha);
    }

    const cv::Mat sub = m_mask(roi);

    cv::Mat pB, pG, pR, pA;
    cv::LUT(sub, lutB, pB);
    cv::LUT(sub, lutG, pG);
    cv::LUT(sub, lutR, pR);
    cv::LUT(sub, lutA, pA);

    cv::Mat rgba;
    const std::vector<cv::Mat> planes { pR, pG, pB, pA };   // RGBA 顺序
    cv::merge(planes, rgba);

    // 逐行拷进 QImage 的目标 ROI（QImage 的行跨距可能大于 ROI 宽度）
    // m_overlay 固定为 Format_RGBA8888，即每像素 4 字节
    for (int y = 0; y < h; ++y) {
        const uchar *src = rgba.ptr<uchar>(y);
        uchar *dst = m_overlay.scanLine(roi.y + y) + 4 * roi.x;
        std::memcpy(dst, src, static_cast<size_t>(w) * 4);
    }

    // 只刷新受影响区域，避免整幅重绘
    update(QRectF(roi.x, roi.y, w, h));
}

QRectF MaskItem::boundingRect() const
{
    return m_rect;
}

void MaskItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{
    if (m_overlay.isNull() || m_mask.empty())
        return;
    painter->setRenderHint(QPainter::SmoothPixmapTransform, false);
    painter->drawImage(QPointF(0, 0), m_overlay);
}

// ===========================================================================
//  ShapeItem
// ===========================================================================
ShapeItem::ShapeItem(QGraphicsItem *parent) : QGraphicsItem(parent)
{
    setZValue(2.0);
    setFlag(QGraphicsItem::ItemIsSelectable, false);   // 选中状态由画布统一管理
    setAcceptedMouseButtons(Qt::NoButton);             // 鼠标事件全部交给画布
}

double ShapeItem::handleImageSize() const
{
    if (m_viewScale <= 0.0001)
        return kHandleScreenSize;
    return kHandleScreenSize / m_viewScale;
}

void ShapeItem::setViewScale(double scale)
{
    // 缩放变化会改变手柄的图元尺寸，因此包围盒也要重算
    prepareGeometryChange();
    m_viewScale = scale > 0.0001 ? scale : 1.0;
    update();
}

void ShapeItem::setGeometry(const AnnoObject &obj, const QColor &color, bool selected,
                            const QVector<ClassInfo> &classes, bool hasBackground)
{
    prepareGeometryChange();
    m_obj           = obj;
    m_color         = color;
    m_selected      = selected;
    m_classes       = classes;
    m_hasBackground = hasBackground;

    const double pad = handleImageSize() + 2.0;
    m_paintRect = m_obj.boundingRect().adjusted(-pad, -pad, pad, pad);
    update();
}

QRectF ShapeItem::boundingRect() const
{
    return m_paintRect;
}

void ShapeItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *)
{
    if (!m_obj.isValid())
        return;

    // 预标注对象用虚线，提示「尚未人工复核」
    const bool dashed = m_obj.pre;

    QPen pen(m_color);
    pen.setCosmetic(true);
    pen.setWidthF(m_selected ? 2.4 : 1.6);
    if (dashed)
        pen.setStyle(Qt::DashLine);

    painter->setBrush(Qt::NoBrush);
    painter->setPen(pen);

    switch (m_obj.type) {
    case AnnoObject::Rect: {
        const QRectF r = m_obj.rect.normalized();
        painter->drawRect(r);
        break;
    }
    case AnnoObject::Polygon: {
        // 闭合多边形：半透明填充 + 描边，保证能同时看到下方掩码与图像
        QPolygonF poly = m_obj.points;
        QColor fill = m_color;
        fill.setAlpha(60);
        painter->setBrush(fill);
        painter->drawPolygon(poly);
        painter->setBrush(Qt::NoBrush);
        painter->setPen(pen);
        painter->drawPolygon(poly);
        break;
    }
    case AnnoObject::Stroke: {
        QPen spen(m_color);
        spen.setCosmetic(true);
        spen.setWidthF(std::max(1.0, m_obj.radius * 2.0));
        spen.setCapStyle(Qt::RoundCap);
        spen.setJoinStyle(Qt::RoundJoin);
        if (dashed)
            spen.setStyle(Qt::DashLine);
        else if (m_selected) {
            spen.setColor(m_color.lighter(140));
        }
        painter->setPen(spen);
        if (m_obj.points.size() == 1)
            painter->drawPoint(m_obj.points.first());
        else
            painter->drawPolyline(m_obj.points);
        break;
    }
    }

    // 选中时画顶点手柄（屏幕上恒定大小）
    if (m_selected && m_obj.type != AnnoObject::Stroke) {
        const double hs = handleImageSize();
        QPen hp(QColor(0xFF, 0xFF, 0xFF));
        hp.setCosmetic(true);
        hp.setWidthF(1.2);
        painter->setPen(hp);
        painter->setBrush(m_color.darker(120));

        if (m_obj.type == AnnoObject::Rect) {
            const QRectF r = m_obj.rect.normalized();
            const QPointF corners[4] = { r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft() };
            for (const QPointF &c : corners)
                painter->drawRect(QRectF(c.x() - hs / 2.0, c.y() - hs / 2.0, hs, hs));
        } else {
            for (const QPointF &p : m_obj.points)
                painter->drawRect(QRectF(p.x() - hs / 2.0, p.y() - hs / 2.0, hs, hs));
        }
    }
}

bool ShapeItem::hitTest(const QPointF &pt, double tol) const
{
    if (!m_obj.isValid())
        return false;

    switch (m_obj.type) {
    case AnnoObject::Rect:
        return m_obj.rect.normalized().adjusted(-tol, -tol, tol, tol).contains(pt);
    case AnnoObject::Polygon: {
        if (m_obj.points.containsPoint(pt, Qt::OddEvenFill))
            return true;
        // 边界附近也算命中，便于点选细长多边形
        const QPolygonF closed = m_obj.points;
        for (int i = 0; i < closed.size(); ++i) {
            const QPointF a = closed.at(i);
            const QPointF b = closed.at((i + 1) % closed.size());
            const QPointF ab = b - a;
            const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (len2 <= 1e-9)
                continue;
            double t = ((pt - a).x() * ab.x() + (pt - a).y() * ab.y()) / len2;
            t = std::max(0.0, std::min(1.0, t));
            const QPointF proj = a + ab * t;
            const QPointF d = proj - pt;
            if (std::sqrt(d.x() * d.x() + d.y() * d.y()) <= tol)
                return true;
        }
        return false;
    }
    case AnnoObject::Stroke: {
        const double r = std::max(1.0, m_obj.radius) + tol;
        if (m_obj.points.size() == 1) {
            const QPointF d = m_obj.points.first() - pt;
            return std::sqrt(d.x() * d.x() + d.y() * d.y()) <= r;
        }
        for (int i = 1; i < m_obj.points.size(); ++i) {
            const QPointF a = m_obj.points.at(i - 1);
            const QPointF b = m_obj.points.at(i);
            const QPointF ab = b - a;
            const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (len2 <= 1e-9) {
                const QPointF d = a - pt;
                if (std::sqrt(d.x() * d.x() + d.y() * d.y()) <= r)
                    return true;
                continue;
            }
            double t = ((pt - a).x() * ab.x() + (pt - a).y() * ab.y()) / len2;
            t = std::max(0.0, std::min(1.0, t));
            const QPointF proj = a + ab * t;
            const QPointF d = proj - pt;
            if (std::sqrt(d.x() * d.x() + d.y() * d.y()) <= r)
                return true;
        }
        return false;
    }
    }
    return false;
}

int ShapeItem::hitVertex(const QPointF &pt, double tol, bool *isCorner) const
{
    if (isCorner)
        *isCorner = false;

    const double hs = std::max(handleImageSize(), tol);

    if (m_obj.type == AnnoObject::Rect) {
        const QRectF r = m_obj.rect.normalized();
        const QPointF corners[4] = { r.topLeft(), r.topRight(), r.bottomRight(), r.bottomLeft() };
        for (int i = 0; i < 4; ++i) {
            if (QLineF(corners[i], pt).length() <= hs) {
                if (isCorner)
                    *isCorner = true;
                return i;   // Corner 枚举与顺序一致
            }
        }
        return -1;
    }

    if (m_obj.type == AnnoObject::Polygon) {
        for (int i = 0; i < m_obj.points.size(); ++i) {
            if (QLineF(m_obj.points.at(i), pt).length() <= hs)
                return i;
        }
    }
    return -1;
}

} // namespace aq
