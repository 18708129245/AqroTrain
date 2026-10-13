// ============================================================================
//  GraphicsItems.h —— 画布图元
//  ---------------------------------------------------------------------------
//  显示层结构（场景坐标 == 图像像素坐标，原点在图像左上角）：
//      QGraphicsScene
//        ├─ ImageItem   z=0   原图
//        ├─ MaskItem    z=1   分割掩码叠加（支持 ROI 局部刷新）
//        └─ ShapeItem   z=2   标注对象（矩形 / 多边形 / 笔画）
//
//  职责划分：
//    · 图元只负责「画什么」与命中测试；
//    · 谁被选中、鼠标拖到哪里意味着什么、撤销栈 —— 全部由 ImageCanvas 负责。
//      这样做是为了让「对象列表」成为唯一真源，避免图元与数据两份状态互相打架。
//
//  手柄（顶点）尺寸需要随视图缩放换算，保证**屏幕上恒定 9px**，
//  因此 ImageCanvas 在缩放后会调用 setViewScale()。
// ============================================================================
#pragma once

#include <QColor>
#include <QGraphicsItem>
#include <QImage>
#include <QRectF>
#include <QVector>

#include <opencv2/core.hpp>

#include "annotate/AnnoObject.h"
#include "core/Types.h"

namespace aq {

/// 屏幕手柄尺寸（像素）
constexpr double kHandleScreenSize = 9.0;

// ---------------------------------------------------------------------------
//  原图
// ---------------------------------------------------------------------------
class ImageItem : public QGraphicsItem
{
public:
    explicit ImageItem(QGraphicsItem *parent = nullptr);

    void   setImage(const cv::Mat &bgr);
    QImage image() const { return m_image; }
    QSize  imageSize() const { return m_image.isNull() ? QSize() : m_image.size(); }

    QRectF boundingRect() const override;
    void   paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                 QWidget *widget = nullptr) override;

private:
    QImage m_image;
    QRectF m_rect;
};

// ---------------------------------------------------------------------------
//  掩码叠加
// ---------------------------------------------------------------------------
class MaskItem : public QGraphicsItem
{
public:
    explicit MaskItem(QGraphicsItem *parent = nullptr);

    /// 直接就地修改掩码数据（CV_8UC1）。改完必须调用 refreshRect / refreshAll。
    cv::Mat &maskRef() { return m_mask; }
    const cv::Mat &mask() const { return m_mask; }

    void setSize(const QSize &size);
    void setClasses(const QVector<ClassInfo> &classes);
    void setAlpha(int alpha);          ///< 0..255
    void setVisibleOverlay(bool on) { setVisible(on); }

    void refreshAll();
    void refreshRect(const cv::Rect &roi);

    QRectF boundingRect() const override;
    void   paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                 QWidget *widget = nullptr) override;

private:
    void rebuildOverlay(const cv::Rect &roi);

    cv::Mat            m_mask;      ///< CV_8UC1，像素值 = 类别索引
    QImage             m_overlay;   ///< RGBA
    QVector<ClassInfo> m_classes;
    int                m_alpha = 130;
    QRectF             m_rect;
};

// ---------------------------------------------------------------------------
//  标注对象
// ---------------------------------------------------------------------------
class ShapeItem : public QGraphicsItem
{
public:
    explicit ShapeItem(QGraphicsItem *parent = nullptr);

    /// 更新要绘制的几何与样式（selected 时画描边与顶点手柄）
    void setGeometry(const AnnoObject &obj, const QColor &color, bool selected,
                     const QVector<ClassInfo> &classes, bool hasBackground);

    const AnnoObject &object() const { return m_obj; }

    /// 视图缩放系数，用于把顶点手柄换算成屏幕上恒定大小
    void setViewScale(double scale);

    QRectF boundingRect() const override;
    void   paint(QPainter *painter, const QStyleOptionGraphicsItem *option,
                 QWidget *widget = nullptr) override;

    /// 命中测试：点是否落在对象几何内（tol 为图像坐标下的容差）
    bool hitTest(const QPointF &pt, double tol) const;

    /// 返回被命中的顶点下标（Polygon 专用；Rect 用角点枚举），未命中返回 -1
    int hitVertex(const QPointF &pt, double tol, bool *isCorner = nullptr) const;

    /// 手柄在图像坐标下的尺寸
    double handleImageSize() const;

    /// 顶点在手柄命中测试中使用的枚举（Rect 的四个角）
    enum Corner { CornerNone = -1, TopLeft = 0, TopRight = 1, BottomRight = 2, BottomLeft = 3 };

private:
    AnnoObject         m_obj;
    QColor             m_color = QColor(Qt::red);
    QVector<ClassInfo> m_classes;
    bool               m_selected = false;
    bool               m_hasBackground = false;
    double             m_viewScale = 1.0;
    QRectF             m_paintRect;
};

} // namespace aq
