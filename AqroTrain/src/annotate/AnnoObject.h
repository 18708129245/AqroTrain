// ============================================================================
//  AnnoObject.h —— 可编辑标注对象模型
//  ---------------------------------------------------------------------------
//  设计要点（本工程与 VisionTrain 的核心差异）：
//
//    ★ 标注的真源是「对象列表」，掩码是派生物。
//      画笔画的一笔是 Stroke 对象、多边形是 Polygon 对象、框是 Rect 对象，
//      因此每个对象都能被再次选中、改类别、拖顶点、删除；
//      分割掩码则在每次保存时由对象列表光栅化得到（兼容训练与导出）。
//
//    ★ 撤销/重做统一为「对象列表快照」：对象数通常 < 200，
//      内存可控，逻辑简单且不会出现掩码与对象不一致的中间态。
//
//    ★ 只有掩码没有对象（外部导入 / 旧数据）时，用连通域分析反推对象，
//      并标 derived = true，使其同样可编辑。
//
//  坐标一律为图像像素坐标（与画布场景坐标一致）。
// ============================================================================
#pragma once

#include <QJsonObject>
#include <QPointF>
#include <QPolygonF>
#include <QRectF>
#include <QSize>
#include <QString>
#include <QVector>

#include <opencv2/core.hpp>

#include "core/Types.h"

namespace aq {

struct AnnoObject {
    enum Type {
        Rect    = 0,   ///< 矩形（检测框 / 分割矩形填充）
        Polygon = 1,   ///< 多边形
        Stroke  = 2    ///< 画笔笔画（折线 + 半径）
    };

    Type      type = Rect;
    int       classId = 0;
    QRectF    rect;              ///< Rect 专用
    QPolygonF points;            ///< Polygon 顶点 / Stroke 折线
    double    radius = 12.0;     ///< Stroke 末端半径（像素）
    bool      derived = false;   ///< 由掩码反推而来
    bool      pre = false;       ///< 智能标注产生，尚待人工复核

    /// 对象的包围盒（图像坐标）。无效对象返回空矩形。
    QRectF boundingRect() const;

    /// 是否含有可光栅化的几何
    bool isValid() const;

    /// 顶点数（Rect 返回 0）
    int vertexCount() const;

    QJsonObject toJson() const;
    static AnnoObject fromJson(const QJsonObject &o, bool *ok = nullptr);
};

namespace anno {

/// 光栅化到整幅掩码：CV_8UC1，尺寸 width×height，像素值 = 类别索引。
/// 对象按列表顺序绘制，后面的覆盖前面的（与作画顺序一致）。
void rasterize(const QVector<AnnoObject> &objects, int width, int height, cv::Mat *mask);

/// 增量光栅化：只重算 roi 区域，掩码其余部分保持不动。
/// roi 会被裁剪到掩码范围内；roi 无效时直接返回。
void rasterizeRegion(const QVector<AnnoObject> &objects, const cv::Rect &roi, cv::Mat *mask);

/// 把多个对象的包围盒求并集（用于计算增量刷新区域）
cv::Rect unionBounds(const QVector<AnnoObject> &objects, int width, int height);

/**
 * 折线简化（Ramer–Douglas–Peucker）。
 *
 * 距离用「点到**线段**」而不是点到无限直线，因此能正确处理折返：
 * 一笔「去 A→B、再从 B 折回 A」会被压成 3 个点（A、B、A），
 * 而锯齿/拐角这类真正撑开形状的点会被保留（它们离弦线的距离大）。
 *
 * 保证：简化后的折线离原始**每一个**采样点都不超过 tolerance，
 * 所以描边结果（掩码）的边界变化也在 tolerance 以内、肉眼不可见。
 * 这正是「来回涂抹时采样点不该无上限增长」的正确做法 ——
 * 不要用「跳过采样点」的手段去省点，那会让折线脱离真实笔迹（出现弦线/断续）。
 */
QPolygonF simplifyPolyline(const QPolygonF &points, double tolerance);

/// 由掩码反推对象：每个面积 >= minArea 的连通域生成一个多边形对象。
/// 掩码像素值即类别索引；索引 0 视为背景并跳过。
QVector<AnnoObject> objectsFromMask(const cv::Mat &mask, int minArea = 16);

/// 标注落盘：
///   分类          -> 只写 .json（classId + state）
///   检测 / 分割    -> 写 .json（对象列表 + state），分割另写派生掩码 .png
/// imageSize 为原图尺寸，用于生成与图像同尺寸的分割掩码（必须给出）。
bool writeLabel(const QString &jsonPath, const QString &maskPath, TaskType task,
                const QVector<AnnoObject> &objects, int classId, AnnoState state,
                const QSize &imageSize, QString *err = nullptr);

/// 标注读入。jsonPath 不存在但分割掩码存在时，由掩码反推对象（derived = true）。
bool readLabel(const QString &jsonPath, const QString &maskPath, TaskType task,
               QVector<AnnoObject> *objects, int *classId, AnnoState *state,
               int minArea = 16, QString *err = nullptr);

/// 判断一个类别索引对应的对象是否算「有目标」（供 OK/NG 判定使用）
bool hasTargetObject(const QVector<AnnoObject> &objects, const QVector<ClassInfo> &classes,
                     bool hasBackground);

} // namespace anno
} // namespace aq
