// ============================================================================
//  AnnoObject.cpp —— 对象模型、掩码光栅化与标注落盘
// ============================================================================
#include "annotate/AnnoObject.h"

#include <QFileInfo>
#include <QJsonArray>

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include "core/ImageIO.h"
#include "core/JsonStore.h"

namespace aq {

// ===========================================================================
//  AnnoObject
// ===========================================================================
QRectF AnnoObject::boundingRect() const
{
    switch (type) {
    case Rect:
        return rect.normalized();
    case Polygon: {
        if (points.isEmpty())
            return QRectF();
        return points.boundingRect();
    }
    case Stroke: {
        if (points.isEmpty())
            return QRectF();
        const double r = radius > 0.0 ? radius : 1.0;
        return points.boundingRect().adjusted(-r, -r, r, r);
    }
    }
    return QRectF();
}

bool AnnoObject::isValid() const
{
    switch (type) {
    case Rect: {
        const QRectF r = rect.normalized();
        return r.width() > 0.0 && r.height() > 0.0;
    }
    case Polygon:
        return points.size() >= 3;
    case Stroke:
        return !points.isEmpty();
    }
    return false;
}

int AnnoObject::vertexCount() const
{
    if (type == Rect)
        return 0;
    return points.size();
}

QJsonObject AnnoObject::toJson() const
{
    QJsonObject o;
    switch (type) {
    case Rect:    o.insert(QStringLiteral("type"), QStringLiteral("rect"));    break;
    case Polygon: o.insert(QStringLiteral("type"), QStringLiteral("polygon")); break;
    case Stroke:  o.insert(QStringLiteral("type"), QStringLiteral("stroke"));  break;
    }
    o.insert(QStringLiteral("classId"), classId);

    if (type == Rect) {
        o.insert(QStringLiteral("rect"), jsonio::rectToJson(rect.normalized()));
    } else {
        o.insert(QStringLiteral("points"), jsonio::polygonToJson(points));
        if (type == Stroke)
            o.insert(QStringLiteral("radius"), radius);
    }
    if (derived)
        o.insert(QStringLiteral("derived"), true);
    if (pre)
        o.insert(QStringLiteral("pre"), true);
    return o;
}

AnnoObject AnnoObject::fromJson(const QJsonObject &o, bool *ok)
{
    AnnoObject a;
    const QString t = jsonio::str(o, QStringLiteral("type"), QStringLiteral("rect")).toLower();

    if (t == QLatin1String("rect")) {
        a.type = Rect;
        a.rect = jsonio::rectFromJson(o.value(QStringLiteral("rect")).toArray());
    } else if (t == QLatin1String("polygon")) {
        a.type   = Polygon;
        a.points = jsonio::polygonFromJson(o.value(QStringLiteral("points")).toArray());
    } else if (t == QLatin1String("stroke")) {
        a.type   = Stroke;
        a.points = jsonio::polygonFromJson(o.value(QStringLiteral("points")).toArray());
        a.radius = jsonio::num(o, QStringLiteral("radius"), 12.0);
    } else {
        if (ok) *ok = false;
        return a;
    }

    a.classId = jsonio::integer(o, QStringLiteral("classId"), 0);
    a.derived = jsonio::boolean(o, QStringLiteral("derived"), false);
    a.pre     = jsonio::boolean(o, QStringLiteral("pre"), false);

    if (ok)
        *ok = a.isValid();
    return a;
}

// ===========================================================================
//  光栅化
// ===========================================================================
namespace anno {

namespace {

/// 把一个对象画到 dst 上；origin 为 dst 左上角在原图坐标系中的位置
void drawObject(cv::Mat &dst, const AnnoObject &obj, const cv::Point &origin)
{
    const int value = obj.classId;
    if (value < 0 || value > 255)
        return;

    const int n = obj.type == AnnoObject::Rect ? 2 : obj.points.size();
    if (n <= 0)
        return;

    if (obj.type == AnnoObject::Rect) {
        const QRectF r = obj.rect.normalized();
        const cv::Rect box(cv::Point(int(std::floor(r.left())) - origin.x,
                                     int(std::floor(r.top())) - origin.y),
                           cv::Point(int(std::ceil(r.right())) - origin.x,
                                     int(std::ceil(r.bottom())) - origin.y));
        cv::rectangle(dst, box, cv::Scalar(value), cv::FILLED);
        return;
    }

    std::vector<cv::Point> pts;
    pts.reserve(size_t(n));
    for (const QPointF &p : obj.points)
        pts.emplace_back(int(std::lround(p.x())) - origin.x, int(std::lround(p.y())) - origin.y);

    if (obj.type == AnnoObject::Polygon) {
        if (pts.size() < 3)
            return;
        std::vector<std::vector<cv::Point>> polys { pts };
        cv::fillPoly(dst, polys, cv::Scalar(value), cv::LINE_8);
        return;
    }

    // Stroke：折线 + 圆头。半径 0 时至少画 1 像素宽。
    const int thickness = std::max(1, int(std::lround(obj.radius * 2.0)));
    if (pts.size() == 1) {
        cv::circle(dst, pts[0], std::max(1, int(std::lround(obj.radius))), cv::Scalar(value), cv::FILLED);
        return;
    }
    for (size_t i = 1; i < pts.size(); ++i)
        cv::line(dst, pts[i - 1], pts[i], cv::Scalar(value), thickness, cv::LINE_8);
    // 圆头：折点处补圆，避免关节出现缺口
    const int r = std::max(1, int(std::lround(obj.radius)));
    for (const cv::Point &p : pts)
        cv::circle(dst, p, r, cv::Scalar(value), cv::FILLED);
}

} // namespace

void rasterize(const QVector<AnnoObject> &objects, int width, int height, cv::Mat *mask)
{
    if (!mask || width <= 0 || height <= 0)
        return;

    *mask = cv::Mat::zeros(height, width, CV_8UC1);
    const cv::Point origin(0, 0);
    for (const AnnoObject &o : objects) {
        if (!o.isValid())
            continue;
        drawObject(*mask, o, origin);
    }
}

cv::Rect unionBounds(const QVector<AnnoObject> &objects, int width, int height)
{
    QRectF total;
    bool first = true;
    for (const AnnoObject &o : objects) {
        if (!o.isValid())
            continue;
        const QRectF b = o.boundingRect();
        total = first ? b : total.united(b);
        first = false;
    }
    if (first)
        return cv::Rect();

    int x0 = int(std::floor(total.left())) - 1;
    int y0 = int(std::floor(total.top())) - 1;
    int x1 = int(std::ceil(total.right())) + 1;
    int y1 = int(std::ceil(total.bottom())) + 1;

    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    if (width > 0)
        x1 = std::min(width, x1);
    if (height > 0)
        y1 = std::min(height, y1);

    if (x1 <= x0 || y1 <= y0)
        return cv::Rect();
    return cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}

void rasterizeRegion(const QVector<AnnoObject> &objects, const cv::Rect &roiIn, cv::Mat *mask)
{
    if (!mask || mask->empty() || mask->type() != CV_8UC1)
        return;

    cv::Rect roi = roiIn & cv::Rect(0, 0, mask->cols, mask->rows);
    if (roi.width <= 0 || roi.height <= 0)
        return;

    // 在 ROI 局部画布上重画全部对象（超出 ROI 的部分被画布自然裁掉），
    // 再整块写回，避免逐像素判断，也保证与整幅光栅化的结果一致。
    cv::Mat local = cv::Mat::zeros(roi.height, roi.width, CV_8UC1);
    const cv::Point origin(roi.x, roi.y);
    for (const AnnoObject &o : objects) {
        if (!o.isValid())
            continue;
        drawObject(local, o, origin);
    }
    local.copyTo(mask->operator()(roi));
}

QPolygonF simplifyPolyline(const QPolygonF &points, double tolerance)
{
    const int n = points.size();
    if (n <= 2 || tolerance <= 0.0)
        return points;

    // 点到线段的距离。线段退化（两端重合）时退化为点到点距离 ——
    // 折返笔画的弦线正是这种情况，必须处理，否则会算出无意义的距离。
    auto pointSegDist = [](const QPointF &p, const QPointF &a, const QPointF &b) {
        const QPointF ab = b - a;
        const double len2 = ab.x() * ab.x() + ab.y() * ab.y();
        if (len2 <= 1e-12) {
            const QPointF d = p - a;
            return std::sqrt(d.x() * d.x() + d.y() * d.y());
        }
        double t = ((p - a).x() * ab.x() + (p - a).y() * ab.y()) / len2;
        t = std::max(0.0, std::min(1.0, t));
        const QPointF proj = a + ab * t;
        const QPointF d = proj - p;
        return std::sqrt(d.x() * d.x() + d.y() * d.y());
    };

    QVector<bool> keep(n, false);
    keep[0] = true;
    keep[n - 1] = true;

    // 显式栈：一次涂抹可能有上千个采样点，递归会把栈吃穿
    QVector<QPair<int, int>> stack;
    stack.append(qMakePair(0, n - 1));

    while (!stack.isEmpty()) {
        const QPair<int, int> range = stack.takeLast();
        const int first = range.first;
        const int last  = range.second;
        if (last <= first + 1)
            continue;

        double maxDist = -1.0;
        int    maxIdx  = -1;
        for (int i = first + 1; i < last; ++i) {
            const double d = pointSegDist(points.at(i), points.at(first), points.at(last));
            if (d > maxDist) {
                maxDist = d;
                maxIdx  = i;
            }
        }

        if (maxIdx >= 0 && maxDist > tolerance) {
            keep[maxIdx] = true;
            stack.append(qMakePair(first, maxIdx));
            stack.append(qMakePair(maxIdx, last));
        }
    }

    QPolygonF out;
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (keep[i])
            out.append(points.at(i));
    }
    return out.size() >= 2 ? out : points;
}

QVector<AnnoObject> objectsFromMask(const cv::Mat &mask, int minArea)
{
    QVector<AnnoObject> out;
    if (mask.empty())
        return out;

    cv::Mat idx;
    if (mask.type() != CV_8UC1)
        mask.convertTo(idx, CV_8UC1);
    else
        idx = mask;

    if (minArea < 1)
        minArea = 1;

    // 逐类别提取连通域，保证反推出的对象类别索引正确
    for (int c = 1; c < 256; ++c) {
        cv::Mat bin;
        cv::compare(idx, c, bin, cv::CMP_EQ);
        if (cv::countNonZero(bin) == 0)
            continue;

        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
        for (int i = 1; i < n; ++i) {
            const int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area < minArea)
                continue;

            const int x = stats.at<int>(i, cv::CC_STAT_LEFT);
            const int y = stats.at<int>(i, cv::CC_STAT_TOP);
            const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
            const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

            cv::Mat region = (labels(cv::Rect(x, y, w, h)) == i);

            // 用轮廓近似成多边形，便于后续顶点编辑；近似失败则退化为矩形
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(region, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            AnnoObject obj;
            obj.classId = c;
            obj.derived = true;

            if (!contours.empty() && contours.front().size() >= 3) {
                std::vector<cv::Point> approx;
                const double eps = std::max(1.0, 0.01 * cv::arcLength(contours.front(), true));
                cv::approxPolyDP(contours.front(), approx, eps, true);
                if (approx.size() >= 3) {
                    obj.type = AnnoObject::Polygon;
                    QPolygonF poly;
                    for (const cv::Point &p : approx)
                        poly.append(QPointF(p.x + x, p.y + y));
                    obj.points = poly;
                }
            }

            if (obj.type != AnnoObject::Polygon) {
                obj.type = AnnoObject::Rect;
                obj.rect = QRectF(x, y, w, h);
            }
            out.append(obj);
        }
    }
    return out;
}

// ===========================================================================
//  标注落盘 / 读入
// ===========================================================================
bool writeLabel(const QString &jsonPath, const QString &maskPath, TaskType task,
                const QVector<AnnoObject> &objects, int classId, AnnoState state,
                const QSize &imageSize, QString *err)
{
    QJsonObject root;
    root.insert(QStringLiteral("format"), QStringLiteral("AqroTrain.Label"));
    root.insert(QStringLiteral("version"), 1);
    root.insert(QStringLiteral("task"), taskTypeKey(task));
    root.insert(QStringLiteral("state"), annoStateKey(state));
    root.insert(QStringLiteral("width"), imageSize.width());
    root.insert(QStringLiteral("height"), imageSize.height());

    if (task == TaskType::Classification) {
        root.insert(QStringLiteral("classId"), classId);
    } else {
        root.insert(QStringLiteral("classId"), -1);
        QJsonArray arr;
        for (const AnnoObject &o : objects) {
            if (!o.isValid())
                continue;
            arr.append(o.toJson());
        }
        root.insert(QStringLiteral("objects"), arr);
    }

    if (!jsonio::writeObject(jsonPath, root, err))
        return false;

    // 分割：把对象列表光栅化成派生掩码（训练与导出使用掩码，编辑使用对象列表）
    if (task == TaskType::Segmentation) {
        if (imageSize.width() <= 0 || imageSize.height() <= 0) {
            if (err) *err = QStringLiteral("生成分割掩码失败：图像尺寸无效。");
            return false;
        }
        cv::Mat mask;
        anno::rasterize(objects, imageSize.width(), imageSize.height(), &mask);
        if (mask.empty() || !imageio::imwriteUnicode(maskPath, mask)) {
            if (err) *err = QStringLiteral("写入分割掩码失败：%1").arg(QFileInfo(maskPath).fileName());
            return false;
        }
    }
    return true;
}

bool readLabel(const QString &jsonPath, const QString &maskPath, TaskType task,
               QVector<AnnoObject> *objects, int *classId, AnnoState *state,
               int minArea, QString *err)
{
    if (objects)
        objects->clear();
    if (classId)
        *classId = -1;
    if (state)
        *state = AnnoState::None;

    QJsonObject root;
    if (!jsonio::readObject(jsonPath, &root, err)) {
        // .json 缺失：分割任务可尝试由派生掩码反推对象
        if (task == TaskType::Segmentation && QFileInfo(maskPath).exists()) {
            const cv::Mat mask = imageio::imreadUnicode(maskPath, cv::IMREAD_GRAYSCALE);
            if (!mask.empty()) {
                if (objects)
                    *objects = anno::objectsFromMask(mask, minArea);
                if (classId)
                    *classId = -1;
                if (state)
                    *state = AnnoState::Done;
                return true;
            }
        }
        if (err)
            *err = QStringLiteral("标注文件不存在：%1").arg(QFileInfo(jsonPath).fileName());
        return false;
    }

    if (state)
        *state = annoStateFromKey(jsonio::str(root, QStringLiteral("state"), QStringLiteral("none")));
    if (classId)
        *classId = jsonio::integer(root, QStringLiteral("classId"), -1);

    const QJsonArray arr = root.value(QStringLiteral("objects")).toArray();
    for (int i = 0; i < arr.size(); ++i) {
        bool ok = false;
        const AnnoObject o = AnnoObject::fromJson(arr.at(i).toObject(), &ok);
        if (ok && objects)
            objects->append(o);
    }
    return true;
}

bool hasTargetObject(const QVector<AnnoObject> &objects, const QVector<ClassInfo> &classes,
                     bool hasBackground)
{
    for (const AnnoObject &o : objects) {
        if (!o.isValid())
            continue;
        if (o.classId < 0 || o.classId >= classes.size())
            continue;
        if (hasBackground && o.classId == 0)
            continue;
        return true;
    }
    return false;
}

} // namespace anno
} // namespace aq
