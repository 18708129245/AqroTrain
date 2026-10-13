// ============================================================================
//  ImageIO.h —— 图像读写与 Qt/OpenCV 互转工具
//  ---------------------------------------------------------------------------
//  重点解决两个问题：
//   1. OpenCV 的 cv::imread / cv::imwrite 在 Windows 下无法处理含中文的路径，
//      这里统一走「Qt 读字节 -> cv::imdecode」的方式；
//   2. cv::Mat（BGR）与 QImage（RGB）之间的零歧义互转，供界面显示使用。
// ============================================================================
#pragma once

#include <QImage>
#include <QString>
#include <QStringList>
#include <QVector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "core/Types.h"

namespace aq {
namespace imageio {

// --- 读写 ------------------------------------------------------------------

/// 读取图像（支持中文路径）。flags 同 cv::ImreadModes。失败返回空 Mat。
cv::Mat imreadUnicode(const QString &path, int flags = cv::IMREAD_COLOR);

/// 写入图像（支持中文路径）。失败返回 false。
bool imwriteUnicode(const QString &path, const cv::Mat &image);

/// 被支持作为数据源的图像扩展名（小写，含点）
QStringList supportedImageExtensions();

/// 判断是否为可读入的图像文件（按扩展名）
bool isImageFile(const QString &path);

/// 扫描目录下的图像文件（不递归），按文件名自然排序，返回绝对路径
QStringList listImages(const QString &dir);

// --- 互转 ------------------------------------------------------------------

/// cv::Mat -> QImage（BGR/BGRA/GRAY 自动识别，返回深拷贝）
QImage matToQImage(const cv::Mat &mat);

/// QImage -> cv::Mat（统一输出 CV_8UC3 BGR）
cv::Mat qImageToMat(const QImage &image);

// --- 掩码着色 --------------------------------------------------------------

/**
 * 把类别索引掩码（CV_8UC1，像素值=类别索引）着色成 BGR 图像。
 * alpha 为前景不透明度，0 表示只输出纯色底图。
 */
cv::Mat colorizeMask(const cv::Mat &mask, const QVector<ClassInfo> &classes, double alpha = 1.0);

/**
 * 把索引掩码转成带透明度的 RGBA QImage，用于叠加在原始图像之上。
 * 背景（索引 0 且 classes[0] 未定义时）保持透明。
 */
QImage maskToOverlay(const cv::Mat &mask, const QVector<ClassInfo> &classes, int alpha = 120);

/// 生成「棋盘格」背景图，用于显示透明区域
QImage checkerboard(int width, int height, int cell = 12,
                    const QColor &c1 = QColor(0x2A, 0x2F, 0x3A),
                    const QColor &c2 = QColor(0x22, 0x26, 0x2E));

} // namespace imageio
} // namespace aq
