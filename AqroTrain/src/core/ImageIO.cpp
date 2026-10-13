// ============================================================================
//  ImageIO.cpp —— 图像读写与 Qt/OpenCV 互转实现
// ============================================================================
#include "core/ImageIO.h"

#include <QCollator>
#include <QDir>
#include <QFileInfo>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <vector>

#include <opencv2/core.hpp>

namespace aq {
namespace imageio {

// ---------------------------------------------------------------------------
//  读写
// ---------------------------------------------------------------------------
cv::Mat imreadUnicode(const QString &path, int flags)
{
    // 以二进制方式读入整个文件，再交给 cv::imdecode —— 绕开 OpenCV 的窄字符路径限制
    std::ifstream ifs(path.toStdWString(), std::ios::binary);
    if (!ifs.is_open())
        return cv::Mat();

    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(ifs)),
                                   std::istreambuf_iterator<char>());
    ifs.close();

    if (buf.empty())
        return cv::Mat();

    return cv::imdecode(buf, flags);
}

bool imwriteUnicode(const QString &path, const cv::Mat &image)
{
    if (image.empty())
        return false;

    // 依据扩展名选择编码参数
    const QString ext = QFileInfo(path).suffix().toLower();
    std::vector<int> params;
    if (ext == QLatin1String("png")) {
        params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
    } else if (ext == QLatin1String("jpg") || ext == QLatin1String("jpeg")) {
        params = { cv::IMWRITE_JPEG_QUALITY, 95 };
    } else if (ext == QLatin1String("bmp")) {
        params = {};
    }

    std::vector<unsigned char> buf;
    if (!cv::imencode("." + ext.toStdString(), image, buf, params))
        return false;

    std::ofstream ofs(path.toStdWString(), std::ios::binary);
    if (!ofs.is_open())
        return false;

    ofs.write(reinterpret_cast<const char *>(buf.data()),
              static_cast<std::streamsize>(buf.size()));
    ofs.close();
    return ofs.good();
}

QStringList supportedImageExtensions()
{
    return { QStringLiteral(".jpg"),  QStringLiteral(".jpeg"), QStringLiteral(".png"),
             QStringLiteral(".bmp"),  QStringLiteral(".tif"),  QStringLiteral(".tiff"),
             QStringLiteral(".webp") };
}

bool isImageFile(const QString &path)
{
    const QString ext = QFileInfo(path).suffix().toLower();
    if (ext.isEmpty())
        return false;
    return supportedImageExtensions().contains(QLatin1Char('.') + ext);
}

QStringList listImages(const QString &dir)
{
    QDir d(dir);
    if (!d.exists())
        return {};

    QStringList filters;
    for (const QString &e : supportedImageExtensions())
        filters << QLatin1Char('*') + e;

    const QFileInfoList infos =
        d.entryInfoList(filters, QDir::Files | QDir::Readable, QDir::NoSort);

    QStringList names;
    names.reserve(infos.size());
    for (const QFileInfo &fi : infos)
        names << fi.fileName();

    // 自然排序，保证 img2 排在 img10 前面
    QCollator collator;
    collator.setNumericMode(true);
    collator.setCaseSensitivity(Qt::CaseInsensitive);
    std::sort(names.begin(), names.end(),
              [&collator](const QString &a, const QString &b) { return collator.compare(a, b) < 0; });

    QStringList out;
    out.reserve(names.size());
    for (const QString &n : names)
        out << d.absoluteFilePath(n);
    return out;
}

// ---------------------------------------------------------------------------
//  互转
// ---------------------------------------------------------------------------
QImage matToQImage(const cv::Mat &mat)
{
    if (mat.empty())
        return QImage();

    switch (mat.type()) {
    case CV_8UC1: {
        QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step),
                   QImage::Format_Grayscale8);
        return img.copy();
    }
    case CV_8UC3: {
        cv::Mat rgb;
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
        QImage img(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step),
                   QImage::Format_RGB888);
        return img.copy();
    }
    case CV_8UC4: {
        cv::Mat rgba;
        cv::cvtColor(mat, rgba, cv::COLOR_BGRA2RGBA);
        QImage img(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step),
                   QImage::Format_RGBA8888);
        return img.copy();
    }
    default:
        break;
    }

    // 其余类型统一归一化到 8 位再转换
    cv::Mat tmp;
    if (mat.depth() == CV_32F || mat.depth() == CV_64F)
        mat.convertTo(tmp, CV_8U, 255.0);
    else
        mat.convertTo(tmp, CV_8U);

    if (tmp.channels() == 1)
        return matToQImage(tmp);
    if (tmp.channels() == 4)
        return matToQImage(tmp);
    return QImage();
}

cv::Mat qImageToMat(const QImage &image)
{
    if (image.isNull())
        return cv::Mat();

    QImage src = image;
    if (src.format() != QImage::Format_RGB888 && src.format() != QImage::Format_Grayscale8)
        src = src.convertToFormat(QImage::Format_RGB888);

    if (src.format() == QImage::Format_Grayscale8) {
        cv::Mat gray(src.height(), src.width(), CV_8UC1,
                     const_cast<uchar *>(src.constBits()), static_cast<size_t>(src.bytesPerLine()));
        cv::Mat bgr;
        cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
        return bgr.clone();
    }

    cv::Mat rgb(src.height(), src.width(), CV_8UC3,
                const_cast<uchar *>(src.constBits()), static_cast<size_t>(src.bytesPerLine()));
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr.clone();
}

// ---------------------------------------------------------------------------
//  掩码着色
// ---------------------------------------------------------------------------
cv::Mat colorizeMask(const cv::Mat &mask, const QVector<ClassInfo> &classes, double alpha)
{
    if (mask.empty())
        return cv::Mat();

    cv::Mat idx;
    if (mask.type() != CV_8UC1)
        mask.convertTo(idx, CV_8UC1);
    else
        idx = mask;

    /*
     * 注意：cv::LUT 要求查找表的通道数等于输入通道数或为 1
     *      （见 opencv/modules/core/src/lut.cpp 的
     *        CV_Assert((lutcn == cn || lutcn == 1) && ...)）。
     *      单通道索引图配三通道 LUT 会直接抛异常，
     *      因此这里对 B / G / R 三个平面分别查表，再用 merge 合成彩色图。
     */
    cv::Mat lutB(1, 256, CV_8UC1);
    cv::Mat lutG(1, 256, CV_8UC1);
    cv::Mat lutR(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i) {
        QColor c;
        if (i < classes.size())
            c = classes[i].color;
        else
            c = defaultClassColor(i);
        lutB.at<uchar>(0, i) = static_cast<uchar>(c.blue());
        lutG.at<uchar>(0, i) = static_cast<uchar>(c.green());
        lutR.at<uchar>(0, i) = static_cast<uchar>(c.red());
    }

    cv::Mat planeB, planeG, planeR;
    cv::LUT(idx, lutB, planeB);
    cv::LUT(idx, lutG, planeG);
    cv::LUT(idx, lutR, planeR);

    cv::Mat color;
    const std::vector<cv::Mat> planes { planeB, planeG, planeR };
    cv::merge(planes, color);

    if (alpha >= 0.999)
        return color;

    // 与黑底按 alpha 混合（用于半透明叠加前的准备）
    cv::Mat out;
    color.convertTo(out, CV_32FC3);
    out *= alpha;
    out.convertTo(out, CV_8UC3);
    return out;
}

QImage maskToOverlay(const cv::Mat &mask, const QVector<ClassInfo> &classes, int alpha)
{
    if (mask.empty())
        return QImage();

    cv::Mat idx;
    if (mask.type() != CV_8UC1)
        mask.convertTo(idx, CV_8UC1);
    else
        idx = mask;

    cv::Mat color = colorizeMask(idx, classes, 1.0);
    cv::Mat rgba;
    cv::cvtColor(color, rgba, cv::COLOR_BGR2RGBA);
    rgba.setTo(cv::Scalar(0, 0, 0, 0), idx == 0);   // 索引 0 视作透明背景

    const int a = std::max(0, std::min(255, alpha));
    for (int y = 0; y < rgba.rows; ++y) {
        cv::Vec4b *row = rgba.ptr<cv::Vec4b>(y);
        for (int x = 0; x < rgba.cols; ++x) {
            if (row[x][3] != 0)
                row[x][3] = static_cast<uchar>(a);
        }
    }

    QImage img(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step),
               QImage::Format_RGBA8888);
    return img.copy();
}

QImage checkerboard(int width, int height, int cell, const QColor &c1, const QColor &c2)
{
    if (width <= 0 || height <= 0 || cell <= 0)
        return QImage();

    QImage img(width, height, QImage::Format_RGB32);
    for (int y = 0; y < height; ++y) {
        QRgb *row = reinterpret_cast<QRgb *>(img.scanLine(y));
        for (int x = 0; x < width; ++x) {
            const bool odd = ((x / cell) + (y / cell)) & 1;
            row[x] = odd ? c1.rgb() : c2.rgb();
        }
    }
    return img;
}

} // namespace imageio
} // namespace aq
