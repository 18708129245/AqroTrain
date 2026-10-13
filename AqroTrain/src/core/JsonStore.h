// ============================================================================
//  JsonStore.h —— 轻量 JSON 读写与常用结构序列化
//  ---------------------------------------------------------------------------
//  全部走 Qt 自带的 QJsonDocument，**不使用 OpenCV 的 FileStorage**：
//  FileStorage 会依赖临时文件中转，在受限环境与中文路径下曾出现
//  「Can't open file ... in write mode」的失败（已在 VisionTrain 项目中实测）。
//  这里由 Qt 负责读写盘，路径天然支持中文。
// ============================================================================
#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QPolygonF>
#include <QRectF>
#include <QString>
#include <QVector>

#include "core/Types.h"

namespace aq {
namespace jsonio {

// --- 文件读写 --------------------------------------------------------------

/// 读取 JSON 对象。文件不存在或解析失败返回 false 并写 err。
bool readObject(const QString &path, QJsonObject *out, QString *err = nullptr);

/// 原子写入 JSON 对象（先写 .tmp 再替换，避免中途失败留下半截文件）。
bool writeObject(const QString &path, const QJsonObject &obj, QString *err = nullptr);

/// 读取 JSON 数组（如标注对象列表）。失败返回 false。
bool readArray(const QString &path, QJsonArray *out, QString *err = nullptr);

// --- 常用结构 --------------------------------------------------------------

QJsonArray          classesToJson(const QVector<ClassInfo> &classes);
QVector<ClassInfo>  classesFromJson(const QJsonArray &arr, bool *ok = nullptr);

QJsonArray  polygonToJson(const QPolygonF &poly);
QPolygonF   polygonFromJson(const QJsonArray &arr);

QJsonArray  rectToJson(const QRectF &r);          ///< [x, y, w, h]
QRectF      rectFromJson(const QJsonArray &arr);

// --- 小工具 ----------------------------------------------------------------

double  num(const QJsonObject &o, const QString &key, double def);
int     integer(const QJsonObject &o, const QString &key, int def);
bool    boolean(const QJsonObject &o, const QString &key, bool def);
QString str(const QJsonObject &o, const QString &key, const QString &def = QString());

} // namespace jsonio
} // namespace aq
