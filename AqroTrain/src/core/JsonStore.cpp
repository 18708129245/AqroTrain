// ============================================================================
//  JsonStore.cpp —— 轻量 JSON 读写实现
// ============================================================================
#include "core/JsonStore.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonParseError>
#include <QSaveFile>
#include <QTextStream>

namespace aq {
namespace jsonio {

// ---------------------------------------------------------------------------
//  文件读写
// ---------------------------------------------------------------------------
bool readObject(const QString &path, QJsonObject *out, QString *err)
{
    if (out)
        *out = QJsonObject();

    QFile f(path);
    if (!f.exists()) {
        if (err) *err = QStringLiteral("文件不存在：%1").arg(path);
        return false;
    }
    if (!f.open(QIODevice::ReadOnly)) {
        if (err) *err = QStringLiteral("无法读取文件：%1（%2）").arg(path, f.errorString());
        return false;
    }
    const QByteArray data = f.readAll();
    f.close();

    QJsonParseError pe {};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &pe);
    if (pe.error != QJsonParseError::NoError) {
        if (err) *err = QStringLiteral("JSON 解析失败：%1（偏移 %2）").arg(pe.errorString()).arg(pe.offset);
        return false;
    }
    if (!doc.isObject()) {
        if (err) *err = QStringLiteral("JSON 根节点不是对象：%1").arg(path);
        return false;
    }
    if (out)
        *out = doc.object();
    return true;
}

bool writeObject(const QString &path, const QJsonObject &obj, QString *err)
{
    QDir().mkpath(QFileInfo(path).absolutePath());

    // QSaveFile：写临时文件后原子替换，避免写一半失败留下损坏的 JSON
    QSaveFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (err) *err = QStringLiteral("无法写入文件：%1（%2）").arg(path, f.errorString());
        return false;
    }
    const QByteArray data = QJsonDocument(obj).toJson(QJsonDocument::Indented);
    if (f.write(data) != data.size()) {
        if (err) *err = QStringLiteral("写入不完整：%1").arg(path);
        f.cancelWriting();
        return false;
    }
    if (!f.commit()) {
        if (err) *err = QStringLiteral("提交文件失败：%1（%2）").arg(path, f.errorString());
        return false;
    }
    return true;
}

bool readArray(const QString &path, QJsonArray *out, QString *err)
{
    if (out)
        *out = QJsonArray();

    QFile f(path);
    if (!f.exists()) {
        if (err) *err = QStringLiteral("文件不存在：%1").arg(path);
        return false;
    }
    if (!f.open(QIODevice::ReadOnly)) {
        if (err) *err = QStringLiteral("无法读取文件：%1（%2）").arg(path, f.errorString());
        return false;
    }
    const QByteArray data = f.readAll();
    f.close();

    QJsonParseError pe {};
    const QJsonDocument doc = QJsonDocument::fromJson(data, &pe);
    if (pe.error != QJsonParseError::NoError) {
        if (err) *err = QStringLiteral("JSON 解析失败：%1").arg(pe.errorString());
        return false;
    }
    if (out)
        *out = doc.array();
    return true;
}

// ---------------------------------------------------------------------------
//  类别
// ---------------------------------------------------------------------------
QJsonArray classesToJson(const QVector<ClassInfo> &classes)
{
    QJsonArray arr;
    for (const ClassInfo &c : classes) {
        QJsonObject o;
        o.insert(QStringLiteral("name"), c.name);
        o.insert(QStringLiteral("color"), c.color.name(QColor::HexRgb));
        arr.append(o);
    }
    return arr;
}

QVector<ClassInfo> classesFromJson(const QJsonArray &arr, bool *ok)
{
    QVector<ClassInfo> out;
    out.reserve(arr.size());
    for (int i = 0; i < arr.size(); ++i) {
        const QJsonObject o = arr.at(i).toObject();
        const QString name = str(o, QStringLiteral("name"));
        if (name.isEmpty())
            continue;
        QColor color(str(o, QStringLiteral("color")));
        if (!color.isValid())
            color = defaultClassColor(out.size());
        out.append(ClassInfo(name, color));
    }
    if (ok)
        *ok = true;
    return out;
}

// ---------------------------------------------------------------------------
//  几何
// ---------------------------------------------------------------------------
QJsonArray polygonToJson(const QPolygonF &poly)
{
    QJsonArray arr;
    for (const QPointF &p : poly) {
        QJsonArray pt;
        pt.append(p.x());
        pt.append(p.y());
        arr.append(pt);
    }
    return arr;
}

QPolygonF polygonFromJson(const QJsonArray &arr)
{
    QPolygonF poly;
    for (int i = 0; i < arr.size(); ++i) {
        const QJsonArray pt = arr.at(i).toArray();
        if (pt.size() < 2)
            continue;
        poly.append(QPointF(pt.at(0).toDouble(), pt.at(1).toDouble()));
    }
    return poly;
}

QJsonArray rectToJson(const QRectF &r)
{
    QJsonArray arr;
    arr.append(r.x());
    arr.append(r.y());
    arr.append(r.width());
    arr.append(r.height());
    return arr;
}

QRectF rectFromJson(const QJsonArray &arr)
{
    if (arr.size() < 4)
        return QRectF();
    return QRectF(arr.at(0).toDouble(), arr.at(1).toDouble(),
                  arr.at(2).toDouble(), arr.at(3).toDouble());
}

// ---------------------------------------------------------------------------
//  小工具
// ---------------------------------------------------------------------------
double num(const QJsonObject &o, const QString &key, double def)
{
    const QJsonValue v = o.value(key);
    return v.isDouble() ? v.toDouble() : def;
}

int integer(const QJsonObject &o, const QString &key, int def)
{
    const QJsonValue v = o.value(key);
    return v.isDouble() ? v.toInt() : def;
}

bool boolean(const QJsonObject &o, const QString &key, bool def)
{
    const QJsonValue v = o.value(key);
    return v.isBool() ? v.toBool() : def;
}

QString str(const QJsonObject &o, const QString &key, const QString &def)
{
    const QJsonValue v = o.value(key);
    return v.isString() ? v.toString() : def;
}

} // namespace jsonio
} // namespace aq
