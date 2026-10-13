// ============================================================================
//  SampleIndex.cpp —— 模块级样本状态表实现
// ============================================================================
#include "project/SampleIndex.h"

#include <QJsonArray>
#include <QJsonObject>

#include "core/JsonStore.h"

namespace aq {

QString SampleIndex::fileName()
{
    return QStringLiteral("samples.json");
}

SampleState SampleIndex::state(const QString &imageKey) const
{
    const auto it = m_map.constFind(imageKey);
    return it == m_map.constEnd() ? SampleState() : it.value();
}

void SampleIndex::setState(const QString &imageKey, const SampleState &s)
{
    if (imageKey.isEmpty())
        return;
    m_map.insert(imageKey, s);
}

void SampleIndex::remove(const QString &imageKey)
{
    m_map.remove(imageKey);
}

bool SampleIndex::syncWith(const QStringList &imageKeys)
{
    bool changed = false;

    QHash<QString, bool> present;
    present.reserve(imageKeys.size());
    for (const QString &k : imageKeys) {
        present.insert(k, true);
        if (!m_map.contains(k)) {
            m_map.insert(k, SampleState());
            changed = true;
        }
    }

    const QStringList existing = m_map.keys();
    for (const QString &k : existing) {
        if (!present.contains(k)) {
            m_map.remove(k);
            changed = true;
        }
    }
    return changed;
}

QStringList SampleIndex::keys() const
{
    QStringList out = m_map.keys();
    out.sort(Qt::CaseInsensitive);
    return out;
}

int SampleIndex::count(Split s) const
{
    int n = 0;
    for (auto it = m_map.constBegin(); it != m_map.constEnd(); ++it) {
        if (it.value().split == s)
            ++n;
    }
    return n;
}

int SampleIndex::okCount() const
{
    int n = 0;
    for (auto it = m_map.constBegin(); it != m_map.constEnd(); ++it) {
        if (it.value().ok)
            ++n;
    }
    return n;
}

int SampleIndex::annotatedCount() const
{
    int n = 0;
    for (auto it = m_map.constBegin(); it != m_map.constEnd(); ++it) {
        if (it.value().state != AnnoState::None)
            ++n;
    }
    return n;
}

bool SampleIndex::load(const QString &path, QString *err)
{
    m_map.clear();

    QJsonObject o;
    if (!jsonio::readObject(path, &o, err)) {
        // 文件不存在属于正常情况（新建模块尚无任何划分），不作为错误
        return false;
    }

    const QJsonObject samples = o.value(QStringLiteral("samples")).toObject();
    for (auto it = samples.constBegin(); it != samples.constEnd(); ++it) {
        const QJsonObject s = it.value().toObject();
        SampleState st;
        bool ok = false;
        st.split = splitFromKey(jsonio::str(s, QStringLiteral("split"), QStringLiteral("none")), &ok);
        st.ok    = jsonio::boolean(s, QStringLiteral("ok"), false);
        st.state = annoStateFromKey(jsonio::str(s, QStringLiteral("state"), QStringLiteral("none")), &ok);
        m_map.insert(it.key(), st);
    }
    return true;
}

bool SampleIndex::save(const QString &path, QString *err) const
{
    QJsonObject samples;
    const QStringList ks = keys();   // 排序后写入，落盘内容稳定
    for (const QString &k : ks) {
        const SampleState &st = m_map[k];
        QJsonObject s;
        s.insert(QStringLiteral("split"), splitKey(st.split));
        s.insert(QStringLiteral("ok"), st.ok);
        s.insert(QStringLiteral("state"), annoStateKey(st.state));
        samples.insert(k, s);
    }

    QJsonObject root;
    root.insert(QStringLiteral("format"), QStringLiteral("AqroTrain.Samples"));
    root.insert(QStringLiteral("version"), 1);
    root.insert(QStringLiteral("samples"), samples);
    return jsonio::writeObject(path, root, err);
}

} // namespace aq
