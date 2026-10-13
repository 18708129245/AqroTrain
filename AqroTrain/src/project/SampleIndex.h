// ============================================================================
//  SampleIndex.h —— 模块级样本状态表（samples.json）
//  ---------------------------------------------------------------------------
//  记录「某张图在本模块下的」数据划分、OK 标记与标注状态。
//  键为图像池中的文件名（工程内唯一），值见 SampleState。
//
//  该文件是派生状态的集中存放处：即使标注文件被外部删除，
//  也能据此发现不一致并给出提示。
// ============================================================================
#pragma once

#include <QHash>
#include <QString>
#include <QStringList>
#include <QVector>

#include "core/Types.h"

namespace aq {

class SampleIndex
{
public:
    static QString fileName();   ///< samples.json

    SampleIndex() = default;

    void clear() { m_map.clear(); }

    bool contains(const QString &imageKey) const { return m_map.contains(imageKey); }
    int  size() const { return m_map.size(); }

    SampleState state(const QString &imageKey) const;
    void        setState(const QString &imageKey, const SampleState &s);
    void        remove(const QString &imageKey);

    /// 与图像池对齐：补上缺失的条目，删掉已不存在的条目。返回是否有变化。
    bool syncWith(const QStringList &imageKeys);

    /// 排序后的键列表（保证落盘内容稳定，便于比对）
    QStringList keys() const;

    int count(Split s) const;
    int okCount() const;
    int annotatedCount() const;

    bool load(const QString &path, QString *err = nullptr);
    bool save(const QString &path, QString *err = nullptr) const;

private:
    QHash<QString, SampleState> m_map;
};

} // namespace aq
