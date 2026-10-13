// ============================================================================
//  Project.h —— 工程模型（对应 AIDI 的「工程」）
//  ---------------------------------------------------------------------------
//  工程目录结构：
//      <工程根>/
//        project.json          工程描述（名称 / 图片格式 / 模块列表）
//        images/               工程级图像池（导入时统一转码为工程图片格式）
//        thumbs/               缩略图缓存
//        modules/<moduleId>/   各模块的数据、模型与结果（见 Module.h）
//
//  「图像池属于工程、标注属于模块」是与 AIDI 一致的关键约定：
//  同一张图可以在多个模块里各有独立标注。
// ============================================================================
#pragma once

#include <QJsonObject>
#include <QString>
#include <QVector>

#include "core/Types.h"

namespace aq {

class Project
{
public:
    /// 工程描述文件名
    static QString fileName();
    /// 格式标识，用于校验读入的文件确属本软件
    static QString formatTag();

    /// 模块引用：工程只记录模块的「身份与顺序」，具体内容在模块自己的 module.json 里
    struct ModuleRef {
        QString  id;
        QString  name;
        TaskType type  = TaskType::Classification;
        int      order = 0;
    };

    Project() = default;

    // --- 基本信息 ---------------------------------------------------------
    QString     root() const { return m_root; }
    void        setRoot(const QString &dir) { m_root = dir; }

    QString     name() const { return m_name; }
    void        setName(const QString &n) { m_name = n; }

    ImageFormat imageFormat() const { return m_imageFormat; }
    void        setImageFormat(ImageFormat f) { m_imageFormat = f; }

    QString     created() const { return m_created; }
    void        setCreated(const QString &t) { m_created = t; }

    QString     description() const { return m_description; }
    void        setDescription(const QString &d) { m_description = d; }

    // --- 模块引用 ---------------------------------------------------------
    const QVector<ModuleRef> &moduleRefs() const { return m_modules; }
    void setModuleRefs(const QVector<ModuleRef> &refs) { m_modules = refs; }

    void appendModuleRef(const ModuleRef &ref);
    /// 删除下标 index 及其之后的全部模块引用（AIDI 的串联删除语义）
    void removeModuleRefsFrom(int index);
    /// 重新编号 order 字段，使其等于下标
    void renumberModuleRefs();

    // --- 路径 -------------------------------------------------------------
    QString filePath() const;
    QString imagesDir() const;
    QString thumbsDir() const;
    QString modulesDir() const;
    QString moduleDir(const QString &moduleId) const;

    // --- 序列化 -----------------------------------------------------------
    QJsonObject toJson() const;
    static Project fromJson(const QJsonObject &o, const QString &root);

    bool save(QString *err) const;
    static bool load(const QString &root, Project *out, QString *err);

private:
    QString             m_root;
    QString             m_name;
    ImageFormat         m_imageFormat = ImageFormat::PNG;
    QString             m_created;
    QString             m_description;
    QVector<ModuleRef>  m_modules;
};

} // namespace aq
