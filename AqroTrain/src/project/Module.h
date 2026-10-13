// ============================================================================
//  Module.h —— 模块模型（对应 AIDI 的「模块」）
//  ---------------------------------------------------------------------------
//  模块目录结构：
//      <工程>/modules/<moduleId>/
//        module.json     模块描述（类型 / 类别 / 训练与测试参数）
//        samples.json    每张图的 数据划分 + OK 标记 + 标注状态
//        labels/         该模块的标注（.json 为真源；分割另含派生 .png 掩码）
//        models/         训练产物 model.* + model.json 元数据
//        results/        metrics.json + <图名>.json（每图预测）
//        logs/           训练日志
//
//  类别索引约定（全局统一，与 AIDI 一致）：
//      · 图像分类：classes 只含真实类别，索引 0..N-1，无背景类
//      · 目标检测 / 图像分割：索引 0 固定为「背景」，目标类别从 1 开始
// ============================================================================
#pragma once

#include <QJsonObject>
#include <QString>
#include <QVector>

#include "core/Types.h"

namespace aq {

class Module
{
public:
    static QString fileName();          ///< module.json
    static QString samplesFileName();   ///< samples.json
    static QString labelsDirName();     ///< labels
    static QString modelsDirName();     ///< models
    static QString resultsDirName();    ///< results
    static QString logsDirName();       ///< logs

    /// 背景类别名（检测 / 分割的索引 0）
    static QString backgroundClassName();

    Module() = default;

    // --- 基本信息 ---------------------------------------------------------
    QString  id() const { return m_id; }
    void     setId(const QString &v) { m_id = v; }

    QString  name() const { return m_name; }
    void     setName(const QString &v) { m_name = v; }

    TaskType type() const { return m_type; }
    void     setType(TaskType t) { m_type = t; }

    int      order() const { return m_order; }
    void     setOrder(int v) { m_order = v; }

    QString  root() const { return m_root; }
    void     setRoot(const QString &dir) { m_root = dir; }

    // --- 类别 -------------------------------------------------------------
    const QVector<ClassInfo> &classes() const { return m_classes; }
    void setClasses(const QVector<ClassInfo> &c) { m_classes = c; }
    QVector<ClassInfo> &classesRef() { return m_classes; }

    /// 类别数（含背景）
    int classCount() const { return m_classes.size(); }
    /// 检测 / 分割是否包含背景类
    bool hasBackground() const { return m_type != TaskType::Classification; }
    /// 可作为标注目标的类别数（分类 = 类别数；检测/分割 = 类别数 - 1）
    int targetClassCount() const;
    /// 可标注类别的最小索引（分类 = 0；检测/分割 = 1）
    int firstTargetClass() const { return hasBackground() ? 1 : 0; }
    /// 该索引是否可用于标注（不是背景类，且在范围内）
    bool isTargetClass(int classId) const;
    /// 索引对应的类别名（越界返回空串）
    QString className(int classId) const;
    /// 索引对应的颜色（越界用默认调色板）
    QColor  classColor(int classId) const;

    /// 按任务类型补齐默认类别（分类至少 1 个；检测/分割自动插入背景类）
    void ensureDefaultClasses();

    // --- 参数 -------------------------------------------------------------
    const TrainConfig &train() const { return m_train; }
    TrainConfig       &trainRef() { return m_train; }
    void setTrain(const TrainConfig &t) { m_train = t; }

    const TestConfig &test() const { return m_test; }
    TestConfig       &testRef() { return m_test; }
    void setTest(const TestConfig &t) { m_test = t; }

    // --- 路径 -------------------------------------------------------------
    QString filePath() const;        ///< module.json
    QString samplesFilePath() const; ///< samples.json
    QString labelsDir() const;
    QString modelsDir() const;
    QString resultsDir() const;
    QString logsDir() const;
    QString modelPath() const;       ///< models/model.<ext>
    QString modelMetaPath() const;   ///< models/model.<ext>.json
    QString metricsPath() const;     ///< results/metrics.json
    /// 某张图的标注真源路径（.json）
    QString labelPath(const QString &imageKey) const;
    /// 某张图的分割掩码路径（.png，派生物）
    QString maskPath(const QString &imageKey) const;
    /// 某张图的预测结果路径（results/<图名>.json）
    QString resultPath(const QString &imageKey) const;
    /// 训练视图的根目录（等于模块根目录；训练后端据此定位 logs/ 等）
    QString datasetRoot() const { return m_root; }

    // --- 序列化 -----------------------------------------------------------
    QJsonObject toJson() const;
    static Module fromJson(const QJsonObject &o, const QString &root);

    bool save(QString *err) const;
    static bool load(const QString &moduleDir, Module *out, QString *err);

private:
    QString            m_id;
    QString            m_name;
    TaskType           m_type = TaskType::Classification;
    int                m_order = 0;
    QString            m_root;
    QVector<ClassInfo> m_classes;
    TrainConfig        m_train;
    TestConfig         m_test;
};

} // namespace aq
