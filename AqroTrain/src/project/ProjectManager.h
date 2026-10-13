// ============================================================================
//  ProjectManager.h —— 工程与模块的生命周期、图像池、样本状态、标注读写
//  ---------------------------------------------------------------------------
//  这是数据层唯一的对外门面：界面层只与本类交互，不直接碰文件。
//
//  AIDI 语义上的三处关键实现：
//    1. **串联删除**：removeModule(i) 会删除模块 i 及其之后的全部模块
//       （含它们的数据、模型、测试结果），与 AIDI 一致；
//    2. **图像池属于工程**：同一张图在多个模块下各有独立标注；
//    3. **数据划分**：每张图在每个模块下有 未划分/训练集/测试集 三态，
//       另有独立的 OK 图标记（无缺陷正例）。
//
//  目录创建的失败诊断（逐层 QDir::mkdir + 精确原因）也集中在这里，
//  因为曾实测遇到第三方安全软件拒绝创建目录而 ACL 一切正常的情况。
// ============================================================================
#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QVector>

#include "core/Types.h"
#include "annotate/AnnoObject.h"
#include "project/Module.h"
#include "project/Project.h"
#include "project/SampleIndex.h"

namespace aq {

class ProjectManager : public QObject
{
    Q_OBJECT

public:
    explicit ProjectManager(QObject *parent = nullptr);
    ~ProjectManager() override;

    // -----------------------------------------------------------------------
    //  静态工具：目录创建与属性诊断
    // -----------------------------------------------------------------------

    /// 一次目录创建尝试的结果（用于把「为什么失败」说清楚）
    struct DirCreateAttempt {
        QString strategy;          ///< 用了哪个 API
        bool    ok = false;        ///< 是否成功
        quint32 win32Error = 0;    ///< 失败时的 Win32 错误码（0 表示无/不适用）
        QString message;           ///< 失败原因（中文）
    };

    /**
     * 逐层单层创建目录，失败时给出可诊断的详细原因。
     *
     * 刻意不使用 QDir::mkpath（无法定位失败层级），也**不只用 QDir::mkdir**：
     * 实测在本机会出现「QDir::mkdir / CreateDirectoryW 失败，而同样的路径
     * 由 .NET Directory.CreateDirectory 却能建出来」的情况
     * （例如 D:\datasets\1\1，叶子名是纯数字）。因此这里按顺序尝试多种 Win32/CRT/外壳
     * API，任一成功即算成功，并把每个策略的真实错误码记进 attempts，便于定位。
     */
    static bool ensureDir(const QString &path, QString *err = nullptr,
                          QVector<DirCreateAttempt> *attempts = nullptr);

    /// 单层创建目录：依次尝试多种 API，任一成功即返回。
    /// 目录已存在（含竞态）视为成功。
    static bool createOneLevel(const QString &path, QVector<DirCreateAttempt> *attempts = nullptr,
                              QString *err = nullptr);

    /// 本进程能否往该目录写文件。
    /// 刻意用「真的写一个临时文件」来探测，而不是看 Windows 只读属性 /
    /// QFileInfo::isWritable() —— 实测目录带只读属性时这些接口的结论会互相矛盾，
    /// 而真正的拒绝（安全软件、策略）只能在真正写入时暴露。
    static bool probeWritable(const QString &dir, QString *err = nullptr);

    /**
     * 递归删除目录（含只读项），**支持超过 MAX_PATH 的长路径**。
     *
     * 为什么不用 QDir::removeRecursively：它内部走 Qt 的文件引擎，
     * 在 Windows 上不会加 `\\?\` 前缀，路径超过 260 字符就删不掉 ——
     * 于是出现「程序能建出长路径、却和 rmdir/资源管理器一样删不掉」的死局。
     * 这里用 FindFirstFileW / DeleteFileW / RemoveDirectoryW 自己递归，
     * 全程带长路径前缀。目录不存在视为删除成功。
     */
    static bool removeDirRecursive(const QString &path);

    /// 把一个 Win32 错误码翻译成中文说明
    static QString describeWin32Error(quint32 code);

    /// 目录是否带 Windows「只读」属性（只对已存在的路径判断）
    static bool looksReadOnly(const QString &path);

    /// 权限摘要（rwx 形式）
    static QString permissionString(const QString &path);

    /// 清除目录的只读属性。
    /// 刻意不基于 QFileInfo::permissions() 做前置判断——实测同一进程内对同一目录
    /// 可能先返回 r-x、后返回 rwx，且与 isWritable() 结论冲突，据此判断只会误导。
    static bool clearReadOnlyAttribute(const QString &dir, QString *err);

    /// 训练视图中的一个样本（契约见 core/Types.h 的 TrainSample）
    using TrainSample = aq::TrainSample;

    // -----------------------------------------------------------------------
    //  工程
    // -----------------------------------------------------------------------
    bool createProject(const QString &root, const QString &name, ImageFormat fmt, QString *err);
    bool openProject(const QString &root, QString *err);
    void closeProject();
    bool isOpen() const { return m_open; }

    const Project &project() const { return m_project; }
    QString        root() const { return m_project.root(); }
    QString        name() const { return m_project.name(); }
    QString        imagesDir() const { return m_project.imagesDir(); }
    ImageFormat    imageFormat() const { return m_project.imageFormat(); }

    /// 保存 project.json 与当前模块的 module.json / samples.json
    bool saveAll(QString *err = nullptr);

    // -----------------------------------------------------------------------
    //  模块
    // -----------------------------------------------------------------------
    int           moduleCount() const { return m_modules.size(); }
    const Module &moduleAt(int index) const;
    int           currentModuleIndex() const { return m_currentModule; }
    const Module *currentModule() const;

    /// 添加模块并创建其目录结构。index 返回新模块下标。
    bool addModule(TaskType type, const QString &name, QString *err, int *index = nullptr);

    /// 串联删除：删除下标 index 及其之后的全部模块（数据 / 模型 / 测试结果）
    bool removeModule(int index, QString *err);

    /// 切换当前模块（会载入其 module.json 与 samples.json）
    bool setCurrentModule(int index, QString *err);

    /// 重命名当前模块（同时更新 module.json 与工程里的模块名）
    bool renameCurrentModule(const QString &name, QString *err = nullptr);

    /// 保存当前模块的 module.json 与 samples.json
    bool saveCurrentModule(QString *err = nullptr);

    /// 修改当前模块的训练 / 测试参数并落盘。
    /// 界面层与自检都通过它写参数，避免直接改内部模块对象。
    bool setCurrentConfig(const TrainConfig &train, const TestConfig &test,
                          QString *err = nullptr);

    /// 只改当前模块的测试参数（置信度 / NMS / IoU / 最小面积）
    bool setCurrentTestConfig(const TestConfig &test, QString *err = nullptr);

    /// 只改当前模块的类别集合
    bool setCurrentClasses(const QVector<ClassInfo> &classes, QString *err = nullptr);

    /// 生成一个未被占用的模块 id
    QString makeModuleId() const;

    // -----------------------------------------------------------------------
    //  图像池
    // -----------------------------------------------------------------------
    const QStringList &imagePaths() const { return m_images; }
    int     imageCount() const { return m_images.size(); }
    /// 下标对应的图像池键（文件名）
    QString imageKey(int index) const;
    /// 由文件名反查下标；找不到返回 -1
    int     indexOfKey(const QString &imageKey) const;
    QSize   imageSize(int index, bool *ok = nullptr) const;

    /// 导入图像（按工程格式统一转码后落盘到 images/）。
    /// added 返回新增数量；failed 返回失败的文件名与原因。
    bool importImages(const QStringList &files, QString *err, int *added = nullptr,
                      QStringList *failed = nullptr);

    /// 导入整个文件夹（不递归）
    bool importFolder(const QString &dir, QString *err, int *added = nullptr,
                      QStringList *failed = nullptr);

    /// 从工程中删除一张图（同时删除各模块下它的标注与结果）
    bool removeImage(int index, QString *err);

    /// 重新扫描 images/ 目录，与内存中的列表对齐
    bool rescanImages(QString *err = nullptr);

    // -----------------------------------------------------------------------
    //  样本状态
    // -----------------------------------------------------------------------
    const SampleIndex &sampleIndex() const { return m_samples; }
    SampleState        sampleState(int index) const;
    /// 设置单张图的状态（会立即写入 samples.json）
    bool setSampleState(int index, const SampleState &s);
    /// 批量设置划分
    bool setSplit(const QVector<int> &indexes, Split split);
    /// 批量设置 OK 图标记
    bool setOk(const QVector<int> &indexes, bool ok);

    DatasetStats stats() const;

    // -----------------------------------------------------------------------
    //  标注读写（作用于当前模块）
    // -----------------------------------------------------------------------

    /// 载入某张图在当前模块下的标注。返回 false 表示尚无标注（不是错误）。
    bool loadAnnotation(int index, QVector<AnnoObject> *objects, int *classId, AnnoState *state,
                        QString *err = nullptr) const;

    /// 保存某张图的标注，并同步 samples.json 中的标注状态
    bool saveAnnotation(int index, const QVector<AnnoObject> &objects, int classId,
                        AnnoState state, QString *err = nullptr);

    /// 删除某张图的标注文件（含分割掩码）
    bool removeAnnotationFile(int index, QString *err = nullptr);

    /// 写入/读取某张图的单图预测结果（供「显示测试结果」使用）
    bool savePrediction(int index, const PredictionResult &pr, QString *err = nullptr) const;
    bool loadPrediction(int index, PredictionResult *pr, QString *err = nullptr) const;

    /// 评价指标
    bool       saveMetrics(const Metrics &m, QString *err = nullptr) const;
    Metrics    loadMetrics() const;
    bool       hasModel() const;

    // -----------------------------------------------------------------------
    //  训练视图
    // -----------------------------------------------------------------------
    /// 当前模块中划分到训练集、且已有标注的样本
    QVector<TrainSample> trainSamples() const;
    /// 当前模块中划分到测试集的样本（含未标注者，用于 OK/NG 评价）
    QVector<TrainSample> testSamples() const;

    /// 当前模块下尚无标注的图像下标（供智能标注批量处理）
    QVector<int> unannotatedIndexes() const;

signals:
    void projectOpened();
    void projectClosed();
    void modulesChanged();
    void currentModuleChanged(int index);
    void samplesChanged();
    void classesChanged();

private:
    bool createModuleDirs(const Module &m, QString *err) const;
    /// 把源图转码并落盘到 images/，返回新的文件名（失败返回空串）
    QString importOneImage(const QString &srcPath, QString *err);
    void rebuildImageList();
    void ensureSampleEntry(int index);

    bool           m_open = false;
    Project        m_project;
    QVector<Module> m_modules;
    int            m_currentModule = -1;
    SampleIndex    m_samples;
    QStringList    m_images;        ///< 绝对路径，与 m_samples 的键一一对应
};

} // namespace aq
