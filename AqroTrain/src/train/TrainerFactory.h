// ============================================================================
//  TrainerFactory.h —— 训练后端工厂
//  ---------------------------------------------------------------------------
//  按优先级挑选可用的训练后端：
//    LibTorch（若已安装并在编译期启用） > OpenCV 轻量后端
// ============================================================================
#pragma once

#include <QString>

class QObject;

namespace aq {

class ITrainer;

class TrainerFactory
{
public:
    enum Backend {
        Auto     = 0,   ///< 自动：优先 LibTorch，不可用则退回 OpenCV
        LibTorch = 1,   ///< 强制 LibTorch，不可用时返回 nullptr
        OpenCV   = 2    ///< 强制 OpenCV 轻量后端
    };

    /// 由配置里的后端名（"opencv" / "libtorch" / "auto"）解析
    static Backend fromKey(const QString &key);

    /// 创建训练后端实例；失败返回 nullptr（由调用方负责 delete）
    static ITrainer *create(QObject *parent = nullptr, Backend backend = Auto);

    /// 各后端可用性说明（多行文本，供「关于 / 环境信息」展示）
    static QString backendStatus();

    /// 当前是否会使用 LibTorch 后端
    static bool libTorchAvailable(QString *reason = nullptr);
};

} // namespace aq
