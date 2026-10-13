// ============================================================================
//  TrainerFactory.cpp
// ============================================================================
#include "train/TrainerFactory.h"

#include <QStringList>

#include "train/ITrainer.h"
#include "train/LibTorchTrainer.h"
#include "train/OpenCvTrainer.h"

namespace aq {

TrainerFactory::Backend TrainerFactory::fromKey(const QString &key)
{
    const QString k = key.trimmed().toLower();
    if (k == QLatin1String("libtorch") || k == QLatin1String("torch"))
        return LibTorch;
    if (k == QLatin1String("opencv") || k == QLatin1String("cv"))
        return OpenCV;
    return Auto;
}

bool TrainerFactory::libTorchAvailable(QString *reason)
{
    LibTorchTrainer probe;
    return probe.isAvailable(reason);
}

ITrainer *TrainerFactory::create(QObject *parent, Backend backend)
{
    switch (backend) {
    case OpenCV:
        return new OpenCvTrainer(parent);

    case LibTorch: {
        auto *lt = new LibTorchTrainer(parent);
        QString reason;
        if (lt->isAvailable(&reason))
            return lt;
        delete lt;
        return nullptr;
    }

    case Auto:
    default: {
        auto *lt = new LibTorchTrainer(parent);
        QString reason;
        if (lt->isAvailable(&reason))
            return lt;
        delete lt;
        return new OpenCvTrainer(parent);
    }
    }
}

QString TrainerFactory::backendStatus()
{
    QStringList lines;

    LibTorchTrainer lt;
    QString reason;
    if (lt.isAvailable(&reason)) {
        lines << QStringLiteral("[可用]  LibTorch 深度学习后端（将优先使用）");
    } else {
        lines << QStringLiteral("[不可用] LibTorch 深度学习后端");
        lines << QStringLiteral("    原因：%1").arg(reason);
        lines << QStringLiteral("    启用方法：下载 LibTorch 后设置环境变量 LIBTORCH_DIR 指向解压目录，");
        lines << QStringLiteral("              或用 MSBuild 参数 /p:LibTorch_DIR=<路径>，然后重新编译。");
    }

    lines << QStringLiteral("[可用]  OpenCV 轻量训练后端（随机森林，始终可用，适合快速验证流程）");
    return lines.join(QLatin1Char('\n'));
}

} // namespace aq
