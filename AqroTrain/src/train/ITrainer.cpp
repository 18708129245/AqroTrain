// ============================================================================
//  ITrainer.cpp —— 训练后端接口的公共实现
// ============================================================================
#include "train/ITrainer.h"

namespace aq {

ITrainer::ITrainer(QObject *parent) : QObject(parent) {}
ITrainer::~ITrainer() = default;

} // namespace aq
