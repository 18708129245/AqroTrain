// ============================================================================
//  SelfTest.h —— 无人值守自检
//  ---------------------------------------------------------------------------
//  三个入口，覆盖「界面 -> 端到端数据流 -> 标注交互」三层验证：
//
//   · runAutoTrain   端到端：建工程 → 三模块 → 合成数据 → 脚本化标注 → 划分
//                    → 训练 → 自动评价 → 导出模型 → 重新加载推理
//   · runAnnoSelfTest 标注交互：向画布 viewport 派发鼠标事件，验证对象写入、
//                    顶点编辑、撤销/重做，并断言「对象光栅化掩码 == 落盘掩码」
//   · runMkdirDiagnostic 目录创建诊断：逐层报告失败位置与原因
//
//  三者均返回 0 表示通过；report 填入可读的中文报告。
// ============================================================================
#pragma once

#include <QString>

namespace aq {

/// 端到端自动训练与评价自检。workDir 为工作目录（会被创建/覆盖）。
int runAutoTrain(const QString &workDir, QString *report);

/// 标注交互自检（对象模型、掩码一致性、顶点编辑、撤销重做、背景类拦截）。
int runAnnoSelfTest(QString *report);

/// 目录创建诊断：逐层探测并报告路径状态与真实失败原因。
int runMkdirDiagnostic(const QString &dir, QString *report);

/**
 * 把自检报告写到指定路径。
 *
 * 刻意带**回退**：实测存在「某些目录下本进程无法新建文件（被安全软件/策略拦截）」的
 * 情况，而报告写不出去时原先只是静默失败 —— 使用者拿不到报告，退出码却仍是成功，
 * 等于把问题藏起来了。这里按「指定路径 → 工作目录 → 程序目录」依次尝试，
 * 并返回**实际写入的路径**（全部失败返回空串），由调用方打印到 stderr。
 */
QString writeReport(const QString &requestedPath, const QString &text,
                    const QString &workDirHint = QString());

} // namespace aq
