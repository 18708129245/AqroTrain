// ============================================================================
//  main.cpp —— AqroTrain 入口
//  ---------------------------------------------------------------------------
//  AqroTrain：对标阿丘科技 AIDI 的视觉标注训练平台
//  工具链：Visual Studio 2022 (v143) + Qt 5.12.10 + OpenCV 4.6.0
//
//  命令行模式（用于无人值守验证，均在处理完后直接返回退出码）：
//    --selftest <截图路径> [工程根目录]   构建主窗口、（可选）打开工程、截图后退出
//    --autotrain <工作目录> [报告路径]    端到端：建工程 → 三模块 → 标注 → 训练 → 评价 → 导出
//    --anno-selftest [报告路径]           标注交互自检（对象 / 掩码一致性、顶点编辑、撤销）
//    --diag-mkdir <目录> [报告路径]       目录创建诊断（逐层探测失败原因）
//  ============================================================================
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFont>
#include <QPixmap>
#include <QTextStream>
#include <QTimer>

#include <cstdio>

#include "SelfTest.h"
#include "ui/MainWindow.h"

namespace {

// 读取并应用 Qt 样式表（style.qss 已编入 qrc 资源）
void applyStyleSheet(QApplication &app)
{
    QFile f(QStringLiteral(":/style.qss"));
    if (f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream ts(&f);
        ts.setCodec("UTF-8");
        app.setStyleSheet(ts.readAll());
        f.close();
    }
}

// 把报告中转写到文件（带回退），并把实际路径/失败原因写到 stderr。
// 之所以要打印：报告写不出去时若静默失败，使用者拿不到报告却发现退出码是成功的。
bool dumpReport(const QString &reportPath, const QString &report, const QString &workDirHint)
{
    const QString used = aq::writeReport(reportPath, report, workDirHint);
    if (used.isEmpty()) {
        std::fprintf(stderr, "[AqroTrain] 报告写入失败：%s（也已尝试工作目录与程序目录）\n",
                     reportPath.toUtf8().constData());
        return false;
    }
    if (used != reportPath) {
        std::fprintf(stderr,
                     "[AqroTrain] 指定的报告路径无法写入，已改写到：%s\n",
                     used.toUtf8().constData());
    }
    return true;
}

// 取「开关之后、下一个以 - 开头之前」的参数
QString argAfter(const QStringList &args, int idx, const QString &def = QString())
{
    if (idx + 1 < args.size() && !args.at(idx + 1).startsWith(QLatin1Char('-')))
        return args.at(idx + 1);
    return def;
}

// 自检模式：显示窗口 ->（可选）打开工程 -> 延时截图 -> 退出
int runSelfTest(QApplication &app, const QString &shotPath, const QString &projectRoot)
{
    auto *w = new aq::MainWindow();
    w->resize(1560, 960);
    w->show();

    if (!projectRoot.isEmpty())
        w->openProjectAt(projectRoot);

    int exitCode = 3;   // 默认失败：截图未成功

    QTimer::singleShot(1500, [&app, w, shotPath, &exitCode]() {
        const QPixmap pm = w->grab();
        QDir().mkpath(QFileInfo(shotPath).absolutePath());
        exitCode = pm.save(shotPath, "PNG") ? 0 : 4;
        app.quit();
    });

    app.exec();
    return exitCode;
}

} // namespace

int main(int argc, char *argv[])
{
    // 高 DPI 支持（必须在 QApplication 构造前设置）
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

    QApplication app(argc, argv);
    app.setApplicationName(QStringLiteral("AqroTrain"));
    app.setApplicationDisplayName(QStringLiteral("AqroTrain 视觉标注训练平台"));
    app.setOrganizationName(QStringLiteral("AqroTrain"));
    app.setApplicationVersion(QStringLiteral("1.0.0"));

    QFont font = app.font();
    font.setFamily(QStringLiteral("Microsoft YaHei"));
    font.setPointSize(9);
    app.setFont(font);

    applyStyleSheet(app);

    const QStringList args = app.arguments();
    const int selIdx = args.indexOf(QStringLiteral("--selftest"));
    const int autIdx = args.indexOf(QStringLiteral("--autotrain"));
    const int annoIdx = args.indexOf(QStringLiteral("--anno-selftest"));
    const int mkIdx  = args.indexOf(QStringLiteral("--diag-mkdir"));

    // ---- 界面冒烟自检 ----
    if (selIdx >= 0) {
        const QString shot = argAfter(args, selIdx, QStringLiteral("selftest.png"));
        const QString root = argAfter(args, selIdx + 1);
        return runSelfTest(app, shot, root);
    }

    // ---- 端到端自动训练自检 ----
    if (autIdx >= 0) {
        const QString workDir = argAfter(args, autIdx, QStringLiteral("autotrain"));
        const QString reportPath =
            argAfter(args, autIdx + 1, QDir(workDir).absoluteFilePath(QStringLiteral("selftest_report.txt")));

        QString report;
        const int rc = aq::runAutoTrain(workDir, &report);
        dumpReport(reportPath, report, workDir);
        return rc;
    }

    // ---- 标注交互自检 ----
    if (annoIdx >= 0) {
        const QString reportPath =
            argAfter(args, annoIdx, QDir::temp().absoluteFilePath(QStringLiteral("aq_anno_report.txt")));
        QString report;
        const int rc = aq::runAnnoSelfTest(&report);
        dumpReport(reportPath, report, QDir::tempPath());
        return rc;
    }

    // ---- 目录创建诊断 ----
    if (mkIdx >= 0) {
        const QString dir = argAfter(args, mkIdx);
        const QString reportPath =
            argAfter(args, mkIdx + 1, QDir::temp().absoluteFilePath(QStringLiteral("aq_mkdir_diag.txt")));
        QString report;
        const int rc = aq::runMkdirDiagnostic(dir, &report);
        dumpReport(reportPath, report, QDir::cleanPath(QFileInfo(dir).absolutePath()));
        return rc;
    }

    // ---- 正常启动 ----
    aq::MainWindow w;
    w.show();
    return app.exec();
}
