// ============================================================================
//  ModuleBar.cpp
// ============================================================================
#include "ui/ModuleBar.h"

#include <QCursor>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenu>
#include <QToolButton>

namespace aq {

namespace {

QString moduleButtonText(const Project::ModuleRef &ref)
{
    return QStringLiteral("%1\n%2").arg(ref.name, taskTypeName(ref.type));
}

} // namespace

ModuleBar::ModuleBar(QWidget *parent) : QWidget(parent)
{
    auto *outer = new QHBoxLayout(this);
    outer->setContentsMargins(6, 4, 6, 4);
    outer->setSpacing(6);

    m_host = new QWidget(this);
    m_layout = new QHBoxLayout(m_host);
    m_layout->setContentsMargins(0, 0, 0, 0);
    m_layout->setSpacing(6);
    m_layout->addStretch(1);

    outer->addWidget(new QLabel(QStringLiteral("模块："), this));
    outer->addWidget(m_host, 1);

    m_btnAdd = new QToolButton(this);
    m_btnAdd->setText(QStringLiteral("＋ 添加模块"));
    m_btnAdd->setToolTip(QStringLiteral("在当前模块之后添加一个新模块（分类 / 检测 / 分割）"));
    m_btnAdd->setObjectName(QStringLiteral("moduleBarAction"));
    connect(m_btnAdd, &QToolButton::clicked, this, &ModuleBar::addRequested);

    m_btnRemove = new QToolButton(this);
    m_btnRemove->setText(QStringLiteral("－ 删除模块"));
    m_btnRemove->setToolTip(QStringLiteral("删除当前模块，并连同其后的所有模块一起删除"
                                           "（数据、模型与测试结果都会一并删除）"));
    m_btnRemove->setObjectName(QStringLiteral("moduleBarAction"));
    connect(m_btnRemove, &QToolButton::clicked, this, [this]() {
        if (m_current >= 0)
            emit removeRequested(m_current);
    });

    outer->addWidget(m_btnAdd);
    outer->addWidget(m_btnRemove);
}

void ModuleBar::setModules(const QVector<Project::ModuleRef> &modules, int current)
{
    m_modules = modules;
    m_current = current;
    rebuild();
}

void ModuleBar::rebuild()
{
    for (QToolButton *b : m_buttons) {
        m_layout->removeWidget(b);
        b->deleteLater();
    }
    m_buttons.clear();

    for (int i = 0; i < m_modules.size(); ++i) {
        const Project::ModuleRef &ref = m_modules.at(i);
        auto *btn = new QToolButton(m_host);
        btn->setText(moduleButtonText(ref));
        btn->setCheckable(true);
        btn->setChecked(i == m_current);
        btn->setToolTip(QStringLiteral("第 %1 个模块：%2（%3）\n单击切换；右键可重命名")
                            .arg(i + 1)
                            .arg(ref.name, taskTypeName(ref.type)));
        btn->setObjectName(QStringLiteral("moduleCard"));
        btn->setProperty("moduleIndex", i);
        // 右键重命名（QToolButton 没有双击信号，用菜单更稳妥）
        btn->setContextMenuPolicy(Qt::CustomContextMenu);

        connect(btn, &QToolButton::clicked, this, [this, i]() { emit moduleSelected(i); });
        connect(btn, &QToolButton::customContextMenuRequested, this, [this, i](const QPoint &pos) {
            QMenu menu;
            QAction *actRename = menu.addAction(QStringLiteral("重命名模块…"));
            QAction *actSelect = menu.addAction(QStringLiteral("切换到该模块"));
            QAction *chosen = menu.exec(QCursor::pos());
            if (!chosen)
                return;
            if (chosen == actRename)
                emit renameRequested(i);
            else if (chosen == actSelect)
                emit moduleSelected(i);
            Q_UNUSED(pos)
        });

        m_layout->insertWidget(i, btn);
        m_buttons.append(btn);
    }

    m_btnRemove->setEnabled(m_current >= 0);
    m_btnAdd->setEnabled(true);
}

} // namespace aq
