// C# 5 / .NET Framework 4.8, compiled x64 /unsafe /optimize+ in a temp dir.
// Dedicated compact sibling of ldname.cs: LDE1 is NOT the LDN1 disk format.
// Wire/disk layout is documented in bench/engines/eng_csharp.py.
using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Diagnostics;

internal unsafe sealed class LdEngine : IDisposable
{
    private MemoryMappedFile mapping;
    private MemoryMappedViewAccessor view;
    private byte* address;
    private bool acquired;
    private byte* low;
    private byte* original;
    private bool rowMode;
    private uint* rowLengths;
    private byte* rowFlags;
    private HashSet<int> skipped = new HashSet<int>();
    private uint* offsets;
    private uint* originalOffsets;
    private int* parents;
    private int rows;
    private string root;
    private byte[] rootLow;
    private readonly ParallelOptions parallel = new ParallelOptions {
        MaxDegreeOfParallelism = Math.Min(32, Environment.ProcessorCount) };
    private static readonly Encoding Utf8 = new UTF8Encoding(false, true);

    public LdEngine(string file)
    {
        try
        {
            long fileBytes = new FileInfo(file).Length;
            if (fileBytes < 16)
                throw new InvalidDataException("invalid LDE1 header");
            mapping = MemoryMappedFile.CreateFromFile(file, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            // A zero-sized view maps to the end and can expose a page-rounded
            // capacity on Windows. The disk format describes actual file bytes.
            view = mapping.CreateViewAccessor(0, fileBytes, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            acquired = true;
            address = pointer + view.PointerOffset;
            if (*(uint*)address == 0x3149444cU) // LDI1 product row snapshot
            {
                rowMode = true;
                rows = *(int*)(address + 4);
                long blobBytes = *(long*)(address + 8);
                long metadataBytes = 16L + 17L * rows;
                if (rows < 0 || blobBytes < 0 || blobBytes > uint.MaxValue ||
                    metadataBytes + blobBytes != fileBytes)
                    throw new InvalidDataException("invalid LDI1 length");
                offsets = (uint*)(address + 16);
                rowLengths = offsets + rows;
                parents = (int*)(rowLengths + rows);
                rowFlags = (byte*)(parents + rows);
                low = address + metadataBytes;
                for (int r = 0; r < rows; r++)
                    if ((ulong)offsets[r] + rowLengths[r] > (ulong)blobBytes ||
                        parents[r] < -1 || parents[r] >= rows)
                        throw new InvalidDataException("invalid LDI1 row");
                int warm = 0;
                for (long i = 0; i < fileBytes; i += 4096) warm ^= address[i];
                GC.KeepAlive(warm);
                return;
            }
            if (fileBytes < 32 || view.Capacity < fileBytes || *(uint*)address != 0x3145444cU)
                throw new InvalidDataException("invalid LDE1 header");
            rows = *(int*)(address + 4);
            long lowBytes = *(long*)(address + 8), originalBytes = *(long*)(address + 16);
            int rootBytes = *(int*)(address + 24), rootLowBytes = *(int*)(address + 28);
            if (rows < 0 || lowBytes < 0 || lowBytes > uint.MaxValue || originalBytes < 0 ||
                originalBytes > uint.MaxValue || rootBytes < 0 || rootLowBytes < 0)
                throw new InvalidDataException("invalid LDE1 sizes");
            long metadata = 32L + rootBytes + rootLowBytes + 8L * (rows + 1L) + 5L * rows;
            long expectedBytes = metadata + lowBytes + originalBytes;
            if (expectedBytes != fileBytes)
                throw new InvalidDataException("invalid LDE1 length: expected " + expectedBytes +
                    " bytes, file has " + fileBytes + " bytes");
            root = Decode(address + 32, rootBytes);
            rootLow = new byte[rootLowBytes];
            System.Runtime.InteropServices.Marshal.Copy(new IntPtr(address + 32L + rootBytes), rootLow, 0, rootLowBytes);
            offsets = (uint*)(address + 32L + rootBytes + rootLowBytes);
            originalOffsets = offsets + rows + 1L;
            parents = (int*)(originalOffsets + rows + 1L);
            low = address + metadata;
            original = low + lowBytes;
            if (offsets[0] != 0 || originalOffsets[0] != 0 || offsets[rows] != lowBytes || originalOffsets[rows] != originalBytes)
                throw new InvalidDataException("invalid LDE1 offsets");
            // Touch all mapped pages in build/load, rather than charging demand
            // paging to the first query. No measured query is cached or primed.
            int touched = 0;
            for (long i = 0; i < fileBytes; i += 4096) touched ^= address[i];
            GC.KeepAlive(touched);
        }
        catch { Dispose(); throw; }
    }

    private static string Decode(byte* p, int length)
    {
        return Utf8.GetString(p, length);
    }

    // SWAR memchr: safely skip eight bytes when none equals the needle's first
    // byte. A positive word mask is only a candidate; scalar verification avoids
    // false positives from the subtract/borrow zero-byte detection idiom.
    private static bool Contains(byte* text, int length, byte[] needle)
    {
        int m = needle.Length;
        if (m == 0) return true;
        int end = length - m, pos = 0;
        ulong repeated = 0x0101010101010101UL * needle[0];
        while (pos <= end)
        {
            if (pos + 7 <= end)
            {
                ulong v = *(ulong*)(text + pos) ^ repeated;
                if (((v - 0x0101010101010101UL) & ~v & 0x8080808080808080UL) == 0)
                { pos += 8; continue; }
            }
            if (text[pos] == needle[0])
            {
                int j = 1;
                while (j < m && text[pos + j] == needle[j]) j++;
                if (j == m) return true;
            }
            pos++;
        }
        return false;
    }

    private sealed class PathToken
    {
        public readonly byte[] Bytes;
        public readonly int[] Failure;
        public PathToken(string value)
        {
            Bytes = Utf8.GetBytes(value);
            Failure = new int[Bytes.Length];
            for (int i = 1, j = 0; i < Bytes.Length; i++)
            {
                while (j > 0 && Bytes[i] != Bytes[j]) j = Failure[j - 1];
                if (Bytes[i] == Bytes[j]) j++;
                Failure[i] = j;
            }
        }
        public bool Feed(byte value, ref int state)
        {
            while (state > 0 && value != Bytes[state]) state = Failure[state - 1];
            if (value == Bytes[state]) state++;
            return state == Bytes.Length;
        }
    }

    private sealed class Query
    {
        public readonly List<byte[]> Plain = new List<byte[]>();
        public readonly List<byte[]> Suffix = new List<byte[]>();
        public readonly List<PathToken> Paths = new List<PathToken>();
        public Query(string text)
        {
            // Python lowercases before transport, exactly like the oracle.
            foreach (string token in text.Split((char[])null, StringSplitOptions.RemoveEmptyEntries))
            {
                if (token.StartsWith("ext:", StringComparison.Ordinal))
                    Suffix.Add(Utf8.GetBytes("." + token.Substring(4)));
                else if (token.StartsWith("*.", StringComparison.Ordinal))
                    Suffix.Add(Utf8.GetBytes(token.Substring(1)));
                else if (token.StartsWith("path:", StringComparison.Ordinal))
                    Paths.Add(new PathToken(token.Substring(5)));
                else Plain.Add(Utf8.GetBytes(token));
            }
            Plain.Sort(delegate(byte[] a, byte[] b) { return b.Length.CompareTo(a.Length); });
        }
    }

    private void Chain(int row, List<int> chain)
    {
        chain.Clear();
        for (int i = row; i >= 0; i = parents[i]) chain.Add(i);
    }

    private bool PathMatches(PathToken token, List<int> chain)
    {
        if (token.Bytes.Length == 0) return true;
        int state = 0;
        foreach (byte value in rootLow) if (token.Feed(value, ref state)) return true;
        bool separator = rootLow.Length > 0 && rootLow[rootLow.Length - 1] != (byte)'\\' && rootLow[rootLow.Length - 1] != (byte)'/';
        for (int c = chain.Count - 1; c >= 0; c--)
        {
            int row = chain[c];
            if (row == 0 && parents[row] < 0) continue; // paths_of() root convention
            if (separator && token.Feed((byte)Path.DirectorySeparatorChar, ref state)) return true;
            byte* start = low + offsets[row];
            int length = (int)(offsets[row + 1] - offsets[row] - 1);
            for (int j = 0; j < length; j++) if (token.Feed(start[j], ref state)) return true;
            separator = true;
        }
        return false;
    }

    private string FullPath(List<int> chain)
    {
        var path = new StringBuilder(root);
        for (int c = chain.Count - 1; c >= 0; c--)
        {
            int row = chain[c];
            if (row == 0 && parents[row] < 0) continue;
            if (path.Length > 0 && path[path.Length - 1] != '\\' && path[path.Length - 1] != '/')
                path.Append(Path.DirectorySeparatorChar);
            path.Append(Decode(original + originalOffsets[row], (int)(originalOffsets[row + 1] - originalOffsets[row] - 1)));
        }
        return path.ToString();
    }

    private List<string> SearchRows(string text, int limit)
    {
        // Deterministic IDs; Python ranks the COMPLETE candidate set. The extra
        // sentinel row makes saturation explicit, rather than silently losing
        // a better match outside the transport budget.
        var result = new List<string>();
        byte[] ancestor = null;
        string scanText = text;
        if (text.StartsWith("\u0001", StringComparison.Ordinal))
        {
            int split = text.IndexOf('\n');
            ancestor = Utf8.GetBytes(split < 0 ? text.Substring(1) : text.Substring(1, split - 1));
            scanText = split < 0 ? "" : text.Substring(split + 1);
        }
        var query = new Query(scanText);
        object gate = new object();
        int done = 0;
        int chunks = (int)((rows + 8191L) / 8192);
        Parallel.For(0, chunks, parallel, delegate(int chunk, ParallelLoopState loop) {
            int end = (int)Math.Min((long)rows, (chunk + 1L) * 8192);
            for (int row = chunk * 8192; row < end; row++)
            {
                if (Volatile.Read(ref done) != 0) { loop.Stop(); break; }
                if ((rowFlags[row] & 8) != 0 || skipped.Contains(row)) continue;
                byte* start = low + offsets[row];
                int length = (int)rowLengths[row];
                bool match = true;
                foreach (byte[] suffix in query.Suffix)
                {
                    if (suffix.Length > length) { match = false; break; }
                    int at = length - suffix.Length;
                    for (int k = 0; k < suffix.Length; k++)
                        if (start[at + k] != suffix[k]) { match = false; break; }
                    if (!match) break;
                }
                if (!match) continue;
                foreach (byte[] needle in query.Plain)
                    if (!Contains(start, length, needle)) { match = false; break; }
                if (!match) continue;
                if (ancestor != null)
                {
                    bool found = false;
                    int current = row;
                    for (int depth = 0; current >= 0 && depth < 4096; depth++)
                    {
                        if (Contains(low + offsets[current], (int)rowLengths[current], ancestor))
                        { found = true; break; }
                        current = parents[current];
                    }
                    if (!found) continue;
                }
                lock (gate)
                {
                    if (done != 0) break;
                    result.Add(row.ToString(System.Globalization.CultureInfo.InvariantCulture));
                    if (result.Count == limit) Volatile.Write(ref done, 1);
                }
            }
        });
        return result;
    }

    public List<string> Search(string text, int limit)
    {
        if (rowMode) return SearchRows(text, limit);
        var result = new List<string>(Math.Min(limit, 200));
        if (limit <= 0 || rows == 0) return result;
        var query = new Query(text);
        var seen = new HashSet<string>(StringComparer.Ordinal);
        object gate = new object();
        int done = 0;
        const int chunkSize = 8192;
        int chunks = (int)((rows + (long)chunkSize - 1) / chunkSize);
        Parallel.For<List<int>>(0, chunks, parallel,
            delegate { return new List<int>(64); },
            delegate(int chunk, ParallelLoopState loop, List<int> chain)
            {
                int end = (int)Math.Min((long)rows, (chunk + 1L) * chunkSize);
                for (int row = chunk * chunkSize; row < end; row++)
                {
                    if (Volatile.Read(ref done) != 0) { loop.Stop(); break; }
                    byte* start = low + offsets[row];
                    int length = (int)(offsets[row + 1] - offsets[row] - 1);
                    bool match = true;
                    foreach (byte[] suffix in query.Suffix)
                    {
                        if (suffix.Length > length) { match = false; break; }
                        int at = length - suffix.Length;
                        for (int k = 0; k < suffix.Length; k++)
                            if (start[at + k] != suffix[k]) { match = false; break; }
                        if (!match) break;
                    }
                    if (!match) continue;
                    foreach (byte[] needle in query.Plain)
                        if (!Contains(start, length, needle)) { match = false; break; }
                    if (!match) continue;
                    Chain(row, chain);
                    foreach (PathToken token in query.Paths)
                        if (!PathMatches(token, chain)) { match = false; break; }
                    if (!match) continue;
                    lock (gate)
                    {
                        if (done != 0) break;
                        // Only accepted matches reach full-path materialisation.
                        string path = FullPath(chain);
                        if (seen.Add(path)) result.Add(path);
                        if (result.Count == limit) Volatile.Write(ref done, 1);
                    }
                }
                return chain;
            }, delegate(List<int> chain) { });
        return result;
    }

    public void Dispose()
    {
        if (acquired) { view.SafeMemoryMappedViewHandle.ReleasePointer(); acquired = false; }
        if (view != null) { view.Dispose(); view = null; }
        if (mapping != null) { mapping.Dispose(); mapping = null; }
    }

    private static void Reply(BinaryWriter output, int status, List<string> paths)
    {
        output.Write(status);
        using (Process self = Process.GetCurrentProcess()) output.Write(self.WorkingSet64);
        output.Write(paths.Count);
        foreach (string path in paths)
        {
            byte[] bytes = Utf8.GetBytes(path);
            output.Write(bytes.Length);
            output.Write(bytes);
        }
        output.Flush();
    }

    public static int Main(string[] args)
    {
        // Parallel.For runs on the thread pool: without this the pool injects ~1 thread per 0.5 s when the first queries
        // arrive (measured on the real volume: 600-1200 ms per query for the first ~12 s, 12-20 ms afterwards).
        int _w, _io; System.Threading.ThreadPool.GetMinThreads(out _w, out _io);
        System.Threading.ThreadPool.SetMinThreads(Math.Max(_w, 64), Math.Max(_io, 64));
        try
        {
            if (args.Length != 1) throw new ArgumentException("expected snapshot path");
            using (var engine = new LdEngine(args[0]))
            using (var input = new BinaryReader(Console.OpenStandardInput(), Utf8))
            using (var output = new BinaryWriter(new BufferedStream(Console.OpenStandardOutput(), 65536), Utf8))
            {
                Reply(output, 0, new List<string>());
                while (true)
                {
                    int length;
                    try { length = input.ReadInt32(); } catch (EndOfStreamException) { break; }
                    int limit = input.ReadInt32();
                    if (length == -3 && engine.rowMode)
                    {
                        if (limit < 0 || limit > 20000) throw new InvalidDataException("invalid skip count");
                        var skip = new HashSet<int>();
                        for (int i = 0; i < limit; i++)
                        {
                            int row = input.ReadInt32();
                            if (row >= 0 && row < engine.rows) skip.Add(row);
                        }
                        engine.skipped = skip;
                        continue; // skip update and following query share one response
                    }
                    if (length == -2) break;
                    if (length == -1) { Reply(output, 0, new List<string>()); continue; }
                    if (length < 0 || length > 1024 * 1024 || limit < 0 || limit > 1000000)
                        throw new InvalidDataException("invalid request bounds");
                    byte[] bytes = input.ReadBytes(length);
                    if (bytes.Length != length) throw new EndOfStreamException();
                    var watch = System.Diagnostics.Stopwatch.StartNew();
                    int hits = -1;
                    try { var found = engine.Search(Utf8.GetString(bytes), limit); hits = found.Count; Reply(output, 0, found); }
                    catch (Exception error) { Reply(output, -1, new List<string> { error.Message }); }
                    string logPath = Environment.GetEnvironmentVariable("LDENG_LOG");
                    if (!String.IsNullOrEmpty(logPath))
                    {
                        try { System.IO.File.AppendAllText(logPath, String.Format("{0:F1} ms q={1} hits={2} t={3:F1}s{4}", watch.Elapsed.TotalMilliseconds, Utf8.GetString(bytes), hits, (DateTime.Now - System.Diagnostics.Process.GetCurrentProcess().StartTime).TotalSeconds, Environment.NewLine)); }
                        catch (Exception) { }
                    }
                }
            }
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
