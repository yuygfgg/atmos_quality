using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using Cavern;
using Cavern.Channels;
using Cavern.Format;
using Cavern.Format.Common;

namespace CavernRender;

internal static class Program
{
    private sealed record Layout(string Name, ReferenceChannel[] Channels);

    private static readonly Dictionary<string, Layout> NamedLayouts = new(StringComparer.OrdinalIgnoreCase)
    {
        ["2.0"] = new("2.0", ChannelPrototype.ref200),
        ["3.1.2"] = new("3.1.2", ChannelPrototype.ref312),
        ["5.1"] = new("5.1", ChannelPrototype.ref510),
        ["5.1.2"] = new("5.1.2", ChannelPrototype.ref512),
        ["5.1.4"] = new("5.1.4", ChannelPrototype.ref514),
        ["7.1"] = new("7.1", ChannelPrototype.ref710),
        ["7.1.2"] = new("7.1.2", ChannelPrototype.ref712),
        ["7.1.4"] = new("7.1.4", ChannelPrototype.ref714),
        ["7.1.6"] = new("7.1.6", ChannelPrototype.wav716),
        ["7.1.6-sides"] = new("7.1.6-sides", ChannelPrototype.ref716),
        ["9.1.4"] = new("9.1.4", ChannelPrototype.ref914),
        ["9.1.6"] = new("9.1.6", ChannelPrototype.wav916),
        ["9.1.6-sides"] = new("9.1.6-sides", ChannelPrototype.ref916),
    };

    private static int Main(string[] args)
    {
        if (args.Length < 3)
        {
            Usage();
            return 1;
        }

        string input = args[0];
        string output = args[1];
        string layoutSpec = args[2];
        double maxSeconds = 0.0;
        double gainDb = 0.0;
        int updateRate = 256;
        bool raw = false;

        try
        {
            for (int i = 3; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--max-seconds":
                        maxSeconds = ParseDouble(args, ref i);
                        break;
                    case "--gain-db":
                        gainDb = ParseDouble(args, ref i);
                        break;
                    case "--update-rate":
                        updateRate = ParseInt(args, ref i);
                        break;
                    case "--raw":
                        raw = true;
                        break;
                    case "-h":
                    case "--help":
                        Usage();
                        return 0;
                    default:
                        Console.Error.WriteLine($"unknown option: {args[i]}");
                        Usage();
                        return 1;
                }
            }
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }

        if (!double.IsFinite(maxSeconds) || maxSeconds < 0.0)
        {
            Console.Error.WriteLine("--max-seconds must be a non-negative finite number");
            return 1;
        }
        if (!double.IsFinite(gainDb))
        {
            Console.Error.WriteLine("--gain-db must be finite");
            return 1;
        }
        if (updateRate < 16)
        {
            Console.Error.WriteLine("--update-rate must be at least 16");
            return 1;
        }

        Layout layout;
        try
        {
            layout = ResolveLayout(layoutSpec);
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return 1;
        }

        try
        {
            return Render(input, output, layout, maxSeconds, gainDb, updateRate, raw);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"render failed: {ex.Message}");
            return 2;
        }
    }

    private static int Render(
        string input,
        string output,
        Layout layout,
        double maxSeconds,
        double gainDb,
        int updateRate,
        bool raw)
    {
        Listener.ReplaceChannels(ChannelPrototype.ToLayout(layout.Channels));

        using AudioReader reader = input.EndsWith(".damf", StringComparison.OrdinalIgnoreCase)
            ? new DolbyAtmosMasterFormatReader(input)
            : AudioReader.Open(input);
        reader.ReadHeader();

        Console.WriteLine(
            $"input: {reader.ChannelCount} ch, {reader.Length} samples, " +
            $"{reader.SampleRate} Hz, {reader.Length / (double)reader.SampleRate:F3}s");

        var renderer = reader.GetRenderer();
        Console.WriteLine(
            $"renderer: {renderer.GetType().Name}, objects={renderer.Objects.Count}, " +
            $"objectBased={renderer.HasObjects}, beds={renderer.GetChannels().Length}");

        if (renderer.Objects.Count == 0)
        {
            throw new InvalidOperationException("no renderable objects or channels were found");
        }

        var listener = new Listener(false)
        {
            SampleRate = reader.SampleRate,
            UpdateRate = updateRate,
            AudioQuality = QualityModes.Perfect,
            Volume = (float)Math.Pow(10.0, gainDb / 20.0),
            LFESeparation = true,
        };
        listener.AttachSources(renderer.Objects);

        long total = reader.Length;
        if (maxSeconds > 0.0)
        {
            total = Math.Min(total, (long)Math.Round(maxSeconds * reader.SampleRate));
        }
        if (total <= 0)
        {
            throw new InvalidOperationException("nothing to render");
        }

        AudioWriter? writer = null;
        FileStream? rawStream = null;
        bool channelMask = false;
        try
        {
            if (raw)
            {
                rawStream = new FileStream(output, FileMode.Create, FileAccess.Write, FileShare.Read);
            }
            else
            {
                channelMask = NamedLayouts.ContainsKey(layout.Name) && SupportsWaveMask(layout.Channels);
                writer = channelMask
                    ? new RIFFWaveWriter(
                        output, layout.Channels, total, reader.SampleRate, BitDepth.Float32)
                    : new RIFFWaveWriter(
                        output, layout.Channels.Length, total, reader.SampleRate, BitDepth.Float32);
                writer.WriteHeader();
            }

            long rendered = 0;
            long nextProgress = reader.SampleRate * 5L;
            while (rendered < total)
            {
                float[]? block = listener.Render();
                if (block is null)
                {
                    throw new InvalidOperationException("Cavern returned no samples");
                }

                int frames = (int)Math.Min(updateRate, total - rendered);
                int count = frames * layout.Channels.Length;
                if (writer is not null)
                {
                    writer.WriteBlock(block, 0, count);
                }
                else
                {
                    rawStream!.Write(MemoryMarshal.AsBytes(block.AsSpan(0, count)));
                }

                rendered += frames;
                if (rendered >= nextProgress || rendered == total)
                {
                    Console.Error.WriteLine(
                        $"rendered {rendered / (double)reader.SampleRate:F1}s / " +
                        $"{total / (double)reader.SampleRate:F1}s");
                    nextProgress += reader.SampleRate * 5L;
                }
            }
        }
        finally
        {
            writer?.Dispose();
            rawStream?.Dispose();
        }

        var sidecar = new
        {
            input = Path.GetFullPath(input),
            output = Path.GetFullPath(output),
            layout = layout.Name,
            channels = layout.Channels.Select(ShortLabel).ToArray(),
            sample_rate = reader.SampleRate,
            samples = total,
            duration_s = total / (double)reader.SampleRate,
            source_channels = reader.ChannelCount,
            source_samples = reader.Length,
            source_sample_rate = reader.SampleRate,
            renderer = renderer.GetType().Name,
            object_based = renderer.HasObjects,
            object_count = renderer.Objects.Count,
            bed_channels = renderer.GetChannels().Length,
            raw,
            wav_channel_mask = !raw && channelMask,
            gain_db = gainDb,
            update_rate = updateRate,
            cavern_version = "2.1.0",
            created_at = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture),
        };
        File.WriteAllText(
            output + ".json",
            JsonSerializer.Serialize(sidecar, new JsonSerializerOptions { WriteIndented = true }));

        Console.WriteLine(
            $"wrote {output}: {layout.Channels.Length} ch, {total} samples, " +
            $"{total / (double)reader.SampleRate:F3}s");
        return 0;
    }

    private static Layout ResolveLayout(string spec)
    {
        if (NamedLayouts.TryGetValue(spec, out Layout? named))
        {
            return named;
        }

        string[] names = spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (names.Length == 0)
        {
            throw new ArgumentException("layout must be a named layout or a comma-separated channel list");
        }

        var channels = new ReferenceChannel[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            channels[i] = ParseChannel(names[i]);
        }
        return new Layout(spec, channels);
    }

    private static bool SupportsWaveMask(ReferenceChannel[] channels) => channels.All(
        channel => channel is
            ReferenceChannel.FrontLeft or
            ReferenceChannel.FrontRight or
            ReferenceChannel.FrontCenter or
            ReferenceChannel.ScreenLFE or
            ReferenceChannel.RearLeft or
            ReferenceChannel.RearRight or
            ReferenceChannel.SideLeft or
            ReferenceChannel.SideRight or
            ReferenceChannel.TopFrontLeft or
            ReferenceChannel.TopFrontRight or
            ReferenceChannel.TopFrontCenter or
            ReferenceChannel.TopRearLeft or
            ReferenceChannel.TopRearRight or
            ReferenceChannel.TopRearCenter);

    private static ReferenceChannel ParseChannel(string name) => name.ToLowerInvariant() switch
    {
        "l" or "fl" or "frontleft" => ReferenceChannel.FrontLeft,
        "r" or "fr" or "frontright" => ReferenceChannel.FrontRight,
        "c" or "fc" or "frontcenter" => ReferenceChannel.FrontCenter,
        "lfe" or "sub" => ReferenceChannel.ScreenLFE,
        "ls" or "sl" or "sideleft" => ReferenceChannel.SideLeft,
        "rs" or "sr" or "sideright" => ReferenceChannel.SideRight,
        "lrs" or "rl" or "rearleft" => ReferenceChannel.RearLeft,
        "rrs" or "rr" or "rearright" => ReferenceChannel.RearRight,
        "ltf" or "tfl" or "topfrontleft" => ReferenceChannel.TopFrontLeft,
        "rtf" or "tfr" or "topfrontright" => ReferenceChannel.TopFrontRight,
        "ctf" or "tfc" or "topfrontcenter" => ReferenceChannel.TopFrontCenter,
        "ltr" or "trl" or "toprearleft" => ReferenceChannel.TopRearLeft,
        "rtr" or "trr" or "toprearright" => ReferenceChannel.TopRearRight,
        "ctr" or "trc" or "toprearcenter" => ReferenceChannel.TopRearCenter,
        "lts" or "tsl" or "topsideleft" => ReferenceChannel.TopSideLeft,
        "rts" or "tsr" or "topsideright" => ReferenceChannel.TopSideRight,
        "lw" or "wl" or "wideleft" => ReferenceChannel.WideLeft,
        "rw" or "wr" or "wideright" => ReferenceChannel.WideRight,
        "bs" or "bottomsurround" => ReferenceChannel.BottomSurround,
        "cb" or "rc" or "rearcenter" => ReferenceChannel.RearCenter,
        "lc" or "frontleftcenter" => ReferenceChannel.FrontLeftCenter,
        "rc2" or "frontrightcenter" => ReferenceChannel.FrontRightCenter,
        _ => throw new ArgumentException($"unknown channel name: {name}"),
    };

    private static string ShortLabel(ReferenceChannel channel) => channel switch
    {
        ReferenceChannel.FrontLeft => "L",
        ReferenceChannel.FrontRight => "R",
        ReferenceChannel.FrontCenter => "C",
        ReferenceChannel.ScreenLFE => "LFE",
        ReferenceChannel.SideLeft => "Ls",
        ReferenceChannel.SideRight => "Rs",
        ReferenceChannel.RearLeft => "Lrs",
        ReferenceChannel.RearRight => "Rrs",
        ReferenceChannel.TopFrontLeft => "Ltf",
        ReferenceChannel.TopFrontRight => "Rtf",
        ReferenceChannel.TopFrontCenter => "Ctf",
        ReferenceChannel.TopRearLeft => "Ltr",
        ReferenceChannel.TopRearRight => "Rtr",
        ReferenceChannel.TopRearCenter => "Ctr",
        ReferenceChannel.TopSideLeft => "Lts",
        ReferenceChannel.TopSideRight => "Rts",
        ReferenceChannel.WideLeft => "Lw",
        ReferenceChannel.WideRight => "Rw",
        ReferenceChannel.BottomSurround => "Bs",
        ReferenceChannel.RearCenter => "Cb",
        ReferenceChannel.FrontLeftCenter => "Lc",
        ReferenceChannel.FrontRightCenter => "Rc",
        _ => channel.ToString(),
    };

    private static double ParseDouble(string[] args, ref int index)
    {
        if (index + 1 >= args.Length)
        {
            throw new ArgumentException($"{args[index]} requires a value");
        }
        index++;
        if (!double.TryParse(args[index], NumberStyles.Float, CultureInfo.InvariantCulture, out double value))
        {
            throw new ArgumentException($"{args[index - 1]} must be a number");
        }
        return value;
    }

    private static int ParseInt(string[] args, ref int index)
    {
        if (index + 1 >= args.Length)
        {
            throw new ArgumentException($"{args[index]} requires a value");
        }
        index++;
        if (!int.TryParse(args[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out int value))
        {
            throw new ArgumentException($"{args[index - 1]} must be an integer");
        }
        return value;
    }

    private static void Usage()
    {
        Console.WriteLine(
            """
            usage: cavern-render <input> <output> <layout> [options]

            input     .atmos/.damf master, .ec3/.eac3, WAV, or CAF
            output    WAV (float32) or raw f32le with --raw
            layout    2.0, 3.1.2, 5.1, 5.1.2, 5.1.4, 7.1, 7.1.2, 7.1.4,
                      7.1.6, 7.1.6-sides, 9.1.4, 9.1.6, 9.1.6-sides,
                      or a comma-separated list such as
                      L,R,C,LFE,Ls,Rs,Lrs,Rrs,Ltf,Rtf,Ltr,Rtr

            options
              --max-seconds N   render only the first N seconds
              --gain-db N       output gain in dB (default 0)
              --update-rate N   Cavern render block size (default 256)
              --raw             write interleaved f32le instead of WAV
              -h, --help        show this help

            A JSON sidecar is written to <output>.json with channel labels,
            source information, and render provenance.
            """);
    }
}
