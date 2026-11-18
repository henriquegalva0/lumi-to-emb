# Lumi-Embed — resultados

Gerado por `lumi_embed report` a partir de `results/embed/`. Células "—" = sem resultado ainda. IC 95 % por bootstrap sobre tarefas; p-valores por bootstrap pareado (unidade = tarefa), como no MTEB-BR [16].

## Tabela 12 — Visão geral

| Checkpoint | MTEB-BR Mean22 [IC 95%] | PT STS | PT Cls | PT Ret | PT Rer | MMTEB | MTEB eng v2 | LongEmbed |
|---|---|---|---|---|---|---|---|---|
| R0 | — | — | — | — | — | — | — | — |
| R0e | — | — | — | — | — | — | — | — |
| R1 | — | — | — | — | — | — | — | — |
| C1 | — | — | — | — | — | — | — | — |
| C2 | — | — | — | — | — | — | — | — |
| C2-real | — | — | — | — | — | — | — | — |
| C2-syn | — | — | — | — | — | — | — | — |
| C2-nohn | — | — | — | — | — | — | — | — |
| C2-naivehn | — | — | — | — | — | — | — | — |
| C2-ib | — | — | — | — | — | — | — | — |
| C2-direct | — | — | — | — | — | — | — | — |
| C3-long | — | — | — | — | — | — | — | — |
| B2a | — | — | — | — | — | — | — | — |
| B2f | — | — | — | — | — | — | — | — |
| M-C | — | — | — | — | — | — | — | — |
| M-B | — | — | — | — | — | — | — | — |
| lumi-emb | — | — | — | — | — | — | — | — |

## Tabela 13 — MTEB-BR por categoria

| Checkpoint | Classificação (4) | Multilabel (1) | Pares (2) | STS (2) | Clusterização (5) | Retrieval (6) | Reranking (2) |
|---|---|---|---|---|---|---|---|
| R0 | — | — | — | — | — | — | — |
| R0e | — | — | — | — | — | — | — |
| R1 | — | — | — | — | — | — | — |
| C1 | — | — | — | — | — | — | — |
| C2 | — | — | — | — | — | — | — |
| C2-real | — | — | — | — | — | — | — |
| C2-syn | — | — | — | — | — | — | — |
| C2-nohn | — | — | — | — | — | — | — |
| C2-naivehn | — | — | — | — | — | — | — |
| C2-ib | — | — | — | — | — | — | — |
| C2-direct | — | — | — | — | — | — | — |
| C3-long | — | — | — | — | — | — | — |
| B2a | — | — | — | — | — | — | — |
| B2f | — | — | — | — | — | — | — |
| M-C | — | — | — | — | — | — | — |
| M-B | — | — | — | — | — | — | — |
| lumi-emb | — | — | — | — | — | — | — |

## Tabela 13b — MTEB-BR por tarefa (seleção; completo em CSV)

| Tarefa | R0 | R1 | C2 | B2a | B2f | M-C | M-B | lumi-emb |
|---|---|---|---|---|---|---|---|---|
| HateBR | — | — | — | — | — | — | — | — |
| ToxSynPT | — | — | — | — | — | — | — | — |
| FactckBrClassification | — | — | — | — | — | — | — | — |
| PortuLexRRIP | — | — | — | — | — | — | — | — |
| BrighterEmotionMultilabelClassification | — | — | — | — | — | — | — | — |
| AssinRTE | — | — | — | — | — | — | — | — |
| InferBR | — | — | — | — | — | — | — | — |
| AssinSTS | — | — | — | — | — | — | — | — |
| Assin2STS | — | — | — | — | — | — | — | — |
| WikipediaPTCategoriesClusteringP2P | — | — | — | — | — | — | — | — |
| MedPTClustering | — | — | — | — | — | — | — | — |
| JurisTCUClusteringP2P | — | — | — | — | — | — | — | — |
| SciELOClusteringP2P | — | — | — | — | — | — | — | — |
| StackoverflowPtClustering | — | — | — | — | — | — | — | — |
| Quati | — | — | — | — | — | — | — | — |
| JurisTCU | — | — | — | — | — | — | — | — |
| BRTaxQAR | — | — | — | — | — | — | — | — |
| FaQuADIR | — | — | — | — | — | — | — | — |
| MedPTRetrieval | — | — | — | — | — | — | — | — |
| FaqBacenRetrieval | — | — | — | — | — | — | — | — |
| QuatiReranking | — | — | — | — | — | — | — | — |
| JurisTCUReranking | — | — | — | — | — | — | — | — |

## Tabela 14 — Comparação com modelos correspondentes

| Modelo | Relação com o lumi-emb | MTEB-BR Mean22 | MTEB-PT (STS/Cls/Ret/Rer) | MMTEB | LongEmbed |
|---|---|---|---|---|---|
| lumi-emb (este trabalho) | — | — | — / — / — / — | — | — |
| R0 (este trabalho) | Ponto de partida | — | — / — / — / — | — | — |
| Qwen3-Embedding-4B | Mesmo porte, Transformer, líder aberto em PT | 0,6621 [16] | a medir | a medir | a medir |
| Qwen3-Embedding-8B | Líder aberto no MTEB-BR | 0,6704 [16] | a medir | 70,58 [10] | a medir |
| Qwen3-Embedding-0.6B | Baseline pequeno | a medir | 80,7 / 52,3 / 78,9 / 85,8 [17] | 64,33 [9] | a medir |
| multilingual-e5-large-instruct | Baseline multilíngue clássico | a medir | 81,3 / 53,0 / 70,1 / 84,5 [17] | a medir | a medir |
| gte-multilingual-base | Baseline multilíngue | a medir | 80,4 / 50,2 / 76,9 / 84,2 [17] | a medir | a medir |
| llama-embed-nemotron-8b | Mesma família NVIDIA; receita aberta | a medir | a medir | 69,46 [10] | a medir |
| Mamba2 1.3B / Codestral Mamba2 7B | Recorrentes puros | — | — | 55,2 / 59,4 (eixo tarefa) [5] | 40,8 / 44,5 [5] |

Valores com referência entre colchetes vêm da literatura (ver o plano); "(medido)" = avaliado aqui com o mesmo protocolo; "a medir" = ainda não avaliado.

## Tabela 15 — Ablações (MTEB-BR)

| ID | Pergunta | Comparação | Δ Mean22 [IC 95%] | p | Tarefas |
|---|---|---|---|---|---|
| A1 | O estágio S1 é necessário? | C2 − C2-direct | — | — | 0 |
| A2 | Dados reais (mesma escala) | C2 − C2-real | — | — | 0 |
| A2 | Dados sintéticos (mesma escala) | C2 − C2-syn | — | — | 0 |
| A3 | Sem negativos difíceis | C2 − C2-nohn | — | — | 0 |
| A3 | Top-k ingênuo | C2 − C2-naivehn | — | — | 0 |
| A4 | In-batch + máscara de falso negativo | C2 − C2-ib | — | — | 0 |
| A5 | Bidirecional (atenção) | B2a − C2 | — | — | 0 |
| A5 | Bidirecional (atenção + BiMamba) | B2f − C2 | — | — | 0 |
| A7 | Echo embeddings sem treino | R0e − R0 | — | — | 0 |
| A9 | Merge causal vs C2 | M-C − C2 | — | — | 0 |
| A13 | Contexto longo | C3-long − C2 | — | — | 0 |

