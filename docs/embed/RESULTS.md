# Lumi-Embed — resultados

Gerado por `lumi_embed report` a partir de `results/embed/`. Notas × 100. Células "—" = sem resultado ainda. IC 95 % por bootstrap sobre tarefas; p-valores por bootstrap pareado (unidade = tarefa), como no MTEB-BR [16]. Mesmo protocolo para todos os modelos: mesmas instruções em português (só na query) e comprimento máximo de 512 tokens.

O **lumi-emb** é o checkpoint `M-C` (M-C-base0.75), escolhido no dev pela regra: merge se dev(merge) >= dev(melhor individual); senão o melhor individual (dev mean nDCG@10, desempate Borda). Benchmarks não entram na escolha.

O **lumi-emb-long** é o checkpoint `M-L` (M-L-slerp-C3-long-lumi-emb-t0.25), escolhido no dev curto + longo pela mesma regra.

## Tabela 12 — Visão geral

| Modelo | MTEB-BR Mean22 [IC 95%] | PT STS | PT Cls | PT Ret | PT Rer | MMTEB | MTEB eng v2 | LongEmbed |
|---|---|---|---|---|---|---|---|---|
| R0 | 30.65 [20.79, 40.75] | 42.71 | 43.61 | 0.17 | 52.01 | — | — | — |
| R0e | 28.36 [19.37, 37.57] | 40.63 | 41.61 | 0.20 | 53.80 | — | — | — |
| R1 | — | — | — | — | — | — | — | — |
| C1 | 61.78 [53.61, 69.43] | 77.28 | 55.92 | 77.25 | 85.32 | — | — | — |
| C2 | 61.97 [53.83, 69.65] | 78.62 | 55.78 | 77.15 | 85.07 | — | — | — |
| C2-s2 | 61.71 [53.43, 69.50] | 78.42 | 55.83 | 77.17 | 85.16 | — | — | — |
| C2-s3 | 61.82 [53.64, 69.52] | 78.12 | 55.55 | 77.03 | 85.03 | — | — | — |
| C2-real | 61.74 [53.52, 69.44] | 78.16 | 55.65 | 76.63 | 85.00 | — | — | — |
| C2-syn | 61.14 [52.84, 68.99] | 77.49 | 55.54 | 76.55 | 84.86 | — | — | — |
| C2-nohn | — | — | — | — | — | — | — | — |
| C2-naivehn | — | — | — | — | — | — | — | — |
| C2-ib | — | — | — | — | — | — | — | — |
| C2-direct | — | — | — | — | — | — | — | — |
| C3-long | — | — | — | — | — | — | — | 57.54 |
| B2a | — | — | — | — | — | — | — | — |
| B2f | — | — | — | — | — | — | — | — |
| M-C | 62.41 [54.27, 70.14] | 79.49 | 55.40 | 77.84 | 85.35 | — | 63.30 | 47.50 |
| M-B | — | — | — | — | — | — | — | — |
| M-L | 62.48 [54.40, 70.10] | 78.80 | 56.21 | 77.87 | 84.91 | — | — | 56.06 |
| lumi-emb | 62.41 [54.27, 70.14] | 79.49 | 55.40 | 77.84 | 85.35 | — | 63.30 | 47.50 |
| lumi-emb-long | 62.48 [54.40, 70.10] | 78.80 | 56.21 | 77.87 | 84.91 | — | — | 56.06 |
| Qwen3-Embedding-4B | 65.53 [57.33, 73.25] | 83.94 | 58.24 | 77.14 | 84.56 | — | — | 80.12 |
| Qwen3-Embedding-8B | 67.35 [59.41, 74.86] | 84.66 | 62.61 | 78.85 | 85.99 | — | — | — |
| Qwen3-Embedding-0.6B | 61.67 [53.93, 69.10] | 79.72 | 52.41 | 74.53 | 83.82 | — | — | — |
| multilingual-e5-large-instruct | 63.65 [55.38, 71.51] | 81.85 | 56.41 | 76.57 | 86.05 | — | — | — |

## Tabela 13 — MTEB-BR por categoria

| Modelo | Classificação (4) | Multilabel (1) | Pares (2) | STS (2) | Clusterização (5) | Retrieval (6) | Reranking (2) |
|---|---|---|---|---|---|---|---|
| R0 | 59.85 | 21.39 | 46.11 | 44.49 | 37.37 | 1.44 | 18.44 |
| R0e | 55.59 | 21.80 | 42.59 | 38.86 | 34.00 | 1.88 | 17.75 |
| R1 | — | — | — | — | — | — | — |
| C1 | 66.88 | 29.57 | 83.75 | 71.89 | 52.78 | 61.75 | 58.17 |
| C2 | 66.64 | 29.62 | 84.56 | 73.66 | 52.85 | 61.83 | 57.71 |
| C2-s2 | 66.79 | 29.84 | 84.55 | 73.51 | 51.63 | 61.86 | 57.63 |
| C2-s3 | 66.66 | 29.22 | 84.26 | 73.41 | 52.59 | 61.74 | 57.72 |
| C2-real | 67.01 | 29.51 | 83.96 | 72.58 | 53.30 | 61.09 | 57.29 |
| C2-syn | 66.60 | 29.92 | 84.19 | 72.83 | 50.32 | 61.39 | 57.38 |
| C2-nohn | — | — | — | — | — | — | — |
| C2-naivehn | — | — | — | — | — | — | — |
| C2-ib | — | — | — | — | — | — | — |
| C2-direct | — | — | — | — | — | — | — |
| C3-long | — | — | — | — | — | — | — |
| B2a | — | — | — | — | — | — | — |
| B2f | — | — | — | — | — | — | — |
| M-C | 67.34 | 29.63 | 85.03 | 74.78 | 52.90 | 62.18 | 58.36 |
| M-B | — | — | — | — | — | — | — |
| M-L | 67.73 | 29.92 | 84.22 | 73.49 | 53.31 | 62.56 | 58.17 |
| lumi-emb | 67.34 | 29.63 | 85.03 | 74.78 | 52.90 | 62.18 | 58.36 |
| lumi-emb-long | 67.73 | 29.92 | 84.22 | 73.49 | 53.31 | 62.56 | 58.17 |
| Qwen3-Embedding-4B | 63.71 | 36.46 | 87.16 | 80.64 | 64.93 | 62.29 | 58.21 |
| Qwen3-Embedding-8B | 67.13 | 38.96 | 88.24 | 81.24 | 65.27 | 64.81 | 60.07 |
| Qwen3-Embedding-0.6B | 62.66 | 32.56 | 81.38 | 76.22 | 59.64 | 58.22 | 55.39 |
| multilingual-e5-large-instruct | 63.57 | 36.55 | 84.19 | 80.63 | 60.50 | 60.21 | 58.04 |

## Tabela 13b — MTEB-BR por tarefa (seleção; completo em CSV)

| Tarefa | R0 | C1 | C2 | C2-real | C2-syn | lumi-emb | lumi-emb-long | Qwen3-Embedding-4B |
|---|---|---|---|---|---|---|---|---|
| HateBR | 66.57 | 84.79 | 84.82 | 84.64 | 85.24 | 85.26 | 85.62 | 82.81 |
| ToxSynPT | 80.23 | 90.09 | 89.84 | 90.07 | 89.95 | 90.16 | 90.36 | 85.18 |
| FactckBrClassification | 49.15 | 54.09 | 53.13 | 54.17 | 53.32 | 54.56 | 54.67 | 53.75 |
| PortuLexRRIP | 43.44 | 38.55 | 38.76 | 39.15 | 37.89 | 39.38 | 40.27 | 33.09 |
| BrighterEmotionMultilabelClassification | 21.39 | 29.57 | 29.62 | 29.51 | 29.92 | 29.63 | 29.92 | 36.46 |
| AssinRTE | 52.40 | 81.00 | 82.12 | 81.47 | 82.06 | 83.34 | 81.98 | 86.03 |
| InferBR | 39.81 | 86.50 | 87.00 | 86.46 | 86.32 | 86.73 | 86.46 | 88.29 |
| AssinSTS | 45.40 | 68.27 | 70.75 | 69.02 | 70.03 | 72.67 | 70.85 | 79.77 |
| Assin2STS | 43.59 | 75.51 | 76.57 | 76.14 | 75.62 | 76.89 | 76.13 | 81.52 |
| WikipediaPTCategoriesClusteringP2P | 49.58 | 60.89 | 61.86 | 62.36 | 58.19 | 59.24 | 60.65 | 76.21 |
| MedPTClustering | 55.35 | 68.15 | 66.38 | 66.88 | 65.36 | 65.15 | 64.77 | 81.50 |
| JurisTCUClusteringP2P | 11.69 | 31.81 | 32.03 | 30.47 | 29.96 | 32.21 | 31.48 | 28.27 |
| SciELOClusteringP2P | 38.76 | 60.82 | 59.39 | 62.45 | 55.53 | 62.52 | 62.78 | 74.08 |
| StackoverflowPtClustering | 31.49 | 42.24 | 44.61 | 44.35 | 42.57 | 45.37 | 46.85 | 64.58 |
| Quati | 0.00 | 60.06 | 60.67 | 59.40 | 60.71 | 61.14 | 61.81 | 60.76 |
| JurisTCU | 0.78 | 60.90 | 61.08 | 59.83 | 60.62 | 60.91 | 60.42 | 61.36 |
| BRTaxQAR | 1.37 | 21.25 | 21.07 | 20.15 | 20.34 | 21.18 | 22.13 | 28.86 |
| FaQuADIR | 3.41 | 81.31 | 82.35 | 80.07 | 83.17 | 81.43 | 82.42 | 80.70 |
| MedPTRetrieval | 2.13 | 78.88 | 77.62 | 79.33 | 76.77 | 79.40 | 80.06 | 77.62 |
| FaqBacenRetrieval | 0.95 | 68.08 | 68.22 | 67.76 | 66.72 | 69.03 | 68.54 | 64.42 |
| QuatiReranking | 19.65 | 60.55 | 60.14 | 60.16 | 59.48 | 60.96 | 60.65 | 60.57 |
| JurisTCUReranking | 17.23 | 55.78 | 55.28 | 54.43 | 55.28 | 55.77 | 55.69 | 55.85 |

## Variação entre sementes (C2)

| Métrica | C2 | C2-s2 | C2-s3 | Média | Desvio-padrão |
|---|---|---|---|---|---|
| MTEB-BR Mean22 | 61.97 | 61.71 | 61.82 | 61.83 | 0.13 |
| Classificação (4) | 66.64 | 66.79 | 66.66 | 66.70 | 0.08 |
| Multilabel (1) | 29.62 | 29.84 | 29.22 | 29.56 | 0.31 |
| Pares (2) | 84.56 | 84.55 | 84.26 | 84.46 | 0.17 |
| STS (2) | 73.66 | 73.51 | 73.41 | 73.52 | 0.12 |
| Clusterização (5) | 52.85 | 51.63 | 52.59 | 52.36 | 0.64 |
| Retrieval (6) | 61.83 | 61.86 | 61.74 | 61.81 | 0.07 |
| Reranking (2) | 57.71 | 57.63 | 57.72 | 57.69 | 0.05 |
| PT STS | 78.62 | 78.42 | 78.12 | 78.39 | 0.25 |
| PT Cls | 55.78 | 55.83 | 55.55 | 55.72 | 0.15 |
| PT Ret | 77.15 | 77.17 | 77.03 | 77.12 | 0.08 |
| PT Rer | 85.07 | 85.16 | 85.03 | 85.09 | 0.07 |

Mesma receita e mesmos dados; muda só a semente. Diferenças entre checkpoints menores que cerca de 2 desvios-padrão não se distinguem da variação entre execuções.

## Tabela 14 — Comparação com modelos correspondentes

| Modelo | Relação | MTEB-BR Mean22 | MTEB-PT (STS/Cls/Ret/Rer) | MMTEB | LongEmbed |
|---|---|---|---|---|---|
| lumi-emb-long (este trabalho) | Modelo publicado (até 8k tokens) | 62.48 | 78.8 / 56.2 / 77.9 / 84.9 | — | 56.06 |
| lumi-emb (este trabalho) | Versão curta (até 512 tokens) | 62.41 | 79.5 / 55.4 / 77.8 / 85.4 | — | 47.50 |
| R0 (este trabalho) | Ponto de partida | 30.65 | 42.7 / 43.6 / 0.2 / 52.0 | — | — |
| Qwen3-Embedding-4B | Mesmo porte, Transformer, líder aberto em PT | 65.53 (medido) | 83.9 / 58.2 / 77.1 / 84.6 (medido) | a medir | 80.12 (medido) |
| Qwen3-Embedding-8B | Líder aberto no MTEB-BR | 67.35 (medido) | 84.7 / 62.6 / 78.9 / 86.0 (medido) | 70.58 [10] | a medir |
| Qwen3-Embedding-0.6B | Baseline pequeno | 61.67 (medido) | 79.7 / 52.4 / 74.5 / 83.8 (medido) | 64.33 [9] | a medir |
| multilingual-e5-large-instruct | Baseline multilíngue clássico | 63.65 (medido) | 81.8 / 56.4 / 76.6 / 86.0 (medido) | a medir | a medir |
| gte-multilingual-base | Baseline multilíngue | a medir | 80.4 / 50.2 / 76.9 / 84.2 [17] | a medir | a medir |
| llama-embed-nemotron-8b | Mesma família NVIDIA; receita aberta | a medir | a medir | 69.46 [10] | a medir |
| Mamba2 1.3B / Codestral Mamba2 7B | Recorrentes puros | — | — | 55.2 / 59.4 (eixo tarefa) [5] | 40.8 / 44.5 [5] |

Valores com referência entre colchetes vêm da literatura (ver o plano); "(medido)" = avaliado aqui com o mesmo protocolo; "a medir" = ainda não avaliado.

## Tabela 14b — Comparação pareada no MTEB-BR

| Comparação | Δ Mean22 [IC 95%] | p | Vitórias (tarefas) |
|---|---|---|---|
| lumi-emb-long − Qwen3-Embedding-4B | -3.05 [-6.04, -0.25] | 0.033 | 10/22 |
| lumi-emb-long − Qwen3-Embedding-8B | -4.87 [-7.33, -2.50] | < 0.001 | 3/22 |
| lumi-emb-long − Qwen3-Embedding-0.6B | +0.81 [-1.74, +3.21] | 0.518 | 13/22 |
| lumi-emb-long − multilingual-e5-large-instruct | -1.17 [-3.68, +1.20] | 0.350 | 10/22 |
| lumi-emb-long − lumi-emb | +0.07 [-0.28, +0.41] | 0.671 | 12/22 |
| lumi-emb − Qwen3-Embedding-4B | -3.13 [-6.21, -0.29] | 0.030 | 10/22 |
| lumi-emb − Qwen3-Embedding-8B | -4.95 [-7.57, -2.53] | < 0.001 | 3/22 |
| lumi-emb − Qwen3-Embedding-0.6B | +0.74 [-1.92, +3.14] | 0.560 | 14/22 |
| lumi-emb − multilingual-e5-large-instruct | -1.25 [-3.74, +1.05] | 0.309 | 10/22 |
| C2 − Qwen3-Embedding-4B | -3.56 [-6.60, -0.76] | 0.012 | 6/22 |

## MLDR-pt com documentos de até 8k tokens

| Modelo | MLDR-pt, 512 tokens | MLDR-pt, 8k tokens |
|---|---|---|
| lumi-emb | 61.99 | 47.88 |
| C3-long | — | 75.68 |
| lumi-emb-long | 63.99 | 73.94 |
| Qwen3-Embedding-4B | 66.92 | 80.23 |

MultiLongDocRetrieval em português (documentos longos da Wikipédia e do mC4). No MTEB-PT ele é avaliado com 512 tokens, como todo o protocolo; aqui, também com os documentos em até 8k tokens.

## LongEmbed por comprimento (tokens)

| Modelo | Tarefa | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| lumi-emb | Passkey | 96 | 96 | 100 | 100 | 100 | 100 | 66 | 48 |
| lumi-emb | Needle | 80 | 82 | 72 | 78 | 50 | 22 | 0 | 4 |
| C3-long | Passkey | 90 | 96 | 100 | 100 | 100 | 100 | 96 | 48 |
| C3-long | Needle | 88 | 86 | 88 | 90 | 82 | 48 | 2 | 4 |
| lumi-emb-long | Passkey | 90 | 96 | 100 | 100 | 100 | 100 | 92 | 50 |
| lumi-emb-long | Needle | 86 | 86 | 86 | 90 | 78 | 40 | 2 | 4 |
| Qwen3-Embedding-4B | Passkey | 100 | 98 | 98 | 94 | 100 | 96 | 90 | 96 |
| Qwen3-Embedding-4B | Needle | 84 | 76 | 58 | 60 | 86 | 82 | 74 | 82 |

## Sensibilidade à contaminação

A auditoria posterior (`results/embed/contamination/audit.md`) achou textos de treino que coincidem com textos destas tarefas, que o índice usado no filtro não cobria: Quati, BRTaxQAR, FaQuADIR, QuatiReranking. São no máximo 0,1 % dos exemplos de cada conjunto. Para medir o efeito, a média é refeita sem essas tarefas, e a comparação pareada é repetida.

| Modelo | Mean22 | Mean18 (sem as 4) | lumi-emb-long − modelo (22) | lumi-emb-long − modelo (18) | p (18) |
|---|---|---|---|---|---|
| lumi-emb-long | 62.48 | 63.75 | — | — | — |
| lumi-emb | 62.41 | 63.79 | +0.07 | -0.04 [-0.44, +0.35] | 0.847 |
| C2 | 61.97 | 63.28 | +0.51 | +0.47 [-0.11, +1.10] | 0.118 |
| Qwen3-Embedding-4B | 65.53 | 67.27 | -3.05 | -3.52 [-7.12, -0.12] | 0.042 |
| Qwen3-Embedding-8B | 67.35 | 68.85 | -4.87 | -5.10 [-8.00, -2.32] | < 0.001 |
| Qwen3-Embedding-0.6B | 61.67 | 63.15 | +0.81 | +0.60 [-2.46, +3.45] | 0.673 |
| multilingual-e5-large-instruct | 63.65 | 65.89 | -1.17 | -2.14 [-4.99, +0.69] | 0.139 |

## Tabela 15 — Ablações (MTEB-BR)

| ID | Pergunta | Comparação | Δ Mean22 [IC 95%] | p | Tarefas |
|---|---|---|---|---|---|
| A1 | O estágio S1 é necessário? | C2 − C2-direct | — | — | 0 |
| S2 | O S2 (negativos difíceis) melhora o C1? | C2 − C1 | +0.19 [-0.24, +0.64] | 0.394 | 22 |
| A2 | Só dados reais no S2 | C2 − C2-real | +0.23 [-0.27, +0.68] | 0.349 | 22 |
| A2 | Só dados sintéticos no S2 | C2 − C2-syn | +0.83 [+0.38, +1.36] | < 0.001 | 22 |
| A2 | Reais contra sintéticos | C2-real − C2-syn | +0.60 [-0.17, +1.49] | 0.138 | 22 |
| A3 | Sem negativos difíceis | C2 − C2-nohn | — | — | 0 |
| A3 | Top-k ingênuo | C2 − C2-naivehn | — | — | 0 |
| A4 | In-batch + máscara de falso negativo | C2 − C2-ib | — | — | 0 |
| A5 | Bidirecional (atenção) | B2a − C2 | — | — | 0 |
| A5 | Bidirecional (atenção + BiMamba) | B2f − C2 | — | — | 0 |
| A7 | Echo embeddings sem treino | R0e − R0 | -2.30 [-4.54, -0.32] | 0.021 | 22 |
| A9 | Merge causal vs C2 | M-C − C2 | +0.44 [-0.05, +0.90] | 0.074 | 22 |
| A13 | Contexto longo | C3-long − C2 | — | — | 0 |

No A2, os conjuntos do S2 não têm o mesmo tamanho: S2-real e S2-syn ficaram menores que o S2 completo e diferentes entre si (tamanhos em `data/embed/sets/manifest.json` e no DATA_CARD). A versão na mesma escala, prevista no plano, fica para a fase de ablações.

## Hipóteses pré-registradas: situação

| ID | Hipótese | Teste | Δ [IC 95%] | p | Situação |
|---|---|---|---|---|---|
| H1 | Paridade prática com Transformers de porte similar | C2 − Qwen3-Embedding-4B | -3.56 [-6.60, -0.76] | 0.012 | Refutada |
| H2 | O S1 melhora o resultado final | C2 − C2-direct | — | — | — |
| H3 | Reais ajudam mais nas tarefas nativas | C2-real − C2-syn | +0.60 [-0.17, +1.49] | 0.138 | Não refutada |
| H4 | Negativos positive-aware superam o top-k ingênuo | C2 − C2-naivehn | — | — | — |
| H5 | O bidirecional melhora a média | B2a − C2; B2f − C2 | —; — | — | — |
| H6 | O merge supera o melhor checkpoint individual (dev) | M-C (M-C-base0.75) − C2-real, nDCG@10 no dev | +0.08 | — | Não refutada |
| H7 | Vantagem relativa em entradas longas | C3-long − Qwen3-Embedding-4B, LongEmbed | -22.58 | — | Refutada |

"Refutada" / "Não refutada" aplicam o critério de falseamento do plano; "—" = falta executar. H6 usa o dev (critério do plano), sem IC.

## Dev set (seleção de checkpoints)

| Checkpoint | dev nDCG@10 médio | Passos | Minutos de treino | Observação |
|---|---|---|---|---|
| C1 | 0.9740 | 139 | 274.0 |  |
| C2 | 0.9767 | 937 | 246.6 |  |
| C2-s2 | 0.9768 | 937 | 245.8 |  |
| C2-s3 | 0.9764 | 937 | 247.3 |  |
| C2-real | 0.9771 | 711 | 187.6 |  |
| C2-syn | 0.9726 | 579 | 157.8 |  |
| C3-long | 0.9756 | 755 | 850.4 |  |
| M-C | 0.9779 | — | — | = M-C-base0.75 |
| M-L | 0.9756 | — | — | = M-L-slerp-C3-long-lumi-emb-t0.25 |
| lumi-emb | 0.9779 | — | — | = M-C (M-C-base0.75) |
| lumi-emb-long | 0.9756 | — | — | = M-L (M-L-slerp-C3-long-lumi-emb-t0.25) |

## Grade do merge M-C (dev)

| Candidato | dev nDCG@10 | Borda |  |
|---|---|---|---|
| M-C-base0.75 | 0.9779 | 21.0 | escolhido |
| M-C-slerp-C2-real-C2-syn-t0.25 | 0.9778 | 25.5 |  |
| C2-real | 0.9771 | 23.0 | melhor individual |
| M-C-slerp-C2-real-C2-syn-t0.5 | 0.9770 | 21.0 |  |
| C2-s2 | 0.9768 | 23.0 |  |
| C2 | 0.9767 | 20.5 |  |
| C2-s3 | 0.9764 | 16.0 |  |
| M-C-avg | 0.9764 | 18.0 |  |
| M-C-slerp-C2-real-C2-syn-t0.75 | 0.9741 | 10.0 |  |
| M-C-base0.5 | 0.9739 | 11.0 |  |
| C2-syn | 0.9726 | 9.0 |  |
| M-C-base0.25 | 0.9608 | 0.0 |  |

## Grade do merge M-L (dev)

| Candidato | dev nDCG@10 | Borda |  |
|---|---|---|---|
| C3-long | 0.9756 | 8.5 | melhor individual |
| M-L-slerp-C3-long-lumi-emb-t0.25 | 0.9756 | 7.5 | escolhido |
| M-L-slerp-C3-long-lumi-emb-t0.5 | 0.9753 | 12.0 |  |
| M-L-avg | 0.9748 | 13.0 |  |
| M-L-slerp-C3-long-lumi-emb-t0.75 | 0.9689 | 7.5 |  |
| lumi-emb | 0.9572 | 11.5 |  |

