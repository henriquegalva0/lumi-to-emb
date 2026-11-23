---
language:
  - pt
  - en
license: other
license_name: uso-nao-comercial
license_link: https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DATA_CARD.md#licenças-o-que-isto-implica-para-o-modelo-decisão-d2
base_model: ceia-research/lumi-4b
base_model_relation: finetune
pipeline_tag: feature-extraction
library_name: sentence-transformers
tags:
  - sentence-transformers
  - feature-extraction
  - embeddings
  - retrieval
  - mamba
  - mamba2
  - hybrid
  - nemotron_h
  - portuguese
  - mteb
datasets:
  - wikimedia/wikipedia
  - unicamp-dl/mmarco
  - HuggingFaceFW/fineweb-2
  - HuggingFaceFW/fineweb-edu
  - sentence-transformers/gooaq
  - sentence-transformers/all-nli
  - sentence-transformers/parallel-sentences-talks
  - sentence-transformers/parallel-sentences-tatoeba
---

# Lumi-Embed (lumi-4b-embedding)

> **Versão de pesquisa.** O modelo publicado (branch `main`) é o **lumi-emb**, um merge de checkpoints escolhido no dev. No MTEB-BR (22 tarefas nativas em PT-BR), ele fica em **62,41**: empatado estatisticamente com o multilingual-e5-large-instruct e o Qwen3-Embedding-0.6B e abaixo do Qwen3-Embedding-4B (65,53). Em retrieval, empata com o Qwen3-Embedding-4B, e tem a maior média em classificação entre os modelos comparados. A maior distância está em clusterização e STS e, sobretudo, em **documentos longos**: o treino usou só 512 tokens, e no LongEmbed (47,50 contra 80,12) e no MLDR-pt com documentos de até 8k tokens (47,88 contra 80,23) ele fica muito atrás. Para RAG, use trechos de até 512 tokens; o estágio de contexto longo está em treino. As tabelas abaixo são geradas a partir dos arquivos de avaliação. As ablações restantes (trilho bidirecional, contexto longo, variantes de negativos) ainda não foram executadas.

**English summary.** Lumi-Embed turns [Lumi-4B](https://huggingface.co/ceia-research/lumi-4b), a 4B hybrid Mamba-2 + attention LLM for Brazilian Portuguese, into a text-embedding model:

- last-token (EOS) pooling, with a bidirectional track as an ablation;
- staged contrastive training: a weakly supervised stage and a supervised stage with positive-aware hard negatives;
- training data: native PT-BR synthetic data from three open generators, plus natural pairs;
- checkpoint merging, with checkpoints selected on a held-out dev set;
- evaluation on MTEB-BR (22 native tasks), MTEB-PT, LongEmbed and MTEB (eng, v2), under a clean zero-shot contamination policy (a filter bug left 13 tasks unchecked; a post-hoc audit found overlaps in at most 0.1 % of the training examples, and the conclusions hold without the 4 affected tasks).

**Result:** 62.41 on MTEB-BR (22 native PT-BR tasks), statistically tied with multilingual-e5-large-instruct and Qwen3-Embedding-0.6B and below Qwen3-Embedding-4B (65.53). It matches Qwen3-Embedding-4B in retrieval and has the highest classification average among the compared models; it is weakest in clustering, STS and especially long documents (trained on 512 tokens only: 47.50 vs 80.12 on LongEmbed). For RAG, use chunks of up to 512 tokens; a long-context stage is in training. MTEB (eng, v2): 63.30.

As far as we know, it is one of the first evaluations of a hybrid Mamba-Transformer as a text embedder. Code, configs and data recipes: [github.com/henriquegalva0/lumi-to-emb](https://github.com/henriquegalva0/lumi-to-emb).

## Visão geral

| | |
|---|---|
| Backbone | [`ceia-research/lumi-4b`](https://huggingface.co/ceia-research/lumi-4b) (revisão `27b30f4`): 42 camadas, das quais 21 Mamba-2, 17 MLP e 4 de atenção; sem RoPE; derivado do Nemotron 3 Nano 4B |
| Embedding | estado oculto do último token (`</s>`), depois da normalização final; L2; similaridade de cosseno; dimensão 3136 |
| Formato | query: `Instruct: {instrução}\nQuery: {query}</s>`; documento: `{texto}</s>`, sem instrução |
| Modelo publicado | `lumi-emb` = M-C-base0.75: 0,75 × média de cinco checkpoints do S2 (C2, C2-real, C2-syn, C2-s2, C2-s3) + 0,25 × Lumi-4B sem treino (R0) |
| Comprimento | treinado com até 512 tokens; **para RAG, use trechos de até 512 tokens** (no MLDR-pt, ler o documento em até 8k piora o resultado: 61,99 → 47,88). O estágio de contexto longo (C3-long) está em treino |
| Idiomas | português brasileiro (principal) e inglês |
| Carregamento | transformers ≥ 5.23 nativo (`NemotronHModel`), **sem** `trust_remote_code` |

## Uso

O tokenizer salvo já anexa o `</s>` e faz padding à direita. Use sempre o mesmo formato de instrução nas queries e deixe os documentos sem instrução. Em RAG, divida os documentos em trechos de até 512 tokens, o comprimento de treino: com textos mais longos, a qualidade cai.

**Sentence-Transformers** (o prompt `query` traz a instrução de busca padrão):

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("henriquegalvao/lumi-4b-embedding", model_kwargs={"dtype": "bfloat16"})
queries = ["quais as regras para dedução de despesas médicas no imposto de renda?"]
docs = ["As despesas médicas podem ser deduzidas da base de cálculo do imposto de renda."]
q = model.encode(queries, prompt_name="query")
d = model.encode(docs)
print(model.similarity(q, d))
# Outra tarefa: prompt=f"Instruct: {instrucao}\nQuery: "
```

**Transformers:**

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

repo = "henriquegalvao/lumi-4b-embedding"
tok = AutoTokenizer.from_pretrained(repo)                      # anexa </s>, padding à direita
model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()

def embed(texts, instruction=None):
    if instruction:
        texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
    batch = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        h = model(**batch, use_cache=False).last_hidden_state
    last = batch["attention_mask"].sum(dim=1) - 1               # posição do </s>
    return F.normalize(h[torch.arange(h.shape[0], device=h.device), last].float(), dim=-1)

q = embed(["o que é inflação?"], "Dada uma pergunta, recupere passagens que a respondam")
d = embed(["Inflação é o aumento generalizado e contínuo dos preços."])
print(q @ d.T)
```

**Kernels Mamba-2 otimizados** (CUDA, torch 2.12, `pip install kernels`). Sem eles, o modelo roda no caminho PyTorch puro: o resultado é o mesmo, mas fica mais lento (na B200, num passo de treino com sequências de 512 tokens, os kernels processaram 4,5× mais tokens por segundo; o caminho PyTorch puro só coube com lote 4). Ative-os depois de mover o modelo para a GPU, porque o `from_pretrained(..., use_kernels=True)` escolhe os kernels pelo dispositivo do carregamento:

```python
from kernels import Mode

model = AutoModel.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
model.set_use_kernels(True, mode=Mode.TRAINING)   # o modo de treino também serve à inferência
# Sentence-Transformers: model[0].auto_model.set_use_kernels(True, mode=Mode.TRAINING)
```

Os checkpoints intermediários ficam em branches com o nome de cada um. Exemplo: `revision="C2"`.

Os trechos desta seção são executados contra este repositório pelo passo `hubcheck` do código: os embeddings do Sentence-Transformers, do Transformers e dos kernels batem com os do checkpoint local (cosseno ≥ 0,9999), e cada branch tem só os próprios pesos.

## Contexto: a janela efetiva do Lumi-4B (Etapa 1)

Antes do treino, medimos até quantos tokens o `lumi-4b` recupera informação. O teste foi *needle-in-a-haystack* em PT-BR, de 2k a 255k tokens, com 2.610 prompts por modelo e o modelo-base como controle. [Relatório completo](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/stage1/README.md).

| Modelo | L\* (1 agulha) | L\* (multi-chave) | L\* (multi-valor) |
|---|---|---|---|
| Lumi-4B | 32k | **16k** | **16k** |
| Nemotron-3-Nano-4B (base) | 255k | 255k | 128k |

O Lumi colapsa a partir de 64k: as respostas entram em laço e saem sem número. Já em 32k, ele confunde chaves e valores. A perda vem do pré-treino contínuo em janelas de 8k, porque a base mantém 255k. Por isso o embedder treina com 512 tokens. A extensão até 8k fica para um estágio separado (C3-long), ainda não treinado.

## Arquitetura e checkpoints

Cada checkpoint é definido pelos eixos da Tabela 4 do [plano](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/plano_experimento.pdf). Os nomes seguem o plano: R = referência, C = causal (EOS), B = bidirecional (média), M = merge.

| ID | Parte de | Direção / pooling | Treino | Dados | Negativos | Pergunta |
|---|---|---|---|---|---|---|
| R0 | lumi-4b | C / EOS | nenhum | — | — | Quanto o lumi-4b serve como embedder sem treino? |
| R0e | lumi-4b | C / EOS + echo | nenhum | — | — | O echo ajuda sem treino? (A7) |
| R1 | lumi-4b | C / EOS | 1 estágio, τ = 0,01 | mMARCO-PT | 63 BM25 | Receita do Mamba Retriever |
| C1 | lumi-4b | C / EOS | S1 | naturais + sintético + mMARCO-PT + EN | 1 + batch | O pré-treino contrastivo sozinho já serve? |
| **C2** | C1 | C / EOS | S2 | misto | 4, PercPos | **Modelo causal principal** |
| C2-real / C2-syn | C1 | C / EOS | S2 | só real / só sintético | 4, PercPos | Valor de cada tipo de dado (A2) |
| C2-nohn / C2-naivehn / C2-ib | C1 | C / EOS | S2 | misto | batch / top-k ingênuo / 4 + batch | Negativos e perda (A3, A4) |
| C2-direct | lumi-4b | C / EOS | S2 direto | misto | 4, PercPos | O S1 é necessário? (A1) |
| C3-long | C2 | C / EOS | S-long (2k→8k) | documentos longos | 4 | Contexto longo (A13) |
| B0a / B0f | lumi-4b | Ba / Bf | MNTP | texto cru | — | A adaptação bidirecional funciona num híbrido? |
| B1a/f → B2a/f | B0 | B / média | S1 → S2 | como C1/C2 | como C1/C2 | Bidirecional (A5) |
| M-C / M-B | ver abaixo | C / B | merge | — | — | O merge supera o melhor individual? (A9) |

Ba deixa só as 4 camadas de atenção bidirecionais, com as camadas Mamba causais. Bf acrescenta uma varredura reversa nos mixers Mamba-2, com os mesmos pesos, invertendo só os tokens válidos. O trilho B é **experimental**: não há validação prévia dessa adaptação em um LLM Mamba ou híbrido pré-treinado.

**Merge.** Só se mesclam checkpoints do mesmo trilho, com o LoRA já fundido. Para o M-C, os candidatos são C2, C2-real, C2-syn, C2-s2 e C2-s3. A grade inclui:

- média linear;
- SLERP entre C2-real e C2-syn, com t ∈ {0,25; 0,5; 0,75};
- interpolação da média com o R0, com λ ∈ {0,25; 0,5; 0,75}.

A escolha é feita no dev, nunca nos testes. O modelo publicado é o melhor no dev. Ele fica no `main` e no branch `M-C`, com a receita no `meta.json`; as notas de todos os candidatos no dev estão na tabela "Grade do merge M-C", abaixo, e no [RESULTS.md](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/RESULTS.md). O escolhido foi o **M-C-base0.75**, com nDCG@10 de 0,9779 no dev, contra 0,9771 do melhor checkpoint individual (C2-real). Nos benchmarks, ele ficou 0,44 ponto acima do C2 no MTEB-BR (p = 0,074 no bootstrap por tarefa) e 0,58 acima da média das três sementes do C2, cujo desvio-padrão é 0,13.

## Dados

Detalhes na [ficha dos dados](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DATA_CARD.md).

- **Reais:**
  - pares estruturais da Wikipédia PT (título↔lead, seção↔texto);
  - amostra do mMARCO-PT;
  - bitexto PT↔EN;
  - uma fração em inglês (GooAQ, AllNLI).
- **Sintéticos, em PT-BR nativo:** três geradores abertos (Qwen3-30B-A3B-Instruct-2507, Mistral-Small-3.2-24B, gpt-oss-20b), com personas, tipos de consulta, dificuldades e registros sorteados, mais STS com paráfrase e contraste.
- **Negativos:** minerados com o Qwen3-Embedding-4B pelo critério **TopK-PercPos** (score < 0,95 × score do positivo). Os sintéticos passam por um filtro de cosseno > 0,7 e precisam ter o positivo no top-10 do teacher.
- **Contaminação:** política **zero-shot limpo**. Nenhum dataset de treino é tarefa de avaliação. Também saem os textos que coincidem com os do MTEB-BR, MTEB-PT, LongEmbed ou MTEB eng v2, inteiros ou com ≥ 30 % de 13-gramas em comum. Não há dados rotulados em PT, porque todos os candidatos são tarefas de avaliação.
- **Falha no filtro e auditoria posterior:** o índice usado no filtro deixou de fora 13 tarefas, que guardam os textos num formato antigo do mteb: as 8 de retrieval e reranking do MTEB-BR e 5 do LongEmbed. Depois do treino, uma auditoria com o índice corrigido achou coincidências em no máximo 0,1 % dos exemplos de cada conjunto, quase todas com passagens web do Quati e com parágrafos do FaQuAD-IR (estes, sobretudo como negativos). Nenhuma com FaqBacen, JurisTCU ou MedPT. Sem as 4 tarefas afetadas, as conclusões da comparação não mudam (ver "Sensibilidade à contaminação" abaixo). Detalhes na [ficha dos dados](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DATA_CARD.md#auditoria-posterior).

## Avaliação

Protocolo igual para todos os modelos:

- instrução só na query (nos dois lados nas tarefas simétricas), com as mesmas instruções em português por tipo de tarefa;
- documentos sem instrução;
- mesmo comprimento máximo: 512 tokens (32k no LongEmbed).

O benchmark principal é o **MTEB-BR**, 22 tarefas nativas ([arXiv:2607.04581](https://arxiv.org/abs/2607.04581)). Completam o quadro o **MTEB-PT** ([arXiv:2607.04071](https://arxiv.org/abs/2607.04071)), o LongEmbed e o MTEB (eng, v2); o MMTEB é opcional. Os ICs de 95 % e os p-valores usam bootstrap com a tarefa como unidade, como no MTEB-BR. Uma diferença só é declarada com p < 0,05.

<!-- RESULTS:START -->
### Resultados

Notas × 100, geradas pelos arquivos de avaliação ([tabelas completas](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/RESULTS.md)). Mesmo protocolo para todos os modelos: mesmas instruções em português, só na query, e até 512 tokens. IC 95 % e p-valores por bootstrap sobre as tarefas, como no MTEB-BR.

#### MTEB-BR (22 tarefas nativas em PT-BR)

| Modelo | Mean22 [IC 95%] | Classificação (4) | Multilabel (1) | Pares (2) | STS (2) | Clust. (5) | Retrieval (6) | Reranking (2) |
|---|---|---|---|---|---|---|---|---|
| **lumi-emb** (este trabalho) | 62.41 [54.27, 70.14] | 67.3 | 29.6 | 85.0 | 74.8 | 52.9 | 62.2 | 58.4 |
| Qwen3-Embedding-8B | 67.35 [59.41, 74.86] | 67.1 | 39.0 | 88.2 | 81.2 | 65.3 | 64.8 | 60.1 |
| Qwen3-Embedding-4B | 65.53 [57.33, 73.25] | 63.7 | 36.5 | 87.2 | 80.6 | 64.9 | 62.3 | 58.2 |
| multilingual-e5-large-instruct | 63.65 [55.38, 71.51] | 63.6 | 36.5 | 84.2 | 80.6 | 60.5 | 60.2 | 58.0 |
| Qwen3-Embedding-0.6B | 61.67 [53.93, 69.10] | 62.7 | 32.6 | 81.4 | 76.2 | 59.6 | 58.2 | 55.4 |
| Lumi-4B sem treino (R0) | 30.65 [20.79, 40.75] | 59.8 | 21.4 | 46.1 | 44.5 | 37.4 | 1.4 | 18.4 |

#### Comparação pareada no MTEB-BR

| Comparação | Δ Mean22 [IC 95%] | p | Vitórias (tarefas) |
|---|---|---|---|
| lumi-emb − Qwen3-Embedding-4B | -3.13 [-6.21, -0.29] | 0.030 | 10/22 |
| lumi-emb − Qwen3-Embedding-8B | -4.95 [-7.57, -2.53] | < 0.001 | 3/22 |
| lumi-emb − Qwen3-Embedding-0.6B | +0.74 [-1.92, +3.14] | 0.560 | 14/22 |
| lumi-emb − multilingual-e5-large-instruct | -1.25 [-3.74, +1.05] | 0.309 | 10/22 |
| C2 − Qwen3-Embedding-4B | -3.56 [-6.60, -0.76] | 0.012 | 6/22 |

#### MTEB-PT (por família)

| Modelo | PT STS | PT Cls | PT Ret | PT Rer |
|---|---|---|---|---|
| **lumi-emb** (este trabalho) | 79.5 | 55.4 | 77.8 | 85.4 |
| Qwen3-Embedding-8B | 84.7 | 62.6 | 78.9 | 86.0 |
| Qwen3-Embedding-4B | 83.9 | 58.2 | 77.1 | 84.6 |
| multilingual-e5-large-instruct | 81.8 | 56.4 | 76.6 | 86.0 |
| Qwen3-Embedding-0.6B | 79.7 | 52.4 | 74.5 | 83.8 |
| Lumi-4B sem treino (R0) | 42.7 | 43.6 | 0.2 | 52.0 |

#### Outras suítes

| Modelo | MMTEB | MTEB eng v2 | LongEmbed |
|---|---|---|---|
| **lumi-emb** (este trabalho) | — | 63.30 | 47.50 |
| Qwen3-Embedding-4B | — | — | 80.12 |

#### Sensibilidade à contaminação

A auditoria posterior (`results/embed/contamination/audit.md`) achou textos de treino que coincidem com textos destas tarefas, que o índice usado no filtro não cobria: Quati, BRTaxQAR, FaQuADIR, QuatiReranking. São no máximo 0,1 % dos exemplos de cada conjunto. Para medir o efeito, a média é refeita sem essas tarefas, e a comparação pareada é repetida.

| Modelo | Mean22 | Mean18 (sem as 4) | lumi-emb − modelo (22) | lumi-emb − modelo (18) | p (18) |
|---|---|---|---|---|---|
| lumi-emb | 62.41 | 63.79 | — | — | — |
| C2 | 61.97 | 63.28 | +0.44 | +0.51 [-0.07, +1.06] | 0.085 |
| Qwen3-Embedding-4B | 65.53 | 67.27 | -3.13 | -3.48 [-7.18, -0.07] | 0.047 |
| Qwen3-Embedding-8B | 67.35 | 68.85 | -4.95 | -5.06 [-8.15, -2.24] | < 0.001 |
| Qwen3-Embedding-0.6B | 61.67 | 63.15 | +0.74 | +0.64 [-2.56, +3.51] | 0.659 |
| multilingual-e5-large-instruct | 63.65 | 65.89 | -1.25 | -2.10 [-4.99, +0.67] | 0.139 |

#### MLDR-pt com documentos de até 8k tokens

| Modelo | MLDR-pt, 512 tokens | MLDR-pt, 8k tokens |
|---|---|---|
| lumi-emb | 61.99 | 47.88 |
| Qwen3-Embedding-4B | 66.92 | 80.23 |

MultiLongDocRetrieval em português (documentos longos da Wikipédia e do mC4). No MTEB-PT ele é avaliado com 512 tokens, como todo o protocolo; aqui, também com os documentos em até 8k tokens.

#### LongEmbed por comprimento (tokens)

| Modelo | Tarefa | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|
| lumi-emb | Passkey | 96 | 96 | 100 | 100 | 100 | 100 | 66 | 48 |
| lumi-emb | Needle | 80 | 82 | 72 | 78 | 50 | 22 | 0 | 4 |
| Qwen3-Embedding-4B | Passkey | 100 | 98 | 98 | 94 | 100 | 96 | 90 | 96 |
| Qwen3-Embedding-4B | Needle | 84 | 76 | 58 | 60 | 86 | 82 | 74 | 82 |

#### Checkpoints deste trabalho

| Checkpoint | MTEB-BR Mean22 [IC 95%] | dev nDCG@10 | Observação |
|---|---|---|---|
| R0 | 30.65 [20.79, 40.75] | — |  |
| R0e | 28.36 [19.37, 37.57] | — |  |
| C1 | 61.78 [53.61, 69.43] | 0.9740 |  |
| C2 | 61.97 [53.83, 69.65] | 0.9767 |  |
| C2-s2 | 61.71 [53.43, 69.50] | 0.9768 |  |
| C2-s3 | 61.82 [53.64, 69.52] | 0.9764 |  |
| C2-real | 61.74 [53.52, 69.44] | 0.9771 |  |
| C2-syn | 61.14 [52.84, 68.99] | 0.9726 |  |
| M-C | 62.41 [54.27, 70.14] | 0.9779 | = M-C-base0.75 |
| lumi-emb | 62.41 [54.27, 70.14] | 0.9779 | = M-C (M-C-base0.75) |

#### Grade do merge M-C (dev)

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

#### Variação entre sementes (C2)

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

#### Ablações (MTEB-BR)

| ID | Pergunta | Comparação | Δ Mean22 [IC 95%] | p | Tarefas |
|---|---|---|---|---|---|
| S2 | O S2 (negativos difíceis) melhora o C1? | C2 − C1 | +0.19 [-0.24, +0.64] | 0.394 | 22 |
| A2 | Só dados reais no S2 | C2 − C2-real | +0.23 [-0.27, +0.68] | 0.349 | 22 |
| A2 | Só dados sintéticos no S2 | C2 − C2-syn | +0.83 [+0.38, +1.36] | < 0.001 | 22 |
| A2 | Reais contra sintéticos | C2-real − C2-syn | +0.60 [-0.17, +1.49] | 0.138 | 22 |
| A7 | Echo embeddings sem treino | R0e − R0 | -2.30 [-4.54, -0.32] | 0.021 | 22 |
| A9 | Merge causal vs C2 | M-C − C2 | +0.44 [-0.05, +0.90] | 0.074 | 22 |

No A2, os conjuntos do S2 não têm o mesmo tamanho: S2-real e S2-syn ficaram menores que o S2 completo e diferentes entre si (tamanhos em `data/embed/sets/manifest.json` e no DATA_CARD). A versão na mesma escala, prevista no plano, fica para a fase de ablações.

#### Hipóteses pré-registradas

| ID | Hipótese | Teste | Δ [IC 95%] | p | Situação |
|---|---|---|---|---|---|
| H1 | Paridade prática com Transformers de porte similar | C2 − Qwen3-Embedding-4B | -3.56 [-6.60, -0.76] | 0.012 | Refutada |
| H2 | O S1 melhora o resultado final | C2 − C2-direct | — | — | — |
| H3 | Reais ajudam mais nas tarefas nativas | C2-real − C2-syn | +0.60 [-0.17, +1.49] | 0.138 | Não refutada |
| H4 | Negativos positive-aware superam o top-k ingênuo | C2 − C2-naivehn | — | — | — |
| H5 | O bidirecional melhora a média | B2a − C2; B2f − C2 | —; — | — | — |
| H6 | O merge supera o melhor checkpoint individual (dev) | M-C (M-C-base0.75) − C2-real, nDCG@10 no dev | +0.08 | — | Não refutada |
| H7 | Vantagem relativa em entradas longas | lumi-emb − Qwen3-Embedding-4B, LongEmbed | -32.62 | — | Refutada |

"Refutada" / "Não refutada" aplicam o critério de falseamento do plano; "—" = ainda não executado.

<!-- RESULTS:END -->

### Critérios de falseamento das hipóteses

A situação de cada hipótese está na tabela "Hipóteses pré-registradas", acima.

| ID | Hipótese | Como falsear |
|---|---|---|
| H1 | O híbrido com EOS atinge paridade prática com Transformers de porte similar no MTEB-BR | C2 abaixo do Qwen3-Embedding-4B por mais que o IC |
| H2 | O S1 melhora o resultado final | C2-direct ≥ C2 |
| H3 | Dados reais ajudam mais dentro do domínio; sintéticos, na cobertura | C2-syn ≥ C2-real nas tarefas nativas |
| H4 | Negativos positive-aware superam o top-k ingênuo | C2-naivehn ≥ C2 |
| H5 | O bidirecional melhora a média, mas o Bf pode não compensar sobre o Ba | B2f ≤ B2a, ou ambos ≤ C2 |
| H6 | O merge supera o melhor checkpoint individual | M-C ≤ melhor C2 no dev |
| H7 | Vantagem relativa em entradas longas | C3-long ou C2 abaixo de Transformers de porte similar no LongEmbed |

## Limitações

- **Clusterização e STS:** é onde está a maior distância para o Qwen3-Embedding-4B; só a clusterização responde por 2,7 dos 3,1 pontos da diferença média. A mistura de treino tem quase só pares de busca e de frases semelhantes, sem dados que ensinem "mesmo tema", e o treino quase não moveu a clusterização.
- **Documentos longos:** o treino usou até 512 tokens, e o modelo não aprendeu a representar textos longos. No MLDR-pt, passar de 512 para 8k tokens derruba o lumi-emb de 61,99 para 47,88, enquanto o Qwen3-Embedding-4B sobe de 66,92 para 80,23. No LongEmbed, ele fica em 47,50 contra 80,12. Só o passkey, um fato isolado, sobrevive: 100 % de 1k a 8k tokens (a janela do pré-treino contínuo do Lumi), 66 % em 16k e 48 % em 32k; o needle cai a partir de 4k. Na Etapa 1, o backbone colapsou na geração a partir de 64k.
- **Arquitetura:** o embedder híbrido Mamba-Transformer não tem validação prévia na literatura, e o trilho bidirecional é experimental.
- **Inglês:** 63,30 no MTEB (eng, v2), com 41 tarefas. Os baselines ainda não foram medidos em inglês com o nosso protocolo.
- **Dependência do teacher:** os negativos e o filtro dos sintéticos vêm do Qwen3-Embedding-4B.
- **Escala reduzida:** centenas de milhares de pares, contra dezenas de milhões na literatura.
- **Dev saturado:** o dev (1.500 consultas, com 30 mil distratores fixos) fica perto de 0,977 nos candidatos ao merge. As diferenças entre eles são de milésimos, perto da variação entre sementes (0,0004 de amplitude). Ele serve para descartar candidatos ruins, mas separa pouco os bons.
- **Contaminação residual:** ver a auditoria posterior, na seção Dados.
- **Ablação A2 em escalas diferentes:** S2-real (91 mil exemplos, dos quais só 22,5 mil em PT nativo; o resto é mMARCO traduzido e bitexto) e S2-syn (74 mil) não têm o tamanho do S2 (120 mil).
- **Uso não comercial:** entre os dados de treino está o mMARCO/MS MARCO, de uso não comercial.

## Reprodução

Todo o pipeline, com dados, treino, merge, avaliação e publicação, está em [github.com/henriquegalva0/lumi-to-emb](https://github.com/henriquegalva0/lumi-to-emb). O roteiro passo a passo é o [PIPELINE.md](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/PIPELINE.md), e as escolhas de implementação estão no [DECISIONS.md](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DECISIONS.md). Cada checkpoint traz em `meta.json` a semente, o commit do código e a configuração completa.

## Licença

**Uso não comercial.** O modelo deriva do `lumi-4b` e do Nemotron 3 Nano 4B (NVIDIA Nemotron Open Model License). Foi treinado com dados de licenças diversas, inclusive o mMARCO/MS MARCO, que é de uso não comercial. Detalhes na [ficha dos dados](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DATA_CARD.md#licenças-o-que-isto-implica-para-o-modelo-decisão-d2).

## Citação

```bibtex
@misc{lumiembed2026,
  title        = {Lumi-Embed: um modelo de embeddings para português brasileiro a partir de um LLM híbrido Mamba-2},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/henriquegalvao/lumi-4b-embedding}}
}
@misc{ceiaufg2026lumi4b,
  title        = {Lumi 4B},
  author       = {{NLP Team, CEIA/UFG}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/ceia-research/lumi-4b}}
}
```

Referências centrais da receita:

- NVIDIA, *Nemotron-H* ([arXiv:2504.03624](https://arxiv.org/abs/2504.03624));
- Zhang et al., *Mamba Retriever* ([arXiv:2408.08066](https://arxiv.org/abs/2408.08066));
- *Qwen3 Embedding* ([arXiv:2506.05176](https://arxiv.org/abs/2506.05176));
- *Llama-Embed-Nemotron-8B* ([arXiv:2511.07025](https://arxiv.org/abs/2511.07025));
- *NV-Retriever* ([arXiv:2407.15831](https://arxiv.org/abs/2407.15831));
- *LLM2Vec* ([arXiv:2404.05961](https://arxiv.org/abs/2404.05961));
- *BidirLM* ([arXiv:2604.02045](https://arxiv.org/abs/2604.02045));
- *MTEB-BR* ([arXiv:2607.04581](https://arxiv.org/abs/2607.04581)).

A lista completa está no apêndice B do plano.
