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

**Um modelo de embeddings para português brasileiro, construído sobre o [Lumi-4B](https://huggingface.co/ceia-research/lumi-4b), um LLM híbrido Mamba-2 + atenção.** O modelo publicado (branch `main`) é o **lumi-emb-long**. Ele lê documentos inteiros de até 8k tokens, foi treinado com dados em PT-BR nativo e foi avaliado com o mesmo protocolo dos modelos comparados. Todas as tabelas e figuras deste card são geradas direto dos arquivos de avaliação.

- **Documentos longos:** no MLDR-pt, ler o documento inteiro (até 8k tokens) dá 73,94 de nDCG@10, contra 47,88 do modelo antes do estágio longo.
- **MTEB-BR (22 tarefas nativas):** 62,48. É o dobro do Lumi-4B sem treino (30,65) e empata estatisticamente com o multilingual-e5-large-instruct e o Qwen3-Embedding-0.6B.
- **Onde ele se destaca:** maior média em classificação entre os modelos comparados e retrieval acima do Qwen3-Embedding-4B, de mesmo porte. Ele supera esse modelo em 10 das 22 tarefas do MTEB-BR.
- **Versão curta:** o **lumi-emb**, treinado só com até 512 tokens, continua disponível em `revision="lumi-emb"`.

**English summary.** Lumi-Embed turns [Lumi-4B](https://huggingface.co/ceia-research/lumi-4b), a 4B hybrid Mamba-2 + attention LLM for Brazilian Portuguese, into a text-embedding model for search and RAG. The released model, **lumi-emb-long**, reads whole documents of up to 8k tokens. It was built with:

- last-token (EOS) pooling;
- staged contrastive LoRA training: a weakly supervised stage, a supervised stage with positive-aware hard negatives, and a 2k → 4k → 8k long-context stage with short-text replay;
- native PT-BR synthetic data from three open generators, plus natural pairs;
- checkpoint merges selected on a held-out dev set, never on test data.

**Results:**
- MLDR-pt with whole documents up to 8k tokens: 73.94, up from 47.88 before the long stage.
- MTEB-BR, 22 native PT-BR tasks: 62.48. That is twice the untrained backbone (30.65), and statistically tied with multilingual-e5-large-instruct and Qwen3-Embedding-0.6B.
- The highest classification average among the compared models, and retrieval above Qwen3-Embedding-4B.
- Ablations: the two-stage recipe and positive-aware hard-negative filtering each add more than one point; making the hybrid bidirectional does not help.
- Still behind Qwen3-Embedding-4B on average (clustering, STS) and beyond 8k tokens.

Evaluation follows a clean zero-shot contamination policy. A post-hoc audit found overlaps in at most 0.1 % of the training examples, and the conclusions hold without the affected tasks.

As far as we know, it is one of the first hybrid Mamba-Transformer text embedders evaluated in depth in Portuguese. Code, configs and data recipes: [github.com/henriquegalva0/lumi-to-emb](https://github.com/henriquegalva0/lumi-to-emb).

## O que é um modelo de embeddings, em poucas palavras

Um modelo de *embeddings* transforma um texto numa lista de números, um vetor, que representa o seu significado. Textos com sentidos parecidos viram vetores próximos, mesmo quando usam palavras diferentes. Por exemplo, "como declarar gastos com médico?" fica perto de "despesas médicas podem ser deduzidas do imposto de renda". Com isso, dá para buscar por significado, e não só por palavras iguais.

No **RAG** (geração aumentada por recuperação), esse é o primeiro passo:
1. Os documentos de uma base (manuais, leis, contratos, perguntas frequentes) são convertidos em vetores uma única vez.
2. A cada pergunta, o modelo de embeddings encontra os trechos mais próximos dela.
3. Só esses trechos vão para o LLM que escreve a resposta.

Se a busca erra, o LLM responde com base no texto errado, e por isso a qualidade do modelo de embeddings pesa tanto. Além disso, quanto mais texto ele lê de uma vez, menos é preciso picotar os documentos. O lumi-emb-long lê até 8k tokens de uma vez, cerca de uma dezena de páginas.

## Destaques

Os termos técnicos estão explicados no [glossário](#glossário).

<!-- HIGHLIGHTS:START -->
|  | lumi-emb-long | Comparação |
|---|---|---|
| Documentos longos em PT (MLDR-pt, até 8k tokens) | **73.94** | 47.88 sem o estágio longo (+26.06); 63.99 lendo só os primeiros 512 tokens |
| MTEB-BR (22 tarefas nativas em PT-BR) | **62.48** | 30.65 do Lumi-4B sem treino |
| Classificação no MTEB-BR (4 tarefas) | **67.73** | maior média entre os 5 modelos comparados (depois: Qwen3-Embedding-8B, 67.13) |
| Retrieval no MTEB-BR (6 tarefas) | **62.56** | acima do Qwen3-Embedding-4B, de mesmo porte (62.29) |
| Retrieval no MTEB-PT | **77.87** | acima do Qwen3-Embedding-4B, de mesmo porte (77.14) |
| Reranking no MTEB-PT | **84.91** | acima do Qwen3-Embedding-4B, de mesmo porte (84.56) |

![MLDR-pt: nDCG@10 em 512 tokens e com o documento inteiro (até 8k)](docs/figures/mldr-pt-light.png)

- **Empate estatístico** na média do MTEB-BR com Qwen3-Embedding-0.6B (Δ +0.81, p = 0.518) e multilingual-e5-large-instruct (Δ -1.17, p = 0.350).
- **O contexto longo não custou nada nos textos curtos:** lumi-emb-long − lumi-emb = +0.07 no MTEB-BR (p = 0.671).

### Onde ele se destaca

![MTEB-BR por categoria: lumi-emb-long, Qwen3-Embedding-4B e Qwen3-Embedding-8B](docs/figures/mteb-br-categories-light.png)

Nas tarefas abaixo (10 das 22 do MTEB-BR), o lumi-emb-long supera o Qwen3-Embedding-4B, um Transformer de mesmo porte, e as marcadas com ✓ superam também o Qwen3-Embedding-8B, com o dobro de parâmetros.

| Tarefa | Categoria | lumi-emb-long | Qwen3-Embedding-4B | Δ | > 8B |
|---|---|---|---|---|---|
| PortuLexRRIP | Classificação | **40.27** | 33.09 | +7.18 | ✓ |
| ToxSynPT | Classificação | **90.36** | 85.18 | +5.18 | ✓ |
| FaqBacenRetrieval | Retrieval | **68.54** | 64.42 | +4.12 | ✓ |
| JurisTCUClusteringP2P | Clusterização | **31.48** | 28.27 | +3.21 |  |
| HateBR | Classificação | **85.62** | 82.81 | +2.81 |  |
| MedPTRetrieval | Retrieval | **80.06** | 77.62 | +2.44 |  |
| FaQuADIR | Retrieval | **82.42** | 80.70 | +1.72 |  |
| Quati | Retrieval | **61.81** | 60.76 | +1.05 |  |
| FactckBrClassification | Classificação | **54.67** | 53.75 | +0.92 |  |
| QuatiReranking | Reranking | **60.65** | 60.57 | +0.08 |  |

![MTEB-BR: média das 22 tarefas por modelo](docs/figures/mteb-br-ranking-light.png)

### Contexto longo

O estágio longo treinou com documentos de até 8k tokens. Nessa faixa, o modelo encontra a informação escondida bem melhor do que antes dele.

![LongEmbed Passkey e Needle por comprimento](docs/figures/longembed-length-light.png)

<!-- HIGHLIGHTS:END -->

## Visão geral

| | |
|---|---|
| Backbone | [`ceia-research/lumi-4b`](https://huggingface.co/ceia-research/lumi-4b) (revisão `27b30f4`): 42 camadas, das quais 21 Mamba-2, 17 MLP e 4 de atenção; sem RoPE; derivado do Nemotron 3 Nano 4B |
| Embedding | estado oculto do último token (`</s>`), depois da normalização final; L2; similaridade de cosseno; dimensão 3136 |
| Formato | query: `Instruct: {instrução}\nQuery: {query}</s>`; documento: `{texto}</s>`, sem instrução |
| Modelo publicado | `lumi-emb-long` = M-L-slerp-C3-long-lumi-emb-t0.25: SLERP entre o C3-long (estágio longo) e o lumi-emb, a 25 % do caminho, ou seja, cerca de 75 % C3-long |
| Versão curta | `lumi-emb` (`revision="lumi-emb"`) = M-C-base0.75: 0,75 × média de cinco checkpoints do S2 (C2, C2-real, C2-syn, C2-s2, C2-s3) + 0,25 × Lumi-4B sem treino (R0) |
| Comprimento | treinado com até 512 tokens (S1/S2) e depois com 2k → 4k → 8k (S-long); **para RAG, use trechos de até 8k tokens** |
| Idiomas | português brasileiro (principal) e inglês |
| Carregamento | transformers ≥ 5.23 nativo (`NemotronHModel`), **sem** `trust_remote_code` |

## Uso

O tokenizer salvo já anexa o `</s>` e faz padding à direita. Use sempre o mesmo formato de instrução nas queries e deixe os documentos sem instrução. Em RAG, o modelo aproveita trechos grandes, de até 8k tokens, o comprimento máximo do treino. Documentos maiores devem ser divididos.

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
    batch = tok(texts, padding=True, truncation=True, max_length=8192, return_tensors="pt").to("cuda")
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

A versão curta e os checkpoints intermediários ficam em branches com o nome de cada um. Exemplos: `revision="lumi-emb"`, `revision="C2"`.

Os trechos desta seção são executados contra este repositório pelo passo `hubcheck` do código: os embeddings do Sentence-Transformers, do Transformers e dos kernels batem com os do checkpoint local (cosseno ≥ 0,9999), e cada branch tem só os próprios pesos.

## Contexto: a janela efetiva do Lumi-4B (Etapa 1)

Antes do treino, medimos até quantos tokens o `lumi-4b` recupera informação. O teste foi *needle-in-a-haystack* em PT-BR, de 2k a 255k tokens, com 2.610 prompts por modelo e o modelo-base como controle. [Relatório completo](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/stage1/README.md).

| Modelo | L\* (1 agulha) | L\* (multi-chave) | L\* (multi-valor) |
|---|---|---|---|
| Lumi-4B | 32k | **16k** | **16k** |
| Nemotron-3-Nano-4B (base) | 255k | 255k | 128k |

O Lumi colapsa a partir de 64k: as respostas entram em laço e saem sem número. Já em 32k, ele confunde chaves e valores. A perda vem do pré-treino contínuo em janelas de 8k, porque a base mantém 255k. Por isso o embedder treina primeiro com 512 tokens e depois estende o contexto até 8k num estágio separado (C3-long), bem dentro da faixa em que o backbone funciona.

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
| C3-long | lumi-emb | C / EOS | S-long (2k→4k→8k) | documentos longos + 30 % de replay do S2 | 4, PercPos | Estágio longo do modelo publicado |
| B0a / B0f | lumi-4b | Ba / Bf | MNTP | texto cru | — | A adaptação bidirecional funciona num híbrido? |
| B1a/f → B2a/f | B0 | B / média | S1 → S2 | como C1/C2 | como C1/C2 | Bidirecional (A5) |
| M-C / M-B | ver abaixo | C / B | merge | — | — | O merge supera o melhor individual? (A9) |
| M-L | C3-long + lumi-emb | C / EOS | merge | — | — | Um modelo só, bom nos textos curtos e longos |

Ba deixa só as 4 camadas de atenção bidirecionais, com as camadas Mamba causais. Bf acrescenta uma varredura reversa nos mixers Mamba-2, com os mesmos pesos, invertendo só os tokens válidos. O trilho B é **experimental**: não há validação prévia dessa adaptação em um LLM Mamba ou híbrido pré-treinado.

**Merge.** Só se mesclam checkpoints do mesmo trilho, com o LoRA já fundido. Para o M-C, os candidatos são C2, C2-real, C2-syn, C2-s2 e C2-s3. A grade inclui:

- média linear;
- SLERP entre C2-real e C2-syn, com t ∈ {0,25; 0,5; 0,75};
- interpolação da média com o R0, com λ ∈ {0,25; 0,5; 0,75}.

A escolha é feita no dev, nunca nos testes. O melhor no dev vira o lumi-emb. Ele fica no branch `lumi-emb` e no branch `M-C`, com a receita no `meta.json`; as notas de todos os candidatos no dev estão na tabela "Grade do merge M-C", abaixo, e no [RESULTS.md](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/RESULTS.md). O escolhido foi o **M-C-base0.75**, com nDCG@10 de 0,9779 no dev, contra 0,9771 do melhor checkpoint individual (C2-real). Nos benchmarks, ele ficou 0,44 ponto acima do C2 no MTEB-BR (p = 0,074 no bootstrap por tarefa) e 0,58 acima da média das três sementes do C2, cujo desvio-padrão é 0,13.

**Estágio longo e merge M-L.** O C3-long parte do lumi-emb e treina com documentos de 2k, depois 4k e depois 8k tokens, com 30 % de exemplos curtos do S2 repetidos. O dev soma uma tarefa longa (300 consultas e 1.000 documentos longos como distratores) às tarefas curtas. O M-L faz a grade entre o C3-long e o lumi-emb (média e SLERP com t ∈ {0,25; 0,5; 0,75}), escolhida pela mesma regra no dev curto + longo. O escolhido, **M-L-slerp-C3-long-lumi-emb-t0.25**, empatou com o C3-long no dev (0,9756). Pela regra, o empate favorece o merge. Ele é o lumi-emb-long, no `main`.

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
- mesmo comprimento máximo: 512 tokens no MTEB-BR e no MTEB-PT, 32k no LongEmbed e 8k no MLDR-pt longo.

O benchmark principal é o **MTEB-BR**, 22 tarefas nativas ([arXiv:2607.04581](https://arxiv.org/abs/2607.04581)). Completam o quadro o **MTEB-PT** ([arXiv:2607.04071](https://arxiv.org/abs/2607.04071)), o LongEmbed e o MTEB (eng, v2); o MMTEB é opcional. Os ICs de 95 % e os p-valores usam bootstrap com a tarefa como unidade, como no MTEB-BR. Uma diferença só é declarada com p < 0,05.

<!-- RESULTS:START -->
### Resultados

Notas × 100, geradas pelos arquivos de avaliação ([tabelas completas](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/RESULTS.md)). Mesmo protocolo para todos os modelos: mesmas instruções em português, só na query, e até 512 tokens. IC 95 % e p-valores por bootstrap sobre as tarefas, como no MTEB-BR.

#### MTEB-BR (22 tarefas nativas em PT-BR)

| Modelo | Mean22 [IC 95%] | Classificação (4) | Multilabel (1) | Pares (2) | STS (2) | Clust. (5) | Retrieval (6) | Reranking (2) |
|---|---|---|---|---|---|---|---|---|
| **lumi-emb-long** (este trabalho) | 62.48 [54.40, 70.10] | 67.7 | 29.9 | 84.2 | 73.5 | 53.3 | 62.6 | 58.2 |
| **lumi-emb** (este trabalho) | 62.41 [54.27, 70.14] | 67.3 | 29.6 | 85.0 | 74.8 | 52.9 | 62.2 | 58.4 |
| Qwen3-Embedding-8B | 67.35 [59.41, 74.86] | 67.1 | 39.0 | 88.2 | 81.2 | 65.3 | 64.8 | 60.1 |
| Qwen3-Embedding-4B | 65.53 [57.33, 73.25] | 63.7 | 36.5 | 87.2 | 80.6 | 64.9 | 62.3 | 58.2 |
| multilingual-e5-large-instruct | 63.65 [55.38, 71.51] | 63.6 | 36.5 | 84.2 | 80.6 | 60.5 | 60.2 | 58.0 |
| Qwen3-Embedding-0.6B | 61.67 [53.93, 69.10] | 62.7 | 32.6 | 81.4 | 76.2 | 59.6 | 58.2 | 55.4 |
| Lumi-4B sem treino (R0) | 30.65 [20.79, 40.75] | 59.8 | 21.4 | 46.1 | 44.5 | 37.4 | 1.4 | 18.4 |

#### Comparação pareada no MTEB-BR

| Comparação | Δ Mean22 [IC 95%] | p | Vitórias (tarefas) |
|---|---|---|---|
| lumi-emb-long − Qwen3-Embedding-4B | -3.05 [-6.04, -0.25] | 0.033 | 10/22 |
| lumi-emb-long − Qwen3-Embedding-8B | -4.87 [-7.33, -2.50] | < 0.001 | 3/22 |
| lumi-emb-long − Qwen3-Embedding-0.6B | +0.81 [-1.74, +3.21] | 0.518 | 13/22 |
| lumi-emb-long − multilingual-e5-large-instruct | -1.17 [-3.68, +1.20] | 0.350 | 10/22 |
| lumi-emb-long − lumi-emb | +0.07 [-0.28, +0.41] | 0.671 | 12/22 |

#### MTEB-PT (por família)

| Modelo | PT STS | PT Cls | PT Ret | PT Rer |
|---|---|---|---|---|
| **lumi-emb-long** (este trabalho) | 78.8 | 56.2 | 77.9 | 84.9 |
| **lumi-emb** (este trabalho) | 79.5 | 55.4 | 77.8 | 85.4 |
| Qwen3-Embedding-8B | 84.7 | 62.6 | 78.9 | 86.0 |
| Qwen3-Embedding-4B | 83.9 | 58.2 | 77.1 | 84.6 |
| multilingual-e5-large-instruct | 81.8 | 56.4 | 76.6 | 86.0 |
| Qwen3-Embedding-0.6B | 79.7 | 52.4 | 74.5 | 83.8 |
| Lumi-4B sem treino (R0) | 42.7 | 43.6 | 0.2 | 52.0 |

#### Outras suítes

| Modelo | MMTEB | MTEB eng v2 | LongEmbed |
|---|---|---|---|
| **lumi-emb-long** (este trabalho) | — | — | 56.06 |
| **lumi-emb** (este trabalho) | — | 63.30 | 47.50 |
| Qwen3-Embedding-4B | — | — | 80.12 |

#### Sensibilidade à contaminação

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

#### MLDR-pt com documentos de até 8k tokens

| Modelo | MLDR-pt, 512 tokens | MLDR-pt, 8k tokens |
|---|---|---|
| lumi-emb | 61.99 | 47.88 |
| C3-long | — | 75.68 |
| lumi-emb-long | 63.99 | 73.94 |
| Qwen3-Embedding-4B | 66.92 | 80.23 |

MultiLongDocRetrieval em português (documentos longos da Wikipédia e do mC4). No MTEB-PT ele é avaliado com 512 tokens, como todo o protocolo; aqui, também com os documentos em até 8k tokens.

#### LongEmbed por comprimento (tokens)

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
| C2-nohn | 62.51 [54.27, 70.37] | 0.9768 |  |
| C2-naivehn | 60.69 [52.54, 68.40] | 0.9580 |  |
| C2-ib | 61.88 [53.69, 69.61] | 0.9778 |  |
| C2-direct | 60.73 [52.50, 68.57] | 0.9707 |  |
| C3-long | — | 0.9756 |  |
| B2a | 61.74 [53.51, 69.50] | 0.9760 |  |
| B2f | 60.89 [52.71, 68.68] | 0.9746 |  |
| M-C | 62.41 [54.27, 70.14] | 0.9779 | = M-C-base0.75 |
| M-L | 62.48 [54.40, 70.10] | 0.9756 | = M-L-slerp-C3-long-lumi-emb-t0.25 |
| lumi-emb | 62.41 [54.27, 70.14] | 0.9779 | = M-C (M-C-base0.75) |
| lumi-emb-long | 62.48 [54.40, 70.10] | 0.9756 | = M-L (M-L-slerp-C3-long-lumi-emb-t0.25) |

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

#### Grade do merge M-L (dev)

| Candidato | dev nDCG@10 | Borda |  |
|---|---|---|---|
| C3-long | 0.9756 | 8.5 | melhor individual |
| M-L-slerp-C3-long-lumi-emb-t0.25 | 0.9756 | 7.5 | escolhido |
| M-L-slerp-C3-long-lumi-emb-t0.5 | 0.9753 | 12.0 |  |
| M-L-avg | 0.9748 | 13.0 |  |
| M-L-slerp-C3-long-lumi-emb-t0.75 | 0.9689 | 7.5 |  |
| lumi-emb | 0.9572 | 11.5 |  |

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
| A1 | O estágio S1 é necessário? | C2 − C2-direct | +1.24 [+0.58, +1.98] | < 0.001 | 22 |
| S2 | O S2 (negativos difíceis) melhora o C1? | C2 − C1 | +0.19 [-0.24, +0.64] | 0.394 | 22 |
| A2 | Só dados reais no S2 | C2 − C2-real | +0.23 [-0.27, +0.68] | 0.349 | 22 |
| A2 | Só dados sintéticos no S2 | C2 − C2-syn | +0.83 [+0.38, +1.36] | < 0.001 | 22 |
| A2 | Reais contra sintéticos | C2-real − C2-syn | +0.60 [-0.17, +1.49] | 0.138 | 22 |
| A3 | Sem negativos difíceis | C2 − C2-nohn | -0.55 [-1.12, +0.02] | 0.058 | 22 |
| A3 | Top-k ingênuo | C2 − C2-naivehn | +1.28 [+0.61, +1.97] | < 0.001 | 22 |
| A4 | In-batch + máscara de falso negativo | C2 − C2-ib | +0.09 [-0.13, +0.35] | 0.459 | 22 |
| A5 | Bidirecional (atenção) | B2a − C2 | -0.23 [-0.79, +0.28] | 0.392 | 22 |
| A5 | Bidirecional (atenção + BiMamba) | B2f − C2 | -1.08 [-1.82, -0.34] | 0.005 | 22 |
| A7 | Echo embeddings sem treino | R0e − R0 | -2.30 [-4.54, -0.32] | 0.021 | 22 |
| A9 | Merge causal vs C2 | M-C − C2 | +0.44 [-0.05, +0.90] | 0.074 | 22 |

No A2, os conjuntos do S2 não têm o mesmo tamanho: S2-real e S2-syn ficaram menores que o S2 completo e diferentes entre si (tamanhos em `data/embed/sets/manifest.json` e no DATA_CARD). A versão na mesma escala, prevista no plano, ainda não rodou.

#### Hipóteses pré-registradas

| ID | Hipótese | Teste | Δ [IC 95%] | p | Situação |
|---|---|---|---|---|---|
| H1 | Paridade prática com Transformers de porte similar | C2 − Qwen3-Embedding-4B | -3.56 [-6.60, -0.76] | 0.012 | Refutada |
| H2 | O S1 melhora o resultado final | C2 − C2-direct | +1.24 [+0.58, +1.98] | < 0.001 | Não refutada |
| H3 | Reais ajudam mais nas tarefas nativas | C2-real − C2-syn | +0.60 [-0.17, +1.49] | 0.138 | Não refutada |
| H4 | Negativos positive-aware superam o top-k ingênuo | C2 − C2-naivehn | +1.28 [+0.61, +1.97] | < 0.001 | Não refutada |
| H5 | O bidirecional melhora a média | B2a − C2; B2f − C2 | -0.23 [-0.79, +0.28]; -1.08 [-1.82, -0.34] | 0.392; 0.005 | Refutada |
| H6 | O merge supera o melhor checkpoint individual (dev) | M-C (M-C-base0.75) − C2-real, nDCG@10 no dev | +0.08 | — | Não refutada |
| H7 | Vantagem relativa em entradas longas | C3-long − Qwen3-Embedding-4B, LongEmbed | -22.58 | — | Refutada |

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

## O que as ablações mostraram

Cada ablação treina o modelo de novo com uma única peça trocada e compara as notas no MTEB-BR. Três lições:

- **A receita em dois estágios se paga.** Sem o S1, ou com os negativos difíceis escolhidos pelo top-k ingênuo (que tende a trazer falsos negativos para o treino), a média cai mais de um ponto.
- **O Lumi causal já basta.** Tornar o modelo bidirecional, como no LLM2Vec, exige um estágio a mais (MNTP) e não melhorou a média. Com o BiMamba, que roda o Mamba nos dois sentidos, a média piorou. Para este híbrido, a receita causal, com pooling pelo último token, é suficiente.
- **Os negativos difíceis não mudaram a média.** O C2-nohn, treinado só com os negativos do próprio lote, empatou com o C2 e foi melhor em STS. Ele teve a maior média do MTEB-BR entre os checkpoints treinados até aqui, no nível do modelo publicado, e cada passo é mais barato (2 textos por exemplo em vez de 6). Por isso, é o candidato para o próximo merge.

<!-- ABLATIONS:START -->
| O que foi testado | Comparação | Δ MTEB-BR [IC 95%] | p | Efeito |
|---|---|---|---|---|
| Estágio S1 (pares fracos em grande escala) antes do S2 | C2 − C2-direct | +1.24 [+0.58, +1.98] | < 0.001 | **ajuda** |
| Negativos difíceis filtrados (PercPos) em vez do top-k ingênuo | C2 − C2-naivehn | +1.28 [+0.61, +1.97] | < 0.001 | **ajuda** |
| Dados reais e sintéticos juntos no S2, em vez de só sintéticos¹ | C2 − C2-syn | +0.83 [+0.38, +1.36] | < 0.001 | **ajuda** |
| Dados reais e sintéticos juntos no S2, em vez de só reais¹ | C2 − C2-real | +0.23 [-0.27, +0.68] | 0.349 | empate |
| Merge de checkpoints (M-C) em vez do C2 sozinho | M-C − C2 | +0.44 [-0.05, +0.90] | 0.074 | empate |
| 4 negativos difíceis por exemplo no S2, em vez de só os do lote | C2 − C2-nohn | -0.55 [-1.12, +0.02] | 0.058 | empate |
| Negativos do lote somados aos difíceis, com máscara de falso negativo | C2-ib − C2 | -0.09 [-0.35, +0.13] | 0.459 | empate |
| Atenção bidirecional (Ba, com MNTP antes) | B2a − C2 | -0.23 [-0.79, +0.28] | 0.392 | empate |
| Atenção bidirecional + BiMamba (Bf, com MNTP antes) | B2f − C2 | -1.08 [-1.82, -0.34] | 0.005 | **piora** |
| Echo embeddings (texto repetido), sem treino | R0e − R0 | -2.30 [-4.54, -0.32] | 0.021 | **piora** |

Δ = média das 22 tarefas do MTEB-BR com o que foi testado menos a média sem ele (× 100), com IC de 95 % e p por bootstrap pareado nas tarefas. "Ajuda" ou "piora" quando o IC não cruza o zero. Para comparar, o C2 treinado com 3 sementes varia 0.13 (desvio-padrão) na média.

¹ Conjuntos de tamanhos diferentes: o S2-real e o S2-syn são menores que o S2 completo.

<!-- ABLATIONS:END -->

## Limitações

- **Clusterização e STS:** é onde está a maior distância para o Qwen3-Embedding-4B, que fica cerca de 3 pontos acima na média do MTEB-BR; a clusterização responde pela maior parte dessa diferença. A mistura de treino tem quase só pares de busca e de frases semelhantes, sem dados que ensinem "mesmo tema", e o treino quase não moveu a clusterização.
- **Acima de 8k tokens:** o estágio longo treinou até 8k, e o ganho não se estende além disso. No Needle do LongEmbed, o acerto cai para perto de zero em 16k. Documentos maiores devem ser divididos em trechos de até 8k. No LongEmbed, que é em inglês e vai até 32k, o lumi-emb-long fica em 56,06, contra 80,12 do Qwen3-Embedding-4B. No MLDR-pt em 8k, fica em 73,94, contra 80,23.
- **Arquitetura:** o embedder híbrido Mamba-Transformer não tem validação prévia na literatura. O trilho bidirecional (Ba e Bf) foi testado e não superou o causal.
- **Inglês:** o lumi-emb fica em 63,30 no MTEB (eng, v2), com 41 tarefas; o lumi-emb-long e os baselines ainda não foram medidos nele com o nosso protocolo.
- **Dependência do teacher:** os negativos e o filtro dos sintéticos vêm do Qwen3-Embedding-4B.
- **Escala reduzida:** centenas de milhares de pares, contra dezenas de milhões na literatura.
- **Dev saturado:** o dev (1.500 consultas, com 30 mil distratores fixos) fica perto de 0,977 nos candidatos ao merge. As diferenças entre eles são de milésimos, perto da variação entre sementes (0,0004 de amplitude). Ele serve para descartar candidatos ruins, mas separa pouco os bons.
- **Contaminação residual:** ver a auditoria posterior, na seção Dados.
- **Ablação A2 em escalas diferentes:** S2-real (91 mil exemplos, dos quais só 22,5 mil em PT nativo; o resto é mMARCO traduzido e bitexto) e S2-syn (74 mil) não têm o tamanho do S2 (120 mil).
- **Uso não comercial:** entre os dados de treino está o mMARCO/MS MARCO, de uso não comercial.

## Glossário

| Termo | O que significa |
|---|---|
| **Embedding** | Vetor de números que representa o significado de um texto. Textos parecidos viram vetores próximos. |
| **RAG** | Geração aumentada por recuperação: busca trechos relevantes numa base de documentos e os entrega a um LLM junto com a pergunta. |
| **Query e documento** | A pergunta de busca e o texto que pode respondê-la. Aqui, a query leva uma instrução curta, e o documento não. |
| **Similaridade de cosseno** | Medida de proximidade entre dois embeddings: quanto maior, mais parecidos os significados. |
| **Token** | Pedaço de palavra, a unidade que o modelo lê. 8k tokens equivalem a cerca de uma dezena de páginas. |
| **Janela de contexto e L\*** | Quanto texto o modelo consegue ler de uma vez. O L\* é o maior comprimento em que o Lumi-4B ainda acha uma informação escondida com pelo menos 90 % de acerto (Etapa 1). |
| **Transformer** | A arquitetura da maioria dos LLMs e modelos de embeddings, como o Qwen3-Embedding, baseada em camadas de atenção. |
| **Mamba-2 e modelo híbrido** | O Mamba-2 é um tipo de camada que lê o texto em sequência, guardando uma memória de tamanho fixo. O Lumi-4B é híbrido: mistura camadas Mamba-2 com algumas camadas de atenção. |
| **Pooling pelo último token (EOS)** | O embedding é o estado interno do modelo no token final (`</s>`), que resume o texto lido até ali. |
| **Treino contrastivo (InfoNCE)** | Ensina o modelo a aproximar a query do documento certo e a afastá-la dos errados. |
| **Negativos difíceis (TopK-PercPos)** | Documentos parecidos com o certo, mas que não respondem à pergunta; tornam o treino mais exigente. O critério TopK-PercPos descarta os candidatos parecidos demais com o certo, que provavelmente também respondem. |
| **LoRA** | Forma econômica de ajustar um modelo grande: em vez de mudar todos os pesos, treina pequenas matrizes extras. |
| **Checkpoint** | Uma versão salva do modelo durante o projeto (ex.: C2, C3-long). |
| **Merge e SLERP** | Combinar os pesos de modelos já treinados num só. O SLERP é uma interpolação entre dois modelos que segue a curvatura dos vetores de pesos. |
| **Dev** | Conjunto de exemplos separado, usado só para escolher entre checkpoints. Os benchmarks nunca entram nessa escolha. |
| **MTEB-BR, MTEB-PT** | Conjuntos de avaliação em português: o MTEB-BR tem 22 tarefas nativas do Brasil (classificação, busca, agrupamento e outras); o MTEB-PT reúne tarefas em português por família. |
| **LongEmbed e MLDR-pt** | Avaliações com documentos longos: o LongEmbed é em inglês, com documentos de até 32k tokens; o MLDR-pt é busca em documentos longos em português. |
| **Passkey e Needle** | Testes do LongEmbed em que uma informação curta é escondida num texto longo, e o modelo precisa achar o documento que a contém. |
| **nDCG@10** | Nota de 0 a 100 que mede se os documentos certos aparecem entre os 10 primeiros resultados da busca, valendo mais quanto mais perto do topo. |
| **Classificação, clusterização, STS, reranking** | Tipos de tarefa: atribuir rótulos a textos; agrupar textos por tema; dar nota à semelhança entre duas frases; reordenar uma lista de candidatos. |
| **IC 95 %, p-valor e empate estatístico** | O IC é a margem de incerteza de uma média. O p-valor é a chance de uma diferença daquele tamanho aparecer só por acaso. Com p ≥ 0,05, nenhuma diferença é declarada: os modelos estão em empate estatístico. |
| **Contaminação** | Quando textos dos testes aparecem no treino e inflam as notas. Aqui, eles são filtrados antes do treino, e uma auditoria confere o resultado. |

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
