---
language:
  - pt
  - en
license: other
license_name: pendente
license_link: https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DATA_CARD.md
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

> **Em desenvolvimento.** Este repositório vai receber os checkpoints e o modelo final do Lumi-Embed. Ainda **não há resultados de embedding**: as tabelas abaixo são geradas automaticamente a partir dos arquivos de avaliação, e as células "—" indicam o que ainda não foi executado. Os resultados já obtidos são os da Etapa 1 (janela de contexto do `lumi-4b`).

**English summary.** Lumi-Embed turns [Lumi-4B](https://huggingface.co/ceia-research/lumi-4b), a 4B hybrid Mamba-2 + attention LLM for Brazilian Portuguese, into a text-embedding model:

- last-token (EOS) pooling, with a bidirectional track as an ablation;
- staged contrastive training: a weakly supervised stage and a supervised stage with positive-aware hard negatives;
- training data: native PT-BR synthetic data from three open generators, plus natural pairs;
- checkpoint merging, with checkpoints selected on a held-out dev set;
- evaluation on MTEB-BR (22 native tasks), MTEB-PT, LongEmbed and MTEB (eng, v2), under a clean zero-shot contamination policy.

As far as we know, it is one of the first evaluations of a hybrid Mamba-Transformer as a text embedder. Code, configs and data recipes: [github.com/henriquegalva0/lumi-to-emb](https://github.com/henriquegalva0/lumi-to-emb).

## Visão geral

| | |
|---|---|
| Backbone | [`ceia-research/lumi-4b`](https://huggingface.co/ceia-research/lumi-4b) (revisão `27b30f4`): 42 camadas, das quais 21 Mamba-2, 17 MLP e 4 de atenção; sem RoPE; derivado do Nemotron 3 Nano 4B |
| Embedding | estado oculto do último token (`</s>`), depois da normalização final; L2; similaridade de cosseno; dimensão 3136 |
| Formato | query: `Instruct: {instrução}\nQuery: {query}</s>`; documento: `{texto}</s>`, sem instrução |
| Comprimento | treinado até 512 tokens (S1/S2); o checkpoint de contexto longo vai até 8k |
| Idiomas | português brasileiro (principal) e inglês |
| Carregamento | transformers ≥ 5.23 nativo (`NemotronHModel`), **sem** `trust_remote_code` |

## Uso

O tokenizer salvo já anexa o `</s>` e faz padding à direita. Use sempre o mesmo formato de instrução nas queries e deixe os documentos sem instrução.

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
    return F.normalize(h[torch.arange(h.shape[0]), last].float(), dim=-1)

q = embed(["o que é inflação?"], "Dada uma pergunta, recupere passagens que a respondam")
d = embed(["Inflação é o aumento generalizado e contínuo dos preços."])
print(q @ d.T)
```

Para usar os kernels Mamba-2 otimizados (CUDA, torch 2.12): `pip install kernels` e `AutoModel.from_pretrained(..., use_kernels=True)`. Os checkpoints intermediários ficam em branches com o nome de cada um. Exemplo: `revision="C2"`.

## Contexto: a janela efetiva do Lumi-4B (Etapa 1)

Antes do treino, medimos até quantos tokens o `lumi-4b` recupera informação. O teste foi *needle-in-a-haystack* em PT-BR, de 2k a 255k tokens, com 2.610 prompts por modelo e o modelo-base como controle. [Relatório completo](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/stage1/README.md).

| Modelo | L\* (1 agulha) | L\* (multi-chave) | L\* (multi-valor) |
|---|---|---|---|
| Lumi-4B | 32k | **16k** | **16k** |
| Nemotron-3-Nano-4B (base) | 255k | 255k | 128k |

O Lumi colapsa a partir de 64k: as respostas entram em laço e saem sem número. Já em 32k, ele confunde chaves e valores. A perda vem do pré-treino contínuo em janelas de 8k, porque a base mantém 255k. Por isso o embedder treina com 512 tokens e estende o contexto até 8k num estágio separado (C3-long).

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

A escolha é feita no dev, nunca nos testes. O modelo publicado é o melhor no dev, e todos os merges ficam nos branches com a receita no `meta.json`.

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

## Avaliação

Protocolo igual para todos os modelos:

- instrução só na query (nos dois lados nas tarefas simétricas), com as mesmas instruções em português por tipo de tarefa;
- documentos sem instrução;
- mesmo comprimento máximo.

O benchmark principal é o **MTEB-BR**, 22 tarefas nativas ([arXiv:2607.04581](https://arxiv.org/abs/2607.04581)). Completam o quadro o **MTEB-PT** ([arXiv:2607.04071](https://arxiv.org/abs/2607.04071)), o LongEmbed e o MTEB (eng, v2); o MMTEB é opcional. Os ICs de 95 % e os p-valores usam bootstrap com a tarefa como unidade, como no MTEB-BR. Uma diferença só é declarada com p < 0,05.

<!-- RESULTS:START -->
### Resultados
Gerado por `lumi_embed report` a partir de `results/embed/`. Células "—" = sem resultado ainda. IC 95 % por bootstrap sobre tarefas; p-valores por bootstrap pareado (unidade = tarefa), como no MTEB-BR [16].

#### Tabela 12 — Visão geral

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

#### Tabela 13 — MTEB-BR por categoria

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

#### Tabela 13b — MTEB-BR por tarefa (seleção; completo em CSV)

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

#### Tabela 14 — Comparação com modelos correspondentes

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

#### Tabela 15 — Ablações (MTEB-BR)

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


<!-- RESULTS:END -->

### Hipóteses pré-registradas

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

- **Contexto:** o contexto validado vai só até o comprimento treinado (512 tokens; 8k no C3-long). Na Etapa 1, o backbone colapsou na geração a partir de 64k.
- **Arquitetura:** o embedder híbrido Mamba-Transformer não tem validação prévia na literatura, e o trilho bidirecional é experimental.
- **Retrieval em inglês:** é onde modelos recorrentes mostraram a maior lacuna. É monitorado à parte, no MTEB eng v2.
- **Dependência do teacher:** os negativos e o filtro dos sintéticos vêm do Qwen3-Embedding-4B.
- **Escala reduzida:** centenas de milhares de pares, contra dezenas de milhões na literatura.
- **Licença pendente (D2):** o mMARCO é de uso não comercial, e o `lumi-4b` ainda não declara licença.

## Reprodução

Todo o pipeline, com dados, treino, merge, avaliação e publicação, está em [github.com/henriquegalva0/lumi-to-emb](https://github.com/henriquegalva0/lumi-to-emb). O roteiro passo a passo é o [PIPELINE.md](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/PIPELINE.md), e as escolhas de implementação estão no [DECISIONS.md](https://github.com/henriquegalva0/lumi-to-emb/blob/main/docs/embed/DECISIONS.md). Cada checkpoint traz em `meta.json` a semente, o commit do código e a configuração completa.

## Licença

**Pendente (decisão D2 do plano).** O modelo deriva do `lumi-4b`, que ainda não declara licença, e do Nemotron 3 Nano 4B (NVIDIA Nemotron Open Model License). Foi treinado com dados de licenças diversas, inclusive o mMARCO/MS MARCO, que é **não comercial**. Até a definição, use apenas para pesquisa.

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
