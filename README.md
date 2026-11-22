# Lumi-Embed: do Lumi-4B a um modelo de embeddings para português brasileiro

Este repositório transforma o [Lumi-4B](https://huggingface.co/ceia-research/lumi-4b), um LLM de 4B parâmetros focado em português brasileiro e baseado na arquitetura **híbrida Mamba-2 + atenção** do Nemotron 3 Nano 4B, num modelo de *embeddings* para RAG. Ele cobre todo o caminho:

1. medição da janela de contexto efetiva do backbone;
2. construção dos dados em PT-BR nativo;
3. treino contrastivo em estágios, com dois trilhos (causal e bidirecional);
4. merge de checkpoints;
5. avaliação no MTEB-BR, MTEB-PT, LongEmbed e MTEB (eng, v2), sob uma política de contaminação **zero-shot limpo**.

Até onde sabemos, é uma das primeiras avaliações de um híbrido Mamba-Transformer como embedder. O modelo e os checkpoints são publicados em [`henriquegalvao/lumi-4b-embedding`](https://huggingface.co/henriquegalvao/lumi-4b-embedding).

| Etapa | O que é | Estado | Documentação |
|---|---|---|---|
| **1. Janela de contexto** | Até quantos tokens o Lumi recupera informação de forma confiável (L\*) | ✅ concluída | [docs/stage1/](docs/stage1/README.md) |
| 2. Decisão | Desenho do embedder a partir do L\* | ✅ (abaixo) | — |
| 3. Dados | Pares naturais, sintéticos de 3 geradores, negativos minerados, descontaminação | ✅ S1 e S2; S-long pendente | [ficha dos dados](docs/embed/DATA_CARD.md) |
| 4. Treino | LoRA contrastivo em estágios (S0, S1, S2, S-long), trilhos C e B, merges | ✅ trilho causal (C1, C2 e variantes, merge M-C); trilho B, S-long e demais ablações pendentes | [roteiro](docs/embed/PIPELINE.md) · [decisões](docs/embed/DECISIONS.md) |
| 5. Avaliação | MTEB-BR (22 tarefas nativas), MTEB-PT, LongEmbed, MTEB eng v2, ablações | ✅ MTEB-BR e MTEB-PT; LongEmbed e MTEB eng v2 pendentes | [resultados](docs/embed/RESULTS.md) |

O desenho experimental completo das Etapas 2–5 está no [plano do experimento](docs/embed/plano_experimento.pdf).

---

## Etapa 1 — janela de contexto efetiva (concluída)

O teste foi *needle-in-a-haystack* em português, com prompts de 2k a 261k tokens: uma agulha (S), várias chaves (MK) e vários valores (MV). Foram 2.610 prompts por modelo, idênticos token a token entre o Lumi e o seu modelo-base (controle), numa execução com 1× NVIDIA B200.

| Modelo | L\* (S) | L\* (MK) | L\* (MV) | C0 |
|---|---|---|---|---|
| **Lumi-4B** | **32k** | **16k** | **16k** | 1,00 |
| Nemotron-3-Nano-4B (controle) | 255k | 255k | 128k | 1,00 |

![Acurácia por comprimento](docs/stage1/acc_vs_len.png)

- **Colapso em 64k:** em S, a acurácia do Lumi passa de 0,99 em 32k para 0,00 em 64k. As respostas entram em laço ou saem sem número.
- **Falha antes com várias agulhas:** em 32k, o MK cai para 0,58 (todos os erros são o número de outra cidade) e o MV para 0,65.
- **A perda vem do pré-treino contínuo em janelas de 8k:** a base mantém 255k com os mesmos prompts, e as diferenças pareadas são enormes (p ≈ 10⁻⁶ a 10⁻⁶⁶).

**Decisão da Etapa 2.** O L\* efetivo é 16k, então o embedder treina com 512 tokens (S1/S2), como no plano. O contexto é estendido até 8k num estágio separado (C3-long), e acima disso exigiria um *fine-tuning* de contexto longo. Metodologia completa, tabelas e figuras: [docs/stage1/](docs/stage1/README.md).

---

## Etapas 2–5 — Lumi-Embed

### Desenho

O embedding é o estado oculto do último token, um `</s>` anexado ao texto, depois da normalização final, com L2 e cosseno. A instrução vai só na query: `Instruct: {instrução}\nQuery: {query}`. Documentos não levam instrução.

| Trilho | Direção | Pooling | Estágios |
|---|---|---|---|
| **C** (principal) | causal (como pré-treinado) | último token (EOS) | S1 → S2 (→ S-long) |
| **B** (experimental) | Ba: atenção bidirecional · Bf: + BiMamba (varredura reversa nos mixers Mamba-2, mesmos pesos) | média | S0 (MNTP) → S1 → S2 |

Os estágios:

- **S1 (contrastivo fraco):** InfoNCE (τ = 0,02), 1 negativo difícil por exemplo, batch efetivo de 2048 via GradCache e negativos do batch com máscara de falso negativo.
- **S2 (supervisionado):** 4 negativos difíceis TopK-PercPos e batch 128.
- **S-long:** currículo de 2k → 4k → 8k tokens.

Os checkpoints seguem a Tabela 5 do plano: as referências R0 (zero-shot), R0e (echo) e R1 (receita do Mamba Retriever); C1 e C2, mais as variantes de ablação A1–A13; B0–B2; e os merges M-C e M-B, escolhidos **no dev**. O resumo está no [model card](docs/huggingface/README.md#arquitetura-e-checkpoints).

### Como executar

Tudo roda na imagem `vllm/vllm-openai:v0.29.0`. O [`run_embed.sh`](run_embed.sh) cria dois ambientes:

- o do vLLM, para gerar os dados sintéticos e minerar negativos;
- um venv isolado com torch 2.12.1+cu130, para usar os kernels Mamba-2 pré-compilados do Hub no treino e na avaliação.

```bash
./run_embed.sh setup                                  # ambientes + testes
./run_embed.sh check                                  # vazão com/sem kernels, R0 exportado, T1–T4
./run_embed.sh eval R0 smoke mteb_br mteb_pt          # referência zero-shot
./run_embed.sh sources                                # dados reais e sementes
./run_embed.sh synth --generator qwen3_30b            # (e mistral_24b, gpt_oss_20b)
./run_embed.sh decontam && ./run_embed.sh mine && ./run_embed.sh mix
./run_embed.sh train C1 && ./run_embed.sh train C2    # e variantes / trilho B
./run_embed.sh merge M-C
./run_embed.sh eval M-C smoke mteb_br mteb_pt longembed eng_v2
./run_embed.sh report && ./run_embed.sh publish M-C   # tabelas, model card, Hub
```

O roteiro completo, com a ordem das etapas, as saídas, a retomada e as conferências antes de treinar, está em [docs/embed/PIPELINE.md](docs/embed/PIPELINE.md). Num cluster com o gerenciador de filas usado nos experimentos, cada comando vira um job com `bash jobs/embed/submit.sh <comando> …` ([jobs/README.md](jobs/README.md)).

### Verificações

Antes de publicar, todo checkpoint passa pelas verificações da seção 3.6 do plano:

| Teste | Critério |
|---|---|
| T1 | Invariância a padding e à composição do lote (cosseno ≥ 0,9999) |
| T2 | Causalidade no trilho C (estados anteriores idênticos) e bidirecionalidade no B |
| T3 | Posição do pooling igual ao cálculo manual |
| T4 | Ida e volta pelo `lumi_embed` e pelo Sentence-Transformers |
| T5 | Relatório de contaminação |
| T6 | Suíte *smoke* acima do R0 |

### Resultados

As Tabelas 12–15 do plano saem do comando `report`, **só a partir dos arquivos de resultado do mteb**: nenhuma célula é digitada à mão, e o que não foi executado aparece como "—". Elas ficam em [docs/embed/RESULTS.md](docs/embed/RESULTS.md) e no bloco de resultados do model card.

O modelo publicado, **lumi-emb** (merge M-C-base0.75, escolhido no dev), fica em **62,41 no MTEB-BR**. Empata estatisticamente com o multilingual-e5-large-instruct (63,65) e o Qwen3-Embedding-0.6B (61,67) e fica abaixo do Qwen3-Embedding-4B (65,53; p = 0,03). Em retrieval, iguala o Qwen3-Embedding-4B, e tem a maior média em classificação entre os modelos comparados. A distância está quase toda em clusterização e STS. A descontaminação teve uma falha, corrigida e auditada depois do treino ([ficha dos dados](docs/embed/DATA_CARD.md#auditoria-posterior)).

---

## Estrutura do repositório

```
.
├── run_stage1.sh / run_embed.sh   # pontos de entrada da Etapa 1 e das Etapas 2–5
├── configs/
│   ├── stage1.yaml, smoke.yaml    # Etapa 1
│   └── embed/                     # base, data, eval; train/<ID>.yaml (um por checkpoint); merge/
├── src/
│   ├── lumi_ctx/                  # Etapa 1: palheiro, prompts, vLLM, análise
│   └── lumi_embed/
│       ├── modeling/              # embedder (EOS/média), trilhos Ba/Bf, exportação HF + ST
│       ├── data/                  # fontes, síntese (vLLM), mineração, descontaminação, composição
│       ├── train/                 # perdas, GradCache, treino contrastivo, MNTP
│       ├── merge/                 # média, SLERP, interpolação com a âncora; grade + escolha no dev
│       ├── eval/                  # adaptador do mteb, suítes, dev, estatística, relatório
│       ├── verify.py, publish.py  # T1–T4, bench, R0; envio ao Hub
├── tests/                         # 66 testes sem GPU (NemotronH minúsculo em CPU)
├── jobs/                          # descritores de job do cluster (Etapa 1 e submit.sh do Lumi-Embed)
├── docs/
│   ├── stage1/                    # Etapa 1: metodologia, relatório, figuras, tabelas
│   ├── embed/                     # plano (PDF), roteiro, ficha dos dados, decisões, resultados
│   └── huggingface/README.md      # model card do repositório do modelo
├── requirements*.txt, requirements-embed.lock
└── LICENSE                        # MIT (código)
```

## Reprodutibilidade e testes

- **Revisões fixadas:** modelos, datasets, geradores, teacher, imagem Docker, `requirements-embed.lock` (304 pacotes, resolvido para Linux x86_64) e o MTEB-BR por commit.
- **Determinismo:** sementes em toda a cadeia (palheiro, sementes, sorteios da síntese, composição dos conjuntos, ordem do treino).
- **Rastreabilidade:** cada checkpoint salva em `meta.json` a semente, o commit do código e a configuração completa.
- **Testes:** `./run_stage1.sh selftest` e `./run_embed.sh selftest` rodam os testes sem GPU. Eles usam um NemotronH minúsculo em CPU e cobrem:
  - os testes T1–T4;
  - a equivalência exata do GradCache;
  - os laços de treino com LoRA e completo, a retomada, o MNTP e os merges;
  - o pipeline de dados e a descontaminação;
  - o relatório gerado só a partir de resultados.

## Licenças

- **Código:** [MIT](LICENSE).
- **Modelos derivados:** licença **pendente** (decisão D2 do plano). Eles derivam do `lumi-4b`, que ainda não declara licença, e do Nemotron 3 Nano 4B (NVIDIA Nemotron Open Model License).
- **Dados de treino:** incluem o mMARCO/MS MARCO, de **uso não comercial**. Os detalhes estão na [ficha dos dados](docs/embed/DATA_CARD.md#licenças-o-que-isto-implica-para-o-modelo-decisão-d2).

## Referências

- [Model card do Lumi-4B](https://huggingface.co/ceia-research/lumi-4b) (CEIA/UFG).
- [Model card do Nemotron-3-Nano-4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16).
- Waleffe et al., *An Empirical Study of Mamba-based Language Models* ([arXiv:2406.07887](https://arxiv.org/abs/2406.07887)).
- Hsieh et al., *RULER* ([arXiv:2404.06654](https://arxiv.org/abs/2404.06654)).
- Zhang et al., *Mamba Retriever* ([arXiv:2408.08066](https://arxiv.org/abs/2408.08066)).
- *Qwen3 Embedding* ([arXiv:2506.05176](https://arxiv.org/abs/2506.05176)).
- *Llama-Embed-Nemotron-8B* ([arXiv:2511.07025](https://arxiv.org/abs/2511.07025)).
- *NV-Retriever* ([arXiv:2407.15831](https://arxiv.org/abs/2407.15831)).
- *LLM2Vec* ([arXiv:2404.05961](https://arxiv.org/abs/2404.05961)).
- *MTEB-BR* ([arXiv:2607.04581](https://arxiv.org/abs/2607.04581)).
- *MTEB-PT* ([arXiv:2607.04071](https://arxiv.org/abs/2607.04071)).
- A lista completa está no apêndice B do [plano](docs/embed/plano_experimento.pdf).
