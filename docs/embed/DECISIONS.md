# Lumi-Embed — decisões de implementação

O [plano](plano_experimento.pdf) define o experimento; este documento registra **como** ele foi implementado e onde a implementação precisou escolher, simplificar ou divergir. O que o plano marca como *proposta* ou *hipótese* continua sendo proposta aqui.

## Decisões em aberto do plano (D1–D4)

| ID | Decisão | Estado nesta versão |
|---|---|---|
| D1 | Checkpoint de partida | `ceia-research/lumi-4b` (instruct), revisão `27b30f4`. Um checkpoint pré-SFT, se existir, entra em `model.hf_id`/`revision` (ablação A11) |
| D2 | Licença | **Uso não comercial.** Ver a seção de licenças da [ficha dos dados](DATA_CARD.md#licenças-o-que-isto-implica-para-o-modelo-decisão-d2) (entre os dados de treino, o mMARCO é de uso não comercial) |
| D3 | LoRA × fine-tuning completo | `train.method` em cada configuração. Padrão: LoRA r = 16. A10: `C2-full` (completo) e `C2-attnonly` (LoRA só na atenção) |
| D4 | Pooling do trilho bidirecional | `embedding.pooling`. Padrão: média. A6: `B2a-eos` |

## Lições da Etapa 1 que entram aqui

A [Etapa 1](../stage1/README.md) mediu que o Lumi recupera de forma confiável até ~16k tokens com várias agulhas (L\* efetivo) e colapsa a partir de 64k. Por isso:

- o S1 e o S2 usam 512 tokens (como no plano);
- o S-long (C3-long) vai até 8k, dentro da faixa em que o modelo ainda funciona;
- o LongEmbed (até 32k) é avaliado nos finalistas, com a expectativa registrada de degradação acima de 32k.

## Ambiente e modelo

| Decisão | Motivo |
|---|---|
| **NemotronH nativo do transformers 5.23**, sem `trust_remote_code` | O transformers converte as chaves `backbone.*` → `model.*` do checkpoint. O modelo exportado carrega sem código remoto (`architectures: NemotronHModel`), inclusive no Sentence-Transformers |
| **Kernels Mamba-2 do Hub** (`use_kernels=True`, `kernels-community/mamba-ssm`) com **torch 2.12.1+cu130** num venv isolado | Não há wheels do `mamba-ssm` no PyPI, e o Hub só tem builds até o torch 2.12. A imagem do vLLM traz o 2.13; sem os kernels, o NemotronH usa a implementação em PyTorch puro, que é correta, porém lenta. O `check` mede as duas |
| **Kernels ativados explicitamente e verificados** (`modeling/kernels.py`) | O transformers escolhe os kernels pelo dispositivo e pelo modo do modelo no carregamento, e o `kernelize` cai em silêncio no PyTorch puro quando um kernel não carrega. Aqui eles são ativados depois de o modelo estar na GPU, em modo de treino, e uma passada curta em cada modo conta as chamadas do scan em PyTorch puro; se houver alguma, o carregamento falha (`LUMI_ALLOW_TORCH_MAMBA=1` permite seguir). O resultado vai para o `meta.json` de cada checkpoint |
| **EOS `</s>` anexado pelo pós-processador do tokenizer** | O mesmo formato vale no treino, na avaliação, no Transformers e no Sentence-Transformers. O EOS sobrevive ao truncamento. É configurável (`embedding.eos_token`) |
| **Padding à direita** (obrigatório) | No treino, o kernel fundido do Mamba-2 não aplica a máscara de padding dentro do scan. Com padding à direita, os pads ficam depois do EOS e, por causalidade, não afetam o token de pooling. O teste T1 verifica isso |
| **`out_proj` dos mixers Mamba fora do LoRA** | O kernel fundido recebe `out_proj.weight` diretamente. Um adaptador ali seria ignorado no treino e aplicado na inferência |
| **`use_cache=False` em todo forward** | O config do Lumi liga o cache (geração). No embedder ele só gasta memória e, no trilho Bf, desligaria a varredura reversa (bug encontrado e coberto pelo teste T2) |
| **Bidirecionalidade por *monkeypatch*** | Ba usa máscara só de padding nas 4 atenções e `is_causal=False` (sem isso, o SDPA volta a ser causal quando não há padding). Bf soma as varreduras direta e reversa com os **mesmos pesos**, invertendo só os tokens válidos. Os nomes dos pesos não mudam, então LoRA, checkpoints e merges servem aos dois trilhos |
| **Echo (R0e, A7)** | O texto é repetido (`x\nx`) e o pooling continua no EOS. O plano não fixa a variante; a de [8] com pooling na segunda cópia não foi implementada |

## Treino

| Decisão | Motivo |
|---|---|
| **S1: InfoNCE com negativos do batch + máscara de falso negativo, batch 2048 via GradCache** | O plano pede batch efetivo grande; ele só tem efeito se os negativos do batch entram na perda. O GradCache dá o gradiente exato do batch inteiro com a memória de um micro-lote (testado contra o backward direto) |
| **S2: só os 4 negativos difíceis** | Foi a melhor variante de perda em [10]. A alternativa com batch + máscara é a ablação A4 (`C2-ib`) |
| **LR do S2 com LoRA = 1e-4** (proposta) | O plano fixa só o LR do fine-tuning completo (2e-6) |
| **R1 com LoRA** (proposta) | O Mamba Retriever fez ajuste completo de modelos até 790M; aqui o modelo tem 4B. Os 63 negativos são os BM25 oficiais do MS MARCO (`triples.train.ids.small.tsv`), sem rodar BM25 de novo |
| **Varredura de LR do S1** (`C1-sweep-*`: 4e-4, 1,2e-3, 2,4e-3 em 1/8 dos dados, 24 passos) | O plano recomenda varrer LR em Mamba antes do treino completo |
| **LR do S1 = 2,4e-3** (LoRA r = 16) | Melhor no dev da varredura em todos os componentes (4e-4: 0,942; 1,2e-3: 0,974; 2,4e-3: 0,981), perda caindo sem oscilar. Está na ponta da grade, mas o ganho sobre 1,2e-3 já é pequeno |
| **Dev com distratores fixos** (20 mil documentos de retrieval, 5 mil de STS e 5 mil de bitexto, positivos do treino sorteados por hash, gravados em `sets/dev_distractors.jsonl`) | Sem eles, cada query do dev concorria com 315 a 1.925 documentos e todos os modelos da varredura ficavam entre 0,94 e 0,98: a escolha de checkpoints e de merges cairia no ruído. A varredura de LR usou o dev antigo; do C1 em diante, todos usam o mesmo dev com distratores |
| **Aquecimento proporcional** (`warmup_ratio: 0.1`) em vez de 100 passos fixos | Na escala reduzida, o S1 tem ~195 passos: 100 de aquecimento seriam metade do treino, e na varredura (24 passos) a taxa nunca chegaria ao valor testado |
| **Checkpoints por tempo e `SIGTERM`** | Os jobs têm `max_time`; reenviar continua do passo salvo |
| **Contexto longo no pipeline principal** (não mais ablação A13), antes das ablações | O objetivo do modelo é RAG. Com só 512 tokens no treino, o lumi-emb ficou em 47,50 no LongEmbed contra 80,13 do Qwen3-Embedding-4B |
| **C3-long parte do lumi-emb** (merge M-C-base0.75), não do C2 | É o melhor modelo curto; o estágio longo é aplicado a ele |
| **Replay de 30 % de exemplos curtos do S2 no estágio longo** (`mix.long.replay_s2`), sem os acusados pela auditoria | Evitar que o modelo esqueça os textos curtos ao aprender os longos |
| **Dev longo como tarefa própria** (`retrieval_long`): 300 consultas, 1.000 distratores longos (positivos longos do treino), documentos codificados em 8k para todos os modelos | Os distratores curtos do dev não testam documentos longos; arquivo separado (`dev_distractors_long.jsonl`), e o dev curto continua idêntico |
| **Merge M-L entre o C3-long e o lumi-emb**, escolhido no dev curto + longo | O modelo publicado pode ser um só, bom nos dois comprimentos; o spec (`max_length` 8192) vem do primeiro candidato, o C3-long |
| **`mix --only long`** | Regrava só o conjunto longo; S1/S2 são recalculados (determinísticos) para as exclusões, mas não regravados |
| **S0 (MNTP): token de máscara `<unk>`, 30 % mascarado, posição i prevista pelo logit de i − 1** | O tokenizer do Lumi não tem token de máscara. O LLM2Vec usa um token raro, e 30 % fica no meio da faixa 20–40 % do plano. Só os logits das posições mascaradas são calculados |

## Dados

| Decisão | Motivo |
|---|---|
| **Escala reduzida** (centenas de milhares de pares) | Cabe em jobs de uma B200. As quantidades são parâmetros |
| **Leitura em streaming dos parquets** (`HfFileSystem` + `pyarrow`), sem a biblioteca `datasets` | Funciona nos dois ambientes; mesma abordagem da Etapa 1 |
| **Personas de uma lista fixa** (40 perfis), sem a recuperação das 5 melhores do Persona Hub | Simplificação, para não depender de mais um índice. A lista está na configuração |
| **Sem dados sintéticos de classificação e clustering** nesta versão | O componente "outras tarefas" usa STS (paráfrase + contraste) e o bitexto real. O schema já prevê as outras tarefas |
| **S1 realizado com 284.938 exemplos** (pedido: 400 mil) | O filtro do teacher (cosseno > 0,7) descartou ~1/3 dos sintéticos de retrieval e o estoque de pares naturais se esgotou depois do S2 e do dev. Completar com o mMARCO e o inglês que sobraram levaria o S1 a ~47 % de dados traduzidos ou em inglês; manteve-se a composição centrada em PT-BR nativo (72 %), com 139 passos |
| **`retrieval_long` minerada depois** (`mining.defer_families`) | O primeiro job ficou 8 h sem saída nessa família; ela só alimenta o S-long (C3-long, ablação A13), fora do caminho do modelo publicado |
| **Rotulados PT vazios** | Todos os candidatos do plano são tarefas de avaliação; a política é o zero-shot limpo |
| **Teacher padrão 4B** (8B opcional) | Vazão. Os dois estão entre os melhores modelos abertos no MTEB-BR |
| **Índice de contaminação incompleto na primeira execução; auditoria posterior; checkpoints não retreinados** | 13 tarefas (as 8 de retrieval/reranking do MTEB-BR e 5 do LongEmbed) guardam os textos em `task.corpus`/`task.queries`, e o extrator só lia `task.dataset`: elas entraram vazias no índice, sem erro. Corrigido (índice versão 2; tarefa sem textos agora é erro). O `data audit` achou coincidências em ≤ 0,1 % dos exemplos de cada conjunto, quase todas com o Quati e o FaQuAD-IR, e nenhuma com FaqBacen, JurisTCU ou MedPT. Sem as 4 tarefas afetadas, as conclusões não mudam. A remoção fica para a próxima rodada de treino ([ficha dos dados](DATA_CARD.md#auditoria-posterior)) |

## Avaliação

| Decisão | Motivo |
|---|---|
| **MTEB-BR pelo pacote oficial** (`tardellirs/mteb-br`, commit `5f45dbc`), instalado sem dependências a partir do `.tar.gz` do commit | Ele fixa `numpy<2` sem necessidade técnica para as tarefas; o resto do ambiente usa numpy 2. O `.tar.gz` (mesmos arquivos da instalação via git) evita depender do binário `git`, ausente na imagem do vLLM |
| **Mesmas instruções para todos os modelos com instrução**, por tipo de tarefa e em português (`configs/embed/eval.yaml`), registradas por tarefa | Seção 5.1 do plano. Os baselines com instrução (Qwen3-Embedding, e5-instruct) passam pelo `InstructSentenceTransformerModel` do mteb com essas instruções; os sem instrução usam a implementação oficial do mteb |
| **Mesmo comprimento máximo para todos os modelos** (`eval.max_length: 512`; a suíte pode definir outro, como os 32.768 do LongEmbed) | Seção 5.1 do plano. O comprimento vem do protocolo, não do modelo: sem isso, um baseline com contexto de 32k codificava documentos inteiros nas tarefas MultiLongDoc (e estourava a memória), enquanto o nosso cortava em 512. O lote dos baselines também respeita o da suíte |
| **gte-multilingual-base sem medição própria (por ora)** | O código remoto do modelo (escrito para o transformers 4.x) dispara um assert de CUDA em todas as tarefas com o transformers 5.23 do ambiente. A Tabela 14 mantém os valores publicados no MTEB-PT [17]; a medição no nosso protocolo fica para um ambiente separado com transformers 4.x. O e5-large-instruct cobre o papel de baseline multilíngue clássico medido |
| **IC e p-valores com a tarefa como unidade** | Mesmo método do MTEB-BR. "Vitória" só com p < 0,05 |
| **NanoMTEB-BR substituído por uma suíte `smoke`** (2 tarefas do MTEB-BR + 2 do MTEB-PT) | O NanoMTEB-BR não está no mteb |
| **Tabelas geradas só a partir dos JSONs de resultado** | Regra A.3: nenhuma célula é digitada à mão. Os únicos números escritos no código são valores da literatura, com a referência |
| **Notas × 100, com ponto decimal, em todas as tabelas** | Mesma escala do leaderboard do MTEB; os valores da literatura foram convertidos (0,6621 → 66.21) |
| **Três sementes do C2 avaliadas nos benchmarks** (C2, C2-s2, C2-s3) | O IC por tarefa não mede a variação entre execuções. Com as sementes, o desvio-padrão do Mean22 é 0,13: diferenças menores que ~0,3 entre checkpoints não se distinguem do acaso |
| **Modelo publicado escolhido pela regra do dev** (`jobs/embed/choose_release.py`) | O merge, se empata ou supera o melhor checkpoint individual no dev; senão, esse checkpoint. Os benchmarks não entram, porque escolher pelo MTEB-BR seria selecionar no próprio teste. O escolhido foi o M-C-base0.75 (dev 0,9779; melhor individual, C2-real, 0,9771); a proveniência fica em `checkpoints/lumi-emb/choice.json` |
| **A origem do lumi-emb herda as notas dele no relatório** | São os mesmos pesos: o M-C não é avaliado de novo |
| **MLDR-pt em 8k como suíte própria** (`mldr_pt_long`, cache com sufixo `-L8192`) | No MTEB-PT o MLDR é cortado em 512 tokens, como todo o protocolo. Para medir RAG com documentos longos em português, ele é avaliado também em 8k; o sufixo no nome do modelo impede que esse resultado substitua o de 512 no cache do mteb |
| **Filtro de contaminação refeito com o índice v2 antes do estágio longo** | Os documentos longos passam pelo índice corrigido; o relatório do filtro original fica em `report_index_v1.*` |
| **Pesos do lumi-emb por hard link** | Copiar 8 GB na sessão de login morreu com a queda da conexão SSH; o link é instantâneo e não ocupa disco |

## O que não foi implementado (e por quê)

- **Latent attention** (A6, terceira opção): fora do escopo desta versão; o pooling EOS × média está coberto.
- **Varredura automática dos coeficientes do MNTP no dev:** o S0 roda com valores fixos, e o efeito aparece no B1/B2.
- **Ablação A11** (instruct × pré-SFT): depende de um checkpoint pré-SFT que não é público.
