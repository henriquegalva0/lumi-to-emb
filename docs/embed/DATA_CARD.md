# Lumi-Embed — ficha dos dados

Fontes, transformações e política de contaminação dos dados de treino (seção 2 do [plano](plano_experimento.pdf)). Todas as fontes têm **revisão fixada** em [`configs/embed/data.yaml`](../../configs/embed/data.yaml). Os números realizados de cada execução (exemplos por fonte, removidos por contaminação, tamanho de cada componente) ficam em `data/embed/raw/stats_*.json`, `results/embed/contamination/report.md`, `data/embed/mined/stats.json` e `data/embed/sets/manifest.json`.

## Formato

Um único schema JSONL para todas as tarefas ([`schema.py`](../../src/lumi_embed/data/schema.py)):

| Campo | Conteúdo |
|---|---|
| `id` | hash estável de (tarefa, query, positivo) |
| `task` | `retrieval`, `sts`, `bitext` (`classification`, `clustering` e `reranking` estão previstos no schema, mas não são gerados nesta versão) |
| `lang` | `pt-BR`, `en`, `pt-BR→en`, `en→pt-BR` |
| `instruction` | aplicada só à query; nas tarefas simétricas (STS, bitexto), aos dois lados |
| `query`, `positive` | textos normalizados (Unicode NFC, espaços) |
| `negatives` | `[{text, score}]`: TopK-PercPos do teacher (mais o "contraste" do STS ou o negativo do NLI, quando há) |
| `naive_negatives` | top-k do teacher sem o filtro (ablação A3) |
| `bm25_negatives` | negativos BM25 oficiais do MS MARCO (só R1) |
| `pos_score` | cosseno do teacher entre query e positivo |
| `source` | dataset e revisão, ou gerador, persona, tipo, dificuldade e semente |
| `split` | `train` ou `dev` (não há `test`: os testes são os benchmarks externos) |

Formato de entrada no modelo: query = `Instruct: {instrução}\nQuery: {query}` + EOS; documento = texto + EOS, sem instrução. Comprimento máximo de 512 tokens no S1 e no S2; de 2k a 8k no S-long (ainda não executado nesta versão).

## Fontes reais (D<sub>real</sub>)

| Fonte | Revisão | Licença | Uso | Quantidade padrão |
|---|---|---|---|---|
| [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.pt` | `b04c8d1` | CC BY-SA 3.0 / GFDL | Pares naturais título↔lead e (título: seção)↔texto; sementes | 150 mil pares; 150 mil sementes + 30 mil longas |
| [`unicamp-dl/mmarco`](https://huggingface.co/datasets/unicamp-dl/mmarco) (português) | `6d039c4` | CC BY 4.0 (tradução); **o MS MARCO original é só para pesquisa não comercial** | Amostra no S1 e no S2; receita R1 com 63 negativos BM25 oficiais | 150 mil queries (30 mil na R1) |
| [`sentence-transformers/parallel-sentences-talks`](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks) `en-pt-br` e [`…-tatoeba`](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba) `en-pt` | `0c70bc6`, `cec1343` | Não declarada nos cards. Talks: corpus de palestras do OPUS (**conferir**; transcrições TED costumam ser CC BY-NC-ND). Tatoeba: CC BY 2.0 FR | Bitexto PT↔EN (metade em cada sentido) | 40 mil + 20 mil |
| [`sentence-transformers/gooaq`](https://huggingface.co/datasets/sentence-transformers/gooaq) | `b089f72` | Não declarada no card nem no repositório de origem (allenai/gooaq) — **conferir** | Retenção de inglês (pergunta→resposta) | 30 mil |
| [`sentence-transformers/all-nli`](https://huggingface.co/datasets/sentence-transformers/all-nli) (triplet) | `d482672` | Não declarada no card; origem SNLI (CC BY-SA 4.0) e MultiNLI (licenças mistas) | Retenção de inglês (STS); a contradição entra como negativo | 30 mil |
| [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) `por_Latn` | `af9c133` | ODC-By | Sementes da geração sintética (sem domínios `.pt`, para ficar no português brasileiro) | 150 mil |
| [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `sample-10BT` | `87f0914` | ODC-By | ~20 % multidomínio (inglês) do MNTP | ~13 mil textos |

Artigos que mencionam "número(s) mágico(s)" são excluídos, como na Etapa 1.

**Rotulados em PT (NLI, STS, retrieval):** nenhum. Os candidatos do plano (ASSIN, ASSIN 2, SICK-BR, STS-B pt, MLDR) são tarefas do MTEB-BR ou do MTEB-PT, e a política adotada é o **zero-shot limpo**. Os 20 % que a Tabela 3 reserva a eles no S2 são redistribuídos proporcionalmente entre os outros componentes (registrado em `manifest.json`, campo `redistributed_from`).

## Dados sintéticos (D<sub>syn</sub>)

| Gerador | Revisão | Licença |
|---|---|---|
| [`Qwen/Qwen3-30B-A3B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | `0d7cf23` | Apache 2.0 |
| [`mistralai/Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) | `95a6d26` | Apache 2.0 |
| [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) (raciocínio "low") | `6cee5e8` | Apache 2.0 |

O `lumi-4b` não é usado como gerador, para evitar circularidade. Cada gerador recebe **sementes diferentes**, e as saídas dos três são misturadas.

Procedimento ([`synth.py`](../../src/lumi_embed/data/synth.py)):

1. Para cada semente, sorteiam-se:
   - uma persona, de uma lista fixa de 40 perfis brasileiros (`configs/embed/data.yaml`);
   - um tipo de consulta: palavras-chave, aprender algo, resumo, sim/não ou contexto;
   - uma dificuldade: ensino médio, graduação ou doutorado;
   - um registro: formal ou coloquial;
   - um tamanho máximo.
2. O gerador escreve a consulta em PT-BR nativo e em JSON. O positivo é o próprio documento.
3. **STS:** paráfrase de uma frase da semente, mais um "contraste" (frase parecida com sentido diferente), que vira negativo difícil.
4. **Contexto longo:** consulta sobre uma janela de 300 palavras numa posição sorteada de um documento longo; o positivo é o documento inteiro (S-long).
5. Filtros na geração: JSON válido, tamanho, cópia literal (8-gramas do documento) e idioma. Na mineração, entram ainda o cosseno do teacher > 0,7 e a exigência de o positivo estar no top-10 do teacher.

## Mineração de negativos

- **Teacher:** [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B) (`5cf2132`, Apache 2.0), pelo runner de pooling do vLLM, com instrução nas queries.
- **Método:** **TopK-PercPos**. Um candidato só vira negativo se o seu score for menor que 95 % do score do positivo. Antes disso, saem o próprio positivo e as quase-duplicatas (Jaccard de 3-gramas ≥ 0,8).
- **Armazenamento:** até 8 negativos filtrados e 8 ingênuos por exemplo. O S1 usa 1 negativo; o S2 usa 4.
- **Busca:** produto interno na GPU sobre todos os documentos da mesma família de tarefa: retrieval curto, retrieval longo, STS ou bitexto.

## Contaminação

Política **zero-shot limpo** ([`decontaminate.py`](../../src/lumi_embed/data/decontaminate.py)):

1. **Exclusão por dataset:** a mineração falha se alguma fonte tiver o nome de uma tarefa de avaliação (lista em `decontamination.excluded_datasets`).
2. **Índice da avaliação:** todas as strings das tarefas do MTEB-BR (22), do MTEB-PT (14), do LongEmbed e do MTEB eng v2, carregadas pelo próprio mteb. O índice guarda o hash de cada texto inteiro normalizado e os 13-gramas de palavras, com passo 2.
3. **Filtro:**
   - o exemplo sai se a query ou o positivo aparece inteiro na avaliação, ou se ≥ 30 % dos seus 13-gramas estão no índice;
   - negativos contaminados saem sozinhos.
   - **qualidade:** sai também o texto "spinado" de fazendas de conteúdo (sinônimos trocados automaticamente, com marcas como "situaçãeste" e "atuaçãeste"), encontrado nas sementes da FineWeb-2 ao inspecionar a primeira síntese.
4. **Relatório:** `results/embed/contamination/report.md`, com contagens por arquivo e por tarefa indexada.

### Auditoria posterior

**A falha.** No índice usado pelo filtro, 13 tarefas entraram com 0 textos:
- as 8 de retrieval e reranking do MTEB-BR: Quati, QuatiReranking, JurisTCU, JurisTCUReranking, BRTaxQAR, FaQuADIR, MedPTRetrieval e FaqBacenRetrieval;
- 5 das 6 do LongEmbed: NarrativeQA, Needle, QMSum, SummScreenFD e WikimQA.

Essas tarefas guardam os textos em `task.corpus` e `task.queries`, num formato antigo do mteb, e o extrator só lia `task.dataset`. O erro passou em silêncio. Por isso, o filtro não conferiu essas tarefas.

**A correção.** O extrator lê os dois formatos, uma tarefa sem textos agora é erro, e o índice ganhou versão (2).

**A auditoria.** O passo `data audit` confere os conjuntos já treinados contra o índice corrigido, com o mesmo critério do filtro e sem alterar os dados. Ele atribui cada acerto à tarefa de onde veio. O resultado completo está em `results/embed/contamination/audit.md`.

| Conjunto | Exemplos | Com acerto | Na query | No positivo | Num negativo | Exemplos por tarefa |
|---|---|---|---|---|---|---|
| S1 (treino) | 284.938 | 136 (0,05 %) | 17 | 57 | 65 | Quati 98, QuatiReranking 19, FaQuADIR 17, NarrativeQA 3, BRTaxQAR 2, outra 5 |
| S2 (treino) | 120.000 | 125 (0,10 %) | 25 | 18 | 83 | Quati 75, FaQuADIR 26, QuatiReranking 20, BRTaxQAR 1, NarrativeQA 1, outra 8 |
| S2-real | 91.072 | 68 (0,07 %) | 1 | 10 | 57 | Quati 33, FaQuADIR 20, QuatiReranking 9, NarrativeQA 1, outra 5 |
| S2-syn | 74.142 | 85 (0,11 %) | 30 | 8 | 48 | Quati 53, FaQuADIR 21, QuatiReranking 12, BRTaxQAR 1, outra 5 |
| Dev (S1 e S2) | 1.500 cada | 1 cada | 0 | 0 | 1 | Quati 1 |

As contagens por campo somam acertos, e um exemplo pode ter mais de um. "outra" é um texto que passa do limiar só somando n-gramas de várias tarefas.

**Leitura:**
- Quase tudo vem das passagens web do Quati (ClueWeb22-PT). As sementes da síntese são páginas da FineWeb-2, e as duas fontes coletam a mesma web brasileira.
- No FaQuAD-IR, os parágrafos aparecem sobretudo como negativos.
- Nenhum acerto vem de FaqBacen, JurisTCU, JurisTCUReranking ou MedPT.

**Efeito nas notas.** No lumi-emb e nos checkpoints, as notas das 4 tarefas com acerto em query ou positivo (Quati, QuatiReranking, FaQuADIR e BRTaxQAR) não destoam das dos baselines. Sem elas, a média e a comparação pareada mudam pouco, e as conclusões são as mesmas (seção "Sensibilidade à contaminação" do [RESULTS.md](RESULTS.md)).

**Decisão.** Os checkpoints desta versão não foram retreinados. A remoção desses exemplos (refazer o filtro com o índice corrigido, a mineração, a mistura e o treino) fica para a próxima rodada de treino.

**Dev set:** 300 exemplos por componente, separados **antes** dos conjuntos de treino. Ele escolhe checkpoints, LR e coeficientes de merge. Os benchmarks de teste nunca entram nessa escolha.

## Composição por estágio (Tabela 3 do plano, proposta)

| Componente | S0 (MNTP) | S1 | S2 |
|---|---|---|---|
| Texto cru PT | ~80 % | — | — |
| Multidomínio (EN) | ~20 % | — | — |
| Pares naturais (Wikipédia) | — | 30 % | 15 % |
| Sintético, retrieval | — | 45 % | 25 % |
| Sintético, STS | — | 5 % | 20 % |
| mMARCO-PT | — | 10 % | 10 % |
| Rotulados PT | — | — | 20 % → redistribuído (vazio) |
| Inglês e bitexto | — | 10 % | 10 % |
| Negativos por exemplo | — | 1 | 4 |

S1, S2 e dev não compartilham exemplos. Pelo plano, as variantes S2-real e S2-syn teriam a mesma escala do S2, cada uma só com componentes reais ou só com sintéticos. Nesta versão, ficaram menores e diferentes entre si (abaixo).

## Números realizados (esta versão)

Fontes de `results/embed/contamination/report.md`, `data/embed/mined/stats_*.json` e `data/embed/sets/manifest.json`.

**Filtro de contaminação e qualidade** (entrada → mantidos):

| Arquivo | Entrada | Mantidos | Exato | n-grama | Spinado | Negativos removidos |
|---|---|---|---|---|---|---|
| Wikipédia PT | 150.000 | 145.831 | 838 | 3.331 | 0 | 0 |
| mMARCO-PT | 150.000 | 149.963 | 30 | 7 | 0 | 139 |
| Bitexto PT↔EN | 60.000 | 59.927 | 49 | 24 | 0 | 0 |
| Inglês (GooAQ, AllNLI) | 60.000 | 59.143 | 738 | 119 | 0 | 119 |
| Sintético, Qwen3-30B-A3B | 85.481 | 83.772 | 6 | 1.296 | 407 | 1 |
| Sintético, Mistral-Small-3.2-24B | 86.092 | 84.354 | 12 | 1.294 | 432 | 0 |
| Sintético, gpt-oss-20b | 85.329 | 83.591 | 8 | 1.270 | 460 | 0 |

**Mineração** (por família de tarefa):

| Família | Entrada | Saída | Removidos (cosseno do teacher ≤ 0,7) | Removidos (positivo fora do top-10) | Cosseno médio do positivo |
|---|---|---|---|---|---|
| Retrieval curto | 499.954 | 443.590 | 56.264 | 100 | 0,711 |
| STS | 86.342 | 86.270 | 68 | 4 | 0,901 |
| Bitexto | 59.927 | 59.927 | 0 | 0 | 0,923 |
| Retrieval longo | — | — | — | — | adiada (`mining.defer_families`) |

**Conjuntos** (exemplos de treino por componente):

| Conjunto | Natural | Sint. retrieval | Sint. STS | mMARCO-PT | Inglês e bitexto | Total |
|---|---|---|---|---|---|---|
| S1 (pedido: 400 mil) | 104.824 | 80.114 | 20.000 | 40.000 | 40.000 | 284.938 |
| S2 | 22.500 | 37.500 | 30.000 | 15.000 | 15.000 | 120.000 |
| S2-real | 22.500 | — | — | 34.286 | 34.286 | 91.072 |
| S2-syn | — | 37.500 | 36.642 | — | — | 74.142 |

Dev: 1.500 exemplos por estágio. A R1 tem 29.994 queries do mMARCO-PT. O S-long ficou vazio, porque a família de retrieval longo ainda não foi minerada. No S2-real, só 22,5 mil dos 91 mil exemplos são em PT nativo (Wikipédia); o resto é mMARCO traduzido e inglês/bitexto.

## Licenças: o que isto implica para o modelo (decisão D2)

O modelo herda as condições do `lumi-4b`, cuja licença ainda não está documentada, e do Nemotron 3 Nano 4B (NVIDIA Nemotron Open Model License). Dos dados de treino, o ponto que mais pesa é o **mMARCO/MS MARCO, de uso não comercial**. Se a licença final precisar permitir uso comercial, o mMARCO deve sair da mistura (`mix.s1/s2.proportions.mmarco: 0`, e a R1 deixa de existir). O mesmo vale para o bitexto "Talks", se a origem for confirmada como não comercial. Os três datasets do `sentence-transformers` não declaram licença no card; a origem de cada um precisa ser conferida antes da publicação. Os geradores e o teacher são Apache 2.0.
