# Lumi-Embed — roteiro de execução

Este roteiro leva do `lumi-4b` ao embedder publicado, na ordem em que as etapas dependem umas das outras. Cada etapa é um comando do [`run_embed.sh`](../../run_embed.sh). Num cluster com fila, cada comando vira um job; com o Sauron, use [`jobs/embed/submit.sh`](../../jobs/embed/submit.sh), que monta o job com os recursos certos. O desenho experimental completo está no [plano](plano_experimento.pdf), e as escolhas de implementação estão em [DECISIONS.md](DECISIONS.md).

## Ambientes

Tudo roda dentro da imagem `vllm/vllm-openai:v0.29.0`, com dois ambientes Python criados pelo próprio script:

| Ambiente | Onde | Usado em |
|---|---|---|
| **vLLM** | Python do sistema (vLLM 0.29.0, torch 2.13) + extras de `requirements.txt` em `.venv` | `synth`, `mine` |
| **treino** | venv isolado `.venv-embed`, criado com `uv` a partir de `requirements-embed.lock` (torch 2.12.1+cu130, transformers 5.23.0, peft, sentence-transformers, mteb 2.27.18, MTEB-BR no commit `5f45dbc`) | todo o resto |

O ambiente de treino usa torch 2.12 porque é a versão para a qual existem os **kernels Mamba-2 pré-compilados do Hub** (`kernels-community/mamba-ssm`). Com o torch 2.13 da imagem, o NemotronH cairia na implementação em PyTorch puro, que é correta, porém lenta e pesada em memória no treino.

## Ordem das etapas

```
setup → check → eval R0/R0e (referência zero-shot)
      → sources → synth ×3 → decontam → mine → mix
      → train R1 · C1 (varredura de LR antes) → C2 e variações → merge M-C → eval → report → publish/card
      → contexto longo: decontam (índice v2) → mine (retrieval longo) → mix --only long → train C3-long
        → merge M-L → eval (mldr_pt_long, longembed) → report → publish/card
      → (trilho B) train B0a → B1a → B2a → merge M-B
```

| # | Comando | GPU | O que faz | Saída principal |
|---|---|---|---|---|
| 1 | `setup` | não | Cria os dois ambientes e roda os testes automatizados | `.venv-embed/` |
| 2 | `check` | sim | Mede a vazão com e sem kernels do Hub (`bench`), exporta o `lumi-4b` como **R0** e roda T1–T4 nele | `results/embed/bench.json`, `checkpoints/R0/` |
| 3 | `eval R0 smoke mteb_br mteb_pt` | sim | Referência zero-shot (Tabela 12, linha R0) | `results/embed/mteb/` |
| 4 | `eval R0e smoke mteb_br mteb_pt` | sim | Echo embeddings sem treino (ablação A7) | idem |
| 5 | `sources` | não | Pares naturais (Wikipédia PT), mMARCO-PT, bitexto, inglês, sementes e texto do S0 | `data/embed/raw/`, `seeds/`, `sets/s0/` |
| 6 | `synth --generator qwen3_30b` (e `mistral_24b`, `gpt_oss_20b`) | sim | Consultas sintéticas, STS e consultas sobre documentos longos | `data/embed/synth/<gerador>.jsonl` |
| 7 | `decontam` | não | Índice dos textos das tarefas de avaliação e filtro de contaminação | `data/embed/clean/`, `results/embed/contamination/` |
| 8 | `mine` | sim | Teacher Qwen3-Embedding-4B: filtro de cosseno, positivo no top-10 e negativos TopK-PercPos e ingênuos | `data/embed/mined/` |
| 9 | `mix` | não | Conjuntos S1, S2, S2-real, S2-syn, S2-naivehn, S2-nohn, R1, S-long e dev | `data/embed/sets/`, `sets/manifest.json` |
| 10 | `train C1-sweep-4.0e-4` (e `-1.2e-3`, `-2.4e-3`) | sim | Varredura de LR em 1/8 do S1 (24 passos). Copie o melhor LR no dev para `configs/embed/train/_s1.yaml` | `checkpoints/C1-sweep-*/` |
| 11 | `train R1`, `train C1` | sim | Baseline do artigo e estágio S1 | `checkpoints/<ID>/model/` |
| 12 | `train C2`, `C2-real`, `C2-syn`, `C2-s2`, `C2-s3` | sim | Modelo principal e candidatos do merge | idem |
| 13 | `train C2-direct`, `C2-nohn`, `C2-naivehn`, `C2-ib`, `C2-full`, `C2-attnonly`, `C2-noinst`, `C2-mrl` | sim | Ablações A1, A3, A4, A8, A10 e A12 | idem |
| 14 | `merge M-C` | sim | Média, SLERP e interpolação com o R0; escolha **no dev** | `checkpoints/M-C/`, `merge_report.json` |
| 15 | `train B0a` → `B1a` → `B2a` (e `B2a-s2`, `B2a-eos`, `B0f`→`B1f`→`B2f`) | sim | Trilho bidirecional: MNTP, S1, S2 | idem |
| 16 | `merge M-B` | sim | Merge do trilho Ba | `checkpoints/M-B/` |
| 16a | `mix --only long` | não | Só o conjunto do estágio longo: exemplos longos + 30 % de replay curto do S2 (sem os acusados pela auditoria); S1/S2 ficam intactos | `data/embed/sets/long/` |
| 16b | `train C3-long` | sim | Estágio longo (pipeline principal): parte do `lumi-emb`, currículo 2k → 4k → 8k, dev curto + longo | `checkpoints/C3-long/` |
| 16c | `merge M-L` | sim | Grade entre o C3-long e o lumi-emb, escolhida no dev curto + longo | `checkpoints/M-L/` |
| 17 | `verify <ID>` | sim | T1–T4 em qualquer checkpoint (obrigatório antes de publicar) | `checkpoints/<ID>/verify.json` |
| 17b | `train-eval <ID>[,<ID>...] <suítes>` | sim | Treina os IDs em ordem e avalia o último, num job só (ablações na fila, sem acompanhamento) | `checkpoints/<ID>/`, `results/embed/mteb/` |
| 18 | `eval <ID> smoke mteb_br mteb_pt` (+ `longembed eng_v2 mldr_pt_long` nos finalistas; `mmteb` opcional) | sim | Avaliação. A `mldr_pt_long` avalia o MLDR em português com documentos de até 8k tokens, num cache separado | `results/embed/mteb/` |
| 19 | `baseline <hf_id> mteb_br mteb_pt` | sim | Modelos da Tabela 14 com o mesmo protocolo | idem |
| 20 | `report` | não | Tabelas 12–15, `docs/embed/RESULTS.md`, figuras (`docs/embed/figures/`) e os blocos de destaques (README e card) e de resultados (card) | `results/embed/tables/` |
| 21 | `publish <ID>` / `publish <ID> --release` | não | Envia o checkpoint ao Hub (branch próprio) ou como release (`main`) | repositório do modelo |
| 22 | `card` | não | Envia o model card e a documentação ao Hub | idem |
| 23 | `hubcheck` | sim | Carrega o modelo publicado do jeito que o card ensina (Sentence-Transformers, Transformers, kernels), compara com o checkpoint local e confere os pesos de cada branch | `results/embed/hubcheck.json` |

Cada **release** (`lumi-emb`, a versão curta, e `lumi-emb-long`, a de contexto longo) é o melhor no dev entre o merge e os seus candidatos: o merge, se empata ou supera o melhor checkpoint individual no dev; senão, esse checkpoint. Os benchmarks não entram na escolha. `python3 jobs/embed/choose_release.py` (para o modelo longo: `python3 jobs/embed/choose_release.py M-L lumi-emb-long`) aplica a regra a partir do `merge_report.json`, cria `checkpoints/lumi-emb/` e grava a proveniência em `choice.json`. Depois, rode `verify <release>` e `eval <release> …`. O `main` do Hub recebe a release de `hub.release` (`configs/embed/base.yaml`, hoje o `lumi-emb-long`): `publish lumi-emb-long --release`. A outra vai para um branch próprio: `publish lumi-emb`.

**Disco.** Cada versão de um merge ocupa ~8 GB. O merge e o treino conferem o espaço livre logo no início e param com uma mensagem clara se faltar. O modelo escolhido entra em `checkpoints/<merge>/model` por hard links, sem ocupar espaço a mais. Depois da escolha, `python3 jobs/embed/prune_merge.py <merge>` mostra as versões descartadas e quanto espaço cada uma libera; com `--apply`, apaga essas versões e registra o que apagou em `pruned.json`. As notas no dev continuam no `merge_report.json`, e qualquer versão pode ser refeita a partir dos candidatos.

## Com o Sauron

Na sessão de login, dentro do clone (por exemplo `~/workspace/lumi-inference/lumi-to-emb`):

```bash
bash jobs/embed/submit.sh setup
bash jobs/embed/submit.sh check
bash jobs/embed/submit.sh eval R0 smoke mteb_br mteb_pt
bash jobs/embed/submit.sh sources
bash jobs/embed/submit.sh synth --generator qwen3_30b
# ... e assim por diante, um comando por etapa
```

O `submit.sh` grava o YAML e o script de cada job em `jobs/embed/.generated/` e chama `sauron submit`. Há dois perfis:

- **GPU** (`vram: 160`): uma B200, ~179 GB da cota.
- **CPU** (`vram: 20`): a menor classe de GPU, para as etapas que não usam GPU.

Com uma cota de 268 GB, cabe um job de cada perfil ao mesmo tempo. Por exemplo, `decontam` (CPU) pode rodar junto com um `synth` (GPU).

## Retomada e limites de tempo

- **Treino:** salva o estado a cada 30 min e ao receber `SIGTERM`, que é o que o `sauron cancel` envia. Reenviar o mesmo `train <ID>` continua do último passo. Um checkpoint já concluído não é refeito; apague `checkpoints/<ID>/` para refazê-lo.
- **Síntese:** grava a cada 5 mil pedidos e pula as sementes já processadas.
- **Avaliação:** o mteb pula as tarefas já avaliadas.
- **Fontes, descontaminação e composição:** são determinísticas pela semente. Refazer produz os mesmos arquivos.

## Antes de treinar: confira

1. `results/embed/bench.json`: `hub_kernels.kernel_check.active` deve ser `true` (o scan Mamba-2 em PyTorch puro não roda nem em inferência nem em treino) e `speedup_hub_vs_torch` deve ser bem maior que 1. A fase `torch_fallback` usa lote 4 (o caminho PyTorch puro não cabe com 32 × 512 no backward). Sem os kernels, qualquer carregamento na GPU falha com uma mensagem explícita; `LUMI_ALLOW_TORCH_MAMBA=1` permite seguir assim mesmo.
2. `checkpoints/R0/verify.json` com T1–T4 `ok`.
3. `results/embed/contamination/report.md`: quantos exemplos saíram por sobreposição com a avaliação.
4. `data/embed/mined/stats.json` (exemplos descartados pelo filtro de cosseno e pelo de posição do positivo) e `data/embed/sets/manifest.json` (tamanho realizado de cada componente).

## Escala e custo

As quantidades padrão de `configs/embed/data.yaml` são uma **escala reduzida** em relação à literatura (centenas de milhares de pares, e não dezenas de milhões), escolhida para caber numa GPU. Valores aproximados:

| | Padrão |
|---|---|
| Consultas sintéticas | 3 geradores × (60 mil de retrieval + 20 mil de STS + 8 mil longas) |
| S1 | 400 mil exemplos, batch 2048, 1 negativo |
| S2 | 120 mil exemplos, batch 128, 4 negativos |

O tempo real de cada etapa depende da vazão medida no `check`. Ajuste `max_time` e as quantidades depois de ver o `bench.json` e o primeiro treino.
