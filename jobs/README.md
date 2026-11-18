# Jobs no cluster (Sauron)

Descritores para rodar a Etapa 1 pelo gerenciador de filas Sauron. Cada job usa a
imagem oficial `vllm/vllm-openai:v0.29.0` e só chama o `run_stage1.sh` dentro dela.
Tudo o que é gerado fica dentro da pasta do clone (`results/`, `data/`, `work/`,
`logs/`, `diagnostics/`), que é a pasta de trabalho do job.

| Job | Arquivos | GPUs | `max_time` | Para quê |
|---|---|---|---|---|
| Smoke | `smoke.yaml` + `smoke.sh` | 1 | 1h | Validar tudo em todos os comprimentos (até 261k) e estimar o tempo da execução completa |
| Completo | `full.yaml` + `full.sh` | 1 | 3h | Os 5.220 prompts (~1,6 h estimadas pelo smoke) |
| Diagnóstico | `diagnose.yaml` + `diagnose.sh` | 1 | 15m | Empacotar logs, versões e o estado da GPU num `.tar.gz` |

`vram: 160` força a classe de GPU de ~179 GB (B200). O modelo cabe em placas
menores, mas assim os números de desempenho saem sempre do mesmo hardware.

`workspace: lumi-inference` é a pasta em que o job começa. O clone pode ser essa
pasta ou uma subpasta dela (ex.: `~/workspace/lumi-inference/lumi-to-emb`): os
`jobs/*.sh` procuram o `run_stage1.sh` um nível abaixo. Os resultados ficam dentro
do clone (`<clone>/results/`).

CPU e RAM: os dois pedem 4 CPUs e 32 GB (o smoke teve pico de 11,4 GB). Depois do
smoke, `sauron status <job>` mostra o pico real (`ram pico`); com mais GPUs no
completo, some ~10 GB por GPU extra a esse pico.

Cota: cada B200 gasta ~179 GB da cota de VRAM (`sauron whoami`). Com 268 GB de
cota, só cabe uma GPU por job; pedir mais deixa o job parado na fila. O código usa
quantas GPUs o job tiver (um processo vLLM por GPU), então basta mudar `gpus:` se a
cota aumentar.

## Passo a passo

```bash
# 1. Na sessão de login: clonar na pasta de trabalho
ssh <usuario>@<endereço-do-login>
cd ~/workspace
git clone <url-do-repositório> lumi-inference   # ou numa subpasta: lumi-inference/<nome>
cd lumi-inference                                # (ou cd lumi-inference/<nome>)

# 2. (opcional) token do Hugging Face, só para limites de taxa — nunca no YAML
echo "HF_TOKEN=hf_xxx" > .env

# 3. Smoke (envie da raiz do clone; um job de cada vez, pois smoke e completo
#    compartilham .venv/ e data/)
sauron submit -c jobs/smoke.yaml jobs/smoke.sh
sauron logs <job> -f
```

Quando o smoke terminar, leia `results/<data>_smoke/REPORT.md`:

- **Seção 4 (gates):** os gates G1 a G6 devem estar ✅ para os dois modelos. Se o C0 (G2) falhar, pare e investigue antes da execução completa.
- **Seção 5 (estimativa):** traz o tempo previsto da execução completa com 1, 2, 4 e 8 GPUs. Ajuste `gpus` e `max_time` em `jobs/full.yaml` com folga, sem exagero, porque o `max_time` entra na estimativa de espera da fila.

```bash
# 4. Execução completa
sauron submit -c jobs/full.yaml jobs/full.sh
sauron logs <job> -f
```

O log principal mostra o progresso a cada minuto (prompts concluídos, % dos
tokens e ETA). Os detalhes ficam em `results/<run_id>/logs/`.

## Se algo falhar

- **Retomada:** os lotes concluídos ficam gravados. Reenviar o mesmo job (ou usar
  `sauron retry <job>` dentro do prazo) **retoma automaticamente** a última execução
  incompleta. Para começar do zero, acrescente `env: {FRESH: "1"}` ao YAML.
- **Diagnóstico:** `sauron submit -c jobs/diagnose.yaml jobs/diagnose.sh`. Sem GPU,
  dá também para rodar `bash run_stage1.sh diagnose` direto na sessão de login. O
  pacote sai em `diagnostics/diag_<data>.tar.gz`; copie-o para a sua máquina:
  `scp <usuario>@<endereço-do-login>:~/workspace/lumi-inference/diagnostics/diag_*.tar.gz .`
- Se o `sauron submit` reclamar do caminho `jobs/...`, copie o par de arquivos para
  a raiz do clone (`cp jobs/smoke.* .`) e envie de lá.

## Trazer os resultados

O cluster não tem backup. Copie a pasta da execução para fora:

```bash
rsync -rlt <usuario>@<endereço-do-login>:~/workspace/lumi-inference/results/<run_id> ./results/
```

Para refazer a análise localmente (sem GPU), rode `./run_stage1.sh analyze <run_id>`
na raiz do clone local, com um Python ≥ 3.12. Na primeira vez, o script cria um
venv e instala `requirements-dev.txt`.

## Limpeza

`work/` guarda os prompts pré-tokenizados (~0,7 GB) e a fila de cada execução, e
pode ser apagado depois que a execução terminar. `data/haystack_pt.parquet` é
reaproveitado entre execuções. Os pesos dos modelos ficam no cache compartilhado
do cluster (`HF_HOME`), fora da pasta do projeto.
