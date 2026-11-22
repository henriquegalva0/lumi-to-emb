#!/usr/bin/env bash
# Lumi-Embed (Etapas 2–5): dados, treino, merge, avaliação e publicação.
#
# Uso: ./run_embed.sh <comando> [argumentos]
#
#   setup                        cria os ambientes e roda os testes automatizados
#   check                        GPU, kernels Mamba do Hub (bench), exporta o R0 e roda T1–T4 nele
#   sources [--only a,b]         fontes reais, sementes e texto do S0               (CPU)
#   synth [--generator NOME]     geração sintética com os LLMs geradores            (GPU, vLLM)
#   decontam                     índice da avaliação + filtro de contaminação       (CPU)
#   audit                        confere os conjuntos montados contra o índice     (CPU)
#   mine                         teacher + negativos difíceis                       (GPU, vLLM)
#   mix                          conjuntos S1, S2, ablações, R1, S-long e dev       (CPU)
#   export-base                  R0 = lumi-4b sem treino, como embedder
#   verify ID                    T1–T4 num checkpoint                               (GPU)
#   train ID [--max-minutes N]   treina configs/embed/train/ID.yaml                 (GPU)
#   merge NOME                   grade de merges + escolha no dev (M-C, M-B)        (GPU)
#   eval ID SUÍTE [SUÍTE ...]    mteb: smoke, mteb_br, mteb_pt, longembed, eng_v2, mmteb
#   baseline HF_ID SUÍTE [...]   avalia um modelo de comparação (Tabela 14)
#   report                       Tabelas 12–15, docs/embed/RESULTS.md e model card
#   publish ID [--release]       envia o checkpoint ao Hub (branch ID; --release = main)
#   card                         envia o model card e a documentação ao Hub
#   selftest                     testes automatizados (sem GPU)
#   diagnose                     empacota logs e ambiente (mesmo do run_stage1.sh)
#
# Ambientes (os dois dentro da imagem vllm/vllm-openai:v0.29.0):
#   vLLM   python do sistema + extras de requirements.txt (.venv)   -> synth, mine
#   treino venv isolado (.venv-embed, uv): torch 2.12.1+cu130 e requirements-embed.txt
#          (kernels Mamba-2 do Hub existem para torch 2.12, não para o 2.13 da imagem)
#
# Variáveis: LUMI_WORKDIR (padrão: o repositório), HF_HOME, HF_TOKEN (no .env; só para publicar).
set -euo pipefail

_self="${BASH_SOURCE[0]:-$0}"
REPO_DIR="$(cd "$(dirname "$_self")" 2>/dev/null && pwd || true)"
[ -n "$REPO_DIR" ] && [ -d "$REPO_DIR/src/lumi_embed" ] || REPO_DIR="$(pwd)"
[ -d "$REPO_DIR/src/lumi_embed" ] || { echo "ERRO: rode a partir da raiz do repositório." >&2; exit 2; }
cd "$REPO_DIR"

export LUMI_WORKDIR="${LUMI_WORKDIR:-$REPO_DIR}"
if [ -f "$REPO_DIR/.env" ]; then set -a; . "$REPO_DIR/.env"; set +a; fi
if [ -z "${HF_HOME:-}" ]; then export HF_HOME="$LUMI_WORKDIR/hf_cache"; fi
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="${LUMI_UV_CACHE:-$LUMI_WORKDIR/.uv-cache}"
export MTEB_CACHE="$LUMI_WORKDIR/results/embed/mteb"

CMD="${1:-help}"; [ $# -gt 0 ] && shift
TS="$(date +%Y%m%d-%H%M%S)"
say() { echo "[run_embed $(date +%H:%M:%S)] $*"; }
die() { echo "[run_embed] ERRO: $*" >&2; exit 2; }
usage() { sed -n '2,32p' "$REPO_DIR/run_embed.sh" | sed 's/^# \{0,1\}//'; }

TORCH_INDEX="https://download.pytorch.org/whl/cu130"
# MTEB-BR pelo arquivo do commit (a imagem do vLLM não traz o binário git, exigido por URLs git+https)
MTEB_BR="mteb-pt @ https://github.com/tardellirs/mteb-br/archive/5f45dbc309cbbaf5c244be4b7122d6e332bbebfc.tar.gz"
PY_SYS="$(command -v python3 || true)"

# ------------------------------------------------------------------ trava entre jobs
# Dois jobs podem rodar ao mesmo tempo (um de GPU e um de CPU). A trava impede que ambos
# criem o mesmo ambiente simultaneamente: o segundo espera o primeiro terminar a instalação.
env_lock() {
  command -v flock >/dev/null || { say "flock indisponível; sem trava entre jobs"; return 0; }
  mkdir -p "$LUMI_WORKDIR/logs"
  exec 9>"$LUMI_WORKDIR/logs/.envs.lock"
  flock -n 9 || { say "outro job está criando os ambientes; aguardando..."; flock 9; }
}
env_unlock() { if command -v flock >/dev/null; then flock -u 9 || true; exec 9>&-; fi; }

# ------------------------------------------------------------------ ambiente do vLLM
vllm_env() {
  # reaproveita o setup da Etapa 1 (python do sistema + extras com restrições)
  [ -n "$PY_SYS" ] || return 1
  "$PY_SYS" -c "import vllm" 2>/dev/null || return 1
  env_lock
  bash "$REPO_DIR/run_stage1.sh" setup >/dev/null
  env_unlock
  if [ -x "$REPO_DIR/.venv/bin/python" ]; then PY="$REPO_DIR/.venv/bin/python"; else PY="$PY_SYS"; fi
}

# ------------------------------------------------------------------ ambiente de treino
embed_env() {
  local venv="$REPO_DIR/.venv-embed"
  local stamp="$venv/.ok_$("$PY_SYS" -c 'import hashlib,sys; print(hashlib.sha256((open(sys.argv[1]).read()+sys.argv[2]).encode()).hexdigest()[:12])' requirements-embed.lock "$MTEB_BR")"
  if [ -x "$venv/bin/python" ] && [ -f "$stamp" ]; then PY="$venv/bin/python"; return; fi
  env_lock
  # outro job pode ter terminado a instalação enquanto esperávamos a trava
  if [ -x "$venv/bin/python" ] && [ -f "$stamp" ]; then env_unlock; PY="$venv/bin/python"; return; fi
  command -v uv >/dev/null || "$PY_SYS" -m pip install --user uv
  say "Criando $venv a partir de requirements-embed.lock (torch 2.12.1+cu130)"
  rm -rf "$venv"
  # UV_OVERRIDE da imagem fixa o NCCL do vLLM; não vale para este venv isolado
  env -u UV_OVERRIDE uv venv --python "$PY_SYS" "$venv"
  env -u UV_OVERRIDE uv pip install --python "$venv/bin/python" -r requirements-embed.lock \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
  # o MTEB-BR fixa numpy<2 sem necessidade; instalamos só o pacote (as dependências já estão aqui)
  env -u UV_OVERRIDE uv pip install --python "$venv/bin/python" --no-deps "$MTEB_BR"
  "$venv/bin/python" -c "import torch, transformers, peft, mteb, mteb_pt, sentence_transformers; print('ok', torch.__version__, transformers.__version__)"
  touch "$stamp"
  env_unlock
  PY="$venv/bin/python"
  "$PY" -m pip freeze > "$LUMI_WORKDIR/logs/pip_freeze_embed_${TS}.txt" 2>/dev/null || \
    env -u UV_OVERRIDE uv pip freeze --python "$PY" > "$LUMI_WORKDIR/logs/pip_freeze_embed_${TS}.txt" || true
}

gpu_banner() { command -v nvidia-smi >/dev/null && nvidia-smi -L || say "sem GPU visível"; }
run() { "$PY" -m lumi_embed "$@"; }

main() {
  mkdir -p "$LUMI_WORKDIR/logs"
  case "$CMD" in
    help|-h|--help) usage ;;
    setup)       embed_env; vllm_env || say "ambiente do vLLM indisponível aqui (ok fora da imagem)"; embed_env
                 "$PY" -m pytest -q -p no:cacheprovider "$REPO_DIR/tests" ;;
    selftest)    embed_env; "$PY" -m pytest -q -p no:cacheprovider "$REPO_DIR/tests" "$@" ;;
    check)       embed_env; gpu_banner; run bench; run export-base; run verify R0 ;;
    sources)     embed_env; run data sources "$@" ;;
    synth)       vllm_env || die "vLLM não encontrado: rode este comando na imagem vllm/vllm-openai:v0.29.0."; gpu_banner; run data synth "$@" ;;
    decontam)    embed_env; run data decontam ;;
    audit)       embed_env; run data audit ;;
    mine)        vllm_env || die "vLLM não encontrado: rode este comando na imagem vllm/vllm-openai:v0.29.0."; gpu_banner; run data mine ;;
    mix)         embed_env; run data mix ;;
    export-base) embed_env; run export-base ;;
    verify)      embed_env; gpu_banner; run verify "$@" ;;
    train)       embed_env; gpu_banner; run train "$@" ;;
    merge)       embed_env; gpu_banner; run merge "$@" ;;
    eval)        embed_env; gpu_banner; id="$1"; shift; args=(); for s in "$@"; do args+=(--suite "$s"); done
                 run eval "$id" "${args[@]}" ;;
    baseline)    embed_env; gpu_banner; id="$1"; shift; args=(); for s in "$@"; do args+=(--suite "$s"); done
                 run baseline "$id" "${args[@]}" ;;
    report)      embed_env; run report ;;
    publish)     embed_env; run publish "$@" ;;
    card)        embed_env; run card ;;
    diagnose)    bash "$REPO_DIR/run_stage1.sh" diagnose ;;
    *) usage; die "comando desconhecido: $CMD" ;;
  esac
}

if [ "$CMD" = "help" ] || [ "$CMD" = "-h" ] || [ "$CMD" = "--help" ]; then
  main "$@"
else
  mkdir -p "$LUMI_WORKDIR/logs"
  LOG="$LUMI_WORKDIR/logs/run_embed_${CMD}_${TS}.log"
  say "comando: $CMD $* | workdir: $LUMI_WORKDIR | log: $LOG"
  main "$@" 2>&1 | tee -a "$LOG"
  exit "${PIPESTATUS[0]}"
fi
