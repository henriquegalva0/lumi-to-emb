#!/usr/bin/env bash
# Etapa 1 — teste de janela de contexto (needle-in-a-haystack em PT-BR).
#
# Uso: ./run_stage1.sh <comando> [opções extras para o Python]
#
#   setup             prepara o Python: usa os pacotes da imagem e instala os extras
#                     de requirements.txt num venv (.venv) sobre eles
#   check             confere ambiente, GPUs, tokenizers, chat template e acesso ao Hub
#   prepare           monta o palheiro (data/haystack_pt.parquet) e os prompts (só CPU)
#   smoke             execução curta: todos os comprimentos e tarefas, poucas repetições
#   full              execução completa: 2610 prompts por modelo
#   analyze <run_id>  refaz a análise de uma execução (só CPU)
#   selftest          roda os testes automatizados (sem GPU)
#   diagnose          empacota logs, ambiente e estado das execuções num .tar.gz
#
# Variáveis de ambiente:
#   LUMI_WORKDIR  onde tudo é gravado (padrão: a pasta deste repositório)
#   RUN_ID        nome de uma execução existente, para retomá-la
#   FRESH=1       não retoma automaticamente a última execução incompleta
#   HF_HOME       cache do Hugging Face (padrão: $LUMI_WORKDIR/hf_cache)
#   HF_TOKEN      opcional; coloque-o num arquivo .env, nunca no código
set -euo pipefail

# ------------------------------------------------------------------ localização
_self="${BASH_SOURCE[0]:-$0}"
REPO_DIR="$(cd "$(dirname "$_self")" 2>/dev/null && pwd || true)"
if [ -z "$REPO_DIR" ] || [ ! -d "$REPO_DIR/src/lumi_ctx" ]; then
  REPO_DIR="$(pwd)"
fi
if [ ! -d "$REPO_DIR/src/lumi_ctx" ]; then
  echo "ERRO: não achei src/lumi_ctx. Rode a partir da raiz do repositório." >&2
  exit 2
fi
cd "$REPO_DIR"

export LUMI_WORKDIR="${LUMI_WORKDIR:-$REPO_DIR}"
case "${1:-help}" in
  help|-h|--help) ;;
  *) mkdir -p "$LUMI_WORKDIR/logs" "$LUMI_WORKDIR/results" "$LUMI_WORKDIR/data" "$LUMI_WORKDIR/work" ;;
esac

# Segredos (ex.: HF_TOKEN) ficam num .env fora do Git.
if [ -f "$REPO_DIR/.env" ]; then
  set -a; . "$REPO_DIR/.env"; set +a
fi

if [ -z "${HF_HOME:-}" ]; then
  export HF_HOME="$LUMI_WORKDIR/hf_cache"
  export LUMI_HF_HOME_SOURCE=workdir
else
  export LUMI_HF_HOME_SOURCE="${LUMI_HF_HOME_SOURCE:-external}"
fi
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

CMD="${1:-help}"
[ $# -gt 0 ] && shift
TS="$(date +%Y%m%d-%H%M%S)"
SHELL_LOG="$LUMI_WORKDIR/logs/run_stage1_${CMD}_${TS}.log"

say() { echo "[run_stage1 $(date +%H:%M:%S)] $*"; }
die() { echo "[run_stage1] ERRO: $*" >&2; exit 2; }

usage() { sed -n '2,21p' "$REPO_DIR/run_stage1.sh" | sed 's/^# \{0,1\}//'; }

# ------------------------------------------------------------------ Python
PY_SYS="$(command -v python3 || command -v python || true)"
PY=""

# Os pacotes fixados em requirements.txt já estão instalados neste Python?
reqs_ok() {
  "$1" - "$REPO_DIR/requirements.txt" <<'EOF' >/dev/null 2>&1
import importlib.metadata as md, re, sys
for line in open(sys.argv[1]):
    line = line.split("#")[0].strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line)
    if m and md.version(m.group(1)) != m.group(2):
        sys.exit(1)
EOF
}

has_core() {  # tokenizers/transformers/hub: vêm da imagem do vLLM (ou do requirements-dev.txt local)
  "$1" -c "import transformers, tokenizers, huggingface_hub, numpy, yaml" >/dev/null 2>&1
}

setup_env() {
  [ -n "$PY_SYS" ] || die "python3 não encontrado"
  if reqs_ok "$PY_SYS" && has_core "$PY_SYS"; then
    PY="$PY_SYS"; say "Usando o Python do sistema/imagem ($("$PY" --version 2>&1))"
  else
    local req_hash
    req_hash="$("$PY_SYS" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:12])' requirements.txt)"
    if [ -x "$REPO_DIR/.venv/bin/python" ] && [ -f "$REPO_DIR/.venv/.req_$req_hash" ] && reqs_ok "$REPO_DIR/.venv/bin/python"; then
      PY="$REPO_DIR/.venv/bin/python"
    else
      say "Criando venv em .venv sobre os pacotes da imagem"
      rm -rf "$REPO_DIR/.venv"
      if "$PY_SYS" -m venv --without-pip --system-site-packages "$REPO_DIR/.venv"; then
        PY="$REPO_DIR/.venv/bin/python"
        # Restrições: o pip não pode trocar nenhum pacote que a imagem já traz.
        "$PY" -m pip list --format=freeze 2>/dev/null \
          | grep -iE '^(vllm|torch|torchvision|torchaudio|transformers|tokenizers|huggingface[-_]hub|numpy|triton|flashinfer[-_a-z]*|xformers|safetensors|pyyaml|pillow|fsspec)==' \
          > "$REPO_DIR/.venv/constraints.txt" || true
        "$PY" -m pip install --disable-pip-version-check -r requirements.txt -c "$REPO_DIR/.venv/constraints.txt" \
          || die "pip install falhou (conflito com os pacotes da imagem?). Veja $SHELL_LOG"
        if ! has_core "$PY"; then
          say "transformers/tokenizers ausentes (máquina sem a imagem do vLLM): instalando requirements-dev.txt"
          "$PY" -m pip install --disable-pip-version-check -r requirements-dev.txt -c "$REPO_DIR/.venv/constraints.txt" \
            || die "pip install de requirements-dev.txt falhou"
        fi
        touch "$REPO_DIR/.venv/.req_$req_hash"
      else
        die "não foi possível criar o venv. Instale python3-venv ou use uma imagem com os pacotes de requirements.txt."
      fi
    fi
  fi
  export LUMI_PIP_FREEZE="$LUMI_WORKDIR/logs/pip_freeze_${TS}.txt"
  "$PY" -m pip freeze > "$LUMI_PIP_FREEZE" 2>/dev/null || "$PY" -m pip list --format=freeze > "$LUMI_PIP_FREEZE" 2>/dev/null || true
}

gpu_banner() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    say "nvidia-smi indisponível (sem GPU visível)"
  fi
}

run_py() { "$PY" -m lumi_ctx "$@"; }

resume_args() {
  if [ -n "${RUN_ID:-}" ]; then
    echo "--run-id $RUN_ID"
  elif [ "${FRESH:-0}" != "1" ]; then
    echo "--resume-latest"
  fi
}

# ------------------------------------------------------------------ diagnóstico
diagnose() {
  local name="diag_${TS}"
  local out="$LUMI_WORKDIR/diagnostics"
  local d="$out/$name"
  mkdir -p "$d"
  {
    echo "## data";       date
    echo "## sistema";    uname -a; cat /etc/os-release 2>/dev/null || true
    echo "## cpu";        (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "?")
    echo "## memória";    (free -g 2>/dev/null || true)
    echo "## disco";      df -h "$LUMI_WORKDIR" 2>/dev/null || true
    echo "## workdir";    echo "$LUMI_WORKDIR"; ls -la "$LUMI_WORKDIR" 2>/dev/null || true
    echo "## git";        cat "$REPO_DIR/.git/HEAD" 2>/dev/null || true
    ref="$(sed -n 's/^ref: //p' "$REPO_DIR/.git/HEAD" 2>/dev/null || true)"
    [ -n "$ref" ] && cat "$REPO_DIR/.git/$ref" 2>/dev/null || true
  } > "$d/system.txt" 2>&1
  if command -v nvidia-smi >/dev/null 2>&1; then
    { nvidia-smi; nvidia-smi -q 2>/dev/null | head -300; } > "$d/nvidia-smi.txt" 2>&1 || true
  else
    echo "nvidia-smi indisponível" > "$d/nvidia-smi.txt"
  fi
  # variáveis de ambiente, com segredos redigidos
  env | sort | awk -F= '{ if (toupper($1) ~ /(TOKEN|KEY|SECRET|PASS|CREDENTIAL|AUTH)/) print $1"=<redigido>"; else print $0 }' > "$d/env.txt"
  for py in "$REPO_DIR/.venv/bin/python" "$(command -v python3 || true)"; do
    [ -x "$py" ] || continue
    {
      echo "## $py"; "$py" --version 2>&1
      "$py" - <<'EOF' 2>&1
import importlib.metadata as md
for p in ("vllm", "torch", "transformers", "tokenizers", "huggingface_hub", "numpy", "pandas", "pyarrow", "matplotlib", "PyYAML"):
    try:
        print(p, md.version(p))
    except md.PackageNotFoundError:
        print(p, "-")
try:
    import torch
    print("torch.cuda", torch.version.cuda, "disponível:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())
except Exception as e:
    print("torch:", e)
EOF
      "$py" -m pip list --format=freeze 2>/dev/null || true
    } > "$d/python_$(basename "$(dirname "$(dirname "$py")")")_$(basename "$py").txt" 2>&1
  done
  # arquivos pequenos: logs, as 3 execuções mais recentes e o estado das filas
  local list="$d/files.txt"
  : > "$list"
  ( cd "$LUMI_WORKDIR" && find logs -type f -size -20M 2>/dev/null ) >> "$list" || true
  for r in $(ls -1t "$LUMI_WORKDIR/results" 2>/dev/null | head -3); do
    ( cd "$LUMI_WORKDIR" && find "results/$r" -type f -size -20M 2>/dev/null ) >> "$list" || true
    ( cd "$LUMI_WORKDIR" && find "work/$r" -maxdepth 3 -type f \( -name '*.json' -o -name '*.jsonl' -o -name 'PROMPT_KEY' \) -size -20M 2>/dev/null ) >> "$list" || true
  done
  ( cd "$LUMI_WORKDIR" && find work/prompts -maxdepth 2 -name stats.json 2>/dev/null ) >> "$list" || true
  echo "diagnostics/$name" >> "$list"
  local tarball="$out/$name.tar.gz"
  ( cd "$LUMI_WORKDIR" && COPYFILE_DISABLE=1 tar czf "$tarball" -T "diagnostics/$name/files.txt" ) || die "falha ao criar $tarball"
  rm -rf "$d"
  say "Pacote de diagnóstico: $tarball ($(du -h "$tarball" | cut -f1))"
  say "Copie para a sua máquina, por exemplo: scp <usuario>@<servidor>:$tarball ."
}

# ------------------------------------------------------------------ comandos
main() {
  case "$CMD" in
    help|-h|--help) usage ;;
    setup)    setup_env ;;
    check)    setup_env; gpu_banner; run_py check "$@" ;;
    prepare)  setup_env; run_py prepare "$@" ;;
    smoke)    setup_env; gpu_banner; run_py run --mode smoke $(resume_args) "$@" ;;
    full)     setup_env; gpu_banner; run_py run --mode full $(resume_args) "$@" ;;
    analyze)  setup_env; [ $# -ge 1 ] || die "uso: ./run_stage1.sh analyze <run_id>"; run_py analyze --run-id "$@" ;;
    selftest) setup_env; "$PY" -m pytest -q "$REPO_DIR/tests" "$@" ;;
    diagnose) diagnose ;;
    *) usage; die "comando desconhecido: $CMD" ;;
  esac
}

if [ "$CMD" = "help" ] || [ "$CMD" = "-h" ] || [ "$CMD" = "--help" ] || [ "$CMD" = "diagnose" ]; then
  main "$@"
else
  say "comando: $CMD | workdir: $LUMI_WORKDIR | log: $SHELL_LOG"
  main "$@" 2>&1 | tee -a "$SHELL_LOG"
  exit "${PIPESTATUS[0]}"
fi
