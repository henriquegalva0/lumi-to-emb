#!/bin/bash
# Envia uma etapa do Lumi-Embed ao Sauron. Rode NA SESSÃO DE LOGIN, dentro do clone:
#   bash jobs/embed/submit.sh <comando> [argumentos]      (os comandos do run_embed.sh)
# Exemplos:
#   bash jobs/embed/submit.sh setup
#   bash jobs/embed/submit.sh synth --generator qwen3_30b
#   bash jobs/embed/submit.sh train C1
#   bash jobs/embed/submit.sh eval C2 mteb_br mteb_pt
# Gera jobs/embed/.generated/<nome>.{yaml,sh} com o perfil de recursos da etapa e chama
# `sauron submit`. Perfis: "gpu" = B200 (vram 160, ~179 GB da cota); "cpu" = a menor classe
# de GPU (vram 20), para etapas sem GPU — as duas cabem juntas numa cota de 268 GB.
set -euo pipefail
[ $# -ge 1 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
STEP="$1"; shift
CLONE="$(cd "$(dirname "$0")/../.." && pwd)"
WS_ROOT="$HOME/workspace"
case "$CLONE" in "$WS_ROOT"/*) ;; *) echo "ERRO: o clone precisa estar dentro de ~/workspace (está em $CLONE)" >&2; exit 2 ;; esac
REL="${CLONE#$WS_ROOT/}"          # ex.: lumi-inference/lumi-to-emb
WORKSPACE="${REL%%/*}"            # ex.: lumi-inference  (pasta em que o job começa)
SUBDIR="${REL#$WORKSPACE}"; SUBDIR="${SUBDIR#/}"   # ex.: lumi-to-emb (vazio se o clone for a própria pasta)

profile=gpu; hours=8; mem=96g; cpus=8
case "$STEP" in
  setup|selftest)          profile=cpu; hours=2 ;;
  sources)                 profile=cpu; hours=8; mem=64g ;;
  decontam)                profile=cpu; hours=8; mem=96g ;;
  mix|report)              profile=cpu; hours=2; mem=64g ;;
  publish|card)            profile=cpu; hours=3; mem=16g; cpus=2 ;;
  check|export-base|verify) hours=2 ;;
  synth|mine)              hours=12 ;;
  train|eval|baseline)     hours=12 ;;
  merge)                   hours=6 ;;
esac
vram=160; [ "$profile" = cpu ] && vram=20
NAME="lumi-emb-${STEP}$( [ $# -gt 0 ] && printf -- '-%s' "$@" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40 )"
NAME="$(echo "$NAME" | sed 's/--*/-/g; s/-$//')"
GEN="$CLONE/jobs/embed/.generated"
mkdir -p "$GEN"
cat > "$GEN/$NAME.yaml" <<YAML
# Gerado por jobs/embed/submit.sh — $(date '+%Y-%m-%d %H:%M')
name: $NAME
image: vllm/vllm-openai:v0.29.0
gpus: 1
vram: $vram
cpus: $cpus
memory: $mem
max_time: ${hours}h
workspace: $WORKSPACE
YAML
ARGS=""; for a in "$@"; do ARGS+=" $(printf '%q' "$a")"; done
cat > "$GEN/$NAME.sh" <<SH
#!/bin/bash
set -e
${SUBDIR:+cd $(printf '%q' "$SUBDIR")}
bash run_embed.sh $(printf '%q' "$STEP")$ARGS
SH
echo "Job: $NAME (perfil $profile, até ${hours}h)  →  $GEN/$NAME.yaml"
cd "$CLONE"
sauron submit -c "$GEN/$NAME.yaml" "$GEN/$NAME.sh"
