#!/bin/bash
# Envia uma etapa do Lumi-Embed ao Sauron. Rode NA SESSÃO DE LOGIN, dentro do clone:
#   bash jobs/embed/submit.sh <comando> [argumentos]      (os comandos do run_embed.sh)
# Exemplos:
#   bash jobs/embed/submit.sh setup
#   bash jobs/embed/submit.sh synth --generator qwen3_30b
#   bash jobs/embed/submit.sh train C1
#   bash jobs/embed/submit.sh eval C2 mteb_br mteb_pt
#   LUMI_PROFILE=small bash jobs/embed/submit.sh eval R0 mteb_br   (classe menor de GPU)
# Gera jobs/embed/.generated/<nome>.{yaml,sh} com o perfil de recursos da etapa e chama
# `sauron submit`. Perfis: "gpu" = B200 (vram 160, ~179 GB da cota); "cpu" = a menor classe
# de GPU (vram 20), para etapas sem GPU — as duas cabem juntas numa cota de 268 GB.
set -euo pipefail
trap 'echo "ERRO: submit.sh falhou na linha $LINENO (comando: $BASH_COMMAND)" >&2' ERR
[ $# -ge 1 ] || { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
STEP="$1"; shift
CLONE="$(cd "$(dirname "$0")/../.." && pwd)"
WS_ROOT="$HOME/workspace"
case "$CLONE" in "$WS_ROOT"/*) ;; *) echo "ERRO: o clone precisa estar dentro de ~/workspace (está em $CLONE)" >&2; exit 2 ;; esac
REL="${CLONE#$WS_ROOT/}"          # ex.: lumi-inference/lumi-to-emb
WORKSPACE="${REL%%/*}"            # ex.: lumi-inference  (pasta em que o job começa)
SUBDIR="${REL#$WORKSPACE}"; SUBDIR="${SUBDIR#/}"   # ex.: lumi-to-emb (vazio se o clone for a própria pasta)

# O guia do Sauron trata > 4 CPUs e > 32 GB de RAM como "limite de atenção" (o job é aceito,
# com aviso). Só passam disso as etapas que carregam o modelo de 4B (ou um gerador/teacher)
# e as que montam índices grandes em memória.
profile=gpu; hours=8; mem=64g; cpus=4
case "$STEP" in
  setup|selftest)          profile=cpu; hours=2; mem=16g ;;
  diagnose)                profile=cpu; hours=1; mem=8g; cpus=2 ;;
  sources)                 profile=cpu; hours=8; mem=64g ;;
  decontam|audit)          profile=cpu; hours=8; mem=64g ;;
  mix|report)              profile=cpu; hours=2; mem=32g ;;
  publish|card)            profile=cpu; hours=3; mem=16g; cpus=2 ;;
  check|export-base|verify) hours=2 ;;
  synth|mine)              hours=12 ;;
  train|baseline)          hours=12 ;;
  eval)                    hours=12; mem=32g ;;   # pico medido no MTEB-BR/PT: 7,4 GB
  merge)                   hours=6; mem=64g ;;    # M-C: ram pico 31,7 GB (cache dos safetensors)
esac
# Suítes grandes: o MTEB eng v2 inclui o MindSmallReranking (2,36 milhões de queries; só carregá-lo
# usou ~31 GB de RAM no decontam, e os embeddings somam ~30 GB em float32). Mais RAM e tempo.
if [ "$STEP" = eval ] || [ "$STEP" = baseline ]; then
  case " $* " in *" eng_v2 "*|*" mmteb "*) mem=128g; hours=24 ;; esac
fi
# LUMI_PROFILE=small força a classe menor de GPU (vram 20) numa etapa que normalmente usa a B200.
# Útil para avaliações (inferência de um modelo de 4B cabe com folga) em paralelo com um treino
# ou síntese na B200: 179 + 89 GB cabem juntos numa cota de 268 GB.
case "${LUMI_PROFILE:-}" in
  "") ;;
  small|cpu) profile=cpu ;;
  gpu) profile=gpu ;;
  *) echo "ERRO: LUMI_PROFILE deve ser 'small' ou 'gpu' (recebido: $LUMI_PROFILE)" >&2; exit 2 ;;
esac
vram=160; [ "$profile" = cpu ] && vram=20
SUFFIX=""
if [ $# -gt 0 ]; then SUFFIX="$(printf -- '-%s' "$@" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-40)"; fi
NAME="lumi-emb-${STEP}${SUFFIX}"
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
plabel="B200, vram 160"; [ "$profile" = cpu ] && plabel="classe menor de GPU, vram 20"
echo "Job: $NAME ($plabel, até ${hours}h)  →  $GEN/$NAME.yaml"
cd "$CLONE"
sauron submit -c "$GEN/$NAME.yaml" "$GEN/$NAME.sh"
