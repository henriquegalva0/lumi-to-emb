#!/bin/bash
# Comandos do job (rodam dentro da imagem). O job começa na pasta de `workspace:`;
# o clone pode ser essa pasta ou uma subpasta dela (ex.: lumi-inference/lumi-to-emb).
set -e
if [ ! -f run_stage1.sh ]; then
  found=$(ls -1 */run_stage1.sh 2>/dev/null | head -1)
  [ -n "$found" ] || { echo "ERRO: run_stage1.sh não encontrado em $(pwd) nem nas subpastas" >&2; exit 2; }
  cd "$(dirname "$found")"
fi
bash run_stage1.sh smoke
