# Imagem opcional e autocontida: a imagem oficial do vLLM + os extras fixados.
# O caminho padrão (run_stage1.sh) não precisa dela: instala os extras num venv
# sobre a imagem oficial. Use-a se preferir um ambiente pronto, sem instalação
# no início do job.
#
#   docker build -f docker/Dockerfile -t <usuario>/lumi-ctx:1.0.0 .
#   docker run --gpus all --rm -v "$PWD":/workspace/lumi-inference \
#       -w /workspace/lumi-inference <usuario>/lumi-ctx:1.0.0 bash run_stage1.sh smoke
FROM vllm/vllm-openai:v0.29.0

COPY requirements.txt /opt/lumi-ctx/requirements.txt
# Restringe os pacotes que a imagem já traz, para o pip não trocá-los.
RUN python3 -m pip list --format=freeze \
      | grep -iE '^(vllm|torch|torchvision|torchaudio|transformers|tokenizers|huggingface[-_]hub|numpy|triton|safetensors|pyyaml|pillow|fsspec)==' \
      > /opt/lumi-ctx/constraints.txt \
 && python3 -m pip install --no-cache-dir --break-system-packages \
      -r /opt/lumi-ctx/requirements.txt -c /opt/lumi-ctx/constraints.txt

WORKDIR /workspace/lumi-inference
ENTRYPOINT []
CMD ["bash", "run_stage1.sh", "help"]
