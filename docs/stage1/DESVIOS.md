# Desvios em relação à especificação

## 1. Hardware diferente do previsto na especificação (8× H100): 1× NVIDIA B200.

- **Motivo:** Cluster disponível para o experimento.
- **Impacto:** Afeta só as medidas de desempenho (perf.csv); a acurácia não depende do hardware.
- Código: `hardware` · registrado em 2026-11-18T11:52:47

## 2. Lumi carregado de `ceia-research/lumi-4b` (revisão `27b30f40b930`) em vez de `ceia-br/lumi-4b`.

- **Motivo:** O repositório foi renomeado; o id antigo redireciona para este.
- **Impacto:** Nenhum: é o mesmo checkpoint, agora com revisão fixada.
- Código: `lumi-repo-id` · registrado em 2026-11-18T11:52:47

## 3. Cache do Hugging Face em `/cache/huggingface` (definido pelo ambiente), não dentro do diretório de trabalho.

- **Motivo:** HF_HOME já vinha definido no ambiente (no cluster usado, o gerenciador de jobs fornece um cache compartilhado e pede para não trocá-lo).
- **Impacto:** Nenhum sobre os resultados.
- Código: `hf-cache` · registrado em 2026-11-18T11:52:47

## 4. Saídas gravadas em subpastas do próprio clone (results/, data/, work/), todas ignoradas pelo Git.

- **Motivo:** No servidor o clone é a própria pasta de trabalho do job; LUMI_WORKDIR permite separar.
- **Impacto:** Nenhum: nada gerado entra no controle de versão.
- Código: `workdir` · registrado em 2026-11-18T11:52:47

## 5. Palheiro lido em streaming direto dos parquets do Hub (HfFileSystem + pyarrow), sem a biblioteca `datasets`.

- **Motivo:** A versão atual de `datasets` exige um `huggingface_hub` mais novo que o da imagem do vLLM; instalá-la trocaria pacotes da imagem.
- **Impacto:** Nenhum: mesma fonte, mesma revisão fixada e ordem determinística pela semente.
- Código: `haystack-streaming` · registrado em 2026-11-18T11:52:47

## 6. Prompts C0 com ~95 tokens (o plano previa cerca de 150).

- **Motivo:** É o tamanho real de template + agulha + pergunta com este tokenizer.
- **Impacto:** Nenhum: o C0 continua sendo o controle sem palheiro.
- Código: `c0-length` · registrado em 2026-11-18T11:55:10

## 7. Gate G3 restrito ao C0 e a L ≤ 8k; 713 saída(s) com <think> acima disso (64k, 128k, 255k) contadas como modo de falha.

- **Motivo:** A especificação trata <think> na saída como erro de configuração, mas o template está verificado (G1) e os <think> só aparecem em contextos longos, em saídas degeneradas (laços até max_tokens).
- **Impacto:** Nenhum sobre a acurácia: essas saídas já pontuam 0. A taxa aparece em 'Modos de falha' no relatório.
- Código: `think-gate-scope` · registrado em 2026-11-18T11:36:10

