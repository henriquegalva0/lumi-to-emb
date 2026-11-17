"""Fila de trabalho em disco, compartilhada pelos workers (um por GPU).

Cada lote ("chunk") é um arquivo JSON. Os estados são codificados no nome:
    0007.todo.json  ->  0007.run.<gpu>.json  ->  0007.done.json
A passagem de estado usa os.rename, que é atômico no mesmo sistema de arquivos:
dois workers nunca pegam o mesmo lote. Lotes de um worker que caiu voltam para
"todo" e são refeitos, o que torna a execução retomável.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def create_chunks(qdir: Path, index: list[dict], chunk_tokens: int, max_prompts: int) -> int:
    """Divide os prompts em lotes (mais longos primeiro). Não recria se já existir."""
    qdir.mkdir(parents=True, exist_ok=True)
    if any(qdir.glob("*.json")):
        return len(list(qdir.glob("*.json")))
    order = sorted(range(len(index)), key=lambda i: (-index[i]["n_tokens"], index[i]["uid"]))
    chunks: list[list[int]] = []
    current: list[int] = []
    tokens = 0
    for i in order:
        n = index[i]["n_tokens"]
        if current and (tokens + n > chunk_tokens or len(current) >= max_prompts):
            chunks.append(current)
            current, tokens = [], 0
        current.append(i)
        tokens += n
    if current:
        chunks.append(current)
    for k, items in enumerate(chunks):
        payload = {"chunk": f"{k:04d}", "items": items, "tokens": int(sum(index[i]["n_tokens"] for i in items))}
        tmp = qdir / f"{k:04d}.tmp"
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, qdir / f"{k:04d}.todo.json")
    return len(chunks)


def claim(qdir: Path, label: str) -> tuple[Path, dict] | None:
    for path in sorted(qdir.glob("*.todo.json")):
        chunk_id = path.name.split(".", 1)[0]
        target = qdir / f"{chunk_id}.run.{label}.json"
        try:
            os.rename(path, target)
        except (FileNotFoundError, OSError):
            continue  # outro worker pegou antes
        return target, json.loads(target.read_text())
    return None


def complete(run_path: Path) -> None:
    chunk_id = run_path.name.split(".", 1)[0]
    os.replace(run_path, run_path.parent / f"{chunk_id}.done.json")


def reset_running(qdir: Path, label: str | None = None) -> int:
    """Devolve à fila os lotes em execução (de um worker, ou de todos)."""
    pattern = f"*.run.{label}.json" if label else "*.run.*.json"
    n = 0
    for path in qdir.glob(pattern):
        chunk_id = path.name.split(".", 1)[0]
        try:
            os.rename(path, qdir / f"{chunk_id}.todo.json")
            n += 1
        except OSError:
            pass
    return n


def status(qdir: Path) -> dict:
    st = {"todo": 0, "run": 0, "done": 0, "prompts_done": 0, "prompts_total": 0, "tokens_done": 0, "tokens_total": 0}
    for path in qdir.glob("*.json"):
        kind = path.name.split(".")[1]
        payload = json.loads(path.read_text())
        st[kind] = st.get(kind, 0) + 1
        st["prompts_total"] += len(payload["items"])
        st["tokens_total"] += payload["tokens"]
        if kind == "done":
            st["prompts_done"] += len(payload["items"])
            st["tokens_done"] += payload["tokens"]
    return st
