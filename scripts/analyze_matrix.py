import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIG_ORDER = ["sequential", "static-maxlen", "static-exact", "continuous-maxlen", "continuous-exact"]


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dig(summary: dict[str, Any], key: str) -> float | None:
    head, _, tail = key.partition(".")
    value = summary[head]
    if tail:
        return None if value is None else value[tail]
    return value


def cell(rows: list[dict[str, Any]], key: str, digits: int = 1) -> str:
    values = [v for r in rows if (v := dig(r["summary"], key)) is not None]
    if not values:
        return "n/a"
    mean = statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})"


def table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def matrix_section(rows: list[dict[str, Any]], key: str, title: str, digits: int = 1) -> str:
    grouped: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["mult"], r["config"])].append(r)
    mults = sorted({m for m, _ in grouped})
    header = ["Config"] + [f"{m:g}x" for m in mults]
    body = [
        [name] + [cell(grouped.get((m, name), []), key, digits) for m in mults]
        for name in CONFIG_ORDER
        if any((m, name) in grouped for m in mults)
    ]
    return f"### {title}\n\n{table(header, body)}\n"


def sweep_section(rows: list[dict[str, Any]], key: str, title: str, digits: int = 1) -> str:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[(r["pool_blocks"], r["config"])].append(r)
    pools = sorted({p for p, _ in grouped})
    header = ["Config"] + [f"{p} blocks" for p in pools]
    body = [
        [name] + [cell(grouped.get((p, name), []), key, digits) for p in pools]
        for name in CONFIG_ORDER
        if any((p, name) in grouped for p in pools)
    ]
    return f"### {title}\n\n{table(header, body)}\n"


def report(rows: list[dict[str, Any]]) -> str:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    env = by_kind["environment"][0]
    out = [
        "## Run identity\n",
        f"- Code: `{env['git_sha'][:7]}`, chip {env['chip']}, torch {env['torch']}, "
        f"device {env['device']}, dtype {env['dtype']}",
        f"- {env['requests_per_run']} requests per cell, {env['reps']} repetitions, base seed {env['base_seed']}",
        f"- Block size {env['block_size']}, max_model_len {env['max_model_len']}, "
        f"max_new_tokens {env['max_new_tokens']}, static wait {env['static_wait_s']} s",
    ]
    if by_kind["calibration"]:
        c = by_kind["calibration"][0]
        out.append(
            f"- Sequential capacity: {c['capacity_requests_per_s']:.3f} requests/s, "
            f"{c['summary']['tokens_per_s']:.1f} output tokens/s, "
            f"mean prompt {c['workload']['mean_prompt_tokens']:.0f} tokens"
        )
    out.append("")
    if by_kind["paging_overhead"]:
        p = by_kind["paging_overhead"][0]
        slow = 1 - p["paged_tokens_per_s"] / p["contiguous_tokens_per_s"]
        out.append(
            f"## Paging overhead at batch 1\n\n{p['requests']} requests: contiguous cache "
            f"{p['contiguous_tokens_per_s']:.1f} tokens/s, paged engine {p['paged_tokens_per_s']:.1f} tokens/s "
            f"({slow * 100:+.1f}% slower for paged). Single run, no repetitions.\n"
        )
    matrix = by_kind["matrix"]
    if matrix:
        out.append("## Arrival-rate matrix\n\nCells show the mean across repetitions, with the range in parentheses.\n")
        out.append(matrix_section(matrix, "tokens_per_s", "Output tokens per second"))
        out.append(matrix_section(matrix, "ttft_ms.p50", "Time to first token, median (ms)", 0))
        out.append(matrix_section(matrix, "ttft_ms.p99", "Time to first token, p99 (ms)", 0))
        out.append(matrix_section(matrix, "itl_ms.p50", "Inter-token latency, median (ms)"))
        out.append(matrix_section(matrix, "itl_ms.p99", "Inter-token latency, p99 (ms)"))
        out.append(matrix_section(matrix, "queue_wait_ms.p50", "Queue wait, median (ms)", 0))
        out.append(matrix_section(matrix, "mean_running", "Mean sequences running per step", 2))
        out.append(matrix_section(matrix, "mean_held_rows", "Mean finished-but-held rows per step", 2))
        out.append(matrix_section(matrix, "rejection_rate", "Rejection rate", 3))
    sweep = by_kind["sweep"]
    if sweep:
        out.append("## Pool-size sweep at the highest arrival rate\n")
        out.append(sweep_section(sweep, "tokens_per_s", "Output tokens per second"))
        out.append(sweep_section(sweep, "queue_wait_ms.p50", "Queue wait, median (ms)", 0))
        out.append(sweep_section(sweep, "ttft_ms.p99", "Time to first token, p99 (ms)", 0))
        out.append(sweep_section(sweep, "mean_running", "Mean sequences running per step", 2))
        out.append(sweep_section(sweep, "mean_reserved_blocks", "Mean blocks reserved", 1))
        out.append(sweep_section(sweep, "mean_used_blocks", "Mean blocks holding tokens", 1))
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    print(report(load(parser.parse_args().path)))


if __name__ == "__main__":
    main()
