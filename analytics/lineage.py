"""
analytics/lineage.py — Hand-built Graphviz DOT for a run's data lineage.

Answers "where did this data come from and what happened to it?" as a
left-to-right flow: source → each transform stage → DuckDB table → (optional)
mart.  The DOT string is assembled BY HAND — no graphviz pip package and no
Graphviz binary is required, because ``st.graphviz_chart`` accepts DOT source
text directly.

Everything is read from a run's saved artifacts (profile + docs); this never
touches the database, so it works on any historical run.
"""

from __future__ import annotations

# Design-system colors (kept literal here so the module has no UI dependency).
_EMERALD_FILL = "#ECFDF5"   # verified / persisted states (source, table, mart)
_EMERALD_LINE = "#059669"
_SLATE_LINE = "#E2E8F0"     # transform nodes (work-in-progress, neutral)
_SKY = "#0284c7"            # informational — the flow edges


def _esc(text: object) -> str:
    """Escape a value for safe inclusion in a DOT double-quoted label.

    Backslashes first (so we don't double-escape our own escapes), then quotes.
    Callers join multi-line labels with the literal ``\\n`` DOT line break AFTER
    escaping each part, so real newlines never reach this function.
    """
    return str(text).replace("\\", "\\\\").replace('"', '\\"')


def _label(*parts: object) -> str:
    """Build a DOT label from parts, escaped and joined with DOT line breaks."""
    return "\\n".join(_esc(p) for p in parts if p is not None and str(p) != "")


def build_lineage_dot(run_artifacts: dict, mart_involved: bool = False) -> str:
    """Assemble a Graphviz DOT string describing a run's lineage.

    Nodes:
        source  — dataset name + source type + input row count (emerald = origin)
        t0..tN  — one per transform step (white, slate border), label = action +
                  detail headline (the detail carries the row-count story)
        table   — the DuckDB output table + output row count (emerald)
        mart    — optional reporting_state_health node when *mart_involved*

    Args:
        run_artifacts: A run's artifacts dict (expects ``profile`` and ``docs``).
        mart_involved: When True, append a mart node fed by the table.

    Returns:
        A DOT source string suitable for ``st.graphviz_chart(dot)``.
    """
    docs = run_artifacts.get("docs") or {}
    profile = run_artifacts.get("profile") or {}
    lineage = docs.get("lineage") or {}
    source = lineage.get("source") or {}
    schema = docs.get("schema") or {}
    transforms = lineage.get("transformations") or []

    dataset_name = (
        lineage.get("dataset_name")
        or docs.get("dataset_name")
        or "dataset"
    )

    # --- Source node ---
    src_name = source.get("name") or dataset_name
    src_type = source.get("source_type") or source.get("type") or "source"
    # Prefer the source's own extracted count; fall back to the raw profile rows.
    src_rows = source.get("last_record_count")
    if not isinstance(src_rows, int) or src_rows < 0:
        src_rows = (profile.get("overview") or {}).get("row_count")
    src_rows_txt = f"{src_rows:,} rows" if isinstance(src_rows, int) and src_rows >= 0 else None
    src_label = _label(src_name, f"({src_type})", src_rows_txt)

    # --- Table node ---
    table_name = schema.get("table_name") or dataset_name
    out_rows = schema.get("row_count")
    out_rows_txt = f"{out_rows:,} rows" if isinstance(out_rows, int) else None
    table_label = _label(f"DuckDB: {table_name}", out_rows_txt)

    # --- Assemble DOT ---
    lines: list[str] = [
        "digraph lineage {",
        "  rankdir=LR;",
        "  bgcolor=\"transparent\";",
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10];',
        f'  edge [color="{_SKY}"];',  # flow is informational → sky
        f'  src [label="{src_label}" fillcolor="{_EMERALD_FILL}" color="{_EMERALD_LINE}"];',
    ]

    prev = "src"
    for i, step in enumerate(transforms):
        node_id = f"t{i}"
        action = step.get("action", "step")
        detail = step.get("detail", "")
        node_label = _label(action, detail)
        lines.append(
            f'  {node_id} [label="{node_label}" '
            f'fillcolor="white" color="{_SLATE_LINE}"];'
        )
        lines.append(f"  {prev} -> {node_id};")
        prev = node_id

    lines.append(
        f'  tbl [label="{table_label}" '
        f'fillcolor="{_EMERALD_FILL}" color="{_EMERALD_LINE}"];'
    )
    lines.append(f"  {prev} -> tbl;")

    if mart_involved:
        mart_label = _label("reporting_state_health", "(state health mart)")
        lines.append(
            f'  mart [label="{mart_label}" '
            f'fillcolor="{_EMERALD_FILL}" color="{_EMERALD_LINE}"];'
        )
        lines.append("  tbl -> mart;")

    lines.append("}")
    return "\n".join(lines)
