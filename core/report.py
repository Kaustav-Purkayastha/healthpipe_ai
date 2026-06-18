"""
report.py — HTML report generator.

Produces a single self-contained HTML file (inline CSS, no external
dependencies) that renders the full pipeline output as a styled dashboard:
    - Summary cards (rows, columns, quality grade, completeness)
    - Quality scorecard with pass/fail indicators
    - Data dictionary table
    - Column profile statistics
    - Transformation audit trail
    - Correlations and quality issues

The HTML opens in any browser, prints cleanly, and screenshots well
for portfolios.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import REPORTS_DIR
from core.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CSS — all styles inline so the HTML is fully self-contained
# ---------------------------------------------------------------------------

_CSS = """
:root {
    --primary: #1a365d;
    --primary-light: #2a4a7f;
    --accent: #38a169;
    --accent-warn: #e53e3e;
    --accent-caution: #d69e2e;
    --bg: #f7fafc;
    --card-bg: #ffffff;
    --text: #2d3748;
    --text-light: #718096;
    --border: #e2e8f0;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
    --radius: 8px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Header */
.header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: #fff;
    padding: 2rem 2.5rem;
}
.header h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
.header .subtitle {
    font-size: 0.95rem; opacity: 0.85; margin-top: 0.25rem;
}
.header .meta {
    display: flex; gap: 2rem; margin-top: 1rem; font-size: 0.85rem; opacity: 0.75;
}

/* Container */
.container { max-width: 1200px; margin: 0 auto; padding: 1.5rem 2rem 3rem; }

/* Summary cards */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
}
.card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    box-shadow: var(--shadow);
    border-top: 3px solid var(--primary);
}
.card .label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-light); margin-bottom: 0.25rem;
}
.card .value { font-size: 1.75rem; font-weight: 700; color: var(--primary); }
.card .detail { font-size: 0.8rem; color: var(--text-light); margin-top: 0.25rem; }

/* Grade-specific card colors */
.card.grade-a { border-top-color: var(--accent); }
.card.grade-a .value { color: var(--accent); }
.card.grade-b { border-top-color: #3182ce; }
.card.grade-b .value { color: #3182ce; }
.card.grade-c { border-top-color: var(--accent-caution); }
.card.grade-c .value { color: var(--accent-caution); }
.card.grade-f { border-top-color: var(--accent-warn); }
.card.grade-f .value { color: var(--accent-warn); }

/* Sections */
.section {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    margin-bottom: 1.5rem;
    overflow: hidden;
}
.section-header {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.5rem;
}
.section-header h2 {
    font-size: 1.1rem; font-weight: 600; color: var(--primary);
}
.section-header .badge {
    font-size: 0.7rem; background: var(--primary); color: #fff;
    padding: 0.15rem 0.5rem; border-radius: 10px; font-weight: 600;
}
.section-body { padding: 1.25rem 1.5rem; }

/* Tables */
table {
    width: 100%; border-collapse: collapse; font-size: 0.875rem;
}
thead th {
    text-align: left; padding: 0.75rem 1rem;
    background: #f1f5f9; color: var(--primary);
    font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.03em; border-bottom: 2px solid var(--border);
}
tbody td {
    padding: 0.65rem 1rem; border-bottom: 1px solid var(--border);
    vertical-align: top;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #f8fafc; }

/* Status badges */
.badge-pass {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 10px;
    background: #c6f6d5; color: #22543d;
}
.badge-fail {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 10px;
    background: #fed7d7; color: #742a2a;
}
.badge-info {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 10px;
    background: #bee3f8; color: #2a4365;
}
.badge-warn {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 10px;
    background: #fefcbf; color: #744210;
}
.badge-critical {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 0.15rem 0.5rem; border-radius: 10px;
    background: #fed7d7; color: #742a2a;
}

/* Audit log step numbers */
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 1.5rem; height: 1.5rem; border-radius: 50%;
    background: var(--primary); color: #fff; font-size: 0.7rem;
    font-weight: 700; flex-shrink: 0;
}

/* Mono text for types and values */
.mono { font-family: "SF Mono", "Fira Code", Consolas, monospace; font-size: 0.82rem; }

/* Quality score bar */
.score-bar-container {
    width: 100%; background: #e2e8f0; border-radius: 6px;
    height: 8px; margin-top: 0.4rem; overflow: hidden;
}
.score-bar {
    height: 100%; border-radius: 6px; transition: width 0.5s ease;
}
.score-bar.grade-a { background: var(--accent); }
.score-bar.grade-b { background: #3182ce; }
.score-bar.grade-c { background: var(--accent-caution); }
.score-bar.grade-f { background: var(--accent-warn); }

/* Usage notes */
.note-item {
    padding: 0.6rem 0; border-bottom: 1px solid var(--border);
    font-size: 0.9rem;
}
.note-item:last-child { border-bottom: none; }
.note-item::before {
    content: "\\2022"; color: var(--primary); font-weight: 700;
    margin-right: 0.5rem;
}

/* Correlation bar */
.corr-bar {
    display: inline-block; height: 6px; border-radius: 3px;
    background: var(--primary); vertical-align: middle; margin-left: 0.5rem;
}

/* Empty state */
.empty-state {
    text-align: center; padding: 2rem; color: var(--text-light);
    font-size: 0.9rem;
}

/* Footer */
.footer {
    text-align: center; padding: 1.5rem; color: var(--text-light);
    font-size: 0.8rem; border-top: 1px solid var(--border); margin-top: 1rem;
}

/* Print styles */
@media print {
    body { background: #fff; }
    .header { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .section { box-shadow: none; border: 1px solid var(--border); }
    .card { box-shadow: none; border: 1px solid var(--border); }
}
"""


class ReportGenerator:
    """
    Generates a self-contained HTML dashboard from pipeline outputs.

    Usage:
        report = ReportGenerator()
        report.generate(
            dataset_name="who_life_expectancy",
            overview={...},          # from ProfilerAgent
            scorecard={...},         # from QualityCheckerAgent
            data_dictionary=[...],   # from DocumenterAgent
            column_profiles={...},   # from ProfilerAgent
            transform_log=[...],     # from TransformerAgent
            quality_issues=[...],    # from ProfilerAgent
            correlations=[...],      # from ProfilerAgent
            usage_notes=[...],       # from DocumenterAgent
        )
    """

    def generate(
        self,
        dataset_name: str,
        overview: dict | None = None,
        scorecard: dict | None = None,
        data_dictionary: list[dict] | None = None,
        column_profiles: dict | None = None,
        transform_log: list[dict] | None = None,
        quality_issues: list[dict] | None = None,
        correlations: list[dict] | None = None,
        usage_notes: list[str] | None = None,
    ) -> Path:
        """
        Build and save the HTML report.

        All parameters except dataset_name are optional — the report
        gracefully handles missing sections with "No data available"
        messages.

        Args:
            dataset_name:    Name of the dataset (shown in header).
            overview:        High-level stats from ProfilerAgent.
            scorecard:       Quality scorecard from QualityCheckerAgent.
            data_dictionary: Column docs from DocumenterAgent.
            column_profiles: Per-column stats from ProfilerAgent.
            transform_log:   Audit trail from TransformerAgent.
            quality_issues:  Flagged issues from ProfilerAgent.
            correlations:    Strong correlations from ProfilerAgent.
            usage_notes:     Tips from DocumenterAgent.

        Returns:
            Path to the saved HTML file.
        """
        # Default empty values so every section renders cleanly
        overview = overview or {}
        scorecard = scorecard or {}
        data_dictionary = data_dictionary or []
        column_profiles = column_profiles or {}
        transform_log = transform_log or []
        quality_issues = quality_issues or []
        correlations = correlations or []
        usage_notes = usage_notes or []

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        grade = scorecard.get("grade", "N/A")

        # Build the full HTML from sections
        parts: list[str] = [
            self._html_open(dataset_name, generated_at),
            self._build_header(dataset_name, generated_at, overview),
            '<div class="container">',
            self._build_summary_cards(overview, scorecard),
            self._build_quality_scorecard(scorecard),
            self._build_data_dictionary(data_dictionary),
            self._build_column_profiles(column_profiles),
            self._build_transform_log(transform_log),
            self._build_quality_issues(quality_issues),
            self._build_correlations(correlations),
            self._build_usage_notes(usage_notes),
            self._build_footer(dataset_name, generated_at),
            "</div>",
            "</body></html>",
        ]

        html = "\n".join(parts)

        # Save to outputs/reports/
        output_path = REPORTS_DIR / f"report_{dataset_name}.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report saved to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # HTML skeleton
    # ------------------------------------------------------------------

    def _html_open(self, dataset_name: str, generated_at: str) -> str:
        """Opening HTML tags with embedded CSS."""
        # html.escape is not needed here — dataset_name comes from our
        # own pipeline, not from user-supplied HTML content
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>HealthPipe AI — {_esc(dataset_name)}</title>\n"
            f"<style>{_CSS}</style>\n"
            "</head>\n<body>"
        )

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _build_header(
        self, dataset_name: str, generated_at: str, overview: dict
    ) -> str:
        """Top banner with dataset name, source, and timestamp."""
        rows = overview.get("row_count", "—")
        cols = overview.get("column_count", "—")
        mem = overview.get("memory_usage_mb", "—")

        return (
            '<div class="header">\n'
            f"  <h1>HealthPipe AI Report</h1>\n"
            f'  <div class="subtitle">Dataset: {_esc(dataset_name)}</div>\n'
            f'  <div class="meta">\n'
            f"    <span>{rows} rows &times; {cols} columns</span>\n"
            f"    <span>{mem} MB in memory</span>\n"
            f"    <span>Generated {generated_at}</span>\n"
            f"  </div>\n"
            "</div>"
        )

    # ------------------------------------------------------------------
    # Summary cards
    # ------------------------------------------------------------------

    def _build_summary_cards(self, overview: dict, scorecard: dict) -> str:
        """Row of metric cards at the top of the report."""
        grade = scorecard.get("grade", "N/A")
        score = scorecard.get("score", "—")
        grade_class = f"grade-{grade.lower()}" if grade in "ABCF" else ""

        cards = [
            self._card(
                "Rows",
                f"{overview.get('row_count', '—'):,}" if isinstance(
                    overview.get("row_count"), int
                ) else str(overview.get("row_count", "—")),
                "Total records in dataset",
            ),
            self._card(
                "Columns",
                str(overview.get("column_count", "—")),
                "Fields per record",
            ),
            self._card(
                "Quality Grade",
                grade,
                f"{score}% — {scorecard.get('checks_passed', '?')}/"
                f"{scorecard.get('total_checks', '?')} checks passed",
                extra_class=grade_class,
            ),
            self._card(
                "Completeness",
                f"{overview.get('completeness_score', '—')}%",
                "Non-null cells",
            ),
            self._card(
                "Duplicates",
                str(overview.get("duplicate_rows", "—")),
                f"{overview.get('duplicate_percentage', '—')}% of rows",
            ),
            self._card(
                "Memory",
                f"{overview.get('memory_usage_mb', '—')} MB",
                "In-memory footprint",
            ),
        ]

        return f'<div class="cards">\n{"".join(cards)}\n</div>'

    def _card(
        self,
        label: str,
        value: str,
        detail: str,
        extra_class: str = "",
    ) -> str:
        """Single summary card."""
        cls = f"card {extra_class}".strip()
        return (
            f'<div class="{cls}">\n'
            f'  <div class="label">{_esc(label)}</div>\n'
            f'  <div class="value">{_esc(value)}</div>\n'
            f'  <div class="detail">{_esc(detail)}</div>\n'
            f"</div>\n"
        )

    # ------------------------------------------------------------------
    # Quality scorecard
    # ------------------------------------------------------------------

    def _build_quality_scorecard(self, scorecard: dict) -> str:
        """Quality checks table with pass/fail badges."""
        checks = scorecard.get("checks", [])
        if not checks:
            return self._empty_section("Quality Scorecard", "No checks available")

        grade = scorecard.get("grade", "N/A")
        score = scorecard.get("score", 0)
        grade_class = f"grade-{grade.lower()}" if grade in "ABCF" else ""

        # Score bar at top
        score_bar = (
            f'<div style="margin-bottom: 1rem;">'
            f'  <div style="display:flex; justify-content:space-between; '
            f'font-size:0.85rem; margin-bottom:0.25rem;">'
            f'    <span>Overall Score</span>'
            f'    <strong>{score}%</strong>'
            f'  </div>'
            f'  <div class="score-bar-container">'
            f'    <div class="score-bar {grade_class}" '
            f'style="width:{score}%"></div>'
            f'  </div>'
            f'</div>'
        )

        # Group checks: failed first, then passed
        failed = [c for c in checks if not c.get("passed")]
        passed = [c for c in checks if c.get("passed")]
        ordered = failed + passed

        rows: list[str] = []
        for check in ordered:
            status = check.get("passed", False)
            badge = (
                '<span class="badge-pass">PASS</span>'
                if status
                else '<span class="badge-fail">FAIL</span>'
            )
            rows.append(
                f"<tr>\n"
                f"  <td>{badge}</td>\n"
                f'  <td class="mono">{_esc(str(check.get("check", "")))}</td>\n'
                f"  <td>{_esc(str(check.get('detail', '')))}</td>\n"
                f'  <td class="mono">{_esc(str(check.get("value", "")))}</td>\n'
                f'  <td class="mono">{_esc(str(check.get("threshold", "")))}</td>\n'
                f"</tr>"
            )

        total = scorecard.get("total_checks", 0)
        passed_count = scorecard.get("checks_passed", 0)

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Quality Scorecard</h2>\n'
            f'    <span class="badge">{passed_count}/{total} passed</span>\n'
            f"  </div>\n"
            f'  <div class="section-body">\n'
            f"    {score_bar}\n"
            f"    <table>\n"
            f"      <thead><tr>\n"
            f"        <th>Status</th><th>Check</th><th>Detail</th>"
            f"<th>Value</th><th>Threshold</th>\n"
            f"      </tr></thead>\n"
            f'      <tbody>\n{"".join(rows)}\n      </tbody>\n'
            f"    </table>\n"
            f"  </div>\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Data dictionary
    # ------------------------------------------------------------------

    def _build_data_dictionary(self, dictionary: list[dict]) -> str:
        """Data dictionary table with column metadata and descriptions."""
        if not dictionary:
            return self._empty_section("Data Dictionary", "No dictionary available")

        rows: list[str] = []
        for col in dictionary:
            nullable = "Yes" if col.get("nullable") else "No"
            samples = ", ".join(
                str(s) for s in col.get("sample_values", [])[:3]
            )
            # Truncate long sample strings for clean display
            if len(samples) > 60:
                samples = samples[:57] + "..."
            rows.append(
                f"<tr>\n"
                f'  <td><strong>{_esc(col.get("column_name", ""))}</strong></td>\n'
                f'  <td class="mono">{_esc(col.get("data_type", ""))}</td>\n'
                f"  <td>{nullable}</td>\n"
                f'  <td>{col.get("null_count", 0)}</td>\n'
                f'  <td>{col.get("unique_count", 0)}</td>\n'
                f'  <td class="mono" style="max-width:180px;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;" '
                f'title="{_esc(samples)}">{_esc(samples)}</td>\n'
                f'  <td>{_esc(col.get("description", ""))}</td>\n'
                f"</tr>"
            )

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Data Dictionary</h2>\n'
            f'    <span class="badge">{len(dictionary)} columns</span>\n'
            f"  </div>\n"
            f"  <table>\n"
            f"    <thead><tr>\n"
            f"      <th>Column</th><th>Type</th><th>Nullable</th>"
            f"<th>Nulls</th><th>Unique</th><th>Samples</th><th>Description</th>\n"
            f"    </tr></thead>\n"
            f'    <tbody>\n{"".join(rows)}\n    </tbody>\n'
            f"  </table>\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Column profiles
    # ------------------------------------------------------------------

    def _build_column_profiles(self, profiles: dict) -> str:
        """Per-column statistics — numeric stats, string stats, etc."""
        if not profiles:
            return self._empty_section(
                "Column Profiles", "No profile data available"
            )

        rows: list[str] = []
        for col_name, stats in profiles.items():
            profile_type = stats.get("profile_type", "unknown")
            dtype = stats.get("dtype", "—")
            null_pct = stats.get("null_percentage", 0)

            # Build the stats detail based on column type
            if profile_type == "numeric":
                detail = (
                    f'Mean: {stats.get("mean", "—")} &nbsp;|&nbsp; '
                    f'Median: {stats.get("median", "—")} &nbsp;|&nbsp; '
                    f'Std: {stats.get("std", "—")} &nbsp;|&nbsp; '
                    f'Range: [{stats.get("min", "—")}, {stats.get("max", "—")}]'
                )
                outliers = stats.get("outlier_count", 0)
                outlier_badge = (
                    f' <span class="badge-warn">{outliers} outliers</span>'
                    if outliers > 0 else ""
                )
            elif profile_type == "string":
                detail = (
                    f'Avg length: {stats.get("avg_length", "—")} &nbsp;|&nbsp; '
                    f'Range: [{stats.get("min_length", "—")}, '
                    f'{stats.get("max_length", "—")}]'
                )
                top = stats.get("top_5_values", {})
                if top:
                    top_str = ", ".join(
                        f"{k} ({v})" for k, v in list(top.items())[:3]
                    )
                    if len(top_str) > 80:
                        top_str = top_str[:77] + "..."
                    detail += f" &nbsp;|&nbsp; Top: {_esc(top_str)}"
                outlier_badge = ""
                if stats.get("looks_like_id"):
                    outlier_badge = ' <span class="badge-info">ID column</span>'
            elif profile_type == "datetime":
                detail = (
                    f'Range: {stats.get("min_date", "—")} to '
                    f'{stats.get("max_date", "—")} '
                    f'({stats.get("range_days", "—")} days)'
                )
                outlier_badge = ""
            else:
                detail = stats.get("note", "—")
                outlier_badge = ""

            # Null percentage indicator
            null_badge = ""
            if null_pct > 50:
                null_badge = f' <span class="badge-critical">{null_pct}% null</span>'
            elif null_pct > 20:
                null_badge = f' <span class="badge-warn">{null_pct}% null</span>'

            type_badge = (
                f'<span class="badge-info">{profile_type}</span>'
                if profile_type != "unknown" else ""
            )

            rows.append(
                f"<tr>\n"
                f"  <td><strong>{_esc(col_name)}</strong>{null_badge}</td>\n"
                f'  <td class="mono">{_esc(dtype)}</td>\n'
                f"  <td>{type_badge}</td>\n"
                f"  <td>{detail}{outlier_badge}</td>\n"
                f"</tr>"
            )

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Column Profiles</h2>\n'
            f'    <span class="badge">{len(profiles)} columns</span>\n'
            f"  </div>\n"
            f"  <table>\n"
            f"    <thead><tr>\n"
            f"      <th>Column</th><th>Type</th><th>Category</th>"
            f"<th>Statistics</th>\n"
            f"    </tr></thead>\n"
            f'    <tbody>\n{"".join(rows)}\n    </tbody>\n'
            f"  </table>\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Transform audit log
    # ------------------------------------------------------------------

    def _build_transform_log(self, log: list[dict]) -> str:
        """Transformation steps as a timeline-style table."""
        if not log:
            return self._empty_section(
                "Transformation Audit Trail", "No transformations recorded"
            )

        rows: list[str] = []
        for entry in log:
            step = entry.get("step", "?")
            action = entry.get("action", "—")
            detail = entry.get("detail", "—")
            timestamp = entry.get("timestamp", "—")
            # Show just the time portion for cleaner display
            if "T" in str(timestamp):
                timestamp = str(timestamp).split("T")[1][:8]

            rows.append(
                f"<tr>\n"
                f'  <td><span class="step-num">{step}</span></td>\n'
                f"  <td><strong>{_esc(action)}</strong></td>\n"
                f"  <td>{_esc(detail)}</td>\n"
                f'  <td class="mono" style="color:var(--text-light)">'
                f"{_esc(timestamp)}</td>\n"
                f"</tr>"
            )

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Transformation Audit Trail</h2>\n'
            f'    <span class="badge">{len(log)} steps</span>\n'
            f"  </div>\n"
            f"  <table>\n"
            f"    <thead><tr>\n"
            f"      <th style=\"width:3rem\">Step</th><th>Action</th>"
            f"<th>Detail</th><th>Time</th>\n"
            f"    </tr></thead>\n"
            f'    <tbody>\n{"".join(rows)}\n    </tbody>\n'
            f"  </table>\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Quality issues
    # ------------------------------------------------------------------

    def _build_quality_issues(self, issues: list[dict]) -> str:
        """Quality issues flagged by the profiler."""
        if not issues:
            return self._empty_section(
                "Quality Issues",
                "No quality issues detected — data looks clean",
            )

        rows: list[str] = []
        for issue in issues:
            severity = issue.get("severity", "info")
            badge_class = {
                "critical": "badge-critical",
                "warning": "badge-warn",
                "info": "badge-info",
            }.get(severity, "badge-info")

            rows.append(
                f"<tr>\n"
                f'  <td><span class="{badge_class}">'
                f"{_esc(severity.upper())}</span></td>\n"
                f"  <td><strong>{_esc(issue.get('column', ''))}</strong></td>\n"
                f"  <td>{_esc(issue.get('issue', ''))}</td>\n"
                f"  <td>{_esc(issue.get('detail', ''))}</td>\n"
                f"</tr>"
            )

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Quality Issues</h2>\n'
            f'    <span class="badge">{len(issues)} issues</span>\n'
            f"  </div>\n"
            f"  <table>\n"
            f"    <thead><tr>\n"
            f"      <th>Severity</th><th>Column</th><th>Issue</th><th>Detail</th>\n"
            f"    </tr></thead>\n"
            f'    <tbody>\n{"".join(rows)}\n    </tbody>\n'
            f"  </table>\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Correlations
    # ------------------------------------------------------------------

    def _build_correlations(self, correlations: list[dict]) -> str:
        """Strong correlations table with visual bars."""
        if not correlations:
            return self._empty_section(
                "Strong Correlations",
                "No strong correlations found (threshold: |r| > 0.7)",
            )

        rows: list[str] = []
        for corr in correlations:
            value = corr.get("correlation", 0)
            # Scale the bar width: |r| of 0.7 → 70px, 1.0 → 100px
            bar_width = int(abs(value) * 100)
            bar_color = (
                "var(--accent)" if value > 0 else "var(--accent-warn)"
            )
            direction = "positive" if value > 0 else "negative"

            rows.append(
                f"<tr>\n"
                f'  <td><strong>{_esc(corr.get("column_1", ""))}</strong></td>\n'
                f'  <td><strong>{_esc(corr.get("column_2", ""))}</strong></td>\n'
                f'  <td class="mono">{value:+.4f}\n'
                f'    <span class="corr-bar" style="width:{bar_width}px;'
                f'background:{bar_color}"></span>\n'
                f"  </td>\n"
                f"  <td>{direction}</td>\n"
                f"</tr>"
            )

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Strong Correlations</h2>\n'
            f'    <span class="badge">{len(correlations)} pairs</span>\n'
            f"  </div>\n"
            f"  <table>\n"
            f"    <thead><tr>\n"
            f"      <th>Column A</th><th>Column B</th>"
            f"<th>Correlation (r)</th><th>Direction</th>\n"
            f"    </tr></thead>\n"
            f'    <tbody>\n{"".join(rows)}\n    </tbody>\n'
            f"  </table>\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Usage notes
    # ------------------------------------------------------------------

    def _build_usage_notes(self, notes: list[str]) -> str:
        """Practical tips for data consumers."""
        if not notes:
            return self._empty_section("Usage Notes", "No usage notes available")

        items = "\n".join(
            f'<div class="note-item">{_esc(note)}</div>' for note in notes
        )

        return (
            f'<div class="section">\n'
            f'  <div class="section-header">\n'
            f'    <h2>Usage Notes</h2>\n'
            f'    <span class="badge">{len(notes)} notes</span>\n'
            f"  </div>\n"
            f'  <div class="section-body">\n{items}\n  </div>\n'
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def _build_footer(self, dataset_name: str, generated_at: str) -> str:
        """Page footer with generation info."""
        return (
            f'<div class="footer">\n'
            f"  Generated by <strong>HealthPipe AI</strong> "
            f"&mdash; {_esc(dataset_name)} &mdash; {generated_at}\n"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_section(self, title: str, message: str) -> str:
        """Render a section with a 'no data' placeholder."""
        return (
            f'<div class="section">\n'
            f'  <div class="section-header"><h2>{_esc(title)}</h2></div>\n'
            f'  <div class="empty-state">{_esc(message)}</div>\n'
            f"</div>"
        )


def _esc(text: str) -> str:
    """Escape HTML special characters to prevent rendering issues."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
