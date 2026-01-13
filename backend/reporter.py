from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import json
from jinja2 import Environment, FileSystemLoader, select_autoescape


@dataclass
class Reporter:
    templates_dir: Path

    def __post_init__(self):
        # Disable template caching and enable auto-reload to ensure latest template is used
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(['html', 'xml']),
            auto_reload=True,
            cache_size=0,
        )

    def render_html(self, run_results: Dict[str, Any]) -> str:
        tpl = self.env.get_template('report.html.j2')
        # Expect run_results contains transcript and per-turn metrics; if not, render minimal
        return tpl.render(**run_results)

    def write_html(self, run_results: Dict[str, Any], out_path: Path) -> Path:
        html = self.render_html(run_results)
        out_path.write_text(html, encoding='utf-8')
        return out_path
