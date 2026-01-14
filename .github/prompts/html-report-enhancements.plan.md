# Implementation Plan — Single‑Run HTML Report Enhancements (Self‑Contained)

Final requirements (confirmed)
- Scope: Enhance the existing single‑run HTML report only (no run‑to‑run compare view).
- Packaging: One self‑contained HTML file (HTML + CSS only; no external assets/CDNs). Avoid JavaScript for now.
- Visual goals: Make it visually compelling, similar in vibe to compare reports.
- Include all features: richer overview KPIs, grouped failure summaries, side‑by‑side turn view, sticky headers, zebra tables, inline SVG icons/badges, section index with anchors and back‑to‑top, and print/PDF styling.
- Brand palette: Google brand colors (Blue #4285F4, Red #EA4335, Yellow #FBBC05, Green #34A853, Slate/Dark #202124; Muted gray #5F6368).
- Content: Include all conversations and all turns with no truncation.
- Backward compatibility: Template must tolerate missing fields and older result shapes.

---

### Prompt 1: Base theme, typography, and layout system

- What it implements: Replace inline styles with a cohesive CSS theme using Google brand colors, utility classes, spacing scale, sticky table headers, zebra rows, badges, and layout primitives. Ensure semantic HTML and accessible color contrast.
- Dependency: None.
- Prompt:
```
Write complete and executable code to upgrade the single‑run report template at backend/templates/report.html.j2 with a cohesive CSS theme:
- Define CSS custom properties for Google brand colors (blue #4285F4, red #EA4335, yellow #FBBC05, green #34A853), neutrals (text #202124, muted #5F6368, borders #E5E7EB, bg #FFFFFF/#F8FAFC).
- Establish utility classes for spacing, typography, badges, chips, grids, and sticky table headers; add zebra striping for tables.
- Ensure high‑contrast pass/fail states using green/red, and neutral info/warning states using blue/yellow.
- Remove duplicated inline styles and centralize them in a single <style> block.
- Keep report self‑contained (HTML+CSS only, no JS, no external assets).
Include all relevant template changes in backend/templates/report.html.j2 and ensure Reporter continues to render without changes.
Add tests that render a minimal run_results dict through Reporter and assert the presence of theme classes, sticky header CSS, and no external links.
Execute the tests and show results.
Make sure the code is production‑ready and avoids placeholders or stubs.
```

### Prompt 2: Rich Overview with KPIs and mini‑donuts

- What it implements: Expand the Overview to show key KPIs and multiple donut visuals (Conversations pass, Turns pass) using conic‑gradient, with counts and percentages, styled per the new theme.
- Dependency: Builds on Prompt 1.
- Prompt:
```
Write complete and executable code to enhance the Overview section in backend/templates/report.html.j2:
- Add KPI tiles for: Conversations Pass (count/%), Turns Pass (count/%), Total Conversations, Total Turns, and Failed Turns.
- Render two mini‑donut charts (conversations pass rate, turns pass rate) using pure CSS (conic‑gradient) with accessible labels.
- Ensure responsive grid layout and consistent spacing per the new theme.
Add pytest tests to verify the HTML includes KPI tiles and donut containers with computed percent variables and labels.
Run the tests and show results.
Ensure production quality.
```

### Prompt 3: Group failures by metric with counts

- What it implements: A failure summary panel grouped by metric (exact, semantic, consistency, adherence, hallucination), each showing counts and short reasons, linking to detailed sections via anchors.
- Dependency: Prompts 1–2.
- Prompt:
```
Write complete and executable code to add a "Failures by Metric" section to backend/templates/report.html.j2:
- Compute counts per metric across all turns (failed only) with safe defaults if metrics are missing.
- Show a table or card list with metric name, failed count, and sample reasons (first 2–3 unique reasons per metric), each metric name linking to its anchor in Detailed Report.
- Preserve existing Failure Explanations section; this adds a grouped summary above it.
Add tests that build a synthetic run_results with mixed metric failures and assert count rendering and presence of metric anchors.
Run tests and show results.
Ensure production quality.
```

### Prompt 4: Section index with anchors and back‑to‑top

- What it implements: A Table of Contents at the top with anchor links to Overview, Failures, and each conversation; add back‑to‑top links on major sections. No JS.
- Dependency: Prompts 1–3.
- Prompt:
```
Write complete and executable code to add a Table of Contents (ToC) to backend/templates/report.html.j2:
- Generate anchors for: Run Summary, Overview, Failures by Metric, Failure Explanations, Detailed Report, and each conversation title.
- Add back‑to‑top links at the end of major sections.
- Use inline SVG icons for link arrows; ensure print hides ToC automatically.
Add tests that render a sample with two conversations and assert ToC contains anchors to both conversations and global sections.
Run tests and show results.
Ensure production quality.
```

### Prompt 5: Side‑by‑side turn layout (user vs assistant)

- What it implements: In Detailed Report, present each turn in a two‑column grid: left = user prompt, right = assistant output, with scrollable preformatted blocks and preserved existing metrics tables.
- Dependency: Prompts 1–4.
- Prompt:
```
Write complete and executable code to change the turn rendering in backend/templates/report.html.j2:
- For each turn, render a responsive two‑column layout (stacking on small screens) with labeled blocks: User Prompt and Assistant Output.
- Keep existing full text (no truncation), with pre-wrap and scroll where needed.
- Retain the per-turn pass indicator and the per-turn metrics table below or beside (choose layout that keeps readability and print suitability).
Add tests that render a sample conversation and assert presence of two-column turn wrappers and labels.
Run tests and show results.
Ensure production quality.
```

### Prompt 6: Inline SVG icons and badges

- What it implements: Introduce inline SVG for pass/fail, warning, info, and severity levels; refine badges for PASS/FAIL and high‑severity flags using Google palette.
- Dependency: Prompts 1–5.
- Prompt:
```
Write complete and executable code to add inline SVG icons and improved badges in backend/templates/report.html.j2:
- Provide reusable inline SVG symbols (defs) for check (green), x/stop (red), alert (yellow), info (blue).
- Use these in badges for conversation and turn status, and in failure reason lists.
- Ensure icons scale and respect print styles.
Add tests that assert presence of SVG symbol IDs and their usage in status badges.
Run tests and show results.
Ensure production quality.
```

### Prompt 7: Print/PDF styling

- What it implements: Add @media print rules to optimize PDF export: page breaks between conversations, avoid breaking inside tables/blocks, hide navigation and decorative elements, condense spacing.
- Dependency: Prompts 1–6.
- Prompt:
```
Write complete and executable code to add print styles to backend/templates/report.html.j2:
- Use @media print to: add page-break-before for each conversation card; avoid break-inside for tables and code blocks; hide ToC and sticky effects; compress margins and padding.
- Ensure colors remain legible in print (use print-color-adjust if needed).
Add tests that parse the HTML and verify presence of @media print with required selectors and rules.
Run tests and show results.
Ensure production quality.
```

### Prompt 8: Robustness and backward compatibility

- What it implements: Harden Jinja logic for missing keys, optional sections, and older result shapes; ensure safe defaults and no crashes.
- Dependency: Prompts 1–7.
- Prompt:
```
Write complete and executable code to make backend/templates/report.html.j2 robust to missing fields:
- Guard all lookups with `is defined` checks and default filters; ensure report renders even when metrics blocks or summary fields are absent.
- Add a minimal smoke test that renders with the smallest viable run_results and asserts the report renders key sections with fallback text.
Run tests and show results.
Ensure production quality.
```

### Prompt 9: Documentation and sample artifact

- What it implements: Update README and add a small sample results.json to demonstrate the new report visuals (checked into tests/fixtures). Document how to open and print to PDF.
- Dependency: Prompts 1–8.
- Prompt:
```
Write complete and executable code to:
- Add or update documentation explaining the enhanced single-run HTML report, its sections, and printing guidance.
- Add a minimal sample run_results fixture under backend/tests/fixtures/ and a test that renders report.html to a temp file.
Run tests and show results.
Ensure production quality.
```

---

Download this plan
- Local path: .github/prompts/html-report-enhancements.plan.md
- Click to open/download: [Download the HTML Report Enhancements Plan](./html-report-enhancements.plan.md)
