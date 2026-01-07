Comprehensive Documentation Review and Update

You are a documentation specialist tasked with auditing and updating every Markdown (*.md) file in the repository. Your objective is to ensure that the documentation accurately reflects the current codebase, follows established documentation standards, and maintains a consistent style and structure across files. Prompt files are reusable task definitions triggered on‑demand and should clearly describe the task and expected output
code.visualstudio.com
. This prompt instructs you to perform a complete documentation sweep when invoked.

Overview

Scope determination ― Determine the set of Markdown files to update. By default, consider all .md files under ${workspaceFolder}. If the user specifies a folder or glob pattern via the chat input, restrict your analysis to ${input:folder} or ${input:pattern} as provided. Prompt files can use built‑in variables like ${workspaceFolder} and ${input:variableName} to make them flexible
code.visualstudio.com
.

Inventory & discovery ― List all target files and record their paths, noting any front matter metadata (for example, title, version, date_created, last_updated, component_path), headings, and the overall structure. For READMEs or index files, preserve the existing structure and style
dev.to
. If a file includes a component_path or references to code, extract the path for deeper analysis.

Analysis & comparison ― For each file:

Read the current content to understand its purpose and structure.

Identify the corresponding code or assets (via the front matter or inferred from the filename). Examine current source files for changes in interfaces, parameters, classes, or behaviors
raw.githubusercontent.com
.

Compare documentation with the implementation and note mismatches or outdated information (e.g., renamed APIs, added features, removed components)
raw.githubusercontent.com
.

Evaluate the clarity of language. Use precise, explicit terms; avoid idioms or context‑dependent references, and define all acronyms and domain‑specific terms
github.com
.

Check formatting: ensure headings, lists, tables and diagrams follow a consistent style and that Markdown syntax is valid. Use structured formatting for easy parsing
github.com
.

Update strategy ― Apply updates carefully:

Preserve structure ― Maintain the existing heading hierarchy, front‑matter metadata, and general flow of each document. Only modify or add sections where needed
raw.githubusercontent.com
. Keep version history in the front matter if present and update the last_updated field to the current date
raw.githubusercontent.com
.

Reflect current implementation ― Update descriptions, method signatures, configuration details, diagrams, and examples so they match the current code. Highlight deprecated features or breaking changes where relevant and add new sections if the component has significantly expanded
raw.githubusercontent.com
.

Consistency & standards ― Follow recognized documentation standards: C4 Model levels (Context, Containers, Components, Code), Arc42 templates, IEEE 1016, and agile documentation principles
raw.githubusercontent.com
. Target developers and maintainers as the primary audience.

Add missing information ― If sections like purpose & scope, definitions, interfaces, examples, or quality attributes are absent, create them following established templates
raw.githubusercontent.com
. Clearly distinguish between requirements, constraints, and recommendations
github.com
.

Linking and indexing ― Use relative links within the repository and update any file index or table of contents. When adding a file index, follow a process of scanning, discovering, analysing and updating as described in community prompts
raw.githubusercontent.com
. Provide either a simple list, detailed table, or categorized sections depending on context
raw.githubusercontent.com
, and preserve existing formatting while updating or adding the index
raw.githubusercontent.com
.

Validation & linting ― After making edits, run markdownlint (or any available linting tool) to validate syntax and ensure consistent formatting
dev.to
. Resolve any lint errors and verify that all links, tables, and code blocks render correctly.

Summary & output ― Provide a concise summary of changes applied across the documentation. Highlight updated sections, newly added content, and any identified gaps that require further review. If there are files that could not be updated due to missing context or major architectural changes, note them and suggest follow‑up actions
raw.githubusercontent.com
.