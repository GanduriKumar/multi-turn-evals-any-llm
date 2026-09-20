# LLM Wiki and OKF Reference-Pack Builder

Status: Proposed

The evaluation toolkit should consume only a versioned built-in reference pack.
This utility creates or refreshes that pack from user-supplied source
locations. Source access happens during pack creation, not during normal
evaluation runs.

## 1. Purpose

The utility accepts one or more source locations and produces:

- immutable raw source copies or snapshots,
- an LLM Wiki-style organized knowledge layer,
- OKF-formatted reference files,
- a manifest with source provenance,
- a validation report,
- a pack version or content hash.

The generated pack is then committed, published, or supplied to the toolkit as
a run input. The evaluator does not need live access to the original sources.

## 2. Source locations

Start with simple source types:

- local directory or file,
- Git repository with branch, tag, or commit,
- HTTP or HTTPS document URL.

Do not add connectors for every enterprise content system initially. Add a
connector only when a real source location requires it.

Each source must record:

- source URI,
- source type,
- requested revision,
- resolved revision or retrieval timestamp,
- content hash,
- source title,
- licensing or access note when supplied.

## 3. Output layout

Use a predictable pack layout:

  reference-pack/
    manifest.json
    raw/
    wiki/
    okf/
    validation-report.json

Raw files are preserved for traceability and must not be silently rewritten.
Wiki files are the maintained synthesis layer. OKF files provide structured
metadata and references for consumers.

The manifest links each generated wiki or OKF entry back to its raw source and
records the builder version and generation timestamp.

## 4. Build pipeline

The utility performs one clear step at a time:

1. Validate source configuration.
2. Fetch or read each source.
3. Store the raw source snapshot.
4. Extract readable text and source metadata.
5. Create or update the LLM Wiki pages.
6. Create OKF reference files and metadata.
7. Validate links, required metadata, and file formats.
8. Write the manifest and validation report.
9. Calculate the final pack version and content hash.

The build fails when a required source cannot be read, a citation is missing,
or an output file cannot be validated. It must not produce a partially valid
pack and label it complete.

## 5. LLM Wiki behavior

Keep raw sources immutable. The wiki layer may organize, summarize, link, and
deduplicate information, but every meaningful statement must retain a
reference to its source material.

The builder should create simple Markdown files with stable identifiers,
titles, summaries, links, and source references. Avoid a complex graph database
or a custom query language in the first release.

Human review may edit or approve generated wiki pages before the pack is
published. The published pack should record whether a page is generated,
reviewed, or changed after generation.

## 6. OKF behavior

Generate OKF files from the same source records used by the wiki layer. Do not
maintain two independent extraction pipelines.

Each OKF record should include, where applicable:

- type,
- title,
- description,
- resource or source URI,
- tags,
- version,
- provenance,
- relationships to related records.

Validate the generated files against the selected OKF specification version.
Record that version in the manifest.

## 7. Interfaces

Provide a small command-line utility first:

  python -m backend.reference_builder build +    --sources configs/reference-sources.yaml +    --output references/built-in

The utility may later be exposed through MCP as an administrative operation:

- build_reference_pack
- validate_reference_pack
- inspect_reference_manifest

Pack building should require explicit local authorization because it reads
external locations and can invoke a synthesis model.

## 8. Configuration

The source configuration should contain only locations and build options:

  sources:
    - id: llm-wiki
      type: git
      location: https://example.org/llm-wiki.git
      revision: v1
    - id: okf-reference
      type: url
      location: https://example.org/okf-reference.md

  output:
    format: markdown
    okf_version: "0.2"
    deterministic: true

Secrets must come from environment variables or a secret manager reference.
They must not be written into manifests, wiki pages, reports, or datasets.

## 9. Security and reproducibility

- Allow-list source protocols and hosts.
- Apply download size and file-count limits.
- Reject executable files as source content unless explicitly supported.
- Record every resolved revision and content hash.
- Use deterministic generation settings where possible.
- Preserve the model and prompt configuration used for synthesis.
- Make pack updates produce a new version instead of overwriting an active pack.

## 10. Tests

Test the utility with:

- local Markdown and JSON sources,
- a fake Git source,
- a fake HTTP source,
- missing and inaccessible sources,
- duplicate source content,
- missing citations,
- invalid OKF metadata,
- deterministic rebuilds,
- changed-source hash detection,
- secret redaction,
- partial-build failure.

The most important integration test builds a pack, deletes network access, and
then runs dataset generation using only the resulting pack.

## 11. Completion criteria

The utility is complete when:

- A configured source set produces raw, wiki, OKF, manifest, and validation
  artifacts.
- Every generated entry can be traced back to a source location.
- A pack can be validated and consumed without network access.
- A changed source creates a new identifiable pack version.
- Invalid or incomplete packs cannot be used for evaluation.
- The evaluator depends on the generated pack, not on source connectors.

## 12. Explicit non-goals

- Live source retrieval during an evaluation run.
- Arbitrary enterprise policy-system integrations.
- A general-purpose document-management platform.
- Hidden or untraceable policy synthesis.
- Replacing the LLM Wiki or OKF conventions with a private format.
