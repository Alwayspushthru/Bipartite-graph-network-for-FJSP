# Revision Workspace

This directory contains materials produced during manuscript revision. It is kept separate from the main training code and the original manuscript files.

## Directory structure

- `reviewer_comments/`: original comments and the comment-response tracker.
- `experiments/`: experiment plans, configurations, commands, and execution notes.
- `results/`: verified raw summaries, statistical analyses, tables, and figures.
- `manuscript_changes/`: section-level revision notes and revised manuscript passages.
- `response_materials/`: point-by-point responses, cover-letter notes, and submission checklists.
- `archive/`: superseded drafts retained for traceability.

## Baseline manuscript

- `manuscript_changes/manuscript_0.docx`: editable source of the first submitted manuscript. Use this file as the baseline for preparing the revised clean and marked manuscripts.
- `manuscript_changes/manuscript_0.pdf`: fixed-layout record of the first submitted manuscript. Use this file to verify the original pagination, figures, tables, and submitted wording.

Both files are treated as version `manuscript_0`. Preserve them unchanged. Create later manuscript versions as new files rather than overwriting either baseline file.

## Working rules

1. Assign every editor or reviewer concern a stable ID, such as `E.1` or `R1.1`.
2. Do not report an experiment as completed until its configuration, result file, and manuscript location are recorded.
3. Use `AUTHOR_INPUT_NEEDED` for missing facts and `PENDING_EXPERIMENT` for unfinished experiments.
4. Keep raw outputs unchanged. Put derived tables and statistical summaries in separate files.
5. Record manuscript locations by section until final page and line numbers are available.
6. Move superseded drafts to `archive/` instead of overwriting them when their history may matter.
7. Make substantive edits in a copy of `manuscript_0.docx`; use `manuscript_0.pdf` only as the immutable submitted-layout reference.

## Suggested workflow

1. Register comments in `reviewer_comments/comment_tracker.md`.
2. Define the evidence required for each comment in `experiments/supplementary_experiment_plan.md`.
3. Save verified results in `results/experiment_results.md`.
4. Map accepted changes in `manuscript_changes/change_log.md`.
5. Draft final replies in `response_materials/response_draft.md` only after the supporting evidence is available.
