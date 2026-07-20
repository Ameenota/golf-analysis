# Agent Instructions & Project Context

Welcome! This is the documentation guide for AI agents working on the **Golf Swing Analyzer** project. This file outlines how the project is structured, how to use the documentation in `docs/`, and your permissions boundary.

---

## 1. Project Overview
The **Golf Swing Analyzer** is a computer vision and machine learning pipeline that parses a video of a golf swing, validates it, detects the 8 key swing milestones using a Bidirectional LSTM, calculates biomechanical angles, and produces a synchronized side-by-side comparison video with a matched professional golfer.

---

## 2. Documentation Directory Map
Before starting a task, consult the files in `docs/` depending on your current needs:

* **[docs/backlog.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/backlog.md)**:
  * *When to read*: At the start of every session (in tandem with [docs/session_context.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/session_context.md)).
  * *Purpose*: Track current issues, tasks in progress, completed features, and technical debt. Update this file as you complete tasks.
* **[docs/session_context.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/session_context.md)**:
  * *When to read*: At the start of every session (in tandem with [docs/backlog.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/backlog.md)).
  * *Purpose*: Session-to-session state tracking, active tasks, and action plans.
* **[docs/architecture.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/architecture.md)**:
  * *When to read*: When modifying models, data extraction, coordinate normalization, or inference wrappers.
  * *Purpose*: Detailed breakdown of the MediaPipe extraction pipeline, coordinates, sliding windows, and trained models.
* **[docs/product.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/product.md)**:
  * *When to read*: When designing UI overlays, biomechanical rule thresholds, coaching/drill recommendations, or video synchronization.
  * *Purpose*: User-facing requirements and specifications.
* **[docs/roadmap.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/roadmap.md)**:
  * *When to read*: When you need high-level direction on project phases (Days 1–6 blueprint).
  * *Purpose*: Master chronological development guide.
* **[docs/experiments.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/experiments.md)**:
  * *When to read*: When designing, running, or analyzing ML model ablation studies, feature engineering tests, or model benchmarks.
  * *Purpose*: Dedicated register for ML hypotheses, feature specifications, benchmark performance tables, per-milestone breakdowns, and model promotion decisions.
* **[docs/history.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/history.md)**:
  * *When to read*: When debugging regressions or tracing past changes.
  * *Purpose*: Historical changelog.

---

## 3. Agent Execution Rules & Protocols

For details on sandbox constraints and permissions boundaries, see the [.agents/rules/permissions.md](file:///Users/sagar/Documents/ML/golf-analysis/.agents/rules/permissions.md) rule file.

### Python Environment Policy
* Strictly use `uv` for package management and script execution.
* Never use raw `pip` or `virtualenv`.
* Run all python tools and scripts using `uv run <command>`.
* Add new dependencies using `uv add <package>`.

### Git Commit Protocol
* Before making a git commit, you must view the git diff of your changes.
* Summarize your changes in [docs/history.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/history.md).
* Run `git add docs/history.md` so it is included in your commit.

### Session Management Protocol
* **On Startup**: Always read [docs/session_context.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/session_context.md) and [docs/backlog.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/backlog.md) at the very beginning of the session to align on the current project state.
* **On Wrap-up**: Before finishing a session, update [docs/session_context.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/session_context.md) with a summary of progress made and an explicit, step-by-step "Action Plan for Next Session". Update [docs/backlog.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/backlog.md) to check off completed items or append new ones.

