# Agent Instructions & Project Context

Welcome! This is the documentation guide for AI agents working on the **Golf Swing Analyzer** project. This file outlines how the project is structured, how to use the documentation in `docs/`, and your permissions boundary.

---

## 1. Project Overview
The **Golf Swing Analyzer** is a computer vision and machine learning pipeline that parses a video of a golf swing, validates it, detects the 8 key swing milestones using a Bidirectional LSTM, calculates biomechanical angles, and produces a synchronized side-by-side comparison video with a matched professional golfer.

---

## 2. Documentation Directory Map
Before starting a task, consult the files in `docs/` depending on your current needs:

* **[docs/backlog.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/backlog.md)**:
  * *When to read*: At the start of every session (in tandem with `session_context.md`).
  * *Purpose*: Track current issues, tasks in progress, completed features, and technical debt. Update this file as you complete tasks.
* **[docs/architecture.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/architecture.md)**:
  * *When to read*: When modifying models, data extraction, coordinate normalization, or inference wrappers.
  * *Purpose*: Detailed breakdown of the MediaPipe extraction pipeline, coordinates, sliding windows, and trained models.
* **[docs/product.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/product.md)**:
  * *When to read*: When designing UI overlays, biomechanical rule thresholds, coaching/drill recommendations, or video synchronization.
  * *Purpose*: User-facing requirements and specifications.
* **[docs/roadmap.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/roadmap.md)**:
  * *When to read*: When you need high-level direction on project phases (Days 1–6 blueprint).
  * *Purpose*: Master chronological development guide.
* **[docs/history.md](file:///Users/sagar/Documents/ML/golf-analysis/docs/history.md)**:
  * *When to read*: When debugging regressions or tracing past changes.
  * *Purpose*: Historical changelog.

---

## 3. Agent Execution Rules & Permissions

### Workspace Bounds & Permissions
* **Full Control**: You have full read, write, and execute permissions within the project root: `/Users/sagar/Documents/ML/golf-analysis`. You can edit files and run tests or scripts inside this workspace freely.
* **Pre-Approved Git Commands**: Standard git operations, including `git push`, are pre-approved and allowed to bypass sandbox constraints in this directory.
* **Hard Approval Required**:
  * Any execution of command lines containing `sudo`.
  * Modifying files or running commands outside `/Users/sagar/Documents/ML/golf-analysis` (e.g., system configuration, global files).
  * Any action with structural or environmental risks.

### Python Environment Policy
* Strictly use `uv` for package management and script execution.
* Never use raw `pip` or `virtualenv`.
* Run all python tools and scripts using `uv run <command>`.
* Add new dependencies using `uv add <package>`.

### Git Commit Protocol
* Before making a git commit, you must view the git diff of your changes.
* Summarize your changes in `docs/history.md` (and also update `HISTORY.md` at the root if required by system pre-commit hooks or automatic validators).
* Run `git add` on the updated history files so they are included in your commit.
