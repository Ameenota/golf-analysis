---
trigger: always_on
---

---
name: auto-ledger
description: Use this rule only when making git commits to update the project history log.
---

# Git Commit Protocol

Before you execute any `git commit` command, you must:
1. View the git diff of your changes.
2. Append a 2-3 bullet point summary of what changed to `docs/history.md`.
3. Run `git add docs/history.md` so it is included in the commit.