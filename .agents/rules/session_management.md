---
trigger: always_on
---

---
name: session-management
description: Enforces reading and updating session context files.
---

# Session Management Protocol

1. **On Startup**: Always read `docs/session_context.md` and `docs/backlog.md` at the very beginning of the session to align on the current project state and pick up where the previous session left off.
2. **On Wrap-up**: Before finishing a session, update `docs/session_context.md` with a summary of the progress made and an explicit, step-by-step "Action Plan for Next Session". Update `docs/backlog.md` to check off completed items or append new ones.
