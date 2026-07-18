---
trigger: always_on
---

---
name: agent-permissions-bounds
description: Enforces the agent's permissions boundary and sandbox constraints.
---

# Agent Workspace & Permissions Protocol

1. **Workspace Control**: You have full read, write, and execute permissions within the project root: `/Users/sagar/Documents/ML/golf-analysis`. You can edit files and run tests or scripts inside this workspace freely.
2. **Pre-Approved Git Commands**: Standard git operations, including `git push`, are pre-approved and allowed to bypass sandbox constraints in this directory.
3. **Hard Approval Required**:
   * Any execution of command lines containing `sudo`.
   * Modifying files or running commands outside the repository directory `/Users/sagar/Documents/ML/golf-analysis` (e.g., system configuration, global files).
   * Any action with structural or environmental risks.
