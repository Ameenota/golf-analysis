---
trigger: always_on
---

---
name: python-env-guard
description: Enforces environment and package management rules for Python development.
---

# Python Environment Protocol

1. **Package Management:** Never use raw `pip` or `virtualenv`. This project strictly uses `uv`. 
2. **Adding Dependencies:** If a task requires a new package, explicitly run `uv add <package>` so the `uv.lock` file stays updated.
3. **Execution:** Always execute python scripts or tools using `uv run <command>` to ensure the locked environment is respected.