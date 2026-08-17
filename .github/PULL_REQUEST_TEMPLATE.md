## What does this PR do?

A short description of the problem, the solution, and the user impact. Link the issue it addresses, if any (`Closes #___`).

## Verification

```powershell
uv run ruff format --check .
uv run ruff check .
uv run ty check src
uv run pytest
```

- [ ] GPU/Ollama verification status stated (or "not run" if unavailable hardware)

## Checklist

- [ ] The change solves one clearly described problem.
- [ ] Tests cover new or corrected behavior.
- [ ] Ruff, ty, and pytest pass.
- [ ] Documentation and `CHANGELOG.md` are updated when needed.
- [ ] No credentials, private data, generated artifacts, or unrelated changes are included.
- [ ] I read [CONTRIBUTING.md](../CONTRIBUTING.md)
