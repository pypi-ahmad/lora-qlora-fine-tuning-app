# Security Policy

## Supported versions

Security fixes are provided for the latest `0.1.x` release and the current `main` branch.

| Version | Supported |
| --- | --- |
| `0.1.x` | Yes |
| Older versions | No |

Update to the latest release before confirming or reporting a problem.

## Report a vulnerability privately

Use the repository's **Security → Report a vulnerability** form to submit a private GitHub
Security Advisory. Do not disclose an unpatched vulnerability in a public issue, discussion, pull
request, log, screenshot, or social-media post.

If private vulnerability reporting is temporarily unavailable, open a minimal public issue asking
the maintainer to provide a private channel. Do not include exploit details in that issue.

Include only the information needed to investigate:

- affected version or commit;
- operating system, Python, GPU, and driver versions when relevant;
- the vulnerable boundary and potential impact;
- the smallest safe reproduction;
- whether exploitation requires local access, a malicious model/dataset, or network exposure;
- suggested mitigation, if known; and
- a preferred private contact for follow-up.

Remove access tokens, private source, proprietary datasets, personal information, and unrelated
system details. The maintainer will acknowledge and triage reports on a best-effort basis and will
coordinate disclosure after a fix or mitigation is available.

## Project security model

LoRA Fine-tune Studio is a local, single-user educational application. It does not provide user
accounts, authorization, tenant isolation, remote job scheduling, or production serving controls.

The application crosses these trust boundaries:

- downloads models, tokenizers, metadata, and datasets from Hugging Face;
- optionally uploads an adapter and tokenizer to Hugging Face;
- accepts local CSV, JSON, and JSONL dataset uploads;
- writes uploads, worker configuration, logs, checkpoints, metrics, and adapters to local disk;
- starts a local Streamlit web server and a hidden training worker;
- optionally calls Ollama at `http://localhost:11434`; and
- downloads the `uv` installer when the one-click launcher cannot find `uv`.

Vulnerabilities in an upstream model, dataset, Ollama, PyTorch, CUDA driver, or another dependency
should also be reported to that upstream project through its security process.

## Existing controls

- Only Hugging Face repository IDs or repository-root HTTPS URLs are accepted for remote models
  and datasets.
- Arbitrary URLs, nested repository paths, query strings, and fragments are rejected.
- Transformer remote model code is disabled with `trust_remote_code=False`.
- Model loading requires safetensors.
- Uploaded datasets are limited to CSV, JSON, or JSONL and 200 MB.
- Uploads use content-derived filenames rather than the supplied path.
- Run IDs reject path traversal characters.
- Job-status JSON is replaced atomically.
- `HF_TOKEN` is read at runtime and is not written to saved training configuration or worker logs.
- Only one live training worker is permitted by the local job manager.

These controls reduce risk; they do not make arbitrary models or datasets trustworthy.

## Safe operation

- Run the application only on a trusted computer and user account.
- Do not expose port `8504` to the public internet or an untrusted local network. The application
  has no authentication.
- Keep the operating system, NVIDIA driver, Ollama, `uv`, Python dependencies, and this project
  updated.
- Inspect the one-click launcher before executing it if organizational policy prohibits remote
  installation scripts. Install `uv` manually when required by policy.
- Use the least-privileged Hugging Face token: read access for downloads and write access only for
  an intentional upload.
- Store tokens in the user environment or an ignored local `.streamlit/secrets.toml`; never commit
  them.
- Treat downloaded models, datasets, and tokenizer files as untrusted content. Review their source,
  license, revision, and integrity.
- Do not train secrets, personal information, or restricted data into model weights without an
  approved governance and retention process.
- Protect `.uploads`, `.runs`, Hugging Face caches, checkpoints, adapters, and logs. They may contain
  source data, generated text, paths, or memorized information.
- Review repository visibility and token scope before enabling **Push adapter to Hugging Face Hub**.
- Back up important adapters before deleting a run directory.

## Not security vulnerabilities

General model-quality problems, hallucinations, expected CUDA memory limits, unsupported hardware,
and questions about model or dataset licenses normally belong in a public issue. If a quality
problem exposes private data, bypasses a trust boundary, or enables unintended code execution,
report it privately instead.
