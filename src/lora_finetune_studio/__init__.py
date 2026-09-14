"""LoRA Fine-tune Studio.

Package root for the local Streamlit fine-tuning app. This module must stay free of
heavy imports (torch, transformers, streamlit) since it loads whenever any submodule
is imported, including inside the isolated training worker process.

Start reading at models.py: it defines the shared data contracts (TrainingConfig,
DatasetSpec, JobStatus) that every other module in this package passes across process
boundaries.
"""

# Unclear from this file: nothing in the codebase reads this constant (it is not
# re-exported to pyproject.toml's version, which is 0.5.2). Treat it as stale rather
# than authoritative for the installed app version.
__version__ = "0.1.0"
