# Graph Report - .  (2026-08-13)

## Corpus Check
- 84 files · ~93,929 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 436 nodes · 909 edges · 29 communities (22 shown, 7 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Training Contracts and Tests
- Job Queue and Monitoring
- Tutorial Builder
- Training UI and Hardware
- Model Loading and Training
- App Pages and Inspection
- Curriculum and Governance
- Tutorial Verification
- Evaluation and Course Artifacts
- Dataset UI
- Dataset and Preference Training
- SFT and Research References
- Ollama Integration
- LoRA Formulas and Glossary
- Model Selection UI
- Job Artifacts and Architecture
- Preference Objectives
- System Readiness and Troubleshooting
- Experiment Configuration
- CLI Launcher
- Community Standards
- Linux Launcher
- Release History
- Checkpoint Answers
- Main Runtime Package
- Unsloth Runtime Package

## God Nodes (most connected - your core abstractions)
1. `TrainingConfig` - 59 edges
2. `DatasetSpec` - 47 edges
3. `dispatch_next_run()` - 18 edges
4. `train()` - 17 edges
5. `write_json_atomic()` - 14 edges
6. `JobStatus` - 13 edges
7. `run_path()` - 13 edges
8. `PeftMode` - 12 edges
9. `StatusCallback` - 12 edges
10. `html_to_flowables()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Setup Guide` --semantically_similar_to--> `Unsloth Pip Installation`  [INFERRED] [semantically similar]
  SETUP.md → .firecrawl/unsloth.ai-docs-get-started-install-pip-install.md
- `Usage Guide` --semantically_similar_to--> `Setup Guide`  [INFERRED] [semantically similar]
  USAGE.md → SETUP.md
- `Module 7: Master LoRA, QLoRA, OFT, and QOFT` --semantically_similar_to--> `Zero to Mastery Course`  [INFERRED] [semantically similar]
  docs/07-master-lora-qlora-oft-and-qoft.html → TUTORIAL.md
- `LoRA Fine-tune Studio: Zero to Mastery PDF` --semantically_similar_to--> `Generated LoRA Fine-tune Studio PDF`  [INFERRED] [semantically similar]
  docs/downloads/lora-finetune-studio-zero-to-mastery.pdf → output/pdf/lora-finetune-studio-zero-to-mastery.pdf
- `source_label()` --references--> `DatasetSpec`  [EXTRACTED]
  app_pages/dataset.py → src/lora_finetune_studio/models.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Application Runtime Boundary** — readme_lora_finetune_studio, setup_two_runtimes, firecrawl_unsloth_ai_docs_get_started_install_pip_install_unsloth_core [INFERRED 0.75]
- **Parameter-Efficient Adapter Method Family** — tutorial_lora, tutorial_qlora, tutorial_oft [EXTRACTED 1.00]
- **Preference Training Objectives** — docs_13_lab_b_preference_training_safely_preference_training, docs_09_select_the_post_training_objective_direct_preference_optimization, docs_09_select_the_post_training_objective_reward_modeling [EXTRACTED 1.00]
- **Training Execution Flow** — docs_16_read_and_extend_the_repository_trainingconfig_contract, docs_15_operate_jobs_and_use_artifacts_training_worker, docs_15_operate_jobs_and_use_artifacts_run_artifacts [EXTRACTED 1.00]

## Communities (29 total, 7 thin omitted)

### Community 0 - "Training Contracts and Tests"
Cohesion: 0.09
Nodes (43): apply_preset(), DatasetSpec, Preset, Any, TrainingConfig, FakeWorkerProcess, Path, _running_test_run() (+35 more)

### Community 1 - "Job Queue and Monitoring"
Cohesion: 0.09
Nodes (48): Training monitor and completed-adapter evaluation page., training_monitor(), fragment, active_run(), cancel_active_run(), cancel_run(), create_run(), dispatch_next_run() (+40 more)

### Community 2 - "Tutorial Builder"
Cohesion: 0.10
Nodes (43): BaseDocTemplate, ListFlowable, NavigableString, Paragraph, ParagraphStyle, build_pdf(), build_search_index(), build_site_files() (+35 more)

### Community 3 - "Training UI and Hardware"
Cohesion: 0.07
Nodes (39): Live GPU memory page., approach_changed(), compute_type_changed(), method_changed(), Training configuration page., cuda_memory_stats(), CudaMemoryStats, detect_hardware() (+31 more)

### Community 4 - "Model Loading and Training"
Cohesion: 0.09
Nodes (43): LoraConfig, OFTConfig, ComputeType, resolve_compute_type(), TrainingApproach, load_training_dataset(), _apply_unsloth_trainer_patch(), _dataset_label() (+35 more)

### Community 5 - "App Pages and Inspection"
Cohesion: 0.06
Nodes (26): Model selection and inspection page., Training review and launch page., Read-only system readiness page., LoRA Fine-tune Studio., DatasetInspection, inspect_dataset(), parse_hf_repo(), Dataset (+18 more)

### Community 6 - "Curriculum and Governance"
Cohesion: 0.06
Nodes (37): Changelog, Development Workflow, Contributing Guide, Module 1: Learn the Map Before the Territory, Module 2: Natural Language Becomes Model Input, Module 3: Neural Networks Learn with Vectors and Gradients, Transformer Attention, Module 4: Build a Transformer from Attention Blocks (+29 more)

### Community 7 - "Tutorial Verification"
Cohesion: 0.18
Nodes (6): HTMLParser, PageParser, Path, sha256(), test_generated_manifest_matches_files(), test_generated_pages_have_valid_local_links()

### Community 8 - "Evaluation and Course Artifacts"
Cohesion: 0.25
Nodes (9): Falsifiable Evaluation Claim, Evaluation Layers, Module 14: Evaluate like an experimenter, Data Card, Defensible Adapter Experiment, Module 17: Capstone train a defensible specialist, LoRA Fine-tune Studio: Zero to Mastery PDF, LoRA Fine-tune Studio: Zero to Mastery (+1 more)

### Community 9 - "Dataset UI"
Cohesion: 0.39
Nodes (7): clear_pending(), is_compatible(), Dataset collection, inspection, and mapping page., save_pending_dataset(), source_identity(), source_key(), source_label()

### Community 10 - "Dataset and Preference Training"
Cohesion: 0.25
Nodes (8): Data Leakage, Module 8: Engineer datasets that teach the intended behavior, Preference Dataset Schema, SFT Dataset Schema, Direct Preference Optimization, Module 13: Lab B preference training safely, Preference Quality Rubric, Preference Training

### Community 11 - "SFT and Research References"
Cohesion: 0.25
Nodes (8): Supervised Fine-Tuning, Baseline Comparison, Module 12: Lab A complete an SFT smoke run, SFT Smoke Run, Appendix E: Official references, Attention Is All You Need, Direct Preference Optimization, QLoRA: Efficient Finetuning of Quantized LLMs

### Community 12 - "Ollama Integration"
Cohesion: 0.38
Nodes (5): Independent local Ollama playground page., generate(), list_models(), Small Ollama HTTP client; no additional dependency required., _request()

### Community 13 - "LoRA Formulas and Glossary"
Cohesion: 0.29
Nodes (7): Appendix B: Formula and settings reference, Effective Batch Size, LoRA Update Formula, Adapter, Appendix D: Glossary, Evaluation Set, LoRA: Low-Rank Adaptation of Large Language Models

### Community 14 - "Model Selection UI"
Cohesion: 0.29
Nodes (7): Inspect Model, LoRA Studio Model Configuration Interface, Model Repository, Model Revision, Model Selection, Qwen/Qwen3-0.6B, Training Workflow

### Community 15 - "Job Artifacts and Architecture"
Cohesion: 0.33
Nodes (6): Module 15: Operate jobs and use artifacts, Run Artifacts, Training Worker Process, Module 16: Read and extend the repository, Runtime Architecture, TrainingConfig Contract

### Community 16 - "Preference Objectives"
Cohesion: 0.40
Nodes (5): KTO, Module 9: Select the post-training objective, ORPO, Proximal Policy Optimization, Reward Modeling

### Community 17 - "System Readiness and Troubleshooting"
Cohesion: 0.40
Nodes (5): Hardware Readiness, Module 10: Prepare the local training system, Unsloth Runtime, Appendix C: Troubleshooting by boundary, Boundary Troubleshooting

### Community 18 - "Experiment Configuration"
Cohesion: 0.67
Nodes (3): Module 11: Translate intent into app settings, One-Variable Experiments, Training Configuration

## Knowledge Gaps
- **41 isolated node(s):** `Launch LoRA Studio.sh script`, `lora-finetune-studio`, `TrainingRecipe`, `lora-finetune-studio-unsloth-runtime`, `Unsloth Core` (+36 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrainingConfig` connect `Training Contracts and Tests` to `Job Queue and Monitoring`, `Training UI and Hardware`, `Model Loading and Training`, `App Pages and Inspection`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `DatasetSpec` connect `Training Contracts and Tests` to `Job Queue and Monitoring`, `Training UI and Hardware`, `Model Loading and Training`, `App Pages and Inspection`, `Dataset UI`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `PeftMode` connect `Training UI and Hardware` to `Training Contracts and Tests`, `Job Queue and Monitoring`, `Model Loading and Training`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `TrainingConfig` (e.g. with `StatusCallback` and `FakeWorkerProcess`) actually correct?**
  _`TrainingConfig` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `DatasetSpec` (e.g. with `StatusCallback` and `FakeWorkerProcess`) actually correct?**
  _`DatasetSpec` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Launch LoRA Studio.sh script`, `lora-finetune-studio`, `TrainingRecipe` to the rest of the system?**
  _41 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Training Contracts and Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.09220779220779221 - nodes in this community are weakly interconnected._