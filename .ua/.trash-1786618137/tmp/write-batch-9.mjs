import fs from 'node:fs';

const root = 'D:/AI/Github/lora-qlora-fine-tuning-app';
const ua = `${root}/.ua`;
const extracted = JSON.parse(fs.readFileSync(`${ua}/tmp/ua-file-extract-results-9.json`, 'utf8'));

const fileInfo = {
  'src/lora_finetune_studio/ollama.py': ['Small standard-library HTTP client for listing local Ollama models and generating a non-streamed response.', ['ollama', 'http-client', 'inference']],
  'streamlit_app.py': ['Streamlit application entry point that initializes shared session state, advances the training queue, configures navigation, and handles graceful shutdown.', ['entry-point', 'streamlit', 'queue-management', 'ui']],
  'tests/test_app.py': ['End-to-end Streamlit smoke test covering startup without Ollama, navigation, dataset removal, training controls, and application shutdown.', ['test', 'streamlit', 'integration', 'ui']],
  'tests/test_hardware.py': ['Unit tests for hardware detection, CUDA memory helpers, model-size warnings, and platform-specific system scanning.', ['test', 'hardware', 'cuda', 'system']],
  'tests/test_inference.py': ['Regression test ensuring failed model loading releases CUDA memory during inference setup.', ['test', 'inference', 'cuda', 'regression']],
  'tests/test_jobs.py': ['Comprehensive tests for persistent training runs, FIFO dispatch, worker lifecycle handling, cancellation, and Unsloth interpreter selection.', ['test', 'job-queue', 'workers', 'persistence']],
  'tests/test_lifecycle.py': ['Unit test for scheduling the application exit callback on a daemon timer.', ['test', 'lifecycle', 'shutdown', 'timer']],
  'tests/test_models.py': ['Validation tests for training configuration serialization, recipe compatibility, dataset constraints, presets, and compute-type selection.', ['test', 'configuration', 'validation', 'training']],
  'tests/test_queue_dispatcher.py': ['Tests the process handoff helper that waits for a parent worker and then resumes queue dispatch with the base interpreter.', ['test', 'job-queue', 'processes', 'dispatch']],
  'tests/test_queue_ui.py': ['Streamlit integration test confirming that a new training request is queued while another worker remains active.', ['test', 'streamlit', 'job-queue', 'integration']],
  'tests/test_sources.py': ['Tests Hugging Face repository parsing plus dataset inspection, upload validation, and content-addressed upload storage.', ['test', 'datasets', 'huggingface', 'validation']],
  'tests/test_training.py': ['Tests dataset normalization and combination, PEFT and Unsloth loading, trainer configuration, compute dtypes, and conversation formatting.', ['test', 'training', 'unsloth', 'peft']],
  'tests/test_tutorial.py': ['Validates the generated zero-to-mastery tutorial, linked HTML pages, manifest hashes, reference dataset, and matching PDF artifacts.', ['test', 'documentation', 'tutorial', 'artifacts']],
  'tests/test_worker.py': ['Tests that the training worker schedules queue handoff after both successful and failed terminal runs.', ['test', 'worker', 'job-queue', 'failure-handling']],
  'unsloth-runtime/pyproject.toml': ['UV project configuration for the isolated Python 3.13 Unsloth runtime, pinning CUDA-enabled PyTorch, TorchAO, torchvision, and Unsloth.', ['configuration', 'unsloth', 'uv', 'dependencies']],
};

const special = {
  _request: 'Builds and executes an Ollama HTTP request, decoding the JSON response.',
  list_models: 'Retrieves the names of locally available Ollama models.',
  generate: 'Requests a non-streamed completion from a selected local Ollama model.',
  test_app_starts_without_ollama: 'Exercises the Streamlit application workflow with external services stubbed out.',
  test_unsloth_loader_uses_optimized_qlora_settings: 'Checks that the Unsloth model loader applies its optimized QLoRA configuration.',
  test_unsloth_reward_loader_keeps_score_head: 'Checks that reward-model loading preserves the score head when using Unsloth.',
  test_handoff_waits_for_parent_exit_before_dispatch: 'Verifies queue handoff waits for the parent worker before dispatching the next run.',
  test_schedule_handoff_uses_base_project_interpreter: 'Verifies queue handoff uses the base project interpreter rather than the training runtime.',
  test_worker_schedules_queue_handoff_after_terminal_status: 'Verifies a successful worker completion schedules the next queued job.',
  test_worker_failure_still_schedules_queue_handoff: 'Verifies a failed worker completion still schedules the next queued job.',
};

const testSummary = name => special[name] ?? `Verifies ${name.replace(/^test_/, '').replaceAll('_', ' ')}.`;
const nodes = [];
const edges = [];
for (const file of extracted.results) {
  const [summary, tags] = fileInfo[file.path];
  const type = file.fileCategory === 'config' ? 'config' : 'file';
  const prefix = type === 'config' ? 'config' : 'file';
  nodes.push({ id: `${prefix}:${file.path}`, type, name: file.path.split('/').at(-1), filePath: file.path, summary, tags, complexity: file.nonEmptyLines > 200 ? 'complex' : file.nonEmptyLines >= 50 ? 'moderate' : 'simple' });
  for (const fn of file.functions ?? []) {
    const significant = fn.endLine - fn.startLine + 1 >= 10 || (file.exports ?? []).some(e => e.name === fn.name);
    if (!significant) continue;
    const isTest = file.path.startsWith('tests/');
    const summary = isTest ? testSummary(fn.name) : special[fn.name] ?? `Provides the ${fn.name} operation for this module.`;
    nodes.push({ id: `function:${file.path}:${fn.name}`, type: 'function', name: fn.name, filePath: file.path, lineRange: [fn.startLine, fn.endLine], summary, tags: isTest ? ['test', 'verification', 'python'] : ['utility', 'ollama', 'http-client'], complexity: fn.endLine - fn.startLine + 1 >= 30 ? 'moderate' : 'simple' });
    edges.push({ source: `${prefix}:${file.path}`, target: `function:${file.path}:${fn.name}`, type: 'contains', direction: 'forward', weight: 1.0 });
    if ((file.exports ?? []).some(e => e.name === fn.name)) edges.push({ source: `${prefix}:${file.path}`, target: `function:${file.path}:${fn.name}`, type: 'exports', direction: 'forward', weight: 0.8 });
  }
  for (const cls of file.classes ?? []) {
    const significant = cls.methods.length >= 2 || cls.endLine - cls.startLine + 1 >= 20 || (file.exports ?? []).some(e => e.name === cls.name);
    if (!significant) continue;
    const summary = cls.name === 'PageParser' ? 'HTML parser helper that collects page anchors and local links for tutorial validation.' : 'Test double that simulates a managed worker process for job lifecycle assertions.';
    nodes.push({ id: `class:${file.path}:${cls.name}`, type: 'class', name: cls.name, filePath: file.path, lineRange: [cls.startLine, cls.endLine], summary, tags: ['test', 'helper', 'python'], complexity: 'simple' });
    edges.push({ source: `${prefix}:${file.path}`, target: `class:${file.path}:${cls.name}`, type: 'contains', direction: 'forward', weight: 1.0 });
    if ((file.exports ?? []).some(e => e.name === cls.name)) edges.push({ source: `${prefix}:${file.path}`, target: `class:${file.path}:${cls.name}`, type: 'exports', direction: 'forward', weight: 0.8 });
  }
}

// Split according to the required batch-fragment limits and alphabetical file grouping.
const paths = extracted.results.map(f => f.path).sort();
const parts = Math.ceil(Math.max(nodes.length / 60, edges.length / 120));
const chunkSize = Math.ceil(paths.length / parts);
for (let i = 0; i < parts; i++) {
  const group = new Set(paths.slice(i * chunkSize, (i + 1) * chunkSize));
  const partNodes = nodes.filter(n => group.has(n.filePath));
  const ids = new Set(partNodes.map(n => n.id));
  const partEdges = edges.filter(e => ids.has(e.source));
  const out = { nodes: partNodes, edges: partEdges };
  const target = `${ua}/intermediate/batch-9-part-${i + 1}.json`;
  fs.writeFileSync(target, `${JSON.stringify(out, null, 2)}\n`);
  JSON.parse(fs.readFileSync(target, 'utf8'));
}
console.log(JSON.stringify({ parts, nodes: nodes.length, edges: edges.length }));
