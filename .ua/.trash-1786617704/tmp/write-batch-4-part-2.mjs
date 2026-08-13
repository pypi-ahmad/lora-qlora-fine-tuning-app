import { writeFileSync } from "node:fs";
const files = [
  ["docs/index.html", "index.html", "Comprehensive static HTML edition of the project's zero-to-mastery handbook and supporting documentation.", "complex", ["documentation","html","tutorial","training"]],
  ["src/lora_finetune_studio/hardware.py", "hardware.py", "Provides local system scanning, CUDA memory statistics, GPU cleanup, and model-size guidance for the application.", "moderate", ["hardware","cuda","diagnostics","utility"]],
  ["src/lora_finetune_studio/inference.py", "inference.py", "Loads a base model and optional adapter to generate text locally while releasing CUDA resources after execution.", "moderate", ["inference","transformers","peft","cuda"]],
  ["src/lora_finetune_studio/lifecycle.py", "lifecycle.py", "Provides a small lifecycle helper for scheduling a safe application shutdown.", "simple", ["lifecycle","shutdown","utility"]],
  ["streamlit_app.py", "streamlit_app.py", "Application entry point that initializes shared state and assembles the Streamlit fine-tuning workflow.", "moderate", ["entry-point","streamlit","training","ui"]],
  ["tests/test_app.py", "test_app.py", "End-to-end Streamlit test covering application startup, navigation, configuration editing, and shutdown behavior.", "moderate", ["test","streamlit","integration","ui"]],
  ["tests/test_hardware.py", "test_hardware.py", "Unit tests for hardware detection, CUDA-memory helpers, and model-size warning behavior.", "moderate", ["test","hardware","cuda","unit-test"]],
  ["tests/test_inference.py", "test_inference.py", "Unit test ensuring inference cleans up CUDA memory when model loading fails.", "simple", ["test","inference","cuda","unit-test"]],
  ["tests/test_jobs.py", "test_jobs.py", "Unit tests for persistent training-run creation, FIFO dispatch, cancellation, worker handling, and Unsloth runtime selection.", "complex", ["test","job-queue","training","unit-test"]],
  ["tests/test_lifecycle.py", "test_lifecycle.py", "Unit test for the asynchronous application-exit scheduler.", "simple", ["test","lifecycle","shutdown","unit-test"]],
  ["tests/test_models.py", "test_models.py", "Unit tests for training configuration serialization, compatibility rules, bounds, and preset validation.", "complex", ["test","configuration","validation","unit-test"]]
];
const funcs = [
  ["src/lora_finetune_studio/hardware.py","scan_system",69,115,"Collects operating-system, package, command, and resource information without installing dependencies."],
  ["src/lora_finetune_studio/hardware.py","cuda_memory_stats",118,128,"Returns global and process CUDA memory usage, rejecting systems without an available GPU."],
  ["src/lora_finetune_studio/hardware.py","detect_hardware",139,178,"Builds the application hardware profile from CUDA, memory, disk, and processor information."],
  ["src/lora_finetune_studio/hardware.py","model_size_warning",181,192,"Returns a warning when a selected model exceeds the detected GPU-memory budget."],
  ["src/lora_finetune_studio/inference.py","generate_text",15,39,"Runs local generation and guarantees CUDA cleanup after the attempt."],
  ["src/lora_finetune_studio/inference.py","_generate_text",42,85,"Loads tokenizer, base model, and optional PEFT adapter before decoding generated text."],
  ["tests/test_app.py","test_app_starts_without_ollama",11,196,"Exercises the Streamlit application when the optional Ollama runtime is unavailable."],
  ["tests/test_hardware.py","test_model_size_warning_uses_detected_limit",15,28,"Verifies model-size guidance respects the detected hardware limit."],
  ["tests/test_hardware.py","test_cuda_memory_stats_reports_global_and_process_usage",31,47,"Verifies reported CUDA metrics include global and process allocations."],
  ["tests/test_hardware.py","test_release_unused_cuda_memory_runs_gc_and_empties_cache",50,60,"Verifies GPU cleanup triggers collection and CUDA cache release."],
  ["tests/test_hardware.py","test_system_scan_reports_live_resources_without_installing",72,106,"Verifies system scanning reports installed tools and live resource values."],
  ["tests/test_hardware.py","test_system_scan_recognizes_native_linux",109,118,"Verifies platform detection identifies native Linux correctly."],
  ["tests/test_inference.py","test_generate_text_releases_cuda_memory_after_loading_failure",6,25,"Verifies inference releases CUDA memory when model loading raises an error."],
  ["tests/test_jobs.py","test_create_run_persists_safe_config",11,28,"Verifies queued runs persist a safe serialized training configuration."],
  ["tests/test_jobs.py","test_create_run_appends_jobs_in_fifo_order",31,41,"Verifies newly created runs are queued in FIFO order."],
  ["tests/test_jobs.py","test_dispatch_starts_only_first_waiting_run",44,66,"Verifies dispatch launches only the first waiting training run."],
  ["tests/test_jobs.py","test_cancelling_running_job_starts_next_waiting_run",69,110,"Verifies cancellation advances the queue to the next waiting run."],
  ["tests/test_jobs.py","test_dispatch_marks_dead_worker_failed_before_continuing",113,134,"Verifies stale workers are failed before the dispatcher continues."],
  ["tests/test_jobs.py","test_resume_appends_checkpoint_run_to_queue_when_worker_is_active",137,164,"Verifies checkpoint resumes join the queue while another worker is active."],
  ["tests/test_jobs.py","test_list_runs_orders_active_then_waiting_then_history",167,190,"Verifies run listing orders active, waiting, and historical jobs."],
  ["tests/test_jobs.py","test_dispatch_marks_launch_failure_and_continues_queue",193,216,"Verifies a launch failure is recorded without blocking later work."],
  ["tests/test_jobs.py","test_dispatch_waits_for_terminal_worker_process_to_exit",219,242,"Verifies dispatch waits for terminal worker processes to exit."],
  ["tests/test_jobs.py","_running_test_run",268,280,"Creates a persisted running run fixture for process-cancellation tests."],
  ["tests/test_jobs.py","test_cancel_run_stops_only_its_training_worker",283,299,"Verifies cancellation terminates only the matching training worker."],
  ["tests/test_jobs.py","test_cancel_run_rejects_unrelated_process",302,312,"Verifies cancellation rejects a process unrelated to the selected run."],
  ["tests/test_jobs.py","test_cancel_active_run_delegates_to_active_run",315,325,"Verifies active-run cancellation delegates to the selected run handler."],
  ["tests/test_jobs.py","test_launch_run_uses_unsloth_interpreter_and_source_path",328,358,"Verifies Unsloth jobs launch through the configured runtime and source path."],
  ["tests/test_jobs.py","test_launch_run_keeps_current_interpreter_when_unsloth_is_off",361,383,"Verifies standard jobs keep the current Python interpreter."],
  ["tests/test_jobs.py","test_launch_run_preserves_base_interpreter_across_worker_handoffs",386,412,"Verifies worker handoffs preserve the base interpreter selection."],
  ["tests/test_jobs.py","test_launch_run_rejects_missing_unsloth_runtime",415,435,"Verifies launch rejects enabled Unsloth mode without a valid runtime."],
  ["tests/test_lifecycle.py","test_schedule_application_exit_uses_daemon_timer",4,25,"Verifies scheduled shutdown uses a daemon timer."],
  ["tests/test_models.py","test_config_round_trip",18,31,"Verifies training configuration survives serialization and validation."],
  ["tests/test_models.py","test_old_config_migrates_single_dataset_and_standard_backend",34,54,"Verifies legacy configurations migrate to current dataset and backend fields."],
  ["tests/test_models.py","test_config_requires_push_destination",57,66,"Verifies hub push configuration requires a destination."],
  ["tests/test_models.py","test_config_rejects_mixed_and_duplicate_datasets",69,83,"Verifies incompatible or duplicate datasets are rejected."],
  ["tests/test_models.py","test_recipe_matrix_accepts_all_advertised_pairs",88,116,"Verifies every advertised approach and method pairing validates."],
  ["tests/test_models.py","test_config_rejects_approach_dataset_mismatch_and_unsloth_oft",119,138,"Verifies invalid approach datasets and Unsloth OFT combinations are rejected."],
  ["tests/test_models.py","test_config_rejects_unsloth_fp32",141,156,"Verifies Unsloth mode rejects unsupported FP32 computation."],
  ["tests/test_models.py","test_kto_requires_actual_batch_size_above_one",159,179,"Verifies KTO requires an effective batch size greater than one."],
  ["tests/test_models.py","test_config_rejects_learning_rate_outside_supported_range",183,199,"Verifies learning-rate bounds are enforced."],
  ["tests/test_models.py","test_config_rejects_invalid_maximum_samples",203,217,"Verifies maximum sample count must be valid."],
  ["tests/test_models.py","test_config_rejects_invalid_maximum_gradient_norm",221,235,"Verifies gradient-norm bounds are enforced."]
];
const cls = ["FakeWorkerProcess",253,265,"Minimal worker-process double used to test cancellation and process handling."];
const nodes = files.map(([path,name,summary,complexity,tags]) => ({id:`file:${path}`,type:"file",name,filePath:path,summary,tags,complexity}));
for (const [path,name,start,end,summary] of funcs) nodes.push({id:`function:${path}:${name}`,type:"function",name,filePath:path,lineRange:[start,end],summary,tags:path.startsWith("tests/")?["test","validation","training"]:["utility","training","cuda"],complexity:end-start>35?"moderate":"simple"});
nodes.push({id:`class:tests/test_jobs.py:${cls[0]}`,type:"class",name:cls[0],filePath:"tests/test_jobs.py",lineRange:[cls[1],cls[2]],summary:cls[3],tags:["test","process","fixture"],complexity:"simple"});
const edges=[];
for(const n of nodes.filter(n=>n.type==="function"||n.type==="class")){const file=`file:${n.filePath}`;edges.push({source:file,target:n.id,type:"contains",direction:"forward",weight:1.0},{source:file,target:n.id,type:"exports",direction:"forward",weight:0.8});}
edges.push(
 {source:"file:src/lora_finetune_studio/hardware.py",target:"file:src/lora_finetune_studio/models.py",type:"imports",direction:"forward",weight:0.7},
 {source:"file:src/lora_finetune_studio/inference.py",target:"file:src/lora_finetune_studio/hardware.py",type:"imports",direction:"forward",weight:0.7},
 {source:"file:streamlit_app.py",target:"file:tests/test_app.py",type:"tested_by",direction:"forward",weight:0.5},
 {source:"file:src/lora_finetune_studio/hardware.py",target:"file:tests/test_hardware.py",type:"tested_by",direction:"forward",weight:0.5},
 {source:"file:src/lora_finetune_studio/inference.py",target:"file:tests/test_inference.py",type:"tested_by",direction:"forward",weight:0.5},
 {source:"file:src/lora_finetune_studio/lifecycle.py",target:"file:tests/test_lifecycle.py",type:"tested_by",direction:"forward",weight:0.5},
 {source:"file:src/lora_finetune_studio/jobs.py",target:"file:tests/test_jobs.py",type:"tested_by",direction:"forward",weight:0.5},
 {source:"file:src/lora_finetune_studio/models.py",target:"file:tests/test_models.py",type:"tested_by",direction:"forward",weight:0.5}
);
writeFileSync(new URL("../intermediate/batch-4-part-2.json", import.meta.url), JSON.stringify({nodes,edges},null,2));
