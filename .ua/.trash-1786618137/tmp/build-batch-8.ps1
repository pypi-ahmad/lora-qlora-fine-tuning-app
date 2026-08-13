$ErrorActionPreference = 'Stop'
$ua = (Resolve-Path '.ua').Path
$batch = (Get-Content -Raw "$ua\intermediate\batches.json" | ConvertFrom-Json).batches | Where-Object { $_.batchIndex -eq 8 }
$extract = Get-Content -Raw "$ua\tmp\ua-file-extract-results-8.json" | ConvertFrom-Json
$resultByPath = @{}
foreach ($result in $extract.results) { $resultByPath[$result.path] = $result }

$fileSummaries = @{
  '.gitattributes' = 'Defines Git text and attribute handling for repository files.'
  '.python-version' = 'Pins the Python version expected by the project tooling.'
  '.streamlit/config.toml' = 'Configures the local Streamlit server appearance and behavior.'
  'app_pages/dataset.py' = 'Implements the Streamlit dataset page for inspecting, validating, and combining compatible training datasets.'
  'app_pages/gpu_memory.py' = 'Shows live CUDA memory metrics and safely releases unused memory when no training run is active.'
  'app_pages/model.py' = 'Implements the model-selection page and inspects Hugging Face model metadata before training.'
  'app_pages/monitor.py' = 'Implements the training monitor, FIFO queue controls, checkpoint recovery, and completed-adapter testing UI.'
  'app_pages/ollama.py' = 'Provides an independent local Ollama chat playground for already installed models.'
  'app_pages/review.py' = 'Reviews the saved training configuration, reports blockers, and launches or queues a run.'
  'app_pages/system.py' = 'Displays a read-only local runtime, CUDA, hardware, token, and Unsloth readiness assessment.'
  'app_pages/training.py' = 'Implements the training-settings page, including supported approach and PEFT method selection.'
  'docs/.nojekyll' = 'Prevents GitHub Pages from processing the generated documentation site with Jekyll.'
  'docs/assets/search-index.js' = 'Contains the generated client-side documentation search index.'
  'docs/assets/site.css' = 'Provides the styling shared by the generated tutorial documentation site.'
  'docs/assets/site.js' = 'Provides client-side navigation, theme, and search behavior for the generated tutorial site.'
  'examples/preference_sample.jsonl' = 'Supplies a small preference-pair dataset example for alignment-training workflows.'
  'examples/sft_sample.jsonl' = 'Supplies a small supervised fine-tuning dataset example for quick application tests.'
  'scripts/build_tutorial.py' = 'Builds the Zero to Mastery tutorial into a multi-page HTML site and a verified PDF handbook.'
  'scripts/tutorial_assets/site.css' = 'Stores source CSS copied into the generated tutorial website.'
  'scripts/tutorial_assets/site.js' = 'Stores source JavaScript copied into the generated tutorial website.'
  'src/lora_finetune_studio/__init__.py' = 'Defines the lora_finetune_studio Python package namespace.'
  'src/lora_finetune_studio/cli.py' = 'Provides the console entry point that launches the Streamlit application.'
  'src/lora_finetune_studio/hardware.py' = 'Detects local hardware and software readiness, reports CUDA memory, and makes conservative model-size recommendations.'
  'src/lora_finetune_studio/inference.py' = 'Runs short-lived, quantized adapter inference for comparing a completed training run.'
  'src/lora_finetune_studio/lifecycle.py' = 'Provides a delayed process-exit helper so Streamlit can render shutdown feedback.'
}
$functionSummaries = @{
  'source_label'='Formats a dataset source as a compact label for the dataset page.'; 'source_identity'='Builds the stable source identity used to detect duplicate datasets.'; 'source_key'='Hashes a dataset source identity into a short key.'; 'is_compatible'='Checks whether a dataset format is accepted by the selected recipe.'; 'clear_pending'='Clears pending dataset state from the Streamlit session.'; 'save_pending_dataset'='Adds or replaces a validated pending dataset in the selected dataset collection.'
  'training_monitor'='Refreshes run state, queue status, progress, logs, recovery actions, and adapter evaluation controls.'
  'approach_changed'='Resets dependent training defaults after the selected approach changes.'; 'method_changed'='Updates configuration state when the PEFT method changes.'; 'compute_type_changed'='Disables unsupported Unsloth use when the compute precision changes.'; 'apply_preset_defaults'='Applies the selected preset values to the training form state.'
  'slugify'='Converts text into a URL-safe chapter slug.'; 'split_source'='Separates a source reference into display components.'; 'render_markdown'='Renders Markdown input as HTML.'; 'plain_text'='Removes markup for text-only output.'; 'pdf_safe'='Normalizes rendered text for the PDF renderer.'; 'chapter_summary'='Extracts a concise summary from rendered chapter content.'; 'load_course'='Loads and parses the tutorial source into chapter records.'; 'nav_items'='Builds navigation links for tutorial chapters.'; 'site_shell'='Wraps page content in the shared HTML site shell.'; 'index_content'='Builds the tutorial landing-page content.'; 'chapter_content'='Builds the content for one tutorial chapter page.'; 'rewrite_root_links'='Rewrites root-relative links for generated pages.'; 'build_search_index'='Builds the browser search index from tutorial chapters.'; 'build_site_files'='Assembles generated HTML, assets, manifest, and PDF files.'; 'pdf_styles'='Defines ReportLab styles used by the tutorial PDF.'; 'inline_markup'='Converts inline HTML markup into PDF flowables.'; 'paragraph_from_tag'='Converts an HTML tag to a ReportLab paragraph.'; 'table_flowable'='Converts an HTML table into a ReportLab table.'; 'list_flowable'='Converts an HTML list into ReportLab flowables.'; 'html_to_flowables'='Converts rendered tutorial HTML into PDF flowable elements.'; 'cover_flowables'='Builds the tutorial PDF cover pages.'; 'build_pdf'='Generates the tutorial handbook PDF from parsed chapters.'; 'file_digest'='Calculates a content digest for a generated file.'; 'manifest_bytes'='Serializes the generated-site file manifest.'; 'read_manifest'='Reads a prior generated-site manifest.'; 'write_site'='Writes generated tutorial files to the output directory.'; 'compare_files'='Compares generated files with the checked-in output.'; 'pdf_content_signature'='Extracts a stable signature for PDF verification.'; 'run'='Builds or verifies tutorial artifacts and reports mismatches.'; 'main'='Parses command-line arguments and invokes the tutorial build workflow.'
  '_package_status'='Reports whether a named optional package is installed and its version.'; 'scan_system'='Collects a read-only operating-system, runtime, resource, and integration inventory.'; 'cuda_memory_stats'='Reads current first-GPU CUDA memory statistics.'; 'release_unused_cuda_memory'='Requests garbage collection and clears unused PyTorch CUDA cache.'; 'detect_hardware'='Builds the hardware profile used to constrain training choices.'; 'model_size_warning'='Returns a warning when a model is unsuitable for the detected hardware.'
  'generate_text'='Runs adapter inference and ensures CUDA cache cleanup afterward.'; '_generate_text'='Loads a quantized causal model and optional adapter, then generates a response.'; 'schedule_application_exit'='Schedules a daemon timer to end the application process after a short delay.'
}
$classSummaries = @{
  'Chapter'='Stores parsed tutorial chapter metadata and its source filename.'; 'HandbookDocTemplate'='Customizes ReportLab document metadata, cover pages, headers, and footers for the handbook.'
  'CudaMemoryStats'='Immutable value object for first-GPU CUDA memory measurements.'; 'SoftwareStatus'='Immutable value object for optional integration availability and version detail.'; 'SystemScan'='Immutable record of local operating-system, runtime, resource, and software scan results.'
}

function FileType($file) { if ($file.fileCategory -eq 'config') { return 'config' }; if ($file.fileCategory -eq 'markup') { return 'file' }; return 'file' }
function FileTags($file) {
  if ($file.path -like 'app_pages/*') { return @('streamlit','ui-page','training','local-tool') }
  if ($file.path -like 'scripts/*') { return @('tutorial','build-tool','documentation','automation') }
  if ($file.path -like 'docs/*') { return @('documentation','static-site','frontend') }
  if ($file.path -like 'examples/*') { return @('example-data','training-data','jsonl') }
  if ($file.path -like 'src/*') { return @('python','service','local-runtime') }
  if ($file.fileCategory -eq 'config') { return @('configuration','streamlit','runtime') }
  return @('configuration','repository','tooling')
}
function Complexity($result, $file) { if ($result -and $result.nonEmptyLines -gt 200) { return 'complex' }; if (($result -and $result.nonEmptyLines -ge 50) -or $file.sizeLines -ge 50) { return 'moderate' }; return 'simple' }

$allNodes = New-Object System.Collections.Generic.List[object]
$allEdges = New-Object System.Collections.Generic.List[object]
$nodesByFile = @{}
foreach ($file in $batch.files) {
  $result = $resultByPath[$file.path]
  $id = "$(FileType $file):$($file.path)"
  $node = [ordered]@{id=$id;type=(FileType $file);name=(Split-Path $file.path -Leaf);filePath=$file.path;summary=$fileSummaries[$file.path];tags=(FileTags $file);complexity=(Complexity $result $file)}
  $allNodes.Add($node); $nodesByFile[$file.path] = New-Object System.Collections.Generic.List[object]; $nodesByFile[$file.path].Add($node)
  if (-not $result) { continue }
  $exports = @($result.exports | ForEach-Object { $_.name })
  foreach ($fn in $result.functions) {
    if ([string]::IsNullOrWhiteSpace($fn.name)) { continue }
    $significant = (($fn.endLine - $fn.startLine + 1) -ge 10) -or ($exports -contains $fn.name)
    if (-not $significant) { continue }
    $fnId="function:$($file.path):$($fn.name)"
    $summary=$functionSummaries[$fn.name]; if (-not $summary) { $summary="Supports the $($fn.name) operation within $($file.path)." }
    $nodeComplexity = if (($fn.endLine - $fn.startLine + 1) -gt 50) { 'moderate' } else { 'simple' }
    $fnNode=[ordered]@{id=$fnId;type='function';name=$fn.name;filePath=$file.path;lineRange=@([int]$fn.startLine,[int]$fn.endLine);summary=$summary;tags=@('python','function','application-logic');complexity=$nodeComplexity}
    $allNodes.Add($fnNode); $nodesByFile[$file.path].Add($fnNode)
    $allEdges.Add([ordered]@{source=$id;target=$fnId;type='contains';direction='forward';weight=1.0})
    if ($exports -contains $fn.name) { $allEdges.Add([ordered]@{source=$id;target=$fnId;type='exports';direction='forward';weight=0.8}) }
  }
  foreach ($class in $result.classes) {
    if ([string]::IsNullOrWhiteSpace($class.name)) { continue }
    $significant = (($class.endLine - $class.startLine + 1) -ge 20) -or ($class.methods.Count -ge 2) -or ($exports -contains $class.name)
    if (-not $significant) { continue }
    $classId="class:$($file.path):$($class.name)"
    $summary=$classSummaries[$class.name]; if (-not $summary) { $summary="Represents $($class.name) within $($file.path)." }
    $nodeComplexity = if (($class.endLine - $class.startLine + 1) -gt 50) { 'moderate' } else { 'simple' }
    $classNode=[ordered]@{id=$classId;type='class';name=$class.name;filePath=$file.path;lineRange=@([int]$class.startLine,[int]$class.endLine);summary=$summary;tags=@('python','class','data-model');complexity=$nodeComplexity}
    $allNodes.Add($classNode); $nodesByFile[$file.path].Add($classNode)
    $allEdges.Add([ordered]@{source=$id;target=$classId;type='contains';direction='forward';weight=1.0})
    if ($exports -contains $class.name) { $allEdges.Add([ordered]@{source=$id;target=$classId;type='exports';direction='forward';weight=0.8}) }
  }
  foreach ($targetPath in @($batch.batchImportData.($file.path))) { $allEdges.Add([ordered]@{source=$id;target="file:$targetPath";type='imports';direction='forward';weight=0.7}) }
}
$allEdges.Add([ordered]@{source='function:src/lora_finetune_studio/inference.py:generate_text';target='function:src/lora_finetune_studio/hardware.py:release_unused_cuda_memory';type='calls';direction='forward';weight=0.8})

$sortedFiles=@($batch.files | Sort-Object path); $parts=2; $chunkSize=[math]::Ceiling($sortedFiles.Count/$parts)
for($part=1;$part -le $parts;$part++) {
  $start=($part-1)*$chunkSize; $end=[math]::Min($start+$chunkSize-1,$sortedFiles.Count-1); $paths=@($sortedFiles[$start..$end] | ForEach-Object {$_.path})
  $partNodes=@($allNodes | Where-Object {$paths -contains $_.filePath}); $ids=@($partNodes | ForEach-Object {$_.id})
  $partEdges=@($allEdges | Where-Object {$ids -contains $_.source})
  [ordered]@{nodes=$partNodes;edges=$partEdges} | ConvertTo-Json -Depth 8 | Set-Content -NoNewline "$ua\intermediate\batch-8-part-$part.json"
}
"nodes=$($allNodes.Count) edges=$($allEdges.Count) imports=$(@($allEdges|Where-Object {$_.type -eq 'imports'}).Count)"
