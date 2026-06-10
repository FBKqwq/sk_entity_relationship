$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = "C:\Users\zhurunjie\Desktop\snorkel\code"
Set-Location $Root

$RunDir = "data\s_pipeline_run"
$LogDir = Join-Path $RunDir "logs"
$Log = Join-Path $LogDir "pipeline.log"
New-Item -ItemType Directory -Force -Path `
  (Join-Path $RunDir "lv1"), `
  (Join-Path $RunDir "entity_base"), `
  (Join-Path $RunDir "entity_nodes"), `
  (Join-Path $RunDir "relationship_base"), `
  (Join-Path $RunDir "summaries"), `
  $LogDir | Out-Null

function Write-Log {
  param([string]$Message)
  Add-Content -LiteralPath $Log -Encoding UTF8 -Value "[$(Get-Date -Format s)] $Message"
}

function Run-Step {
  param(
    [string]$Name,
    [string[]]$ArgsList
  )
  Write-Log "START $Name"
  Write-Log ("CMD conda run -n snorkel python " + ($ArgsList -join " "))
  & "E:\Python\Anaconda\condabin\conda.bat" run -n snorkel python @ArgsList *>> $Log
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    Write-Log "FAILED $Name exit_code=$exitCode"
    throw "$Name failed with exit code $exitCode"
  }
  Write-Log "DONE $Name"
}

try {
  Write-Log "PRECHECK"
  Run-Step "preflight" @("-c", "import sys; print(sys.executable)")
  Run-Step "lv1" @(
    "scripts\02_snorkel_lv1_label_chunks.py",
    "--chunks-dir", "data\chunks\S\chunk_json",
    "--output-dir", "data\s_pipeline_run\lv1",
    "--summary", "data\s_pipeline_run\summaries\lv1_summary.json",
    "--config", "configs\weak_supervision.yaml",
    "--llm-config", "configs\llm.yaml",
    "--enable-prompted-llm"
  )
  Run-Step "entity_base" @(
    "scripts\03_llm_extract_entity_base.py",
    "--chunks-dir", "data\chunks\S\chunk_json",
    "--lv1-dir", "data\s_pipeline_run\lv1",
    "--output-dir", "data\s_pipeline_run\entity_base",
    "--summary", "data\s_pipeline_run\summaries\entity_base_summary.json",
    "--config", "configs\llm.yaml",
    "--Full_extraction"
  )
  Run-Step "lv2_entity_nodes" @(
    "scripts\04_snorkel_lv2_label_entities.py",
    "--entities-dir", "data\s_pipeline_run\entity_base",
    "--chunks-dir", "data\chunks\S\chunk_json",
    "--output-dir", "data\s_pipeline_run\entity_nodes",
    "--summary", "data\s_pipeline_run\summaries\entity_lv2_summary.json",
    "--config", "configs\llm.yaml"
  )
  Run-Step "relationships" @(
    "scripts\06_llm_extract_relationship_base.py",
    "--entities-dir", "data\s_pipeline_run\entity_nodes",
    "--chunks-dir", "data\chunks\S\chunk_json",
    "--output-dir", "data\s_pipeline_run\relationship_base",
    "--summary", "data\s_pipeline_run\summaries\relationship_summary.json",
    "--config", "configs\llm.yaml",
    "--audit-batch-size", "50",
    "--accepted-only"
  )
  Write-Log "ALL DONE"
} catch {
  Write-Log ("PIPELINE FAILED " + $_.Exception.Message)
  exit 1
}
