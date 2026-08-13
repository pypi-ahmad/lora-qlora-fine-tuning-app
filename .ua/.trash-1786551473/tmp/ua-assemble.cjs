const fs = require("fs");
const path = require("path");

const uaDir = process.argv[2];
const commit = process.argv[3];
const graphPath = path.join(uaDir, "intermediate", "assembled-graph.json");
const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
const layers = JSON.parse(
  fs.readFileSync(path.join(uaDir, "intermediate", "layers.json"), "utf8"),
);
const tour = JSON.parse(
  fs.readFileSync(path.join(uaDir, "intermediate", "tour.json"), "utf8"),
);
const nodeIds = new Set(graph.nodes.map((node) => node.id));

for (const layer of layers) {
  if (!layer.id || !layer.name || !layer.description || !Array.isArray(layer.nodeIds)) {
    throw new Error(`Invalid layer: ${JSON.stringify(layer)}`);
  }
  layer.nodeIds = layer.nodeIds.filter((id) => nodeIds.has(id));
}
for (const step of tour) {
  if (!Number.isInteger(step.order) || !step.title || !step.description || !Array.isArray(step.nodeIds)) {
    throw new Error(`Invalid tour step: ${JSON.stringify(step)}`);
  }
  step.nodeIds = step.nodeIds.filter((id) => nodeIds.has(id));
}
tour.sort((left, right) => left.order - right.order);

const finalGraph = {
  version: "1.0.0",
  project: {
    name: "lora-finetune-studio",
    languages: ["batch", "html", "jsonl", "markdown", "python", "shell", "toml", "unknown", "yaml"],
    frameworks: ["GitHub Actions", "Pytest"],
    description: "Local Streamlit studio for LoRA and QLoRA supervised fine-tuning",
    analyzedAt: new Date().toISOString(),
    gitCommitHash: commit,
  },
  nodes: graph.nodes,
  edges: graph.edges,
  layers,
  tour,
};
fs.writeFileSync(graphPath, JSON.stringify(finalGraph, null, 2));
