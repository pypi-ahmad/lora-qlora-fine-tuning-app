const fs = require("fs");
const path = require("path");

const uaDir = process.argv[2];
const gitCommitHash = process.argv[3];
const source = path.join(uaDir, "intermediate", "assembled-graph.json");
const graph = JSON.parse(fs.readFileSync(source, "utf8"));
fs.writeFileSync(path.join(uaDir, "knowledge-graph.json"), JSON.stringify(graph, null, 2));
fs.writeFileSync(
  path.join(uaDir, "meta.json"),
  JSON.stringify(
    {
      lastAnalyzedAt: new Date().toISOString(),
      gitCommitHash,
      version: "1.0.0",
      analyzedFiles: 43,
    },
    null,
    2,
  ),
);
