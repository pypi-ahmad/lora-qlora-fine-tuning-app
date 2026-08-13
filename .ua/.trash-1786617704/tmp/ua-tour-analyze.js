const fs = require("fs");

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error("Usage: ua-tour-analyze.js <input> <output>");
  const graph = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges) {
    if (fanOut.has(edge.source)) fanOut.set(edge.source, fanOut.get(edge.source) + 1);
    if (fanIn.has(edge.target)) fanIn.set(edge.target, fanIn.get(edge.target) + 1);
  }
  const ranked = (counts, key) => [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 20).map(([id, value]) => ({ id, [key]: value, name: byId.get(id).name }));
  const outValues = [...fanOut.values()].sort((a, b) => a - b);
  const inValues = [...fanIn.values()].sort((a, b) => a - b);
  const highOut = outValues[Math.max(0, Math.ceil(outValues.length * 0.9) - 1)] || 0;
  const lowIn = inValues[Math.max(0, Math.ceil(inValues.length * 0.25) - 1)] || 0;
  const entryPointCandidates = nodes.map((node) => {
    const path = node.filePath || node.name || "";
    const basename = path.split("/").pop();
    const isDoc = node.type === "document";
    let score = 0;
    if (isDoc && path === "README.md") score += 5;
    else if (isDoc && !path.includes("/")) score += 2;
    if (!isDoc && /^(index|main|app|server|mod|manage|wsgi|asgi|run|__main__)\.(ts|js|rs|go|py)$|^(Application\.java|Main\.java|Program\.cs|config\.ru|index\.php|App\.swift|Application\.kt|main\.(cpp|c))$/.test(basename)) score += 3;
    if (!isDoc && path.split("/").length <= 2) score += 1;
    if (!isDoc && fanOut.get(node.id) >= highOut) score += 1;
    if (!isDoc && fanIn.get(node.id) <= lowIn) score += 1;
    return { id: node.id, score, name: node.name, summary: node.summary || "" };
  }).filter((candidate) => candidate.score > 0).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, 5);
  const start = entryPointCandidates.find((candidate) => byId.get(candidate.id).type !== "document");
  const forward = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges) if (["imports", "calls"].includes(edge.type) && forward.has(edge.source) && forward.has(edge.target)) forward.get(edge.source).push(edge.target);
  const order = [], depthMap = {}, byDepth = {};
  if (start) {
    const queue = [start.id]; depthMap[start.id] = 0;
    for (let index = 0; index < queue.length; index++) {
      const id = queue[index], depth = depthMap[id]; order.push(id); (byDepth[depth] ||= []).push(id);
      for (const target of forward.get(id).sort()) if (!(target in depthMap)) { depthMap[target] = depth + 1; queue.push(target); }
    }
  }
  const nonCodeFiles = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const node of nodes) {
    const item = { id: node.id, name: node.name, type: node.type, summary: node.summary || "" };
    if (node.type === "document") nonCodeFiles.documentation.push(item);
    else if (["service", "pipeline", "resource"].includes(node.type)) nonCodeFiles.infrastructure.push(item);
    else if (["table", "schema", "endpoint"].includes(node.type)) nonCodeFiles.data.push(item);
    else if (node.type === "config") nonCodeFiles.config.push(item);
  }
  const reciprocal = new Map();
  const pairs = new Set(edges.filter((edge) => ["imports", "calls"].includes(edge.type)).map((edge) => `${edge.source}\u0000${edge.target}`));
  for (const key of pairs) { const [a, b] = key.split("\u0000"); if (pairs.has(`${b}\u0000${a}`)) reciprocal.set([a,b].sort().join("\u0000"), [a,b]); }
  const clusters = [...reciprocal.values()].slice(0, 10).map((nodes) => ({ nodes: [...new Set(nodes)], edgeCount: 2 }));
  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [node.id, { name: node.name, type: node.type, summary: node.summary || "" }]));
  const result = { scriptCompleted: true, entryPointCandidates, fanInRanking: ranked(fanIn, "fanIn"), fanOutRanking: ranked(fanOut, "fanOut"), bfsTraversal: { startNode: start?.id || null, order, depthMap, byDepth }, nonCodeFiles, clusters, layers: { count: (graph.layers || []).length, list: graph.layers || [] }, nodeSummaryIndex, totalNodes: nodes.length, totalEdges: edges.length };
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + "\n");
} catch (error) { console.error(error.stack || error.message); process.exit(1); }
