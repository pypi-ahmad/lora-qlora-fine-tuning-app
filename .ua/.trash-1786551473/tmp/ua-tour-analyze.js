const fs = require('fs');

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error('usage: ua-tour-analyze.js input output');
  const { nodes = [], edges = [], layers = [] } = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const fanIn = new Map(nodes.map((node) => [node.id, 0]));
  const fanOut = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges) {
    if (fanIn.has(edge.target)) fanIn.set(edge.target, fanIn.get(edge.target) + 1);
    if (fanOut.has(edge.source)) fanOut.set(edge.source, fanOut.get(edge.source) + 1);
  }
  const rank = (counts, key) => [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, 20)
    .map(([id, value]) => ({ id, [key]: value, name: byId.get(id).name }));
  const maxOut = Math.max(...fanOut.values(), 0);
  const entryPointCandidates = nodes.map((node) => {
    let score = 0;
    const path = node.filePath || '';
    if (node.type === 'document' && path === 'README.md') score += 5;
    if (node.type === 'document' && path.endsWith('.md') && !path.includes('/')) score += 2;
    if (node.type === 'file') {
      if (/^(index|main|app|server|run|manage|wsgi|asgi|__main__)\.(?:ts|js|py|rs|go|c|cpp)$/.test(node.name)) score += 3;
      if (path.split('/').length <= 2) score += 1;
      if (fanOut.get(node.id) >= Math.max(1, maxOut * 0.9)) score += 1;
      const values = [...fanIn.values()].sort((a, b) => a - b);
      if ((fanIn.get(node.id) || 0) <= values[Math.floor(values.length * 0.25)] ) score += 1;
    }
    return { id: node.id, score, name: node.name, summary: node.summary };
  }).filter((candidate) => candidate.score > 0).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, 5);
  const start = entryPointCandidates.find((candidate) => byId.get(candidate.id).type === 'file');
  const adjacency = new Map(nodes.map((node) => [node.id, []]));
  for (const edge of edges) if ((edge.type === 'imports' || edge.type === 'calls') && adjacency.has(edge.source) && byId.has(edge.target)) adjacency.get(edge.source).push(edge.target);
  const order = [], depthMap = {}, byDepth = {};
  if (start) {
    const queue = [[start.id, 0]], seen = new Set([start.id]);
    while (queue.length) {
      const [id, depth] = queue.shift(); order.push(id); depthMap[id] = depth; (byDepth[depth] ||= []).push(id);
      for (const target of adjacency.get(id)) if (!seen.has(target)) { seen.add(target); queue.push([target, depth + 1]); }
    }
  }
  const groups = { documentation: [], infrastructure: [], data: [], config: [] };
  for (const node of nodes) {
    const entry = { id: node.id, name: node.name, type: node.type, summary: node.summary };
    if (node.type === 'document') groups.documentation.push(entry);
    else if (['service', 'pipeline', 'resource'].includes(node.type)) groups.infrastructure.push(entry);
    else if (['table', 'schema', 'endpoint'].includes(node.type)) groups.data.push(entry);
    else if (node.type === 'config') groups.config.push(entry);
  }
  const pairCounts = new Map();
  for (const edge of edges) {
    if (!['imports', 'calls'].includes(edge.type)) continue;
    const reverse = edges.some((other) => other.source === edge.target && other.target === edge.source && other.type === edge.type);
    if (reverse) { const key = [edge.source, edge.target].sort().join('|'); pairCounts.set(key, (pairCounts.get(key) || 0) + 1); }
  }
  const clusters = [...pairCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([key, edgeCount]) => ({ nodes: key.split('|'), edgeCount }));
  const nodeSummaryIndex = Object.fromEntries(nodes.map((node) => [node.id, { name: node.name, type: node.type, summary: node.summary }]));
  fs.writeFileSync(outputPath, JSON.stringify({ scriptCompleted: true, entryPointCandidates, fanInRanking: rank(fanIn, 'fanIn'), fanOutRanking: rank(fanOut, 'fanOut'), bfsTraversal: { startNode: start?.id || null, order, depthMap, byDepth }, nonCodeFiles: groups, clusters, layers: { count: layers.length, list: layers.map(({ id, name, description }) => ({ id, name, description })) }, nodeSummaryIndex, totalNodes: nodes.length, totalEdges: edges.length }, null, 2));
} catch (error) { console.error(error.message); process.exit(1); }
