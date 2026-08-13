const fs = require('fs');

try {
  const [inputPath, outputPath] = process.argv.slice(2);
  const { fileNodes, importEdges, allEdges } = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const paths = fileNodes.map((node) => node.filePath || node.name);
  const split = (value) => value.replaceAll('\\', '/').split('/');
  const firstSegments = paths.map((path) => split(path)[0]);
  const commonPrefix = firstSegments.every((segment) => segment === firstSegments[0]) && paths.some((path) => split(path).length > 1)
    ? firstSegments[0]
    : null;
  const groupFor = (node) => {
    const parts = split(node.filePath || node.name);
    const index = commonPrefix && parts[0] === commonPrefix ? 1 : 0;
    return parts[index] || 'root';
  };
  const directoryGroups = {};
  const nodeTypeGroups = {};
  const groupById = {};
  for (const node of fileNodes) {
    const group = groupFor(node);
    groupById[node.id] = group;
    (directoryGroups[group] ||= []).push(node.id);
    (nodeTypeGroups[node.type] ||= []).push(node.id);
  }
  const fanIn = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const fanOut = Object.fromEntries(fileNodes.map((node) => [node.id, 0]));
  const interCounts = new Map();
  const groupTotals = Object.fromEntries(Object.keys(directoryGroups).map((group) => [group, { internalEdges: 0, totalEdges: 0 }]));
  for (const edge of importEdges) {
    fanOut[edge.source]++;
    fanIn[edge.target]++;
    const from = groupById[edge.source], to = groupById[edge.target];
    if (!from || !to) continue;
    groupTotals[from].totalEdges++;
    groupTotals[to].totalEdges++;
    if (from === to) {
      groupTotals[from].internalEdges++;
    } else {
      const key = `${from}\u0000${to}`;
      interCounts.set(key, (interCounts.get(key) || 0) + 1);
    }
  }
  const cross = new Map();
  for (const edge of allEdges) {
    const from = fileNodes.find((node) => node.id === edge.source);
    const to = fileNodes.find((node) => node.id === edge.target);
    if (!from || !to) continue;
    const key = `${from.type}\u0000${to.type}\u0000${edge.type}`;
    const item = cross.get(key) || { fromType: from.type, toType: to.type, edgeType: edge.type, count: 0 };
    item.count++;
    cross.set(key, item);
  }
  const patterns = { app_pages: 'ui', src: 'service', tests: 'test', docs: 'documentation', '.github': 'ci-cd', root: 'config' };
  const infraFiles = fileNodes.filter((node) => node.type === 'pipeline' || /docker|terraform|\.github\/workflows/i.test(node.filePath || '')).map((node) => node.filePath);
  const results = {
    scriptCompleted: true,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges: [...cross.values()],
    interGroupImports: [...interCounts.entries()].map(([key, count]) => { const [from, to] = key.split('\u0000'); return { from, to, count }; }),
    intraGroupDensity: Object.fromEntries(Object.entries(groupTotals).map(([group, data]) => [group, { ...data, density: data.totalEdges ? data.internalEdges / data.totalEdges : 0 }])),
    patternMatches: Object.fromEntries(Object.keys(directoryGroups).map((group) => [group, patterns[group] || 'unclassified'])),
    deploymentTopology: { hasDockerfile: false, hasCompose: false, hasK8s: false, hasTerraform: false, hasCI: infraFiles.length > 0, infraFiles },
    dataPipeline: { schemaFiles: [], migrationFiles: [], dataModelFiles: [], apiHandlerFiles: [] },
    docCoverage: { groupsWithDocs: directoryGroups.docs ? 1 : 0, totalGroups: Object.keys(directoryGroups).length, coverageRatio: directoryGroups.docs ? 1 / Object.keys(directoryGroups).length : 0, undocumentedGroups: Object.keys(directoryGroups).filter((group) => group !== 'docs') },
    dependencyDirection: [...interCounts.entries()].map(([key]) => { const [dependent, dependsOn] = key.split('\u0000'); return { dependent, dependsOn }; }),
    fileStats: { totalFileNodes: fileNodes.length, filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([group, ids]) => [group, ids.length])), nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([type, ids]) => [type, ids.length])) },
    fileFanIn: fanIn,
    fileFanOut: fanOut,
  };
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2));
} catch (error) {
  console.error(error.stack || error.message);
  process.exit(1);
}
