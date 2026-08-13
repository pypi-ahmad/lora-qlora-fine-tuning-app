const fs = require("fs");

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const [inputPath, outputPath] = process.argv.slice(2);
if (!inputPath || !outputPath) fail("Usage: node ua-arch-analyze.js INPUT OUTPUT");

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, "utf8"));
} catch (error) {
  fail(`Cannot read input: ${error.message}`);
}

const files = input.fileNodes || [];
const byId = new Map(files.map((node) => [node.id, node]));
const segments = files.map((node) => (node.filePath || "").split("/").slice(0, -1));
let prefix = segments[0] || [];
for (const parts of segments.slice(1)) {
  let count = 0;
  while (count < prefix.length && count < parts.length && prefix[count] === parts[count]) count++;
  prefix = prefix.slice(0, count);
}
function groupFor(node) {
  const parts = (node.filePath || "").split("/");
  const rest = parts.slice(prefix.length);
  if (rest.length > 1) return rest[0];
  if (parts.length > 1) return parts[0];
  if (/\.(test|spec)\./.test(node.name) || /^test_.*\.py$/.test(node.name)) return "test";
  if (/config|toml|ya?ml|json/.test(node.name)) return "config";
  return "root";
}
const directoryGroups = {};
const nodeTypeGroups = {};
const groups = new Map();
for (const node of files) {
  const group = groupFor(node);
  groups.set(node.id, group);
  (directoryGroups[group] ||= []).push(node.id);
  (nodeTypeGroups[node.type] ||= []).push(node.id);
}
const fanIn = Object.fromEntries(files.map((node) => [node.id, 0]));
const fanOut = Object.fromEntries(files.map((node) => [node.id, 0]));
const inter = new Map();
const internal = new Map();
const involved = new Map();
for (const edge of input.importEdges || []) {
  if (!byId.has(edge.source) || !byId.has(edge.target)) continue;
  fanOut[edge.source]++;
  fanIn[edge.target]++;
  const from = groups.get(edge.source), to = groups.get(edge.target);
  const key = `${from}\u0000${to}`;
  inter.set(key, (inter.get(key) || 0) + 1);
  involved.set(from, (involved.get(from) || 0) + 1);
  involved.set(to, (involved.get(to) || 0) + 1);
  if (from === to) internal.set(from, (internal.get(from) || 0) + 1);
}
const cross = new Map();
for (const edge of input.allEdges || []) {
  const source = byId.get(edge.source), target = byId.get(edge.target);
  if (!source || !target) continue;
  const key = `${source.type}\u0000${target.type}\u0000${edge.type}`;
  cross.set(key, (cross.get(key) || 0) + 1);
}
const patternMap = {
  routes: "api", api: "api", controllers: "api", endpoints: "api", handlers: "api",
  services: "service", core: "service", lib: "service", domain: "service", logic: "service",
  models: "data", db: "data", data: "data", persistence: "data", repository: "data", entities: "data",
  components: "ui", views: "ui", pages: "ui", ui: "ui", layouts: "ui", screens: "ui",
  utils: "utility", helpers: "utility", common: "utility", shared: "utility", tools: "utility",
  config: "config", constants: "config", env: "config", settings: "config", tests: "test", test: "test",
  docs: "documentation", documentation: "documentation", wiki: "documentation", ".github": "ci-cd",
  deploy: "infrastructure", deployment: "infrastructure", infra: "infrastructure", infrastructure: "infrastructure",
};
const patternMatches = Object.fromEntries(Object.keys(directoryGroups).map((group) => [group, patternMap[group] || (group === "root" ? "root" : "unclassified")]));
const infraFiles = files.filter((node) => /(^|\/)(Dockerfile|docker-compose)|\.github\/workflows|\.tf(vars)?$/.test(node.filePath)).map((node) => node.filePath);
const result = {
  scriptCompleted: true,
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges: [...cross].map(([key, count]) => { const [fromType, toType, edgeType] = key.split("\u0000"); return {fromType, toType, edgeType, count}; }),
  interGroupImports: [...inter].map(([key, count]) => { const [from, to] = key.split("\u0000"); return {from, to, count}; }),
  intraGroupDensity: Object.fromEntries(Object.keys(directoryGroups).map((group) => { const totalEdges = involved.get(group) || 0; const internalEdges = internal.get(group) || 0; return [group, {internalEdges, totalEdges, density: totalEdges ? internalEdges / totalEdges : 0}]; })),
  patternMatches,
  deploymentTopology: {hasDockerfile: infraFiles.some((p) => /Dockerfile/.test(p)), hasCompose: infraFiles.some((p) => /docker-compose/.test(p)), hasK8s: infraFiles.some((p) => /k8s|kubernetes|helm|charts/.test(p)), hasTerraform: infraFiles.some((p) => /\.tf/.test(p)), hasCI: infraFiles.some((p) => /\.github\/workflows/.test(p)), infraFiles},
  dataPipeline: {schemaFiles: files.filter((n) => n.type === "schema" || /\.(sql|graphql|gql|proto)$/.test(n.filePath)).map((n) => n.filePath), migrationFiles: files.filter((n) => /migrations\//.test(n.filePath)).map((n) => n.filePath), dataModelFiles: files.filter((n) => /models|schema|data/.test(n.filePath)).map((n) => n.filePath), apiHandlerFiles: files.filter((n) => /routes|api|controllers|handlers/.test(n.filePath)).map((n) => n.filePath)},
  docCoverage: {groupsWithDocs: Object.keys(directoryGroups).filter((g) => directoryGroups[g].some((id) => byId.get(id).type === "document" || /README\.md$/.test(byId.get(id).filePath))).length, totalGroups: Object.keys(directoryGroups).length, coverageRatio: 0, undocumentedGroups: []},
  dependencyDirection: [...inter].filter(([key]) => { const [a,b] = key.split("\u0000"); return a !== b; }).map(([key, count]) => { const [dependent, dependsOn] = key.split("\u0000"); const reverse = inter.get(`${dependsOn}\u0000${dependent}`) || 0; return count > reverse ? {dependent, dependsOn} : null; }).filter(Boolean),
  fileStats: {totalFileNodes: files.length, filesPerGroup: Object.fromEntries(Object.entries(directoryGroups).map(([g, ids]) => [g, ids.length])), nodeTypeCounts: Object.fromEntries(Object.entries(nodeTypeGroups).map(([t, ids]) => [t, ids.length]))},
  fileFanIn: fanIn,
  fileFanOut: fanOut,
};
result.docCoverage.coverageRatio = result.docCoverage.totalGroups ? result.docCoverage.groupsWithDocs / result.docCoverage.totalGroups : 0;
result.docCoverage.undocumentedGroups = Object.keys(directoryGroups).filter((g) => !directoryGroups[g].some((id) => byId.get(id).type === "document" || /README\.md$/.test(byId.get(id).filePath)));
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
