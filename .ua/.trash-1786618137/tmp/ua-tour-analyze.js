const fs = require("fs");
try {
  const [inputPath, outputPath] = process.argv.slice(2);
  if (!inputPath || !outputPath) throw new Error("Usage: ua-tour-analyze.js <input> <output>");
  const graph = JSON.parse(fs.readFileSync(inputPath, "utf8"));
  const nodes = graph.nodes || [], edges = graph.edges || [], byId = new Map(nodes.map(n => [n.id, n]));
  const fanIn = new Map(nodes.map(n => [n.id, 0])), fanOut = new Map(nodes.map(n => [n.id, 0]));
  for (const e of edges) { if (fanIn.has(e.target)) fanIn.set(e.target, fanIn.get(e.target) + 1); if (fanOut.has(e.source)) fanOut.set(e.source, fanOut.get(e.source) + 1); }
  const rank = (counts, field) => [...counts].sort((a,b) => b[1]-a[1] || a[0].localeCompare(b[0])).slice(0,20).map(([id,v]) => ({id,[field]:v,name:byId.get(id).name}));
  const sortedOut = [...fanOut.values()].sort((a,b)=>a-b), sortedIn = [...fanIn.values()].sort((a,b)=>a-b);
  const highOut = sortedOut[Math.max(0,Math.ceil(sortedOut.length*.9)-1)] || 0, lowIn = sortedIn[Math.max(0,Math.ceil(sortedIn.length*.25)-1)] || 0;
  const candidates = nodes.map(n => { const p=n.filePath||n.name||"", base=p.split("/").pop(), doc=n.type==="document"; let score=0; if(doc&&p==="README.md") score+=5; else if(doc&&!p.includes("/"))score+=2; if(!doc&&/^(index|main|app|server|mod|manage|wsgi|asgi|run|__main__)\.(ts|js|rs|go|py)$|^(Application\.java|Main\.java|Program\.cs|config\.ru|index\.php|App\.swift|Application\.kt|main\.(cpp|c))$/.test(base))score+=3; if(!doc&&p.split("/").length<=2)score++; if(!doc&&fanOut.get(n.id)>=highOut)score++; if(!doc&&fanIn.get(n.id)<=lowIn)score++; return {id:n.id,score,name:n.name,summary:n.summary||""}; }).filter(x=>x.score>0).sort((a,b)=>b.score-a.score||a.id.localeCompare(b.id)).slice(0,5);
  const start=candidates.find(c=>byId.get(c.id).type!=="document"), forward=new Map(nodes.map(n=>[n.id,[]]));
  for(const e of edges)if(["imports","calls"].includes(e.type)&&forward.has(e.source)&&forward.has(e.target))forward.get(e.source).push(e.target);
  const order=[],depthMap={},byDepth={}; if(start){const q=[start.id];depthMap[start.id]=0;for(let i=0;i<q.length;i++){const id=q[i],d=depthMap[id];order.push(id);(byDepth[d]??=[]).push(id);for(const t of forward.get(id).sort())if(!(t in depthMap)){depthMap[t]=d+1;q.push(t)}}}
  const nonCodeFiles={documentation:[],infrastructure:[],data:[],config:[]}; for(const n of nodes){const item={id:n.id,name:n.name,type:n.type,summary:n.summary||""};if(n.type==="document")nonCodeFiles.documentation.push(item);else if(["service","pipeline","resource"].includes(n.type))nonCodeFiles.infrastructure.push(item);else if(["table","schema","endpoint"].includes(n.type))nonCodeFiles.data.push(item);else if(n.type==="config")nonCodeFiles.config.push(item)}
  const pairs=new Set(edges.filter(e=>["imports","calls"].includes(e.type)).map(e=>`${e.source}\u0000${e.target}`)), seen=new Set(),clusters=[]; for(const key of pairs){const[a,b]=key.split("\u0000"),id=[a,b].sort().join("\u0000");if(pairs.has(`${b}\u0000${a}`)&&!seen.has(id)){seen.add(id);clusters.push({nodes:[a,b],edgeCount:2})}}
  const nodeSummaryIndex=Object.fromEntries(nodes.map(n=>[n.id,{name:n.name,type:n.type,summary:n.summary||""}]));
  fs.writeFileSync(outputPath,JSON.stringify({scriptCompleted:true,entryPointCandidates:candidates,fanInRanking:rank(fanIn,"fanIn"),fanOutRanking:rank(fanOut,"fanOut"),bfsTraversal:{startNode:start?.id||null,order,depthMap,byDepth},nonCodeFiles,clusters:clusters.slice(0,10),layers:{count:(graph.layers||[]).length,list:graph.layers||[]},nodeSummaryIndex,totalNodes:nodes.length,totalEdges:edges.length},null,2)+"\n");
} catch (error) { console.error(error.stack || error.message); process.exit(1); }
