const fs = require("fs");
const path = require("path");

const projectRoot = process.argv[2];
const uaDir = process.argv[3];
const gitCommitHash = process.argv[4];
const scan = JSON.parse(
  fs.readFileSync(path.join(uaDir, "intermediate", "scan-result.json"), "utf8"),
);
const input = {
  projectRoot,
  sourceFilePaths: scan.files.map((file) => file.path),
  gitCommitHash,
};
fs.writeFileSync(
  path.join(uaDir, "intermediate", "fingerprint-input.json"),
  JSON.stringify(input, null, 2),
);
