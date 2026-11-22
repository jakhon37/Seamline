// Writes constraints.txt at the top of the repository: the versions of everything requirements.txt
// brings in, as uv resolves them for Windows, macOS and Linux at once, which an installed Seamline's
// engine is set up with (src/main/engine.ts). A release then installs what was tested, rather than
// whatever is newest on the day someone first runs it. PyTorch and what only it needs are left to
// the build chosen (--torch-backend), with PyTorch kept below the next version not yet tried.
// Run it from app/ when requirements.txt changes, or to move to newer versions: node scripts/constraints.mjs
import { execFileSync } from 'child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import { fileURLToPath } from 'url'

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')
const TORCH = 'torch>=2.3,<2.16'
const family = (name) => /^(torch|torchvision|triton|setuptools)$|^(nvidia|cuda)-/.test(name)

const requirements = readFileSync(path.join(repo, 'requirements.txt'), 'utf8')
  .split(/\r?\n/).filter((l) => !/^\s*(torch|torchvision)\b/.test(l)).join('\n')
const dir = mkdtempSync(path.join(tmpdir(), 'seamline-constraints-'))
let resolved
try {
  writeFileSync(path.join(dir, 'requirements.in'), requirements)
  resolved = execFileSync('uv', ['pip', 'compile', path.join(dir, 'requirements.in'), '--universal', '--python-version', '3.12',
    '--no-header', '--quiet'], { encoding: 'utf8', env: { ...process.env, UV_NO_CONFIG: '1' } })
} finally {
  rmSync(dir, { recursive: true, force: true })
}

// Each package, with what needs it: "name==1.0 ; marker" then "# via" lines
const packages = []
for (const line of resolved.split('\n')) {
  const pin = line.match(/^([A-Za-z0-9_.-]+)==\S+/)
  if (pin) packages.push({ line: line.trim(), name: pin[1].toLowerCase(), via: new Set() })
  const via = line.match(/^\s+#\s+(?:via\s+)?([A-Za-z0-9_.-]+)\s*$/)
  if (via && packages.length && via[1] !== 'via') packages.at(-1).via.add(via[1].toLowerCase())
}
// What only PyTorch's family needs comes with the build too, as each build names its own versions
const left = new Set(packages.filter((p) => family(p.name)).map((p) => p.name))
for (let changed = true; changed;) {
  changed = false
  for (const p of packages) {
    if (!left.has(p.name) && p.via.size && [...p.via].every((v) => left.has(v))) {
      left.add(p.name)
      changed = true
    }
  }
}
const kept = packages.filter((p) => !left.has(p.name)).map((p) => p.line)
writeFileSync(path.join(repo, 'constraints.txt'), [
  '# The versions an installed Seamline sets its engine up with, for Windows, macOS and Linux: written',
  '# by app/scripts/constraints.mjs from requirements.txt. PyTorch and what only it needs come with the',
  '# build chosen; PyTorch stays below the next version not yet tried.',
  TORCH,
  ...kept,
  ''
].join('\n'))
console.log(`constraints.txt: ${kept.length} packages pinned; left to the PyTorch build: ${[...left].sort().join(', ')}`)
