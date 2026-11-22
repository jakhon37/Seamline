// The engine is Python: PyTorch and the packages in requirements.txt. Run from source, it lives in
// the repo's .venv. Installed, Seamline makes it on first run in a folder of its own with uv, which
// ships with the app, choosing the PyTorch build that fits the computer. PyTorch comes in a build
// for each kind of GPU, 0.3 to 3.9 GB to download, which is why it isn't in the installer.
import { ChildProcess, execFile, spawn } from 'child_process'
import { createHash } from 'crypto'
import { app } from 'electron'
import { chmodSync, existsSync, promises as fs, readFileSync, statSync, writeFileSync, mkdirSync } from 'fs'
import { homedir } from 'os'
import path from 'path'

/** uv's --torch-backend: `auto` takes the newest CUDA build the NVIDIA driver runs, or the CPU build */
export type Backend = 'auto' | 'cu126' | 'cpu'
export const BACKENDS: Backend[] = ['auto', 'cu126', 'cpu']
export type Gpu = { name: string; capability: number; driver: string }
export type EngineInfo = {
  backend: Backend
  torch: string
  cuda: boolean
  device: string
  /** False when PyTorch has no kernels for this GPU, such as 2.15's CUDA builds on a GTX 1080 */
  usable: boolean
  requirements: string
  at: string
}
export type Choice = { backend: Backend; title: string; detail: string; download: string; disk: string }
export type EngineStatus = {
  ready: boolean
  /** Seamline's own environment, which it can set up again; not .venv or MYSEG_PYTHON */
  managed: boolean
  python: string
  info: EngineInfo | null
  /** requirements.txt or constraints.txt changed since the environment was made, as after an update */
  outdated: boolean
  gpu: Gpu | null
  suggested: Backend
  choices: Choice[]
  /** uv was found, so the environment can be made */
  canSetUp: boolean
  /** Windows lacks the Visual C++ runtime PyTorch and ONNX Runtime load */
  runtimeMissing: boolean
  platform: NodeJS.Platform
}
export type Progress = { line?: string; step?: string; downloaded?: number; total?: number }

export const VC_REDIST = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
let settingUp = false
export const isSettingUp = () => settingUp

/** Where the engine's code is: the repo from source, the app's resources once installed */
export function codeRoot() {
  return app.isPackaged ? path.join(process.resourcesPath, 'seamline') : path.resolve(__dirname, '../../..')
}

/**
 * Where Seamline's own engine goes: gigabytes, so not in a folder for settings, nor in a Windows
 * profile's roaming part, which is copied at every sign-in. SEAMLINE_USER_DATA, which tests and the
 * release checks set, keeps it in that profile with the rest.
 */
export function engineHome() {
  if (process.env.SEAMLINE_USER_DATA) return path.join(app.getPath('userData'), 'engine')
  if (process.platform === 'win32') return path.join(process.env.LOCALAPPDATA || app.getPath('appData'), 'Seamline', 'engine')
  if (process.platform === 'linux') {
    return path.join(process.env.XDG_DATA_HOME || path.join(homedir(), '.local', 'share'), 'Seamline', 'engine')
  }
  return path.join(app.getPath('userData'), 'engine')
}
const venvDir = () => path.join(engineHome(), 'venv')
const infoFile = () => path.join(engineHome(), 'engine.json')
export const venvPython = (venv: string) =>
  process.platform === 'win32' ? path.join(venv, 'Scripts', 'python.exe') : path.join(venv, 'bin', 'python')

/** A program as the shell would find it: a path as it is, a bare name on the PATH */
function findProgram(name: string): string | null {
  if (path.isAbsolute(name) || name.includes('/') || name.includes('\\')) return existsSync(name) ? name : null
  const exts = process.platform === 'win32' ? ['', ...(process.env.PATHEXT ?? '.EXE').split(';').map((e) => e.toLowerCase())] : ['']
  for (const dir of (process.env.PATH ?? '').split(path.delimiter).filter(Boolean)) {
    for (const ext of exts) {
      const file = path.join(dir, name + ext)
      if (existsSync(file)) return file
    }
  }
  return null
}

/** MYSEG_PYTHON if set; from source, the repo's .venv while it has one; otherwise Seamline's own */
export function pythonPath() {
  if (process.env.MYSEG_PYTHON) return findProgram(process.env.MYSEG_PYTHON) ?? process.env.MYSEG_PYTHON
  const repoVenv = venvPython(path.join(codeRoot(), '.venv'))
  if (!app.isPackaged && existsSync(repoVenv)) return repoVenv
  return venvPython(venvDir())
}
const isManaged = () => !process.env.MYSEG_PYTHON && pythonPath() === venvPython(venvDir())

/** The engine's Python, with a clear word when it isn't there rather than ENOENT */
export function enginePython() {
  if (settingUp) throw new Error("The engine is being set up: try again once it's ready.")
  const python = pythonPath()
  if (existsSync(python)) return python
  if (process.env.MYSEG_PYTHON) throw new Error(`The engine's Python isn't there: MYSEG_PYTHON names ${process.env.MYSEG_PYTHON}.`)
  throw new Error("The engine isn't set up yet: set it up on the Train page, then try again.")
}

/**
 * For every engine process. Seamline's own engine sees none of the Python settings meant for other
 * Pythons, and as an installed app's code is read-only, its bytecode goes to the engine's folder.
 */
export function engineEnv(extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env, PYTHONUNBUFFERED: '1', ...extra }
  if (isManaged()) {
    delete env.PYTHONHOME
    delete env.PYTHONPATH
  }
  if (app.isPackaged) env.PYTHONPYCACHEPREFIX = path.join(engineHome(), 'pycache')
  return env
}

/** An AppImage runs from a new /tmp/.mount_* folder each time: its bytecode from earlier runs goes */
export function pruneBytecode() {
  if (!process.env.APPIMAGE) return
  fs.rm(path.join(engineHome(), 'pycache', 'tmp'), { recursive: true, force: true, maxRetries: 3 }).catch(() => {})
}

function uvPath(): string | null {
  if (process.env.SEAMLINE_UV) return process.env.SEAMLINE_UV
  const exe = process.platform === 'win32' ? 'uv.exe' : 'uv'
  if (!app.isPackaged) {
    const tools = path.join(codeRoot(), '.tools', exe)
    return existsSync(tools) ? tools : findProgram('uv')
  }
  const shipped = path.join(process.resourcesPath, 'uv', exe)
  if (!existsSync(shipped)) return null
  if (process.platform !== 'darwin') return shipped
  // A Mac app from a downloaded disk image is quarantined, uv inside it too, and macOS may refuse to
  // run a quarantined helper. A copy written byte for byte carries no quarantine, and keeps uv's signature.
  const copy = path.join(engineHome(), 'bin', 'uv')
  try {
    if (!existsSync(copy) || statSync(copy).size !== statSync(shipped).size) {
      mkdirSync(path.dirname(copy), { recursive: true })
      writeFileSync(copy, readFileSync(shipped))
      chmodSync(copy, 0o755)
    }
    return copy
  } catch {
    return shipped
  }
}

/** What the engine is made from: requirements.txt and the versions constraints.txt holds it to, without comments */
export function requirementsHash() {
  const hash = createHash('sha256')
  for (const name of ['requirements.txt', 'constraints.txt']) {
    const file = path.join(codeRoot(), name)
    if (!existsSync(file)) continue
    const lines = readFileSync(file, 'utf8').split(/\r?\n/).map((l) => l.replace(/#.*/, '').trim()).filter(Boolean)
    hash.update(`${name}\n${lines.join('\n')}\n`)
  }
  return hash.digest('hex').slice(0, 16)
}

function readInfo(): EngineInfo | null {
  try {
    return JSON.parse(readFileSync(infoFile(), 'utf8'))
  } catch {
    return null
  }
}

function querySmi(fields: string): Promise<string[] | null> {
  return new Promise((resolve) => {
    execFile('nvidia-smi', [`--query-gpu=${fields}`, '--format=csv,noheader'], { timeout: 10000, windowsHide: true }, (err, stdout) => {
      const first = err ? '' : stdout.trim().split('\n')[0]
      resolve(first ? first.split(',').map((s) => s.trim()) : null)
    })
  })
}

/** The first NVIDIA GPU, as its driver reports it, or null when there's none, or no driver */
export async function detectGpu(): Promise<Gpu | null> {
  if (process.platform === 'darwin') return null
  const found = await querySmi('name,compute_cap,driver_version')
  if (found) return { name: found[0], capability: Number(found[1]) || 0, driver: found[2] ?? '' }
  // Drivers older than about 510 don't know compute_cap: the GPU is there, its generation unknown
  const old = await querySmi('name,driver_version')
  return old ? { name: old[0], capability: 0, driver: old[1] ?? '' } : null
}

/** Windows' Visual C++ runtime, which PyTorch's and ONNX Runtime's libraries load from the system */
function runtimeMissing() {
  if (process.platform !== 'win32') return false
  const system = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32')
  return !['msvcp140.dll', 'vcruntime140_1.dll'].every((dll) => existsSync(path.join(system, dll)))
}

// Download and size on disk, measured from the package files on 2026-11-22 for all of
// requirements.txt with Python 3.12; `auto` as it picks CUDA 13.2. uv's Python adds about 30 MB.
const SIZES: Record<string, Record<Backend, [string, string]>> = {
  linux: { cpu: ['0.4 GB', '1.4 GB'], auto: ['3.2 GB', '6.1 GB'], cu126: ['3.9 GB', '7.2 GB'] },
  win32: { cpu: ['0.3 GB', '1.0 GB'], auto: ['2.3 GB', '3.6 GB'], cu126: ['2.8 GB', '4.7 GB'] },
  darwin: { cpu: ['0.3 GB', '1.0 GB'], auto: ['0.3 GB', '1.0 GB'], cu126: ['0.3 GB', '1.0 GB'] }
}

function choices(): Choice[] {
  const size = SIZES[process.platform] ?? SIZES.linux
  const choice = (backend: Backend, title: string, detail: string): Choice =>
    ({ backend, title, detail, download: size[backend][0], disk: size[backend][1] })
  if (process.platform === 'darwin') {
    return [choice('cpu', 'PyTorch for this Mac', "Seamline trains on the Mac's CPU, or on a GPU machine over SSH")]
  }
  return [
    choice('auto', 'NVIDIA GPU', 'For the RTX 20 or GTX 16 series and newer: the newest CUDA build your driver runs'),
    choice('cu126', 'Older NVIDIA GPU', 'For the GTX 10 and 900 series, Titan V, and Tesla P100 or V100: PyTorch 2.14 with CUDA 12.6'),
    choice('cpu', 'No NVIDIA GPU', 'Trains on the CPU here, or on a GPU machine over SSH. Pre-labelling and export run here either way')
  ]
}

export function suggest(gpu: Gpu | null): Backend {
  if (!gpu || process.platform === 'darwin') return 'cpu'
  return gpu.capability > 0 && gpu.capability < 7.5 ? 'cu126' : 'auto'
}

export async function engineStatus(): Promise<EngineStatus> {
  const python = pythonPath()
  const managed = isManaged()
  const info = managed ? readInfo() : null
  const gpu = await detectGpu()
  return {
    ready: existsSync(python) && (!managed || !!info),
    managed,
    python,
    info,
    outdated: !!info && info.requirements !== requirementsHash(),
    gpu,
    suggested: suggest(gpu),
    choices: choices(),
    canSetUp: !!uvPath(),
    runtimeMissing: runtimeMissing(),
    platform: process.platform
  }
}

// What the engine needs, and whether PyTorch can run on the GPU it found: a GPU too old for the
// build still says CUDA is available, then fails on the first kernel. Kernels built for X.y run on
// X.z with z >= y, as PyTorch's own check has it, and PTX for an older generation runs on newer ones.
const CHECK = `
import json, re, torch, segmentation_models_pytorch, cv2, PIL, onnxruntime, yaml, albumentations
out = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "device": "CPU", "usable": True}
if out["cuda"]:
    major, minor = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    sm = [int(a[3:]) for a in archs if re.fullmatch(r"sm_\\d+", a)]
    ptx = [int(a[8:]) for a in archs if re.fullmatch(r"compute_\\d+", a)]
    out["device"] = torch.cuda.get_device_name(0)
    out["usable"] = any(s // 10 == major and s % 10 <= minor for s in sm) or any(p <= major * 10 + minor for p in ptx)
print(json.dumps(out))
`

export function checkEngine(python = pythonPath()): Promise<{ torch: string; cuda: boolean; device: string; usable: boolean }> {
  return new Promise((resolve, reject) => {
    execFile(python, ['-c', CHECK], { cwd: codeRoot(), env: engineEnv(), windowsHide: true, timeout: 300000 }, (err, stdout, stderr) => {
      if (err) {
        const why = stderr.trim().split('\n').pop() || err.message
        // A clean Windows lacks the C++ runtime that PyTorch's and ONNX Runtime's libraries load
        if (process.platform === 'win32' && /DLL load failed/i.test(stderr)) {
          return reject(new Error(`${why}. Install Microsoft's Visual C++ Redistributable (${VC_REDIST}), then set the engine up again.`))
        }
        return reject(new Error(why))
      }
      try {
        resolve(JSON.parse(stdout.trim().split('\n').pop()!))
      } catch {
        reject(new Error(`The engine's check printed no answer: ${stdout.trim() || stderr.trim()}`))
      }
    })
  })
}

/** Stops a process and everything it started: a Unix process group, or a Windows process tree */
export function stopProcess(proc: ChildProcess | null | undefined) {
  if (!proc?.pid) return
  if (process.platform === 'win32') {
    execFile('taskkill', ['/pid', String(proc.pid), '/t', '/f'], { windowsHide: true }, () => {})
    return
  }
  try {
    process.kill(-proc.pid, 'SIGTERM')
  } catch {
    // Already gone
  }
}

const UNITS: Record<string, number> = { B: 1, KiB: 1024, MiB: 1024 ** 2, GiB: 1024 ** 3 }
let setupProc: ChildProcess | null = null
let setupCanceled = false

// uv prints "Downloading torch (209.5MiB)" as a big download starts and " Downloaded torch" when it
// ends, which is enough for a progress bar over the downloads it names. Its errors span lines.
function runUv(uv: string, args: string[], env: NodeJS.ProcessEnv, onProgress: (p: Progress) => void) {
  return new Promise<void>((resolve, reject) => {
    const proc = spawn(uv, args, { env, windowsHide: true, detached: process.platform !== 'win32' })
    setupProc = proc
    const sizes = new Map<string, number>()
    const done = new Set<string>()
    const recent: string[] = []
    let pending = ''
    const onData = (chunk: Buffer) => {
      pending += chunk.toString()
      let end: number
      while ((end = pending.search(/[\r\n]/)) >= 0) {
        const line = pending.slice(0, end).trim()
        pending = pending.slice(end + 1)
        if (!line) continue
        recent.push(line)
        if (recent.length > 12) recent.shift()
        // Python's line reads "Downloading cpython-3.12.16-linux-x86_64-gnu (download) (32.9MiB)"
        const started = line.match(/^Downloading (\S+)(?: \(download\))? \(([\d.]+)(B|KiB|MiB|GiB)\)/)
        if (started) sizes.set(started[1], Number(started[2]) * UNITS[started[3]])
        const finished = line.match(/^Downloaded (\S+)/)
        if (finished) done.add(finished[1])
        const total = [...sizes.values()].reduce((a, b) => a + b, 0)
        const downloaded = [...done].reduce((a, n) => a + (sizes.get(n) ?? 0), 0)
        onProgress({ line, total, downloaded })
      }
    }
    proc.stdout!.on('data', onData)
    proc.stderr!.on('data', onData)
    proc.on('error', reject)
    // close, not exit: the last of what uv wrote may come after it has exited
    proc.on('close', (code) => {
      setupProc = null
      if (pending.trim()) recent.push(pending.trim())
      if (setupCanceled) reject(new Error('Setup was stopped'))
      else if (code === 0) resolve()
      else reject(new Error(recent.filter((l) => !/^\s*[+-] /.test(l)).slice(-6).join('\n') || `uv exited with ${code}`))
    })
  })
}

/** uv's environment: only Seamline's settings, and of the user's only those for networks */
function uvEnv(home: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...process.env }
  const kept = new Set(['UV_NATIVE_TLS', 'UV_HTTP_TIMEOUT', 'UV_HTTP_RETRIES', 'UV_INSECURE_HOST'])
  for (const key of Object.keys(env)) {
    if ((key.startsWith('UV_') && !kept.has(key)) || ['VIRTUAL_ENV', 'CONDA_PREFIX', 'PYTHONHOME', 'PYTHONPATH'].includes(key)) delete env[key]
  }
  return {
    ...env,
    // SEAMLINE_UV_CACHE shares a cache that already holds the packages, as tests do
    UV_CACHE_DIR: process.env.SEAMLINE_UV_CACHE || path.join(home, 'cache'),
    UV_PYTHON_INSTALL_DIR: path.join(home, 'python'),
    UV_PYTHON_DOWNLOADS: 'automatic',
    UV_NO_CONFIG: '1',
    NO_COLOR: '1'
  }
}

/**
 * Makes Seamline's own engine environment, or brings it up to date with requirements.txt after an
 * update of the app (`update`), then checks it. A new environment is made beside the one in use and
 * takes its place only once it works, so a failed download leaves the old one as it was. Everything
 * goes in the engine's folder: uv's Python, its download cache while it works, and the environment.
 */
export async function setupEngine(backend: Backend, onProgress: (p: Progress) => void, update = false): Promise<EngineInfo> {
  if (!BACKENDS.includes(backend)) throw new Error(`There's no PyTorch build called ${backend}`)
  if (settingUp) throw new Error('The engine is already being set up')
  const uv = uvPath()
  if (!uv) throw new Error("uv, which sets up the engine, isn't here: reinstall Seamline, or set SEAMLINE_UV to a uv program")
  settingUp = true
  setupCanceled = false
  try {
    const home = engineHome()
    await fs.mkdir(home, { recursive: true })
    const env = uvEnv(home)
    const venv = venvDir()
    const fresh = !update || !existsSync(venvPython(venv))
    const target = fresh ? `${venv}.new` : venv
    const constraints = path.join(codeRoot(), 'constraints.txt')
    if (fresh) {
      onProgress({ step: 'Getting Python 3.12' })
      await fs.rm(target, { recursive: true, force: true, maxRetries: 3 })
      await runUv(uv, ['venv', target, '--python', '3.12', '--managed-python', '--relocatable'], env, onProgress)
    }
    onProgress({ step: backend === 'cpu' ? 'Installing PyTorch and the engine' : 'Installing PyTorch for the GPU, and the engine' })
    await runUv(uv, ['pip', 'install', '--python', target, `--torch-backend=${backend}`,
      '-r', path.join(codeRoot(), 'requirements.txt'), ...(existsSync(constraints) ? ['-c', constraints] : [])], env, onProgress)
    onProgress({ step: 'Checking the engine' })
    const found = await checkEngine(venvPython(target))
    if (fresh) {
      const old = `${venv}.old`
      await fs.rm(old, { recursive: true, force: true, maxRetries: 3 })
      if (existsSync(venv)) await fs.rename(venv, old)
      await fs.rename(target, venv)
      await fs.rm(old, { recursive: true, force: true, maxRetries: 3 }).catch(() => {})
    }
    const info: EngineInfo = { backend, ...found, requirements: requirementsHash(), at: new Date().toISOString() }
    await fs.writeFile(infoFile(), JSON.stringify(info, null, 2))
    // The environment has its own copies or links of what it uses: the rest of the download cache,
    // such as another PyTorch build tried before, would only take room
    if (!process.env.SEAMLINE_UV_CACHE) await runUv(uv, ['cache', 'clean'], env, () => {}).catch(() => {})
    return info
  } finally {
    settingUp = false
  }
}

export function cancelSetup() {
  if (!setupProc) return
  setupCanceled = true
  stopProcess(setupProc)
}

/**
 * For the release checks: `--setup-engine <backend>` makes the environment and prints uv's lines,
 * `--check-engine` asks the engine for a small model's estimate. Both print, then exit. On Linux
 * they still need a display, as Electron does, such as xvfb-run's.
 */
export async function runEngineCli(argv: string[]): Promise<number | null> {
  const at = argv.indexOf('--setup-engine')
  if (at >= 0) {
    const backend = (argv[at + 1] ?? 'auto') as Backend
    if (!BACKENDS.includes(backend)) {
      process.stderr.write(`--setup-engine takes auto, cu126 or cpu, not ${backend}\n`)
      return 2
    }
    try {
      const info = await setupEngine(backend, (p) => (p.line || p.step) && process.stdout.write(`${p.step ?? p.line}\n`))
      process.stdout.write(`${JSON.stringify(info)}\n`)
      return 0
    } catch (e: any) {
      process.stderr.write(`${e.message}\n`)
      return 1
    }
  }
  if (argv.includes('--check-engine')) {
    const options = JSON.stringify({ architecture: 'Unet', encoder: 'resnet18', image_size: 256, batch_size: 2 })
    return new Promise((resolve) => {
      execFile(pythonPath(), [path.join(codeRoot(), 'engine', 'studio.py'), 'estimate', '--options', options],
        { cwd: codeRoot(), env: engineEnv(), windowsHide: true, timeout: 600000 }, (err, stdout, stderr) => {
          process.stdout.write(stdout)
          if (err) process.stderr.write(stderr || err.message)
          resolve(err ? 1 : 0)
        })
    })
  }
  return null
}
