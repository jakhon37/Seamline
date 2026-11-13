import { ChildProcess, execFile, spawn } from 'child_process'
import { createHash } from 'crypto'
import { app, BrowserWindow, dialog, ipcMain, net, protocol, shell } from 'electron'
import { constants, existsSync, promises as fs } from 'fs'
import { homedir, tmpdir } from 'os'
import path from 'path'
import { pathToFileURL } from 'url'
import sharp from 'sharp'
import { classesIn, type LabelClass, projectFile } from '../shared/classes'
import { FORMAT_NAMES, type LabelFormat, looksLikeCreateML, readClasses, readCreateML, readVoc, readYolo, YOLO_CLASSES } from '../shared/formats'
import { hasArea, readLabelMe, summarize } from '../shared/labels'
import { nameFold } from '../shared/names'

const REPO = path.resolve(__dirname, '../../..')
const PYTHON = process.env.MYSEG_PYTHON || path.join(REPO, '.venv/bin/python')
const ENGINE = path.join(REPO, 'engine/studio.py')
// What Seamtag lists, so both apps show the same images
const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.jfif', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.avif'])
const PALETTE = ['#4f8cff', '#ff6b6b', '#3ecf8e', '#f7b731', '#b57bff', '#22c7d6', '#ff8f4d', '#ec6fb3']
const TILE = 512
const DIRECT_MAX = 4096
const SEAMTAG_RELEASES = 'https://github.com/jakhon37/Seamtag/releases/latest'
// A Seamtag that can't start stops at once: one still running after this long has started
const SEAMTAG_STARTS = 2000
// Electron sets this for itself once it's ready; Seamtag, an Electron app too, gets it only if Seamline did
const GDK_BACKEND = process.env.GDK_BACKEND
sharp.cache(false)

let root = ''
let win: BrowserWindow | null = null
let job: { proc: ChildProcess; base: string; run: string; offset: number; timer: NodeJS.Timeout } | null = null
type RemoteInfo = { host: string; dir: string; project: string; pid: number; fetched: boolean }
// Only ever the open folder's run. `base` is that folder, as `root` moves on when another opens.
type RemoteJob = { base: string; run: string; info: RemoteInfo; events: number; log: number; timer: NodeJS.Timeout; polling: boolean; lost: boolean; stopped: boolean }
let remoteJob: RemoteJob | null = null
let launching = false
type PrelabelJob = { proc: ChildProcess; stopped: boolean }
let prelabelJob: PrelabelJob | null = null
function queue() {
  let last: Promise<unknown> = Promise.resolve()
  return <T>(fn: () => Promise<T>) => {
    const result = last.then(fn)
    last = result.catch(() => {})
    return result
  }
}
// Label files are read, changed and written back: one at a time, or a pre-label could undo a save
const serial = queue()
// The same for the class list, apart, so a change to it doesn't wait for the folder to be read again
const classTurn = queue()

const SSH_OPTS = [
  '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'ControlMaster=auto',
  '-o', `ControlPath=${path.join(tmpdir(), 'myseg-ssh-%C')}`, '-o', 'ControlPersist=60'
]
// Room left on the remote disk for the environment, checkpoints and logs
const REMOTE_RESERVE_KB = 3 * 1024 * 1024
const quote = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`

protocol.registerSchemesAsPrivileged([
  { scheme: 'myseg', privileges: { standard: true, supportFetchAPI: true, stream: true, secure: true } }
])

const metaIn = (base: string, ...p: string[]) => path.join(base, '.myseg', ...p)
const meta = (...p: string[]) => metaIn(root, ...p)
const posix = (p: string) => p.split(path.sep).join('/')
const stemOf = (name: string) => name.replace(/\.[^./]+$/, '')
const sidecar = (name: string, base = root) => path.join(base, stemOf(name) + '.json')

// A window can still be saving to the last folder as the next one opens, so each
// save names its folder. Only folders opened in this run are accepted.
const opened = new Set<string>()
function imageIn(base: string, name: string) {
  const full = path.resolve(base, name)
  if (!opened.has(base) || !full.startsWith(base + path.sep)) throw new Error('Path outside project')
  return full
}

// Image names come from the renderer: never let one leave the project folder
function inProject(name: string): string {
  const full = path.resolve(root, name)
  if (!root || !full.startsWith(root + path.sep)) throw new Error('Path outside project')
  return full
}

async function readJson(file: string): Promise<any | null> {
  try {
    return JSON.parse(await fs.readFile(file, 'utf8'))
  } catch {
    return null
  }
}

// No file means no labels yet. The hash of its bytes tells one version from another however little
// changed: a rewrite that keeps the file's size and time, as on a coarse file clock, still shows.
async function readHashed(file: string): Promise<{ text: string; hash: string } | null> {
  try {
    const bytes = await fs.readFile(file)
    return { text: bytes.toString('utf8'), hash: createHash('sha1').update(bytes).digest('hex') }
  } catch (e: any) {
    if (e.code === 'ENOENT') return null
    throw new Error(`${path.basename(file)} can't be read (${e.code}), so it's left unchanged.`)
  }
}

// Whole numbers past 2^53, such as ids, aren't exact as numbers: such a file is read keeping their
// digits, and they're written back as they were. Not in lists, so points stay numbers.
const LONG_NUMBER = /\d{16}/
const exactDocs = new WeakSet<object>()
const isRaw = (v: unknown): v is { rawJSON: string } => (JSON as any).isRawJSON(v)
function parseExact(text: string) {
  return JSON.parse(text, function (this: unknown, _key: string, value: unknown, context?: { source?: string }) {
    const source = context?.source ?? ''
    return typeof value === 'number' && !Number.isSafeInteger(value) && !Array.isArray(this) && /^-?\d+$/.test(source)
      ? (JSON as any).rawJSON(source) : value
  })
}
// As the window gets them: numbers, which it can send back
function plain(v: any): any {
  if (isRaw(v)) return Number(v.rawJSON)
  if (Array.isArray(v)) return v.map(plain)
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, plain(x)]))
  return v
}
const shapesOf = (doc: any): any[] => (exactDocs.has(doc) ? plain(doc.shapes) : doc.shapes)

// A file we can't read is someone's labels, so this throws rather than let anyone write over it.
// It must be one the engine and Seamtag can read too, or it would count here and stop training.
function parseLabelMe(text: string, file: string): any {
  let doc: any
  try {
    doc = readLabelMe(text)
  } catch (e: any) {
    throw new Error(`${path.basename(file)} can't be read, so it's left unchanged: ${e.message}`)
  }
  if (!LONG_NUMBER.test(text)) return doc
  // Read again for saving, keeping the digits of long whole numbers the checks took as numbers
  const exact = parseExact(text)
  exactDocs.add(exact)
  return exact
}

async function writeAtomic(file: string, text: string) {
  const tmp = `${file}.${process.pid}.tmp`
  await fs.writeFile(tmp, text)
  await fs.rename(tmp, file)
}

// A folder's listing tells files apart as the system does
const fold = nameFold(process.platform)
const exists = (file: string) => fs.access(file).then(() => true, () => false)

// Where Seamtag looks for an image's labels, in its order: <stem>.json (LabelMe, or a CreateML
// list), <stem>.xml (Pascal VOC), then <stem>.txt beside a classes.txt (YOLO). A whole folder is
// looked up in its listing, as a file at a time would keep a big one waiting; one image on disk.
type Located = { kind: 'json' | 'xml' | 'txt'; file: string }
async function locate(base: string, name: string, listing?: Set<string>): Promise<Located | null> {
  const has = (rel: string) => (listing ? listing.has(fold(rel)) : exists(path.join(base, rel)))
  const stem = stemOf(name)
  const file = (ext: string) => path.join(base, stem + ext)
  if (await has(stem + '.json')) return { kind: 'json', file: file('.json') }
  if (await has(stem + '.xml')) return { kind: 'xml', file: file('.xml') }
  // Other .txt files, such as captions, aren't labels
  const classes = path.posix.join(path.posix.dirname(stem), YOLO_CLASSES)
  if (path.posix.basename(stem) + '.txt' !== YOLO_CLASSES && (await has(stem + '.txt')) && (await has(classes))) {
    return { kind: 'txt', file: file('.txt') }
  }
  return null
}

type Stamp = { size: number; mtime: number }
const statOf = (file: string) => fs.stat(file).then((s): Stamp => ({ size: s.size, mtime: s.mtimeMs }), () => null)

/**
 * The label file's size and time, and a key from them that changes whenever its labels can: YOLO's
 * boxes also take their names from classes.txt and their size from the image. Cheap enough for a
 * whole folder, whose listing reads again only the files whose key changed; saves go by content.
 * Taken before reading, so a change made while a file is read shows as a change.
 */
async function stampOf(base: string, name: string, found: Located | null) {
  const stamp = found && (await statOf(found.file))
  if (!found || !stamp) return { stamp: null, key: 'none' }
  let key = `${found.kind}:${stamp.size}:${stamp.mtime}`
  if (found.kind === 'txt') {
    const classes = await statOf(path.join(path.dirname(found.file), YOLO_CLASSES))
    const image = await statOf(path.join(base, name))
    key += `:${classes?.size}:${classes?.mtime}:${image?.mtime}`
  }
  return { stamp, key }
}

// libvips can't read BMP, but the window can, so only its size is needed here
async function bmpSize(file: string): Promise<[number, number]> {
  const handle = await fs.open(file, 'r')
  try {
    const head = Buffer.alloc(26)
    await handle.read(head, 0, 26, 0)
    if (head.toString('ascii', 0, 2) !== 'BM') throw new Error(`${path.basename(file)} isn't a BMP file`)
    // The oldest BMP header keeps the size in 16 bits; rows stored top down have a negative height
    if (head.readUInt32LE(14) === 12) return [head.readUInt16LE(18), head.readUInt16LE(20)]
    return [Math.abs(head.readInt32LE(18)), Math.abs(head.readInt32LE(22))]
  } finally {
    await handle.close()
  }
}

/** The image's size as the window shows it: EXIF orientations 5-8 swap the sides */
async function uprightSize(file: string): Promise<[number, number]> {
  if (path.extname(file).toLowerCase() === '.bmp') return bmpSize(file)
  const info = await sharp(file, { limitInputPixels: false }).metadata()
  return (info.orientation ?? 1) >= 5 ? [info.height!, info.width!] : [info.width!, info.height!]
}

type Read = { format: LabelFormat | null; shapes: any[]; doc: any; key: string }
/**
 * An image's labels in whichever format it has them, as LabelMe shapes. No file is no labels yet.
 * `key` tells them from any other labels the image had: the label file's content, and for YOLO also
 * classes.txt's and the image's size.
 */
async function readFound(base: string, name: string, found: Located | null): Promise<Read> {
  const none: Read = { format: null, shapes: [], doc: null, key: 'none' }
  const read = found && (await readHashed(found.file))
  if (!found || !read) return none
  const { text } = read
  const key = `${found.kind}:${read.hash}`
  if (found.kind === 'json' && !looksLikeCreateML(text)) {
    const doc = parseLabelMe(text, found.file)
    return { format: 'labelme', shapes: shapesOf(doc), doc, key }
  }
  const format: LabelFormat = found.kind === 'json' ? 'createml' : found.kind === 'xml' ? 'voc' : 'yolo'
  try {
    // CreateML names each image by its file name, and a list that doesn't name it has no labels of it
    if (format === 'createml') return { ...none, format, key, shapes: readCreateML(text, path.basename(name)).shapes }
    if (format === 'voc') return { ...none, format, key, shapes: readVoc(text).shapes }
    const classes = await readHashed(path.join(path.dirname(found.file), YOLO_CLASSES))
    const [w, h] = await uprightSize(path.join(base, name))
    const shapes = readYolo(text, classes?.text.trim() ? readClasses(classes.text) : [], w, h)
    return { ...none, format, shapes, key: `${key}:${classes?.hash ?? 'none'}:${w}x${h}` }
  } catch (e: any) {
    throw new Error(`${path.basename(found.file)} can't be read as ${FORMAT_NAMES[format]}, so it's left unchanged: ${e.message}`)
  }
}

// img2 before img10. One collator for the whole sort: localeCompare would make one per comparison.
const natural = new Intl.Collator(undefined, { numeric: true })

// Every image in the folder and its subfolders, as Seamtag lists them, and every file beside them.
// Hidden folders hold caches, such as .myseg, and folders linked from elsewhere aren't followed.
async function scan(base: string) {
  const images: string[] = []
  const files = new Set<string>()
  const walk = async (sub: string) => {
    // A subfolder that can't be read is left out; the folder itself must be there
    const entries = await fs.readdir(path.join(base, sub), { withFileTypes: true }).catch((e) => {
      if (!sub) throw e
      return []
    })
    for (const e of entries) {
      if (e.name.startsWith('.')) continue
      const rel = sub ? `${sub}/${e.name}` : e.name
      if (e.isDirectory()) {
        await walk(rel)
        continue
      }
      files.add(fold(rel))
      if (IMAGE_EXTS.has(path.extname(e.name).toLowerCase())) images.push(rel)
    }
  }
  await walk('')
  return { images: images.sort(natural.compare), files }
}

// `stamp` is stampOf's key, and `key` readFound's
type Listed = { stamp: string; key: string; item: any; labels: string[] }
// What the image list knows of each folder's labels, so a refresh reads only the files that changed
const listed = new Map<string, Map<string, Listed>>()
// By image, the labels Seamline last read or wrote, by readFound's key: pre-labelling writes nothing over
// any that changed since
const seen = new Map<string, string>()
// By image, the labels the window has: those it took from a load, or saved. Its saves write nothing over any
// that changed since, as when Seamtag saved them. Kept apart from `seen`, as a refresh mustn't make the
// window's copy current, and neither may a load the window didn't take, as when it had moved on meanwhile.
const based = new Map<string, string>()
// By image, fresh labels the window has: taken where there was no label file, or with the one Seamline made
// for them, and not a pre-labelling result. The key of the file Seamline made for them since, 'none', or null
// once pre-labelling wrote over it. With no shapes left, the image has no file again rather than an empty one,
// which reads as labels someone emptied. A file another app made or changed meanwhile fails the save's check
// before that.
const made = new Map<string, string | null>()

// One image as the image list knows it, its labels read again only if they changed since. A file that
// couldn't be read is tried again each time: what stopped it, such as its permissions, may have changed.
async function listOne(base: string, name: string, found: Located | null, before?: Listed): Promise<Listed> {
  const { key: stamp } = await stampOf(base, name, found)
  if (before?.stamp === stamp && !before.item.broken) return before
  try {
    const read = await readFound(base, name, found)
    const item = { name, ...summarize(read.shapes), broken: false, format: read.format ?? undefined }
    // What a new class list starts from: labels that would train, as the dataset check lists them
    return { stamp, key: read.key, item, labels: read.shapes.filter((s) => !s.flags?.ai && hasArea(s)).map((s) => s.label) }
  } catch {
    return { stamp, key: 'broken', item: { name, ...summarize([]), broken: true }, labels: [] }
  }
}

async function listImages(base: string) {
  const { images: names, files } = await scan(base)
  const before = listed.get(base) ?? new Map<string, Listed>()
  const now = new Map<string, Listed>()
  await Promise.all(names.map(async (name) => {
    const entry = await listOne(base, name, await locate(base, name, files), before.get(name))
    seen.set(path.join(base, name), entry.key)
    now.set(name, entry)
  }))
  listed.set(base, now)
  return names.map((name) => now.get(name)!)
}

// By folder, the images Seamline writes while each listing of it is on its way
const listings = new Map<string, Set<Set<string>>>()
// Read again by the listings on their way, and by the next: a save can keep the file's size and time
const wrote = (base: string, name: string) => {
  listed.get(base)?.delete(name)
  listings.get(base)?.forEach((names) => names.add(name))
}

/**
 * The folder's images, read beside Seamline's saves rather than in line with them, so a big folder
 * keeps none waiting. What Seamline wrote meanwhile is read again in line with them, and `use` sees
 * the whole list there, before anything else is written.
 */
async function listFolder<T>(base: string, use: (entries: Listed[]) => T) {
  const written = new Set<string>()
  const open = listings.get(base) ?? new Set<Set<string>>()
  listings.set(base, open.add(written))
  try {
    const entries = await listImages(base)
    return await serial(async () => {
      for (const name of written) {
        const at = entries.findIndex((e) => e.item.name === name)
        if (at < 0) continue
        entries[at] = await listOne(base, name, await locate(base, name))
        seen.set(path.join(base, name), entries[at].key)
        listed.get(base)?.set(name, entries[at])
      }
      return use(entries)
    })
  } finally {
    open.delete(written)
  }
}

async function openProject(dir: string) {
  // The slow reading comes first: the open folder only changes once it's done
  const base = path.resolve(dir)
  const entries = await listFolder(base, (all) => all)
  const images = entries.map((e) => e.item)
  // Starting a run works on the open folder throughout
  if (launching && base !== root) throw new Error('A training run is starting. Open another folder once it has.')
  stopPrelabel()
  prelabelJob = null
  if (remoteJob?.base !== base) detachRemote()
  root = base
  opened.add(base)
  await fs.mkdir(meta(), { recursive: true })
  await resumeRemote(base)
  const project = await classTurn(async () => {
    let doc = await readJson(metaIn(base, 'project.json'))
    let classes = classesIn(doc)
    if (!classes) {
      // Folders labelled in LabelMe, LabelImg or Seamtag open with their classes already in place
      const labels = new Set<string>()
      entries.forEach((e) => e.labels.forEach((l) => l && labels.add(l)))
      const names = labels.size ? [...labels].sort() : ['object']
      classes = names.map((name, i) => ({ name, color: PALETTE[i % PALETTE.length] }))
      doc = projectFile([], classes, doc, path.basename(base))
      await writeAtomic(metaIn(base, 'project.json'), JSON.stringify(doc, null, 2))
    }
    classBase.set(base, classes)
    return { name: typeof doc.name === 'string' ? doc.name : path.basename(base), classes }
  })
  return { root, project, images }
}

// By folder, the class list as Seamline last read or wrote it: what its changes are merged from
const classBase = new Map<string, LabelClass[]>()

// The class list, merged with what another app, such as Seamtag, changed in the file since
function saveClasses(base: string, classes: LabelClass[]) {
  if (!opened.has(base)) throw new Error('That folder is not open')
  return classTurn(async () => {
    const file = metaIn(base, 'project.json')
    const doc = projectFile(classBase.get(base) ?? [], classes, await readJson(file), path.basename(base))
    await writeAtomic(file, JSON.stringify(doc, null, 2))
    classBase.set(base, doc.classes)
    return doc.classes
  })
}

// The class list as it is now, which the window shows: null, and nothing changes, if it can't be read
function readProjectClasses(base: string) {
  if (!opened.has(base)) throw new Error('That folder is not open')
  return classTurn(async () => {
    const classes = classesIn(await readJson(metaIn(base, 'project.json')))
    if (classes) classBase.set(base, classes)
    return classes
  })
}

// Another app may have changed the folder since: new and removed images, and labels that changed
async function refreshProject(base: string) {
  if (!opened.has(base)) throw new Error('That folder is not open')
  return listFolder(base, (entries) => {
    // Images whose labels changed since the window last had them: what it remembers of them is out of date
    const changed = entries.filter((e) => {
      const had = based.get(path.join(base, e.item.name))
      return had !== undefined && had !== e.key
    }).map((e) => e.item.name)
    return { images: entries.map((e) => e.item), changed }
  })
}

async function openImage(name: string) {
  const file = inProject(name)
  const stat = await fs.stat(file)
  const started = Date.now()
  if (path.extname(file).toLowerCase() === '.bmp') {
    const [width, height] = await bmpSize(file)
    return { width, height, file, pyramid: null, ms: Date.now() - started }
  }
  const info = await sharp(file, { limitInputPixels: false }).metadata()
  // EXIF orientations 5-8 swap the sides, and the browser shows photos upright
  const swapped = (info.orientation ?? 1) >= 5
  const width = swapped ? info.height! : info.width!
  const height = swapped ? info.width! : info.height!
  const browserReadable = ['jpeg', 'png'].includes(info.format ?? '')
  if (browserReadable && Math.max(width, height) <= DIRECT_MAX) {
    return { width, height, file, pyramid: null, ms: Date.now() - started }
  }
  // Large, or TIFF, GIF, WebP or AVIF: build a tile pyramid once and reuse it until the file changes.
  // Named by a hash of the image's name, as names a folder can't take whole, such as sub/a.tif and
  // sub_a.tif, mustn't share one.
  const key = `${createHash('sha1').update(name).digest('hex')}-${stat.size}-${Math.floor(stat.mtimeMs)}`
  const dir = meta('cache', key)
  if (!existsSync(path.join(dir, 'done'))) {
    await fs.rm(dir, { recursive: true, force: true })
    await fs.mkdir(dir, { recursive: true })
    await sharp(file, { limitInputPixels: false })
      .rotate()
      .toColourspace('srgb')
      .jpeg({ quality: 92 })
      .tile({ size: TILE, overlap: 0, layout: 'dz' })
      .toFile(path.join(dir, 't'))
    await fs.writeFile(path.join(dir, 'done'), '')
  }
  const maxLevel = Math.ceil(Math.log2(Math.max(width, height)))
  return {
    width, height, file, ms: Date.now() - started,
    pyramid: { dir: path.join(dir, 't_files'), maxLevel, tile: TILE }
  }
}

function run(cmd: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { maxBuffer: 1 << 26 }, (err, stdout, stderr) => {
      if (err) reject(new Error(stderr.trim().split('\n').pop() || err.message))
      else resolve(stdout)
    })
  })
}
const ssh = (host: string, command: string) => run('ssh', [...SSH_OPTS, host, command])
const rsync = (...args: string[]) => run('rsync', ['-a', '-e', ['ssh', ...SSH_OPTS].join(' '), ...args])

async function checkRemote(host: string, dir: string) {
  const out = await ssh(host, `mkdir -p ${dir} && cd ${dir} && df -Pk . | tail -1 | awk '{print $4}'; ` +
    `nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader 2>/dev/null | head -1; ` +
    `nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l; ` +
    `python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1 | tail -1`)
  const [free, gpu, jobs, torch] = out.trim().split('\n')
  return { freeKb: Number(free), gpu: gpu ?? '', gpuJobs: Number(jobs), torch: torch ?? '' }
}

// By the time here, as the engine names a run on this computer: both apps take the greatest name for the newest
function runName(at: Date) {
  const two = (n: number) => String(n).padStart(2, '0')
  const day = `${at.getFullYear()}${two(at.getMonth() + 1)}${two(at.getDate())}`
  return `${day}_${two(at.getHours())}${two(at.getMinutes())}${two(at.getSeconds())}`
}

async function startRemote(options: any, target: { host: string; dir: string }) {
  const { host, dir } = target
  const log = (line: string) => send('train:log', line + '\n')
  log(`Checking ${host}…`)
  const remote = await checkRemote(host, dir)
  if (!/ True$/.test(remote.torch)) throw new Error(`No GPU PyTorch on ${host}: ${remote.torch}`)
  if (remote.gpuJobs > 0) throw new Error(`The GPU on ${host} is busy with ${remote.gpuJobs} other job(s).`)

  log('Preparing dataset…')
  await engine(['prepare', root, '--val-split', String(options.val_split), '--tile', String(options.tile)])
  const sizeKb = Number((await run('du', ['-sk', meta('dataset')])).split('\t')[0])
  if (remote.freeKb < sizeKb + REMOTE_RESERVE_KB) {
    const gb = (kb: number) => (kb / 1048576).toFixed(1)
    throw new Error(`Not enough disk on ${host}: ${gb(remote.freeKb)} GB free, need ${gb(sizeKb + REMOTE_RESERVE_KB)} GB.`)
  }

  // A run here can have started in the same second: the next name free keeps each run's folder its own
  let at = Date.now()
  while (existsSync(meta('runs', runName(new Date(at))))) at += 1000
  const name = runName(new Date(at))
  const project = `${dir}/projects/${name}`
  log('Sending code…')
  await ssh(host, `mkdir -p ${dir}/code`)
  await rsync('--delete', '--exclude=__pycache__', ...['engine', 'scripts', 'src', 'configs'].map((d) => path.join(REPO, d)), `${host}:${dir}/code/`)
  log('Setting up the environment if needed…')
  await ssh(host, `cd ${dir} && python3 code/engine/remote_setup.py venv`)
  await ssh(host, `mkdir -p ${project}/.myseg/runs/${name}`)
  log(`Uploading dataset (${(sizeKb / 1024).toFixed(0)} MB)…`)
  await rsync(meta('dataset'), meta('project.json'), `${host}:${project}/.myseg/`)

  const runDir = meta('runs', name)
  await fs.mkdir(runDir, { recursive: true })
  await fs.writeFile(path.join(runDir, 'options.json'), JSON.stringify({ ...options, target }, null, 2))
  await fs.writeFile(path.join(runDir, 'events.jsonl'), '')
  const absolute = (await ssh(host, `cd ${project} && pwd`)).trim()
  const remoteRun = `.myseg/runs/${name}`
  const pid = Number((await ssh(host,
    `cd ${dir}/code && { setsid nohup ../venv/bin/python -u engine/studio.py train ${quote(absolute)} ` +
    `--options ${quote(JSON.stringify(options))} --prepared --run ${name} ` +
    `> ${quote(absolute)}/${remoteRun}/out.log 2>&1 < /dev/null & echo $!; }`)).trim())
  const info: RemoteInfo = { host, dir, project, pid, fetched: false }
  await fs.writeFile(path.join(runDir, 'remote.json'), JSON.stringify(info))
  attachRemote(root, name, info)
  return name
}

function attachRemote(base: string, name: string, info: RemoteInfo, events = 0) {
  remoteJob = { base, run: name, info, events, log: 0, polling: false, lost: false, stopped: false, timer: setInterval(pollRemote, 3000) }
  pollRemote()
}

// A run still going trains on without us and reattaches when its folder opens. Work still on
// its way sees remoteJob has changed and stops, leaving what it didn't save to be read again then.
function detachRemote() {
  if (!remoteJob) return
  clearInterval(remoteJob.timer)
  remoteJob = null
}

// Counted once saved, so events that didn't reach the disk are read again
async function saveEvents(j: RemoteJob, text: string) {
  const file = metaIn(j.base, 'runs', j.run, 'events.jsonl')
  await fs.appendFile(file, text)
  j.events += Buffer.byteLength(text)
  sendEvents(j.run, await fs.readFile(file, 'utf8'))
}

// An event, so the message stays with the run
const saveFailure = (j: RemoteJob, message: string) => saveEvents(j, JSON.stringify({ type: 'error', message }) + '\n')

async function markFetched(j: RemoteJob) {
  j.info.fetched = true
  await fs.writeFile(metaIn(j.base, 'runs', j.run, 'remote.json'), JSON.stringify(j.info))
}

async function pollRemote() {
  const j = remoteJob
  if (!j || j.polling) return
  j.polling = true
  const { host, project, pid } = j.info
  const dirOf = `${project}/.myseg/runs/${j.run}`
  try {
    const out = await ssh(host, `if cd ${dirOf} 2>/dev/null; then kill -0 ${pid} 2>/dev/null && S=alive || S=dead; ` +
      `tail -c +${j.events + 1} events.jsonl 2>/dev/null; echo @@SEP; tail -c +${j.log + 1} out.log 2>/dev/null; echo @@SEP; echo $S; ` +
      `else echo @@SEP; echo @@SEP; echo gone; fi`)
    if (remoteJob !== j) return
    const [eventText, logText, answer] = out.split('@@SEP\n')
    const state = answer.trim()
    const end = eventText.lastIndexOf('\n')
    if (end >= 0) await saveEvents(j, eventText.slice(0, end + 1))
    if (remoteJob !== j) return
    // A run that's over stays "lost" until its model is back, so a failure is told once, not every retry
    if (state === 'alive') {
      if (j.lost) send('train:log', 'Connection back.\n')
      j.lost = false
    }
    if (logText) {
      j.log += Buffer.byteLength(logText)
      send('train:log', logText)
    }
    if (state === 'dead') await finishRemote(j)
    if (state === 'gone') await abandonRemote(j)
  } catch (e: any) {
    // Training goes on without us; keep trying until the box answers again
    if (remoteJob === j && !j.lost) send('train:log', `Connection lost, training continues on ${host}. Retrying… (${e.message})\n`)
    j.lost = true
  } finally {
    j.polling = false
  }
}

async function finishRemote(j: RemoteJob) {
  const { host, project } = j.info
  const local = metaIn(j.base, 'runs', j.run)
  // One read after the process ended, so the last events aren't lost. A run that died early has no file.
  const rest = await ssh(host, `tail -c +${j.events + 1} ${project}/.myseg/runs/${j.run}/events.jsonl 2>/dev/null || true`)
  if (remoteJob !== j) return
  if (rest.trim()) await saveEvents(j, rest)
  if (remoteJob !== j) return
  if (!j.lost) send('train:log', 'Bringing the model back…\n')
  await rsync('--exclude=checkpoint_*', '--exclude=events.jsonl', '--exclude=options.json',
    `${host}:${project}/.myseg/runs/${j.run}/`, `${local}/`)
  // Reattaching does this over, and only the attached job may go on to remove the remote copy
  if (remoteJob !== j) return
  if (!j.events && !j.stopped) {
    await saveFailure(j, `Training on ${host} ended before it reported anything. Its output is in the log, and in out.log in the run's folder.`)
  }
  await markFetched(j)
  // The run is safe here now; its copy on the remote disk isn't needed
  await ssh(host, `rm -rf ${quote(project)}`)
  if (remoteJob !== j) return
  send('train:log', 'Remote copy removed.\n')
  // Polling stops only now: if any of this failed, the next poll tries again
  detachRemote()
  send('train:exit', { run: j.run, code: 0 })
}

// Its files have left the machine, so there's nothing to wait for or bring back. If the
// model is already here, this is just a removal whose answer never reached us.
async function abandonRemote(j: RemoteJob) {
  if (!j.info.fetched) {
    await saveFailure(j, `This run's files are no longer on ${j.info.host}, so its model can't be brought back.`)
    await markFetched(j)
  }
  if (remoteJob !== j) return
  detachRemote()
  send('train:exit', { run: j.run, code: 0 })
}

// A run can outlive the app or a visit to another folder: pick it up again when its project opens
async function resumeRemote(base: string) {
  if (remoteJob) return
  const dir = metaIn(base, 'runs')
  if (!existsSync(dir)) return
  for (const name of await fs.readdir(dir)) {
    const info: RemoteInfo | null = await readJson(path.join(dir, name, 'remote.json'))
    if (info && !info.fetched) {
      const seen = await fs.stat(path.join(dir, name, 'events.jsonl')).then((s) => s.size).catch(() => 0)
      // Reading takes time, and another folder may have opened meanwhile
      if (root === base && !remoteJob) attachRemote(base, name, info, seen)
      return
    }
  }
}

function engine(args: string[]): Promise<any> {
  return new Promise((resolve, reject) => {
    execFile(PYTHON, [ENGINE, ...args], { cwd: REPO, maxBuffer: 1 << 28 }, (err, stdout, stderr) => {
      if (err) return reject(new Error(stderr.trim().split('\n').pop() || err.message))
      // Its answer is the last line; anything else it printed came before
      try {
        resolve(JSON.parse(stdout.trim().split('\n').pop()!))
      } catch {
        reject(new Error(`The engine gave no answer to ${args[0]}: ${stdout.trim().split('\n').pop() || 'it printed nothing'}`))
      }
    })
  })
}

async function listRuns() {
  const dir = meta('runs')
  const names = existsSync(dir) ? (await fs.readdir(dir)).sort().reverse() : []
  return Promise.all(
    names.map(async (name) => {
      const raw = await fs.readFile(path.join(dir, name, 'events.jsonl'), 'utf8').catch(() => '')
      const files = await fs.readdir(path.join(dir, name)).catch(() => [] as string[])
      return {
        name, events: parseEvents(raw),
        options: await readJson(path.join(dir, name, 'options.json')),
        running: job?.run === name || remoteJob?.run === name,
        // Where the engine's predict finds the model, and what Seamtag's pre-label menu reads
        model: files.some((f) => f.startsWith('run_') && existsSync(path.join(dir, name, f, 'best_model.pth'))),
        exported: files.includes('model.onnx')
      }
    })
  )
}

// What a finished run printed: a remote run's whole output, or else the training log every run keeps
async function runLog(name: string) {
  const dir = path.join(meta('runs'), path.basename(name))
  const output = await fs.readFile(path.join(dir, 'out.log'), 'utf8').catch(() => null)
  if (output !== null) return output
  for (const sub of (await fs.readdir(dir).catch(() => [] as string[])).filter((f) => f.startsWith('run_'))) {
    const logs = (await fs.readdir(path.join(dir, sub)).catch(() => [] as string[])).filter((f) => f.endsWith('.log'))
    if (logs.length) return fs.readFile(path.join(dir, sub, logs[0]), 'utf8')
  }
  return ''
}

function send(channel: string, payload: unknown) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload)
}

// A last line still being written isn't an event yet
const parseEvents = (raw: string) =>
  raw.slice(0, raw.lastIndexOf('\n') + 1).split('\n').filter(Boolean).map((l) => JSON.parse(l))

// The run's whole list each time: the window reads the file as well, whenever it likes,
// so it can't tell which of a few new events it already has
const sendEvents = (run: string, raw: string) => send('train:events', { run, events: parseEvents(raw) })

async function launch(options: any, target: { host: string; dir: string } | null) {
  if (job || remoteJob || launching) throw new Error('A training run is already active')
  launching = true
  try {
    return await (target ? startRemote(options, target) : startTraining(options))
  } finally {
    launching = false
  }
}

function startTraining(options: unknown) {
  const proc = spawn(PYTHON, [ENGINE, 'train', root, '--options', JSON.stringify(options)], {
    cwd: REPO, detached: true, env: { ...process.env, PYTHONUNBUFFERED: '1' }
  })
  return new Promise<string>((resolve, reject) => {
    let pending = ''
    const onLog = (chunk: Buffer) => {
      const text = chunk.toString()
      send('train:log', text)
      if (job) return
      pending += text
      const found = pending.match(/\{"run": "([^"]+)"\}/)
      if (!found) return
      const current = { proc, base: root, run: found[1], offset: 0, timer: setInterval(pollEvents, 1000) }
      job = current
      resolve(found[1])
    }
    proc.stdout!.on('data', onLog)
    proc.stderr!.on('data', onLog)
    proc.on('error', reject)
    proc.on('exit', async (code) => {
      if (!job) return reject(new Error(pending.trim().split('\n').pop() || `Engine exited with ${code}`))
      await pollEvents()
      clearInterval(job.timer)
      const run = job.run
      job = null
      send('train:exit', { run, code })
    })
  })
}

async function pollEvents() {
  const j = job
  if (!j) return
  const raw = await fs.readFile(metaIn(j.base, 'runs', j.run, 'events.jsonl'), 'utf8').catch(() => '')
  const end = raw.lastIndexOf('\n') + 1
  if (end <= j.offset) return
  j.offset = end
  sendEvents(j.run, raw)
}

function stopLocal() {
  if (job?.proc.pid) process.kill(-job.proc.pid, 'SIGTERM')
}

// A local run may have started from another folder, so an attached remote run is the one on screen
async function stopTraining() {
  if (!remoteJob) return stopLocal()
  remoteJob.stopped = true
  await ssh(remoteJob.info.host, `kill -TERM -${remoteJob.info.pid}`)
}

/**
 * Shapes the window sent back as it got them keep the file's own digits. Each shape in the file goes to
 * one shape at most: the one the window says came from there (`from`, by place in the file), as shapes
 * can differ only in a long id's last digits, or else the first left that reads the same, as after an undo.
 */
function exactly(doc: any, shapes: any[], from: number[]) {
  if (!exactDocs.has(doc)) return shapes
  const file: any[] = doc.shapes
  const texts = file.map((s) => JSON.stringify(plain(s)))
  const sent = shapes.map((s) => JSON.stringify(s))
  const taken = new Set<number>()
  const placed = shapes.map((_, i) => {
    const at = from[i]
    if (texts[at] !== sent[i] || taken.has(at)) return -1
    taken.add(at)
    return at
  })
  return shapes.map((s, i) => {
    const at = placed[i] >= 0 ? placed[i] : texts.findIndex((t, k) => t === sent[i] && !taken.has(k))
    if (at < 0) return s
    taken.add(at)
    return file[at]
  })
}

// LabelMe format, so labelme and Seamtag open the same files. Keys Seamline doesn't know stay as they
// were, and so does an image labelme keeps in the file (imageData). Gives the key readFound would.
async function writeLabels(file: string, image: string, doc: any | null, shapes: any[], size: number[], from: number[] = []) {
  const out = doc ?? { version: '5.5.0', flags: {}, shapes: [], imagePath: '', imageData: null }
  Object.assign(out, {
    // From the label file's own folder, as labelme reads it
    shapes: exactly(doc, shapes, from), imagePath: posix(path.relative(path.dirname(file), image)), imageWidth: size[0], imageHeight: size[1]
  })
  const text = JSON.stringify(out, null, 2)
  await writeAtomic(file, text)
  return `json:${createHash('sha1').update(text).digest('hex')}`
}

/**
 * The image's LabelMe file, read to be written over, and readFound's key of what the image has now.
 * The file is null when there's none yet, and undefined when the labels are kept in another format,
 * which Seamline never writes: Seamtag does, and keeps it.
 */
async function ownLabels(base: string, name: string, found: Located | null): Promise<{ doc: any; key: string }> {
  if (!found) return { doc: null, key: 'none' }
  if (found.kind !== 'json') return { doc: undefined, key: found.kind }
  const read = await readHashed(found.file)
  if (!read) return { doc: null, key: 'none' }
  return { doc: looksLikeCreateML(read.text) ? undefined : parseLabelMe(read.text, found.file), key: `json:${read.hash}` }
}

// One at a time with the other label writes. A file changed by another app since the window
// took it, even to the same size and time, is left as it is, and the window reloads it.
function saveLabels(base: string, name: string, shapes: any[], size: number[], from: number[] = []) {
  const image = imageIn(base, name)
  return serial(async () => {
    const { doc, key } = await ownLabels(base, name, await locate(base, name))
    if (key !== (based.get(image) ?? 'none') || doc === undefined) return { changed: true }
    const file = sidecar(name, base)
    // No shapes where the image had no labels: no file, and the one Seamline made for them goes
    if (!shapes.length && (!doc || made.has(image))) {
      if (doc) {
        await fs.rm(file, { force: true })
        wrote(base, name)
      }
      based.set(image, 'none')
      seen.set(image, 'none')
      if (made.has(image)) made.set(image, 'none')
      return { changed: false, stamp: null }
    }
    const now = await writeLabels(file, image, doc, shapes, size, from)
    wrote(base, name)
    based.set(image, now)
    seen.set(image, now)
    if (made.has(image)) made.set(image, now)
    return { changed: false, stamp: await statOf(file) }
  })
}

// The window takes what it loads only if it's still on that image: it says so with haveLabels
function loadLabels(base: string, name: string) {
  const image = imageIn(base, name)
  // Through the same queue as saves, so a load always sees the save sent before it
  return serial(async () => {
    const found = await locate(base, name)
    const { stamp } = await stampOf(base, name, found)
    const read = await readFound(base, name, found)
    // Changed since the window last had them: what it remembers of this image is out of date
    const changed = based.has(image) && based.get(image) !== read.key
    seen.set(image, read.key)
    // No file, or the one Seamline made for fresh labels, as it made it
    const fresh = read.key === 'none' || made.get(image) === read.key
    return { shapes: read.shapes, format: read.format, stamp, key: read.key, changed, fresh }
  })
}

// The window shows the labels it loaded as `key`, and its saves are checked against them. Unless
// Seamline has read or written others since, as pre-labelling does: those stay what it checks against.
// Whether they're fresh goes by the window, which takes a pre-labelling result as not fresh.
function haveLabels(base: string, name: string, key: string, fresh: boolean) {
  const image = path.resolve(base, name)
  if (fresh) made.set(image, key)
  else made.delete(image)
  if (seen.get(image) === key) based.set(image, key)
}

async function predictOne(runName: string, images: string[]) {
  images.forEach(inProject)
  const lines = (await run(PYTHON, [ENGINE, 'predict', root, '--run', runName, '--images', ...images])).trim().split('\n')
  const results: Record<string, any[]> = {}
  let device = ''
  for (const line of lines.filter((l) => l.startsWith('{')).map((l) => JSON.parse(l))) {
    if (line.device) device = line.device
    if (line.error) throw new Error(line.error)
    if (line.name) results[line.name] = line.shapes
  }
  return { device, results }
}

// Saves as it goes, so stopping keeps what's done. Images someone has labelled are left alone, as
// are labels in another format and any that changed since Seamline last read them, and images the
// model finds nothing on.
function prelabelAll(runName: string, names: string[]) {
  if (prelabelJob) throw new Error('Pre-labelling is already running')
  names.forEach(inProject)
  const base = root
  const proc = spawn(PYTHON, [ENGINE, 'predict', root, '--run', runName], {
    cwd: REPO, detached: true, env: { ...process.env, PYTHONUNBUFFERED: '1' }
  })
  const current: PrelabelJob = { proc, stopped: false }
  prelabelJob = current
  const failed: string[] = []
  let done = 0, pending = '', errors = ''
  let queue = Promise.resolve()

  const onResult = async (r: { name: string; shapes?: any[]; size?: number[]; error?: string }) => {
    if (prelabelJob !== current) return
    let shapes: any[] | null = null
    // The key of what was written, which the window takes if it shows the result
    let written: string | undefined
    if (r.error) {
      failed.push(`${r.name}: ${r.error}`)
    } else {
      shapes = await serial(async () => {
        const image = path.join(base, r.name)
        const { doc, key } = await ownLabels(base, r.name, await locate(base, r.name))
        const existing: any[] = doc?.shapes ?? []
        // As in Seamtag, a LabelMe file with no shapes is someone's too, as when they rejected every
        // shape: it stays empty, even if it was emptied after the batch started
        if (doc === undefined || existing.some((s) => !s.flags?.ai) || (doc && !existing.length)) return null
        if (key !== (seen.get(image) ?? 'none')) return null
        // Emptied by the batch, a file would read as emptied by someone: the image stays as it was
        if (!r.shapes!.length) return null
        const now = await writeLabels(sidecar(r.name, base), image, doc, r.shapes!, r.size!)
        wrote(base, r.name)
        based.set(image, now)
        seen.set(image, now)
        // Fresh labels the window still has are saved over this, but a later load takes it as pre-labelling's
        if (made.has(image)) made.set(image, null)
        written = now
        return r.shapes!
      }).catch((e) => {
        failed.push(`${r.name}: ${e.message}`)
        return null
      })
    }
    done++
    send('prelabel:progress', { name: r.name, shapes, key: written, done, total: names.length })
  }

  return new Promise<{ done: number; total: number; stopped: boolean; failed: string[]; error?: string }>((resolve, reject) => {
    proc.stdout!.on('data', (chunk: Buffer) => {
      pending += chunk.toString()
      const lines = pending.split('\n')
      pending = lines.pop()!
      for (const line of lines) {
        if (line.startsWith('{"name"')) queue = queue.then(() => onResult(JSON.parse(line)))
      }
    })
    proc.stderr!.on('data', (chunk: Buffer) => { errors = (errors + chunk.toString()).slice(-2000) })
    proc.stdin!.on('error', () => {})
    proc.on('error', (e) => {
      if (prelabelJob === current) prelabelJob = null
      reject(e)
    })
    proc.on('close', async (code, signal) => {
      await queue
      if (prelabelJob === current) prelabelJob = null
      const crashed = code !== 0 && !current.stopped
      const reason = errors.trim().split('\n').pop()
      const error = crashed ? `Engine stopped (${signal ?? code})${reason ? `: ${reason}` : ''}` : undefined
      if (crashed && !done) reject(new Error(error))
      else resolve({ done, total: names.length, stopped: current.stopped || crashed, failed, error })
    })
    proc.stdin!.end(JSON.stringify(names))
  })
}

function stopPrelabel() {
  if (!prelabelJob?.proc.pid) return
  prelabelJob.stopped = true
  try {
    process.kill(-prelabelJob.proc.pid, 'SIGTERM')
  } catch {
    // Already gone
  }
}

// Seamtag, the label-only app made for the same folders: where the user put it, or where its installers do
const choiceFile = () => path.join(app.getPath('userData'), 'seamtag.json')

function installedPlaces() {
  const onPath = (command: string) => (process.env.PATH ?? '').split(path.delimiter)
    .filter((dir) => path.isAbsolute(dir)).map((dir) => path.join(dir, command))
  const places = [...onPath(process.platform === 'win32' ? 'seamtag.exe' : 'seamtag'), '/usr/bin/seamtag', '/opt/Seamtag/seamtag']
  if (process.platform === 'darwin') places.push('/Applications/Seamtag.app', path.join(homedir(), 'Applications', 'Seamtag.app'))
  if (process.platform === 'win32') {
    const { LOCALAPPDATA, ProgramFiles } = process.env
    if (LOCALAPPDATA) places.push(path.join(LOCALAPPDATA, 'Programs', 'Seamtag', 'Seamtag.exe'))
    if (ProgramFiles) places.push(path.join(ProgramFiles, 'Seamtag', 'Seamtag.exe'))
  }
  return places
}

async function runnable(file: string) {
  try {
    if ((await fs.stat(file)).isDirectory()) return process.platform === 'darwin' && file.endsWith('.app')
    if (process.platform === 'win32') return file.toLowerCase().endsWith('.exe')
    await fs.access(file, constants.X_OK)
    return true
  } catch {
    return false
  }
}

// Every copy that could start, in the order to try them, and the one the user picked
async function findSeamtag() {
  const read: unknown = (await readJson(choiceFile()))?.path
  const chosen = typeof read === 'string' && read ? read : null
  const given = process.env.SEAMLINE_SEAMTAG
  // A copy named in SEAMLINE_SEAMTAG replaces the search, so a test never starts the one installed here
  const places = given ? [given, chosen] : [chosen, ...installedPlaces()]
  const found: string[] = []
  for (const place of new Set(places)) if (place && (await runnable(place))) found.push(place)
  return { found, chosen }
}

// Seamline's environment, less what its Electron and electron-vite's dev server set for Seamline
// alone: Seamtag, an Electron app too, would take them as its own, and even load Seamline's page
function seamtagEnv() {
  const env = { ...process.env }
  for (const key of ['ELECTRON_RENDERER_URL', 'NODE_ENV_ELECTRON_VITE', 'ELECTRON_ENTRY', 'ELECTRON_CLI_ARGS', 'CHROME_DESKTOP']) delete env[key]
  if ('ORIGINAL_XDG_CURRENT_DESKTOP' in env) {
    const desktop = env.ORIGINAL_XDG_CURRENT_DESKTOP
    if (desktop) env.XDG_CURRENT_DESKTOP = desktop
    else delete env.XDG_CURRENT_DESKTOP
    delete env.ORIGINAL_XDG_CURRENT_DESKTOP
  }
  if (GDK_BACKEND === undefined) delete env.GDK_BACKEND
  return env
}

/**
 * On its own, so it outlives Seamline. A macOS app goes through `open`, -n for a window of its own.
 * Why it didn't start, if it stopped at once with an error; null once it's running, or `open` is done.
 */
async function startSeamtag(program: string, target: string) {
  const mac = process.platform === 'darwin' && program.endsWith('.app')
  const options = { detached: true, stdio: 'ignore' as const, env: seamtagEnv() }
  const child = mac ? spawn('open', ['-n', '-a', program, '--args', target], options) : spawn(program, [target], options)
  child.on('error', () => {})
  const ended = new Promise<string | null>((resolve) => child.once('exit', (code, signal) => {
    resolve(signal ? `signal ${signal}` : code ? `exit code ${code}` : null)
  }))
  await new Promise<void>((resolve, reject) => {
    child.once('spawn', resolve)
    child.once('error', reject)
  })
  let timer: NodeJS.Timeout | undefined
  const stopped = await Promise.race([ended, new Promise<null>((resolve) => (timer = setTimeout(resolve, SEAMTAG_STARTS, null)))])
  clearTimeout(timer)
  child.unref()
  if (!stopped) return null
  // Ubuntu 24.04 lets no AppImage start Chromium's sandbox
  const why = process.platform === 'linux' && /\.appimage$/i.test(program)
    ? " Seamtag's AppImage needs --no-sandbox on Ubuntu 24.04 and later, so the .deb is the better install there." : ''
  return `${program} closed right after starting (${stopped}).${why}`
}

/**
 * Starts the first copy found that doesn't close as it starts, such as the .deb's program after an
 * AppImage that can't start any more. A copy the user picked that doesn't start is forgotten once
 * another does, so it isn't tried first every time. Why none started, if none did.
 */
async function launchSeamtag(found: string[], chosen: string | null, target: string) {
  const failed: string[] = []
  for (const [at, program] of found.entries()) {
    const why = await startSeamtag(program, target)
    if (!why) {
      if (chosen && found.slice(0, at).includes(chosen)) await fs.rm(choiceFile(), { force: true })
      return null
    }
    failed.push(why)
  }
  return failed.join(' ')
}

// The image itself, even in a subfolder: Seamtag 3.3 and later open the folder above it that has
// .myseg/project.json, on that image, where Seamtag 3.2.0 opens just the image's own folder
function seamtagTarget(base: string, name: string | null) {
  if (name) return imageIn(base, name)
  if (!opened.has(base)) throw new Error('That folder is not open')
  return base
}

function registerIpc() {
  ipcMain.handle('project:pick', async () => {
    const result = await dialog.showOpenDialog(win!, {
      title: 'Open image folder', properties: ['openDirectory']
    })
    return result.canceled ? null : openProject(result.filePaths[0])
  })
  ipcMain.handle('project:open', (_e, dir: string) => openProject(dir))
  ipcMain.handle('project:refresh', (_e, base: string) => refreshProject(base))
  ipcMain.handle('project:save-classes', (_e, base: string, classes: LabelClass[]) => saveClasses(base, classes))
  ipcMain.handle('project:read-classes', (_e, base: string) => readProjectClasses(base))
  ipcMain.handle('image:open', (_e, name: string) => openImage(name))
  ipcMain.handle('labels:load', (_e, base: string, name: string) => loadLabels(base, name))
  ipcMain.handle('labels:have', (_e, base: string, name: string, key: string, fresh: boolean) => haveLabels(base, name, key, fresh))
  ipcMain.handle('labels:save', (_e, base: string, name: string, shapes: any[], size: number[], from: number[]) =>
    saveLabels(base, name, shapes, size, from)
  )
  ipcMain.handle('seamtag:open', async (_e, base: string, name: string | null) => {
    const target = seamtagTarget(base, name)
    const { found, chosen } = await findSeamtag()
    if (!found.length) return { missing: true }
    const failed = await launchSeamtag(found, chosen, target)
    return failed ? { failed } : { launched: true }
  })
  // The program picked is remembered only once it has started
  ipcMain.handle('seamtag:locate', async (_e, base: string, name: string | null) => {
    const target = seamtagTarget(base, name)
    const result = await dialog.showOpenDialog(win!, { title: 'Locate Seamtag', properties: ['openFile'] })
    const program = result.filePaths[0]
    if (result.canceled || !program) return { canceled: true }
    if (!(await runnable(program))) return { failed: `${path.basename(program)} isn't a program Seamline can start.` }
    // Its installer is named Seamtag-<version>-win-x64.exe
    if (process.platform === 'win32' && /-\d+\.\d+\.\d+/.test(path.basename(program))) {
      return { failed: `${path.basename(program)} looks like Seamtag's installer. Run it, then pick the Seamtag.exe it installs.` }
    }
    const failed = await startSeamtag(program, target)
    if (failed) return { failed }
    await fs.mkdir(path.dirname(choiceFile()), { recursive: true })
    await writeAtomic(choiceFile(), JSON.stringify({ path: program }, null, 2))
    return { launched: true }
  })
  ipcMain.handle('seamtag:download', () => shell.openExternal(SEAMTAG_RELEASES))
  ipcMain.handle('model:export', (_e, name: string) => engine(['export', root, '--run', path.basename(name)]))
  ipcMain.handle('runs:list', listRuns)
  ipcMain.handle('runs:log', (_e, name: string) => runLog(name))
  ipcMain.handle('train:start', (_e, options, target) => launch(options, target))
  ipcMain.handle('remote:check', (_e, host: string, dir: string) => checkRemote(host, dir))
  ipcMain.handle('train:stop', stopTraining)
  ipcMain.handle('train:active', () => remoteJob?.run ?? job?.run ?? null)
  ipcMain.handle('dataset:check', (_e, val: number, tile: number) =>
    engine(['prepare', root, '--val-split', String(val), '--tile', String(tile)])
  )
  ipcMain.handle('model:predict', (_e, runName: string, images: string[]) => predictOne(runName, images))
  ipcMain.handle('prelabel:start', (_e, runName: string, names: string[]) => prelabelAll(runName, names))
  ipcMain.handle('prelabel:stop', stopPrelabel)
}

app.whenReady().then(() => {
  protocol.handle('myseg', (request) => {
    const file = path.resolve(decodeURIComponent(new URL(request.url).pathname.slice(1)))
    if (!root || !file.startsWith(root + path.sep)) return new Response('Forbidden', { status: 403 })
    return net.fetch(pathToFileURL(file).toString())
  })
  registerIpc()
  win = new BrowserWindow({
    width: 1500, height: 940, minWidth: 1000, minHeight: 640,
    backgroundColor: '#14161a', autoHideMenuBar: true, title: 'Seamline',
    icon: path.join(__dirname, '../../resources/icon.png'),
    webPreferences: { preload: path.join(__dirname, '../preload/index.js'), sandbox: false }
  })
  // Saves wait a moment after each change: let the last one reach the disk before closing
  let flushed = false
  win.on('close', (e) => {
    if (flushed || !root) return
    e.preventDefault()
    const close = () => {
      if (flushed) return
      flushed = true
      win?.close()
    }
    ipcMain.once('app:flushed', close)
    setTimeout(close, 3000)
    send('app:flush', null)
  })
  if (process.env.ELECTRON_RENDERER_URL) win.loadURL(process.env.ELECTRON_RENDERER_URL)
  else {
    const project = process.env.MYSEG_PROJECT
    win.loadFile(path.join(__dirname, '../renderer/index.html'), project ? { query: { project } } : {})
  }
})

// Local runs end with the app; remote ones keep going and reattach on next open
app.on('window-all-closed', () => {
  stopLocal()
  stopPrelabel()
  app.quit()
})
