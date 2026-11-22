// Puts uv, which sets up the engine on first run, into build/uv for the installer being made.
// scripts/before-pack.mjs calls it before electron-builder packs each system's app; it can also
// run by hand: node scripts/fetch-uv.mjs [linux|darwin|win32] [x64|arm64]. The version and
// checksums are pinned: uv 0.13.7, as tested with Seamline's requirements.
import { execFileSync } from 'child_process'
import { createHash } from 'crypto'
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const VERSION = '0.13.7'
const BUILDS = {
  'linux-x64': ['uv-x86_64-unknown-linux-gnu.tar.gz', 'c5221ff41b7892bfc223543883600125f6a47fe845093e0f5bdff4f359196d83'],
  'darwin-arm64': ['uv-aarch64-apple-darwin.tar.gz', '81b811b57e4624df7ed3aac148e0359100d9962f3d9dce00aec533dc2e086731'],
  'win32-x64': ['uv-x86_64-pc-windows-msvc.zip', '0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021']
}
const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

export async function fetchUv(platform, arch) {
  const key = `${platform}-${arch}`
  if (!BUILDS[key]) throw new Error(`No uv build is pinned for ${key}`)
  const [asset, sha256] = BUILDS[key]
  const out = path.join(app, 'build', 'uv')
  const exe = platform === 'win32' ? 'uv.exe' : 'uv'
  const stamp = path.join(out, 'build.txt')
  if (existsSync(path.join(out, exe)) && existsSync(stamp) && readFileSync(stamp, 'utf8') === `${VERSION} ${key}`) return
  const cache = path.join(app, 'build', 'downloads')
  mkdirSync(cache, { recursive: true })
  const archive = path.join(cache, `${VERSION}-${asset}`)
  if (!existsSync(archive) || createHash('sha256').update(readFileSync(archive)).digest('hex') !== sha256) {
    const response = await fetch(`https://github.com/astral-sh/uv/releases/download/${VERSION}/${asset}`)
    if (!response.ok) throw new Error(`Downloading ${asset} failed: ${response.status}`)
    const data = Buffer.from(await response.arrayBuffer())
    const got = createHash('sha256').update(data).digest('hex')
    if (got !== sha256) throw new Error(`${asset} has checksum ${got}, not the pinned ${sha256}`)
    writeFileSync(archive, data)
  }
  const unpacked = path.join(cache, `${VERSION}-${key}`)
  rmSync(unpacked, { recursive: true, force: true })
  mkdirSync(unpacked, { recursive: true })
  // tar reads both: GNU tar on Linux, bsdtar on macOS, and Windows' own tar.exe, which opens a .zip
  // where Git's GNU tar, often first on a runner's PATH, can't
  const tar = process.platform === 'win32' ? path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe') : 'tar'
  execFileSync(tar, ['-xf', archive, '-C', unpacked])
  const inner = asset.endsWith('.zip') ? unpacked : path.join(unpacked, asset.replace(/\.tar\.gz$/, ''))
  rmSync(out, { recursive: true, force: true })
  mkdirSync(out, { recursive: true })
  copyFileSync(path.join(inner, exe), path.join(out, exe))
  writeFileSync(stamp, `${VERSION} ${key}`)
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await fetchUv(process.argv[2] ?? process.platform, process.argv[3] ?? process.arch)
  console.log(`uv ${VERSION} in build/uv`)
}
