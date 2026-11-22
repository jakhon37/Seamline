// Writes build/legal/THIRD-PARTY-NOTICES.txt: everything the installers carry that isn't Seamline's
// own code, each part with its licence in full. It reads the node_modules of the computer that
// builds, so each system's installer lists what it holds:
//   - the packages Vite builds into out/ from the devDependencies (BUILT_IN below),
//   - the packages app.asar carries: the dependencies, and theirs, as installed,
//   - libvips and the libraries built into it, from sharp's @img packages,
//   - Electron, and uv, which sets up the engine.
// Texts node_modules lacks are in resources/notices, whose README says where each came from. It
// stops when they no longer match what's installed, as another sharp-libvips would.
// Adapted from Seamtag's scripts/notices.mjs, written for Seamtag's installers.
//
// electron-builder runs it before it packs (beforePack, scripts/before-pack.mjs), so the release
// workflow's builds have it, and stop there if a file extraResources names is missing.

import { downloadArtifact } from '@electron/get'
import AdmZip from 'adm-zip'
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { builtinModules } from 'node:module'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const OUT = path.join(ROOT, 'build', 'legal', 'THIRD-PARTY-NOTICES.txt')
const NOTICES = path.join(ROOT, 'resources', 'notices')

// What Vite builds into out/ from the devDependencies: React, the icons, the Inter font and
// Clipper in the window. Every package src/ imports must be a dependency or here, and each of
// these packages' own dependencies here or in LEFT_OUT.
const BUILT_IN = ['react', 'react-dom', 'scheduler', 'lucide-react', '@fontsource-variable/inter', 'clipper-lib']
// React's build-time envify transform, which nothing in the bundles calls
const LEFT_OUT = ['loose-envify', 'js-tokens']
const ABOUT = { '@fontsource-variable/inter': 'the Inter font' }

const text = (file) => readFileSync(file, 'utf8').replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').trimEnd() + '\n'
const json = (file) => JSON.parse(readFileSync(file, 'utf8'))
const line = (c) => c.repeat(80)
const heading = (title, ...facts) => [line('='), title, ...facts.filter(Boolean), line('-'), '', ''].join('\n')

function packageAt(dir) {
  const pkg = json(path.join(dir, 'package.json'))
  let url = (typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url) ?? ''
  url = url.replace(/^git\+/, '').replace(/^git:\/\//, 'https://').replace(/^(ssh:\/\/)?git@github\.com[:/]/, 'https://github.com/')
    .replace(/^github:/, 'https://github.com/').replace(/\.git$/, '')
  if (/^[\w.-]+\/[\w.-]+$/.test(url)) url = `https://github.com/${url}`
  const licences = readdirSync(dir).filter((f) => /^(licen[cs]e|copying|notice)([-.][\w-]+)?(\.(md|markdown|txt))?$/i.test(f)).sort()
  return { dir, name: pkg.name, version: pkg.version, licence: pkg.license ?? 'none named', url: url || pkg.homepage || '', licences }
}
const title = (p) => `${p.name} ${p.version}${ABOUT[p.name] ? ` (${ABOUT[p.name]})` : ''}`

/** The dependencies app.asar carries, and theirs: the lockfile's packages that aren't for development, as installed here */
function appPackages() {
  const lock = json(path.join(ROOT, 'package-lock.json'))
  return Object.entries(lock.packages)
    .filter(([key, entry]) => key && !entry.dev && !entry.devOptional && existsSync(path.join(ROOT, key, 'package.json')))
    .map(([key]) => packageAt(path.join(ROOT, key)))
}

/** Stops when src/ imports a package that's neither a dependency nor in BUILT_IN, or one in BUILT_IN needs another */
function checkBuiltIn() {
  const known = new Set([...Object.keys(json(path.join(ROOT, 'package.json')).dependencies ?? {}), ...BUILT_IN, 'electron'])
  const unknown = new Set()
  const walk = (dir) => {
    for (const e of readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, e.name)
      if (e.isDirectory()) walk(file)
      else if (/\.(ts|tsx|css)$/.test(e.name)) {
        for (const [, type, spec] of readFileSync(file, 'utf8').matchAll(/\b(?:import|export)\s+(type\s+)?(?:[\w*{}\s,$]*?from\s*)?['"]([^'"]+)['"]/g)) {
          const name = spec.split('/').slice(0, spec.startsWith('@') ? 2 : 1).join('/')
          if (!type && !spec.startsWith('.') && !spec.startsWith('node:') && !builtinModules.includes(name) && !known.has(name)) unknown.add(name)
        }
      }
    }
  }
  walk(path.join(ROOT, 'src'))
  for (const name of BUILT_IN) {
    for (const dep of Object.keys(json(path.join(ROOT, 'node_modules', name, 'package.json')).dependencies ?? {})) {
      if (!BUILT_IN.includes(dep) && !LEFT_OUT.includes(dep)) unknown.add(dep)
    }
  }
  if (unknown.size) {
    const them = unknown.size === 1 ? 'it' : 'them'
    throw new Error(`Vite builds ${[...unknown].join(', ')} into the app: add ${them} to BUILT_IN in scripts/notices.mjs, or to LEFT_OUT if the bundles leave ${them} out`)
  }
}

/** A package's own licence files, or what stands in for them where it has none */
function licenceOf(p) {
  const own = p.licences.map((f) => (p.licences.length > 1 ? `${f}:\n\n` : '') + text(path.join(p.dir, f))).join('\n')
  if (p.name === 'clipper-lib') {
    // It names its licences in clipper.js's opening comments, which the bundles leave out
    const head = /^(\s*\/\*[\s\S]*?\*\/)+/.exec(readFileSync(path.join(p.dir, 'clipper.js'), 'utf8'))[0].trim()
    return [own, `clipper.js begins:\n\n${head}\n`, `The Boost Software License 1.0, which it names:\n\n${text(path.join(NOTICES, 'clipper-lib', 'BSL-1.0.txt'))}`,
      `The licence of Tom Wu's JSBN, part of which it holds:\n\n${text(path.join(NOTICES, 'clipper-lib', 'jsbn-LICENSE'))}`].filter(Boolean).join('\n')
  }
  if (/LGPL/.test(p.licence)) return `${own}${own ? '\n' : ''}It holds libvips: the licences are under libvips below.\n`
  if (!own) throw new Error(`${p.name} ${p.version} carries no licence file: give scripts/notices.mjs its licence text`)
  return own
}

// libvips's copyright line: the one its source files carry most often, as resources/notices/sharp-libvips-1.2.4/README.md says
const libvipsCopyright = () => 'Copyright (C) 1991-2005 The National Gallery and others'

/** The files in a folder, by their paths in it: the top level's first */
function filesIn(dir, at = '') {
  const here = readdirSync(path.join(dir, at), { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))
  return [...here.filter((e) => e.isFile()).map((e) => path.posix.join(at, e.name)),
    ...here.filter((e) => e.isDirectory()).flatMap((e) => filesIn(dir, path.posix.join(at, e.name)))]
}

/** The sharp-libvips release sharp takes libvips from, on every system: Windows's @img/sharp-win32-x64 packs it too */
function sharpLibvips() {
  const sharp = json(path.join(ROOT, 'node_modules', 'sharp', 'package.json'))
  const releases = [...new Set(Object.entries(sharp.optionalDependencies ?? {}).filter(([name]) => name.startsWith('@img/sharp-libvips-')).map(([, v]) => v))]
  if (releases.length !== 1) throw new Error(`sharp ${sharp.version} names ${releases.join(' and ') || 'no'} sharp-libvips version${releases.length ? 's' : ''}: scripts/notices.mjs takes one`)
  return releases[0]
}

/**
 * The licence files of libvips and of each library built into it, from their sources at the
 * versions libvips is built from (resources/notices/sharp-libvips-<release>). It stops when sharp
 * takes libvips from another sharp-libvips, or the libraries or their versions change.
 */
function libraryLicences(release, versions) {
  const folder = `resources/notices/sharp-libvips-${release}`
  const dir = path.join(ROOT, folder)
  const how = 'See resources/notices/README.md'
  if (!existsSync(path.join(dir, 'sources.json'))) {
    throw new Error(`${folder} needs the licence files of libvips and the libraries built into it, from their sources at the versions sharp-libvips ${release} builds. ${how}`)
  }
  const { libraries } = json(path.join(dir, 'sources.json'))
  const changed = [...new Set([...Object.keys(versions), ...Object.keys(libraries)])].sort()
    .filter((key) => versions[key] !== libraries[key]?.version || !existsSync(path.join(dir, key)))
    .map((key) => `${key} ${versions[key] ?? 'not'} in libvips, ${libraries[key]?.version ?? 'not'} in ${folder}${existsSync(path.join(dir, key)) ? '' : ', with no folder'}`)
  if (changed.length) throw new Error(`The libraries in libvips don't match ${folder}: ${changed.join('; ')}. ${how}`)
  // A text given once: the next library with the same words points to it
  const given = new Map()
  const words = (t) => t.replace(/\s+/g, ' ').trim()
  // libvips first, then the libraries built into it
  const order = (a, b) => (b === 'vips') - (a === 'vips') || libraries[a].name.localeCompare(libraries[b].name)
  return Object.keys(libraries).sort(order).map((key) => {
    const lib = libraries[key]
    const files = filesIn(path.join(dir, key)).map((file) => {
      const of = file.replace(/\.excerpt$/, '')
      if (of !== file && !lib.excerpts?.[of]) throw new Error(`${folder}/sources.json gives no lines for ${key}/${file}`)
      const label = of === file ? file : `${of}, lines ${lib.excerpts[of]}`
      const body = text(path.join(dir, key, file))
      if (given.has(words(body))) return `${label}: the same words as ${given.get(words(body))}, above.\n`
      given.set(words(body), `${lib.name}'s ${label}`)
      return `${label}:\n\n${body}`
    })
    return [`${line('.')}\n${lib.name} ${lib.version}, from ${lib.source}\n`, ...files].join('\n')
  })
}

/** libvips and the libraries built into it, from the @img package sharp loads it from */
function libvips(packages) {
  return packages.filter((p) => /LGPL/.test(p.licence)).map((p) => {
    const readme = existsSync(path.join(p.dir, 'README.md')) ? text(path.join(p.dir, 'README.md')) : ''
    const licensing = readme.split(/^(?=## )/m).find((s) => /^## Licen[cs]ing\b/.test(s))
    if (!licensing) throw new Error(`${p.name}'s README has no Licensing section, the list of the libraries in libvips and their licences`)
    const versions = existsSync(path.join(p.dir, 'versions.json')) ? json(path.join(p.dir, 'versions.json')) : {}
    const lib = existsSync(path.join(p.dir, 'lib')) ? readdirSync(path.join(p.dir, 'lib')).filter((f) => /vips/.test(f)).sort() : []
    const vips = versions.vips ? `libvips ${versions.vips}` : 'libvips'
    const release = sharpLibvips()
    return {
      listed: `  ${vips} and the libraries built into it, in ${p.name} ${p.version}: ${p.licence}, and the licences listed there`,
      body: [
        heading(`${vips} and the libraries built into it`, `In ${p.name} ${p.version}. Licence: ${p.licence}`) +
          `libvips is ${libvipsCopyright()}. It is licensed under the GNU Lesser General Public License version 2.1 or later, and used here under version 3 (LGPL-3.0-or-later), which ends this part.`,
        '',
        `sharp loads libvips as a shared library${lib.length ? ` (${lib.join(', ')})` : ''}, from node_modules/${p.name}/lib in app.asar.unpacked, beside app.asar.`,
        ...(Object.keys(versions).length ? ['', `Versions: ${Object.entries(versions).map(([k, v]) => `${k} ${v}`).join(', ')}.`] : []),
        '',
        `Sources: libvips at ${versions.vips ? `https://github.com/libvips/libvips/tree/v${versions.vips}` : 'https://github.com/libvips/libvips'}. https://github.com/lovell/sharp-libvips/tree/v${release} builds it and the libraries listed below, and says where each one's source comes from.`,
        '',
        `${p.name}'s README says:`,
        '',
        licensing.trimEnd(),
        '',
        'This software is based in part on the work of the Independent JPEG Group.',
        'This software is based in part on the work of the FreeType Team (https://freetype.org).',
        'This product includes Radiance software (http://radsite.lbl.gov/) developed by the Lawrence Berkeley National Laboratory (http://www.lbl.gov/).',
        "libtiff's LZW compression (tif_lzw.c) was developed by the University of California, Berkeley.",
        '',
        'The licence files of libvips and of each library built into it follow, as their sources carry them at these versions, with the licences of code from other projects that they build in.',
        '',
        ...libraryLicences(release, versions),
        line('.'),
        'The GNU Lesser General Public License version 3 follows. It adds to the GNU General Public License version 3, at https://www.gnu.org/licenses/gpl-3.0.txt.',
        '',
        text(path.join(NOTICES, 'LGPL-3.0.txt'))
      ].join('\n')
    }
  })
}

// uv, in the installers beside the app, which sets up the engine on first run
function uv() {
  const version = /const VERSION = '([^']+)'/.exec(readFileSync(path.join(ROOT, 'scripts', 'fetch-uv.mjs'), 'utf8'))[1]
  return {
    listed: `  uv ${version}: MIT, or Apache-2.0`,
    body: [
      heading(`uv ${version}`, 'Licence: MIT, or Apache-2.0 (Seamline takes it under MIT)', 'https://github.com/astral-sh/uv') +
        "Seamline sets up its engine with uv, which is in the uv folder beside the app. uv's licence, from its source at this version:",
      '',
      text(path.join(ROOT, 'resources', 'legal', 'uv-LICENSE-MIT.txt'))
    ].join('\n')
  }
}

function electron() {
  const p = packageAt(path.join(ROOT, 'node_modules', 'electron'))
  return {
    listed: `  Electron ${p.version}: ${p.licence}, and Chromium's components under theirs`,
    body: [
      heading(`Electron ${p.version}, and Chromium`, `Licence: ${p.licence}`, p.url) +
        'Seamline runs on Electron, which builds in Chromium and the components Chromium uses, FFmpeg among them. Their licences are in LICENSES.chromium.html, and Electron\'s in LICENSE.electron.txt: in the folder that holds the seamline program on Windows and Linux, and beside this file on macOS. Electron\'s licence:',
      '',
      text(path.join(p.dir, 'LICENSE'))
    ].join('\n')
  }
}

export function notices() {
  checkBuiltIn()
  const byName = (a, b) => a.name.localeCompare(b.name) || a.dir.localeCompare(b.dir)
  const builtIn = BUILT_IN.map((name) => packageAt(path.join(ROOT, 'node_modules', name))).sort(byName)
  // A package installed twice, at one version, as semver is for sharp and global-agent, is listed once
  const app = [...new Map(appPackages().sort(byName).map((p) => [`${p.name}@${p.version}`, p])).values()]
  const libraries = [...libvips(app), electron(), uv()]
  const listed = (p) => `  ${title(p)}: ${p.licence}`
  const entry = (p) => heading(title(p), `Licence: ${p.licence}`, p.url) + licenceOf(p)
  return [
    `Seamline ${json(path.join(ROOT, 'package.json')).version}: third-party notices`,
    '',
    'Seamline is free software under the MIT License, which is in LICENSE, beside this file. THIRD_PARTY_NOTICES.md, also beside it, names the code from other projects in Seamline\'s own source files, the Python packages its engine downloads on first run, which the installers don\'t carry, and the datasets in its pictures.',
    '',
    "This file lists the other software the app carries, and each part's licence as that part's source gives it.",
    '',
    `Built into the app's code (${builtIn.length}):`,
    ...builtIn.map(listed),
    '',
    `In the app's node_modules (${app.length}):`,
    ...app.map(listed),
    '',
    'Libraries:',
    ...libraries.map((l) => l.listed),
    '',
    ...builtIn.map(entry),
    ...app.map(entry),
    ...libraries.map((l) => l.body)
  ].join('\n')
}

export function write() {
  const out = notices()
  mkdirSync(path.dirname(OUT), { recursive: true })
  writeFileSync(OUT, out)
  console.log(`build/legal/THIRD-PARTY-NOTICES.txt: ${Math.round(Buffer.byteLength(out) / 1024)} KB`)
}

// Chromium's licences, which a Mac app carries beside NOTICE.md (mac extraResources in
// electron-builder.yml), as electron-builder leaves Electron's licence files out of it. They're read
// from Electron's own zip for the build's system and processor, through @electron/get, which keeps it
// in its cache, or downloads it checked against Electron's checksums: Electron's npm package fills
// node_modules/electron/dist only when something first loads it, which a release build never does
// before it packs, so on the 3.3.0 tag's Mac runners there was no dist to take them from.
export const CHROMIUM_LICENCES = path.join(ROOT, 'build', 'legal', 'LICENSES.chromium.html')
export async function chromiumLicences(platform, arch) {
  const electron = path.join(ROOT, 'node_modules', 'electron')
  const zip = await downloadArtifact({
    version: JSON.parse(readFileSync(path.join(electron, 'package.json'), 'utf8')).version,
    artifactName: 'electron',
    platform,
    arch,
    cacheRoot: process.env.electron_config_cache,
    checksums: JSON.parse(readFileSync(path.join(electron, 'checksums.json'), 'utf8'))
  })
  const text = new AdmZip(zip).readFile('LICENSES.chromium.html')
  if (!text) throw new Error(`${path.basename(zip)} holds no LICENSES.chromium.html`)
  mkdirSync(path.dirname(CHROMIUM_LICENCES), { recursive: true })
  writeFileSync(CHROMIUM_LICENCES, text)
}
// electron-builder's Arch, by its number
const ARCHES = ['ia32', 'x64', 'armv7l', 'arm64']

/** Stops the build when a file extraResources names is missing: electron-builder would only warn, and leave it out */
export function checkResources(packager) {
  const asList = (v) => (v == null ? [] : Array.isArray(v) ? v : [v])
  const missing = [...asList(packager.config.extraResources), ...asList(packager.platformSpecificBuildOptions.extraResources)]
    .flatMap((r) => (typeof r === 'object' && r.from && !r.from.includes('${') ? [r.from] : []))
    .filter((from) => !existsSync(path.resolve(packager.projectDir, from)))
  if (missing.length) throw new Error(`The installers need ${missing.join(' and ')}, which ${missing.length === 1 ? "isn't" : "aren't"} there`)
}
export { ARCHES }

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) write()
