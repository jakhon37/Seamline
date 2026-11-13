import { _electron as electron, ElectronApplication, expect, Page } from '@playwright/test'
import { chmodSync, copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import path from 'path'
import sharp from 'sharp'

// Folders as Seamtag leaves them, the two apps taking turns on them. Each test has a home folder
// and profile of its own, so nothing reads or writes the settings of the person running it.
const BASE = process.env.E2E_SEAMTAG_DIR || path.join(tmpdir(), 'seamline-e2e-seamtag')
// Label files from Seamtag's tests/fixtures: labelme 5's and LabelImg 1.8.6's own files, and labelme 7's shapes
const FIXTURES = path.join(__dirname, 'fixtures/seamtag')
const BACKGROUND = '#445566'
const SHOTS = process.env.E2E_SHOTS || path.join(BASE, 'shots')

export async function shot(page: Page, name: string) {
  mkdirSync(SHOTS, { recursive: true })
  await page.screenshot({ path: path.join(SHOTS, `${name}.png`) })
}

export function folder(name: string) {
  const dir = path.join(BASE, name)
  rmSync(dir, { recursive: true, force: true })
  mkdirSync(dir, { recursive: true })
  return dir
}

/** A plain image of one colour, in the format its name says */
export async function image(dir: string, name: string, width: number, height: number) {
  const file = path.join(dir, name)
  mkdirSync(path.dirname(file), { recursive: true })
  const ext = path.extname(name).toLowerCase()
  if (ext === '.bmp') return writeFileSync(file, bmp(width, height))
  const plain = sharp({ create: { width, height, channels: 3, background: BACKGROUND } })
  const encoded = ext === '.png' ? plain.png() : ext === '.gif' ? plain.gif() : ext === '.webp' ? plain.webp()
    : ext === '.avif' ? plain.avif() : ext === '.tif' || ext === '.tiff' ? plain.tiff() : plain.jpeg()
  await encoded.toFile(file)
}

// sharp writes no BMP, so this one is made by hand: 24 bits, rows bottom up
function bmp(width: number, height: number) {
  const row = Math.ceil((width * 3) / 4) * 4
  const out = Buffer.alloc(54 + row * height)
  out.write('BM')
  out.writeUInt32LE(out.length, 2)
  out.writeUInt32LE(54, 10)
  out.writeUInt32LE(40, 14)
  out.writeInt32LE(width, 18)
  out.writeInt32LE(height, 22)
  out.writeUInt16LE(1, 26)
  out.writeUInt16LE(24, 28)
  out.writeUInt32LE(row * height, 34)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) out.set([0x66, 0x55, 0x44], 54 + y * row + x * 3)
  }
  return out
}

export function copy(dir: string, fixture: string, to: string) {
  mkdirSync(path.dirname(path.join(dir, to)), { recursive: true })
  copyFileSync(path.join(FIXTURES, fixture), path.join(dir, to))
}

export function classes(dir: string, names: string[]) {
  const palette = ['#4f8cff', '#ff6b6b', '#3ecf8e', '#f7b731', '#b57bff', '#22c7d6', '#ff8f4d', '#ec6fb3']
  mkdirSync(path.join(dir, '.myseg'), { recursive: true })
  writeFileSync(path.join(dir, '.myseg/project.json'), JSON.stringify({
    name: path.basename(dir), classes: names.map((name, i) => ({ name, color: palette[i % palette.length] }))
  }))
}

export const read = (dir: string, name: string) => JSON.parse(readFileSync(path.join(dir, name), 'utf8'))

/** Every file in the folder but Seamline's own, with its bytes */
export function files(dir: string, sub = ''): Record<string, string> {
  const out: Record<string, string> = {}
  for (const e of readdirSync(path.join(dir, sub), { withFileTypes: true })) {
    const rel = sub ? `${sub}/${e.name}` : e.name
    if (e.name === '.myseg') continue
    if (e.isDirectory()) Object.assign(out, files(dir, rel))
    else if (e.isFile()) out[rel] = readFileSync(path.join(dir, rel)).toString('base64')
  }
  return out
}

/**
 * A stand-in for Seamtag that notes what it was started with, and whether the image's labels were saved by then.
 * It ends at once, or keeps running for `stay` seconds, as Seamtag does.
 */
export function fakeSeamtag(home: string, watch = '', stay = 0) {
  const script = path.join(home, 'seamtag')
  const args = path.join(home, 'seamtag-args')
  mkdirSync(home, { recursive: true })
  writeFileSync(script, `#!/bin/sh
printf '%s\\n' "$@" >> '${args}'
${watch ? `if [ -e '${watch}' ]; then echo 'labels saved'; else echo 'labels not saved'; fi >> '${args}'` : ''}
${stay ? `sleep ${stay}` : ''}
`)
  chmodSync(script, 0o755)
  return { script, args: () => (existsSync(args) ? readFileSync(args, 'utf8').split('\n').filter(Boolean) : []) }
}

/** The engine, for the commands these tests need: a few lines of Node rather than PyTorch */
export function fakeEngine(home: string, source: string) {
  const script = path.join(home, 'engine')
  mkdirSync(home, { recursive: true })
  writeFileSync(script, `#!${process.execPath}\n${source}`)
  chmodSync(script, 0o755)
  return script
}

export async function launch(dir: string, env: Record<string, string> = {}) {
  const home = path.join(BASE, 'home', `${path.basename(dir)}-${Date.now()}`)
  mkdirSync(home, { recursive: true })
  // Electron 33 crashes under --ozone-platform=headless, even with an empty window: run these on a
  // virtual X server, such as Xvfb, so no window opens on your screen
  const app = await electron.launch({
    args: [path.resolve(__dirname, '..')],
    env: {
      ...process.env, HOME: home, XDG_CONFIG_HOME: path.join(home, '.config'), MYSEG_PROJECT: dir,
      // Never the Seamtag installed on this machine
      SEAMLINE_SEAMTAG: path.join(home, 'no-seamtag-here'),
      ...env
    } as Record<string, string>
  })
  const page = await app.firstWindow()
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await expect(page.locator('.file').first()).toBeVisible()
  await page.waitForTimeout(300)
  return { app, page, errors, home }
}

export async function finish(app: ElectronApplication, errors: string[]) {
  await app.close()
  expect(errors).toEqual([])
}

export const file = (page: Page, name: string) => page.locator(`.file[title="${name}"]`)

export async function open(page: Page, name: string, size: number[]) {
  await file(page, name).click()
  await expect(page.locator('.file.on')).toHaveAttribute('title', name)
  await expect(page.locator('.status')).toContainText(`${size[0]} × ${size[1]}`)
  await expect(page.locator('.loading', { hasText: 'Preparing' })).toHaveCount(0)
  await page.waitForTimeout(250)
}

export async function onImage(page: Page, size: number[], x: number, y: number) {
  const box = (await page.getByTestId('canvas').boundingBox())!
  const scale = Math.min((box.width - 48) / size[0], (box.height - 48) / size[1])
  return [box.x + (box.width - size[0] * scale) / 2 + x * scale, box.y + (box.height - size[1] * scale) / 2 + y * scale] as const
}

export async function drag(page: Page, size: number[], from: number[], to: number[]) {
  await page.mouse.move(...await onImage(page, size, from[0], from[1]))
  await page.mouse.down()
  await page.mouse.move(...await onImage(page, size, (from[0] + to[0]) / 2, (from[1] + to[1]) / 2), { steps: 6 })
  await page.mouse.move(...await onImage(page, size, to[0], to[1]), { steps: 6 })
  await page.mouse.up()
}

export async function polygon(page: Page, size: number[], points: number[][]) {
  await page.keyboard.press('p')
  for (const [x, y] of points) await page.mouse.click(...await onImage(page, size, x, y))
  await page.keyboard.press('Enter')
}

/** The canvas's colour at a point of the image */
export async function pixel(page: Page, size: number[], x: number, y: number) {
  const [cssX, cssY] = await onImage(page, size, x, y)
  return page.getByTestId('canvas').evaluate((c: HTMLCanvasElement, [px, py]) => {
    const box = c.getBoundingClientRect()
    const at = [(px - box.left) * (c.width / box.width), (py - box.top) * (c.height / box.height)]
    return [...c.getContext('2d')!.getImageData(at[0], at[1], 1, 1).data].slice(0, 3)
  }, [cssX, cssY])
}

export const saved = (page: Page) => expect(page.getByTestId('save-state')).toHaveText('Saved')
