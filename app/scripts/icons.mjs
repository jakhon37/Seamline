// The installers' icons, drawn from the brand's marks (docs/brand): the Linux sizes in resources/icons,
// resources/icon.ico for Windows and resources/icon.icns for macOS. Below 64 px the flat small mark
// is used, as docs/brand/README.md says. Linux and Windows icons fill as much of the square as
// resources/icon.png does; a Mac icon sits on Apple's grid, 824 of 1024 px, so it matches its
// neighbours in the Dock. Run from app/: node scripts/icons.mjs
import { mkdirSync, readFileSync, writeFileSync } from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'
import sharp from 'sharp'

const app = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const brand = path.join(app, '..', 'docs', 'brand')
const MARK = readFileSync(path.join(brand, 'seamline-mark.svg'))
const SMALL = readFileSync(path.join(brand, 'seamline-mark-small.svg'))
const FILL = 944 / 1024
const MAC_FILL = 824 / 1024

/** The mark drawn `fill` of the way across a transparent square of `size` px, as PNG */
async function icon(size, fill, svg = size < 64 ? SMALL : MARK) {
  const inner = Math.max(1, Math.round(size * fill))
  const mark = await sharp(svg, { density: (72 * inner) / 512 }).resize(inner, inner).png().toBuffer()
  const pad = size - inner
  return sharp(mark)
    .extend({ top: Math.floor(pad / 2), bottom: Math.ceil(pad / 2), left: Math.floor(pad / 2), right: Math.ceil(pad / 2), background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png({ compressionLevel: 9 })
    .toBuffer()
}

// An .ico holds PNGs: a 6-byte header, a 16-byte entry for each, then the images
function ico(images) {
  const header = Buffer.alloc(6)
  header.writeUInt16LE(1, 2)
  header.writeUInt16LE(images.length, 4)
  let offset = 6 + 16 * images.length
  const entries = images.map(([size, png]) => {
    const e = Buffer.alloc(16)
    e.writeUInt8(size >= 256 ? 0 : size, 0)
    e.writeUInt8(size >= 256 ? 0 : size, 1)
    e.writeUInt16LE(1, 4)
    e.writeUInt16LE(32, 6)
    e.writeUInt32LE(png.length, 8)
    e.writeUInt32LE(offset, 12)
    offset += png.length
    return e
  })
  return Buffer.concat([header, ...entries, ...images.map(([, png]) => png)])
}

// An .icns holds PNGs by type code, each with its length; ic11 to ic14 are the Retina sizes
function icns(images) {
  const parts = images.map(([type, png]) => {
    const head = Buffer.alloc(8)
    head.write(type, 0, 'ascii')
    head.writeUInt32BE(png.length + 8, 4)
    return Buffer.concat([head, png])
  })
  const head = Buffer.alloc(8)
  head.write('icns', 0, 'ascii')
  head.writeUInt32BE(8 + parts.reduce((n, p) => n + p.length, 0), 4)
  return Buffer.concat([head, ...parts])
}

const resources = path.join(app, 'resources')
const linux = path.join(resources, 'icons')
mkdirSync(linux, { recursive: true })
for (const size of [16, 24, 32, 48, 64, 128, 256, 512, 1024]) {
  writeFileSync(path.join(linux, `${size}x${size}.png`), await icon(size, FILL))
}
const winSizes = [16, 24, 32, 48, 64, 128, 256]
writeFileSync(path.join(resources, 'icon.ico'), ico(await Promise.all(winSizes.map(async (s) => [s, await icon(s, FILL)]))))
// The 16 and 32 pt slots get the small mark at 1x and 2x, as a Mac draws them in lists and the Finder
const mac = [
  ['icp4', 16, 16], ['ic11', 32, 16], ['icp5', 32, 32], ['ic12', 64, 32],
  ['ic07', 128, 128], ['ic13', 256, 128], ['ic08', 256, 256], ['ic14', 512, 256], ['ic09', 512, 512], ['ic10', 1024, 512]
]
const macImages = []
for (const [type, size, points] of mac) {
  macImages.push([type, await icon(size, MAC_FILL, points <= 32 ? SMALL : MARK)])
}
writeFileSync(path.join(resources, 'icon.icns'), icns(macImages))
console.log('resources/icons, icon.ico and icon.icns written')
