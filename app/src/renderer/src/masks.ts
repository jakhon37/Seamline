/** Kept until their pixels pass the budget, then the least recently used go first, apart from those to keep */
export class PixelCache<T> {
  private entries = new Map<string, { value: T; pixels: number }>()
  private total = 0
  constructor(private budget: number) {}

  get(key: string) {
    const entry = this.entries.get(key)
    if (!entry) return undefined
    this.entries.delete(key)
    this.entries.set(key, entry)
    return entry.value
  }

  set(key: string, value: T, pixels: number, keep: ReadonlySet<string> = new Set()) {
    this.delete(key)
    this.entries.set(key, { value, pixels })
    this.total += pixels
    for (const [old, entry] of this.entries) {
      if (this.total <= this.budget || old === key) break
      if (keep.has(old)) continue
      this.entries.delete(old)
      this.total -= entry.pixels
    }
  }

  delete(key: string) {
    const entry = this.entries.get(key)
    if (!entry) return
    this.entries.delete(key)
    this.total -= entry.pixels
  }

  values() {
    return [...this.entries.values()].map((e) => e.value)
  }

  get pixels() {
    return this.total
  }
}

// A labelme mask is a PNG of 0s and 1s: decoded once, then tinted per class colour. Masks big and
// many, a decoded one costs a byte a pixel and a tinted copy four, so decoded ones are kept up to
// a budget, and tinted copies only for the image on screen.
export type Decoded = { width: number; height: number; bits: Uint8Array; tinted: Map<string, HTMLCanvasElement> }
const MASK_PIXELS = 50_000_000
const masks = new PixelCache<Decoded>(MASK_PIXELS)
// Being decoded, or not a PNG the window can read: asked for once
const pending = new Set<string>()
// The masks asked for since another image came on screen. They stay past the budget: one image's masks
// can pass it, and each mask decoded would push out another the next frame decodes again, without end.
const shown = new Set<string>()

/** The mask's pixels once decoded; until then null, and `onReady` is called when they are */
export function decodeMask(mask: string, onReady: () => void) {
  shown.add(mask)
  const known = masks.get(mask)
  if (known || pending.has(mask)) return known ?? null
  pending.add(mask)
  const img = new Image()
  img.onload = () => {
    const c = document.createElement('canvas')
    c.width = img.naturalWidth
    c.height = img.naturalHeight
    const ctx = c.getContext('2d', { willReadFrequently: true })!
    ctx.drawImage(img, 0, 0)
    const px = ctx.getImageData(0, 0, c.width, c.height).data
    const bits = new Uint8Array(c.width * c.height)
    for (let i = 0; i < bits.length; i++) bits[i] = px[i * 4] || px[i * 4 + 1] || px[i * 4 + 2] ? 1 : 0
    pending.delete(mask)
    masks.set(mask, { width: c.width, height: c.height, bits, tinted: new Map() }, bits.length, shown)
    onReady()
  }
  img.src = `data:image/png;base64,${mask}`
  return null
}

/** The decoded mask, if it is: for picking a shape by its pixels */
export const decodedMask = (mask: string) => masks.get(mask)

export function tintedMask(m: Decoded, color: string) {
  let c = m.tinted.get(color)
  if (c) return c
  c = document.createElement('canvas')
  c.width = m.width
  c.height = m.height
  const ctx = c.getContext('2d')!
  const px = ctx.createImageData(m.width, m.height)
  const v = parseInt(color.slice(1), 16)
  for (let i = 0; i < m.bits.length; i++) {
    px.data[i * 4] = (v >> 16) & 255
    px.data[i * 4 + 1] = (v >> 8) & 255
    px.data[i * 4 + 2] = v & 255
    px.data[i * 4 + 3] = m.bits[i] ? 255 : 0
  }
  ctx.putImageData(px, 0, 0)
  m.tinted.set(color, c)
  return c
}

/** Another image is on screen: its masks are tinted again as they're drawn, and the last one's may go */
export function dropTints() {
  for (const m of masks.values()) m.tinted.clear()
  shown.clear()
}
