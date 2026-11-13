/*!
 * Pascal VOC, YOLO and CreateML label files, read as LabelImg 1.8.6 reads them
 * (libs/pascal_voc_io.py, yolo_io.py and create_ml_io.py), so folders labelled in LabelImg or
 * Seamtag train here. Seamline never writes these formats: Seamtag does, keeping each one.
 * Ported from LabelImg (https://github.com/HumanSignal/labelImg) under its MIT licence:
 *
 * Copyright (c) <2015-Present> Tzutalin
 *
 * Copyright (C) 2013  MIT, Computer Science and Artificial Intelligence Laboratory. Bryan Russell, Antonio Torralba, William T. Freeman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

export type LabelFormat = 'labelme' | 'voc' | 'yolo' | 'createml'
export const FORMAT_NAMES: Record<LabelFormat, string> = { labelme: 'LabelMe JSON', voc: 'Pascal VOC', yolo: 'YOLO', createml: 'CreateML' }
/** YOLO's class names, one per line, in the folder of the .txt files */
export const YOLO_CLASSES = 'classes.txt'

/** A box as these formats keep it, in LabelMe's shape: two opposite corners */
export type Box = {
  label: string
  points: number[][]
  shape_type: 'rectangle'
  group_id: null
  description: string
  flags: Record<string, boolean>
}

function fail(message: string): never {
  throw new Error(message)
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)

const box = (label: string, x0: number, y0: number, x1: number, y1: number, difficult = false): Box => ({
  label, points: [[x0, y0], [x1, y1]], shape_type: 'rectangle', group_id: null, description: '',
  flags: difficult ? { difficult: true } : {}
})

// XML, as much as label files use: elements, attributes, text, entities, CDATA and comments

export type XmlElement = { name: string; attrs: Record<string, string>; children: XmlElement[]; text: string }

const ENTITIES: Record<string, string> = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'" }
const unescape = (s: string) => s.replace(/&(#x[0-9a-f]+|#[0-9]+|[a-z]+);/gi, (m, e: string) => {
  if (e[0] !== '#') return ENTITIES[e] ?? m
  const code = e[1] === 'x' || e[1] === 'X' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10)
  return code <= 0x10ffff ? String.fromCodePoint(code) : m
})

export function parseXml(text: string): XmlElement {
  const stack: XmlElement[] = []
  let root: XmlElement | null = null
  let i = text.charCodeAt(0) === 0xfeff ? 1 : 0
  const skipTo = (end: string, from: number, what: string) => {
    const at = text.indexOf(end, from)
    if (at < 0) fail(`It isn't valid XML: ${what} never ends`)
    return at + end.length
  }
  while (i < text.length) {
    const lt = text.indexOf('<', i)
    const chunk = text.slice(i, lt < 0 ? text.length : lt)
    if (stack.length) stack[stack.length - 1].text += unescape(chunk)
    else if (chunk.trim()) fail("It isn't valid XML: there's text outside the top element")
    if (lt < 0) break
    if (text.startsWith('<!--', lt)) { i = skipTo('-->', lt + 4, 'a comment'); continue }
    if (text.startsWith('<![CDATA[', lt)) {
      const end = skipTo(']]>', lt + 9, 'a CDATA section')
      if (stack.length) stack[stack.length - 1].text += text.slice(lt + 9, end - 3)
      i = end
      continue
    }
    if (text.startsWith('<?', lt)) { i = skipTo('?>', lt + 2, 'a declaration'); continue }
    if (text.startsWith('<!', lt)) {
      // A DOCTYPE, with any internal subset in brackets
      let j = lt + 2, depth = 0
      for (; j < text.length; j++) {
        if (text[j] === '[') depth++
        else if (text[j] === ']') depth--
        else if (text[j] === '>' && depth <= 0) break
      }
      i = j + 1
      continue
    }
    // The tag ends at the first ">" outside quotes
    let gt = lt + 1, quote = ''
    for (; gt < text.length; gt++) {
      const c = text[gt]
      if (quote) { if (c === quote) quote = '' }
      else if (c === '"' || c === "'") quote = c
      else if (c === '>') break
    }
    if (gt >= text.length) fail("It isn't valid XML: a tag never ends")
    const body = text.slice(lt + 1, gt)
    i = gt + 1
    if (body.startsWith('/')) {
      const name = body.slice(1).trim()
      const open = stack.pop()
      if (!open || open.name !== name) fail(`It isn't valid XML: </${name}> closes nothing that's open`)
      continue
    }
    const closed = body.endsWith('/')
    const inner = closed ? body.slice(0, -1) : body
    const name = /^[^\s/>]+/.exec(inner)?.[0]
    if (!name) fail("It isn't valid XML: a tag has no name")
    const element: XmlElement = { name, attrs: {}, children: [], text: '' }
    for (const m of inner.slice(name.length).matchAll(/([^\s=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g)) {
      element.attrs[m[1]] = unescape(m[2] ?? m[3])
    }
    if (stack.length) stack[stack.length - 1].children.push(element)
    else if (root) fail("It isn't valid XML: it has two top elements")
    else root = element
    if (!closed) stack.push(element)
  }
  if (stack.length) fail(`It isn't valid XML: <${stack[stack.length - 1].name}> is never closed`)
  if (!root) fail("It isn't valid XML: it has no elements")
  return root
}

const child = (e: XmlElement, name: string) => e.children.find((c) => c.name === name)

/** LabelImg's PascalVocReader, which keeps whole pixels; boxes with fractions keep them here */
export function readVoc(text: string): { shapes: Box[]; verified: boolean } {
  const root = parseXml(text)
  if (root.name !== 'annotation') fail(`It isn't a Pascal VOC file: the top element is <${root.name}>, not <annotation>`)
  const shapes = root.children.filter((c) => c.name === 'object').map((o, k) => {
    const where = `Object ${k + 1}`
    const name = child(o, 'name')
    if (!name) fail(`${where} has no <name>`)
    const bndbox = child(o, 'bndbox')
    if (!bndbox) fail(`${where} has no <bndbox>`)
    const coord = (tag: string) => {
      const v = child(bndbox, tag)?.text.trim() ?? ''
      if (!v || !Number.isFinite(Number(v))) fail(`${where} needs a number in <${tag}>`)
      return Number(v)
    }
    const difficult = /^\s*(1|true|yes)\s*$/i.test(child(o, 'difficult')?.text ?? '')
    return box(name.text.trim(), coord('xmin'), coord('ymin'), coord('xmax'), coord('ymax'), difficult)
  })
  return { shapes, verified: root.attrs.verified === 'yes' }
}

/**
 * classes.txt as LabelImg reads it. Blank lines in the middle stay, so the numbers after them do too.
 * A byte order mark, which some editors write first, is left out of the first name, as the engine
 * leaves it out, where LabelImg and Seamtag 3.2.0 keep it: a dog is still a dog.
 */
export function readClasses(text: string): string[] {
  const names = text.replace(/^[\r\n]+|[\r\n]+$/g, '').split(/\r?\n/)
  names[0] = names[0].replace(/^﻿/, '')
  return names
}

/** Python's round(): halves go to the even number */
export function pyRound(x: number) {
  const f = Math.floor(x)
  return x - f === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(x)
}

/** LabelImg's YoloReader: each line is a class and a box's centre and size, as shares of the image */
export function readYolo(text: string, classes: string[], width: number, height: number): Box[] {
  const shapes: Box[] = []
  text.split(/\r?\n/).forEach((line, k) => {
    if (!line.trim()) return
    const parts = line.trim().split(/\s+/)
    if (parts.length !== 5) fail(`Line ${k + 1} needs a class and four numbers, as YOLO's boxes have`)
    const [c, xc, yc, w, h] = parts.map(Number)
    if (![c, xc, yc, w, h].every(Number.isFinite)) fail(`Line ${k + 1} has something that isn't a number`)
    if (!Number.isInteger(c) || c < 0 || c >= classes.length) fail(`Line ${k + 1}: class ${parts[0]} isn't in ${YOLO_CLASSES}`)
    shapes.push(box(classes[c],
      pyRound(width * Math.max(xc - w / 2, 0)), pyRound(height * Math.max(yc - h / 2, 0)),
      pyRound(width * Math.min(xc + w / 2, 1)), pyRound(height * Math.min(yc + h / 2, 1))))
  })
  return shapes
}

/** LabelImg's CreateMLReader: the entries for this image. Their boxes keep any fractions. */
export function readCreateML(text: string, image: string): { shapes: Box[]; verified: boolean; found: boolean } {
  let list: unknown
  try {
    list = JSON.parse(text)
  } catch (e) {
    fail(`It isn't valid JSON (${(e as Error).message})`)
  }
  if (!Array.isArray(list)) fail("It isn't a CreateML file: the top level must be a list")
  const shapes: Box[] = []
  let verified = false, found = false
  list.forEach((entry, k) => {
    if (!isObject(entry) || typeof entry.image !== 'string' || !Array.isArray(entry.annotations)) {
      fail(`Entry ${k + 1} needs an image name and a list of annotations`)
    }
    if (entry.image !== image) return
    found = true
    if (entry.verified === true) verified = true
    entry.annotations.forEach((a, n) => {
      const c = isObject(a) ? a.coordinates : null
      if (!isObject(a) || typeof a.label !== 'string' || !isObject(c) || !['x', 'y', 'width', 'height'].every((v) => typeof c[v] === 'number')) {
        fail(`Annotation ${n + 1} of ${image} needs a label and coordinates x, y, width and height`)
      }
      const { x, y, width, height } = c as Record<string, number>
      shapes.push(box(a.label, x - width / 2, y - height / 2, x + width / 2, y + height / 2))
    })
  })
  return { shapes, verified, found }
}

/** Is this JSON text a CreateML list rather than a LabelMe file? Decided by its first character. */
export const looksLikeCreateML = (text: string) => /^﻿?\s*\[/.test(text)
