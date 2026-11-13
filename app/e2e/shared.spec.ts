import { expect, test } from '@playwright/test'
import { readFileSync } from 'fs'
import path from 'path'
import { type LabelClass, mergeClasses, projectFile } from '../src/shared/classes'
import { readClasses, readYolo } from '../src/shared/formats'
import { hasArea, isLabelled, notClasses, readLabelMe, summarize } from '../src/shared/labels'
import { nameFold } from '../src/shared/names'
import { PixelCache } from '../src/renderer/src/masks'

// What both processes share, and the window's mask cache: plain functions, tested without the app

// LabelMe files and what Seamtag says of them, which the engine's tests read too
type Check = { name: string; text: string; error: string | null }
const CHECKS: Check[] = JSON.parse(readFileSync(path.join(__dirname, '../../tests/fixtures/seamtag/labelme_checks.json'), 'utf8'))
// Each language's JSON parser words its own errors after these words
const NOT_JSON = "It isn't valid JSON"

const c = (name: string, color = '#111111'): LabelClass => ({ name, color })
const merge = (base: LabelClass[], ours: LabelClass[], theirs: LabelClass[] | null) => mergeClasses(base, ours, theirs)

test('the class list keeps what another app added, recoloured or deleted, and what Seamline did', () => {
  const base = [c('cat'), c('dog')]
  // Added in another app, or here
  expect(merge(base, base, [...base, c('bird', '#222222')])).toEqual([c('cat'), c('dog'), c('bird', '#222222')])
  expect(merge(base, [...base, c('cow')], base)).toEqual([c('cat'), c('dog'), c('cow')])
  // Both: ours first, then theirs
  expect(merge(base, [...base, c('cow')], [...base, c('bird')])).toEqual([c('cat'), c('dog'), c('cow'), c('bird')])
  // Both added the same name: ours, with our colour
  expect(merge(base, [...base, c('owl', '#aaaaaa')], [...base, c('owl', '#bbbbbb')])).toEqual([c('cat'), c('dog'), c('owl', '#aaaaaa')])
  // Recoloured here, or there
  expect(merge(base, [c('cat', '#333333'), c('dog')], base)).toEqual([c('cat', '#333333'), c('dog')])
  expect(merge(base, base, [c('cat'), c('dog', '#444444')])).toEqual([c('cat'), c('dog', '#444444')])
  // Recoloured on both sides: ours
  expect(merge(base, [c('cat', '#333333'), c('dog')], [c('cat', '#444444'), c('dog')])).toEqual([c('cat', '#333333'), c('dog')])
  // Deleted here, or there
  expect(merge(base, [c('dog')], base)).toEqual([c('dog')])
  expect(merge(base, base, [c('dog')])).toEqual([c('dog')])
  // Deleted there but recoloured here: kept, as changed here
  expect(merge(base, [c('cat', '#333333'), c('dog')], [c('dog')])).toEqual([c('cat', '#333333'), c('dog')])
  // Renamed here: the old name doesn't come back
  expect(merge([c('x')], [c('z')], [c('x')])).toEqual([c('z')])
  // Ours keeps its order; theirs' new ones follow in theirs'
  expect(merge(base, [c('dog'), c('cat')], [c('fox'), ...base, c('emu')])).toEqual([c('dog'), c('cat'), c('fox'), c('emu')])
  // With no file to read, nothing is merged: ours as it is
  expect(merge(base, [c('dog')], null)).toEqual([c('dog')])
  // A name the file has twice, new to both lists, is added once, as Seamtag adds it
  expect(merge([c('c')], [c('e'), c('b'), c('b'), c('c', '#222222')], [c('dog'), c('a'), c('a', '#333333'), c('Dog')]))
    .toEqual([c('e'), c('b'), c('b'), c('c', '#222222'), c('dog'), c('a'), c('Dog')])
})

test('the project file keeps its other keys, and is written whole when it has gone', () => {
  const base = [c('cat')]
  const onDisk = { name: 'parts', settings: { tile: 512 }, classes: [c('cat'), c('bird')], by: 'Seamtag' }
  const out = projectFile(base, [c('cat'), c('cow')], onDisk, 'folder')
  expect(out).toEqual({ name: 'parts', settings: { tile: 512 }, classes: [c('cat'), c('cow'), c('bird')], by: 'Seamtag' })
  expect(Object.keys(out)).toEqual(['name', 'settings', 'classes', 'by'])
  // Missing, unreadable, or with no list of classes: ours, and the other keys kept where there are any
  expect(projectFile(base, [c('cow')], null, 'folder')).toEqual({ name: 'folder', classes: [c('cow')] })
  expect(projectFile(base, [c('cow')], [1, 2], 'folder')).toEqual({ name: 'folder', classes: [c('cow')] })
  expect(projectFile(base, [c('cow')], { name: 'parts', classes: 'cat' }, 'folder')).toEqual({ name: 'parts', classes: [c('cow')] })
  // Entries without a name aren't classes; a missing colour reads as none
  expect(projectFile([], [], { classes: [{ name: 'cat' }, 'dog', { color: '#fff' }] }, 'f').classes).toEqual([{ name: 'cat', color: '' }])
})

test('LabelMe files are read, or refused in Seamtag\'s words, as the engine reads and refuses them', () => {
  const outcome = (text: string) => {
    try {
      readLabelMe(text)
      return null
    } catch (e: any) {
      return e.message as string
    }
  }
  const differ = CHECKS.map(({ name, text, error }) => ({ name, error, got: outcome(text) }))
    .filter(({ error, got }) => (error === NOT_JSON ? !got?.startsWith(`${NOT_JSON} (`) : got !== error))
  expect(differ).toEqual([])
  expect(CHECKS.some((c) => c.error === null) && CHECKS.some((c) => c.error !== null)).toBe(true)
  // A file that reads comes back whole, for saving with every key it had
  expect(readLabelMe('{"shapes": [{"label": "cat", "points": [[1, 2]], "id": 7}], "by": "x"}')).toEqual({ shapes: [{ label: 'cat', points: [[1, 2]], id: 7 }], by: 'x' })
})

test('an image is labelled, as training counts it, once it has an accepted shape with an area and a class\'s label', () => {
  const shape = (shape_type: string | undefined, n: number, extra = {}) =>
    ({ label: 'x', shape_type, points: Array.from({ length: n }, (_, i) => [i, i * i]), flags: {}, ...extra })
  expect([
    shape('polygon', 3), shape(undefined, 3), shape('rectangle', 2), shape('circle', 2), shape('oriented_rectangle', 4),
    shape('mask', 2, { mask: 'iVBORw0KGgo=' })
  ].map(hasArea)).toEqual([true, true, true, true, true, true])
  expect([
    shape('line', 2), shape('linestrip', 5), shape('point', 1), shape('points', 4),
    shape('polygon', 2), shape('rectangle', 1), shape('circle', 1), shape('mask', 2), shape('mask', 2, { mask: '' })
  ].map(hasArea)).toEqual([false, false, false, false, false, false, false, false, false])
  // Lines and points count as shapes, and keep pre-labelling away, but don't make the image labelled
  expect(summarize([shape('line', 2), shape('point', 1)])).toMatchObject({ count: 2, ai: 0, areas: {} })
  // AI proposals don't count until accepted
  expect(summarize([shape('polygon', 3, { flags: { ai: true }, score: 0.4 }), shape('line', 2)])).toMatchObject({ count: 1, ai: 1, areas: {}, score: 0.4 })
  expect(summarize([shape('polygon', 3), shape('line', 2)])).toMatchObject({ count: 2, areas: { x: 1 }, classes: { x: 2 } })
  // Shapes with an area, by label: only a class's make the image labelled, as in the engine
  const { areas } = summarize([shape('polygon', 3), shape('rectangle', 2, { label: 'dent' }), shape('circle', 2, { label: 'dent' }), shape('line', 2, { label: 'lane' })])
  expect(areas).toEqual({ x: 1, dent: 2 })
  expect([isLabelled(areas, new Set(['x'])), isLabelled(areas, new Set(['dent'])), isLabelled(areas, new Set(['lane', 'cat'])), isLabelled({}, new Set(['x']))])
    .toEqual([true, true, false, false])
})

test('labels that aren\'t classes are counted for the Train page, most used first', () => {
  const images: { areas: Record<string, number> }[] = [
    { areas: { crack: 2 } }, { areas: { dent: 3 } }, { areas: { dent: 1, scratch: 1, crack: 1 } }, { areas: {} },
    // Unnamed shapes can't make a class, as the engine leaves them out of its list
    { areas: { '': 4 } }, { areas: { ant: 1 } }
  ]
  expect(notClasses(images, new Set(['crack']))).toEqual({ images: 3, labels: [['dent', 4], ['ant', 1], ['scratch', 1]] })
  expect(notClasses(images, new Set(['crack', 'dent', 'scratch', 'ant']))).toEqual({ images: 0, labels: [] })
})

test('classes.txt is read past a byte order mark, as the engine reads it', () => {
  expect(readClasses('\uFEFFdog\r\ncat\n')).toEqual(['dog', 'cat'])
  // The numbers stay: a blank first line is still class 0
  expect(readClasses('\uFEFF\ndog\n')).toEqual(['', 'dog'])
  // Just one, at the very start
  expect(readClasses('\uFEFF\uFEFFdog')).toEqual(['\uFEFFdog'])
  expect(readClasses('dog\n\uFEFFcat')).toEqual(['dog', '\uFEFFcat'])
  expect(readYolo('0 0.5 0.5 0.5 0.5\n', readClasses('\uFEFFdog\n'), 64, 48)[0].label).toBe('dog')
})

test('decoded masks stay within a budget of pixels, the least recently used going first', () => {
  const cache = new PixelCache<string>(100)
  cache.set('a', 'A', 40)
  cache.set('b', 'B', 40)
  expect(cache.get('a')).toBe('A')
  cache.set('c', 'C', 40)
  // b was used least recently
  expect([cache.get('a'), cache.get('b'), cache.get('c')]).toEqual(['A', undefined, 'C'])
  expect(cache.pixels).toBe(80)
  // One larger than the budget pushes out the rest, and stays until the next
  cache.set('huge', 'H', 500)
  expect(cache.values()).toEqual(['H'])
  cache.set('d', 'D', 10)
  expect(cache.values()).toEqual(['D'])
  expect(cache.pixels).toBe(10)
  // Set again, a key counts once
  cache.set('d', 'D2', 30)
  expect(cache.pixels).toBe(30)
})

test("masks on screen stay past the budget, so drawing one never pushes out another", () => {
  const cache = new PixelCache<string>(100)
  const shown = new Set(['a', 'b', 'c'])
  for (const key of shown) cache.set(key, key.toUpperCase(), 40, shown)
  expect(cache.values()).toEqual(['A', 'B', 'C'])
  expect(cache.pixels).toBe(120)
  // Once another image is on screen, the last one's go first
  cache.set('d', 'D', 40, new Set(['d']))
  expect(cache.values()).toEqual(['C', 'D'])
  expect(cache.pixels).toBe(80)
})

test('file names are told apart as the system tells them apart, as Seamtag looks them up', () => {
  // Hangul as Finder writes it, decomposed, and as most other apps write it, composed
  const nfd = '\u1100\u1161.JSON', nfc = '\uAC00.json'
  expect(nfd.normalize('NFC')).toBe('\uAC00.JSON')
  expect(nameFold('darwin')(nfd)).toBe(nameFold('darwin')(nfc))
  expect(nameFold('win32')(nfd)).not.toBe(nameFold('win32')(nfc))
  expect(nameFold('win32')('A.JSON')).toBe(nameFold('win32')('a.json'))
  expect(nameFold('linux')('A.json')).not.toBe(nameFold('linux')('a.json'))
  expect(nameFold('linux')(nfd)).toBe(nfd)
})
