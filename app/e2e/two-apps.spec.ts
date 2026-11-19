import { ElectronApplication, expect, test } from '@playwright/test'
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, utimesSync, writeFileSync } from 'fs'
import path from 'path'
import { classes, drag, fakeEngine, file, finish, folder, image, launch, onImage, open, polygon, read, saved, shot } from './seamtag_folder'

// Seamline and Seamtag on one folder, each saving while the other has it open
const SIZE = [400, 300]
const square = (label: string, x0: number, y0: number, x1: number, y1: number, ai = false) => ({
  label, points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], group_id: null, description: '', shape_type: 'polygon',
  flags: ai ? { ai: true } : {}, ...(ai ? { score: 0.5 } : {})
})
const labels = (name: string, shapes: unknown[], extra = {}) => JSON.stringify({
  version: '5.5.0', flags: {}, shapes, imagePath: `${name}.png`, imageData: null, imageHeight: SIZE[1], imageWidth: SIZE[0], ...extra
}, null, 2)
// As Seamtag saves: a whole new file put in place of the old one
function seamtagSaves(dir: string, name: string, text: string) {
  const tmp = path.join(dir, `${name}.json.seamtag.tmp`)
  writeFileSync(tmp, text)
  renameSync(tmp, path.join(dir, `${name}.json`))
}
const focus = (page: import('@playwright/test').Page) => page.evaluate(() => window.dispatchEvent(new Event('focus')))
// On a coarse file clock, such as FAT's 2 s, saves within one tick keep the file's time
const TICK = new Date('2026-01-01T00:00:00Z')
const pin = (file: string) => utimesSync(file, TICK, TICK)
// labels:load answers these images this many ms late, as for big files on a slow disk, and notes what
// it was asked for and what it answered
async function slowLoads(app: ElectronApplication, late: Record<string, number>) {
  await app.evaluate(({ ipcMain }, delays) => {
    const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: any[]) => Promise<unknown>>
    const g = globalThis as any
    g.load ??= handlers.get('labels:load')
    g.asked = []
    g.answered = []
    handlers.set('labels:load', async (e, base, name: string) => {
      g.asked.push(name)
      const result = await g.load(e, base, name)
      await new Promise((done) => setTimeout(done, delays[name] ?? 0))
      g.answered.push(name)
      return result
    })
  }, late)
}
const loads = (app: ElectronApplication) => app.evaluate(() => ({ asked: (globalThis as any).asked, answered: (globalThis as any).answered }))
// labels:save waits while held, so what the window shows before a save is done can be looked at
async function holdSaves(app: ElectronApplication, held: boolean) {
  await app.evaluate(({ ipcMain }, on) => {
    const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: any[]) => Promise<unknown>>
    const g = globalThis as any
    g.save ??= handlers.get('labels:save')
    g.held = on
    handlers.set('labels:save', async (...args) => {
      while (g.held) await new Promise((done) => setTimeout(done, 20))
      return g.save(...args)
    })
  }, held)
}

test('a save never writes over labels another app saved meanwhile: the window shows theirs', async () => {
  const dir = folder('guard')
  classes(dir, ['cat', 'dog'])
  await image(dir, 'g.png', SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'g.json'), labels('g', [square('cat', 50, 50, 150, 150)]))
  const { app, page, errors } = await launch(dir)
  await open(page, 'g.png', SIZE)
  await expect(page.locator('.shape')).toHaveCount(1)

  const theirs = labels('g', [
    square('cat', 60, 60, 160, 160),
    { label: 'dog', points: [[300, 200], [330, 200]], group_id: null, description: '', shape_type: 'circle', flags: {}, mask: null }
  ], { seamtag: 'saved here' })
  seamtagSaves(dir, 'g', theirs)
  // An edit on what the window still shows
  await page.keyboard.press('v')
  await drag(page, SIZE, [100, 100], [200, 150])
  await expect(page.locator('.notice')).toHaveText("These labels changed in another app, so they were reloaded. Your last change wasn't saved.")
  await expect(page.locator('.shape')).toHaveCount(2)
  await expect(page.locator('.shape').nth(1)).toContainText('circle')
  await shot(page, 'reloaded')
  await saved(page)
  await page.waitForTimeout(600)
  expect(readFileSync(path.join(dir, 'g.json'), 'utf8')).toBe(theirs)
  // Undo can't bring back what was there before
  await page.keyboard.press('Control+z')
  await page.waitForTimeout(600)
  expect(readFileSync(path.join(dir, 'g.json'), 'utf8')).toBe(theirs)

  // From here the window is up to date: the next edit is saved, on their labels
  await drag(page, SIZE, [110, 110], [130, 110])
  await saved(page)
  await expect.poll(() => read(dir, 'g.json').shapes[0].points[0].map(Math.round)).toEqual([80, 60])
  const after = read(dir, 'g.json')
  expect(after.shapes[1]).toEqual(JSON.parse(theirs).shapes[1])
  expect(after.seamtag).toBe('saved here')
  await finish(app, errors)
})

test("a save never writes over another app's change that kept the file's size and time", async () => {
  const dir = folder('same-size')
  classes(dir, ['cat', 'dog'])
  await image(dir, 's.png', SIZE[0], SIZE[1])
  const ours = labels('s', [square('cat', 50, 50, 150, 150)])
  writeFileSync(path.join(dir, 's.json'), ours)
  pin(path.join(dir, 's.json'))
  const { app, page, errors } = await launch(dir)
  await open(page, 's.png', SIZE)
  // Within the same tick, another app makes the cat a dog, in as many bytes
  const theirs = ours.replace('"cat"', '"dog"')
  expect(theirs.length).toBe(ours.length)
  seamtagSaves(dir, 's', theirs)
  pin(path.join(dir, 's.json'))
  await page.keyboard.press('v')
  await drag(page, SIZE, [100, 100], [200, 150])
  await expect(page.locator('.notice')).toHaveText("These labels changed in another app, so they were reloaded. Your last change wasn't saved.")
  await expect(page.locator('.shape .name')).toHaveText(['dog'])
  await saved(page)
  await page.waitForTimeout(600)
  expect(readFileSync(path.join(dir, 's.json'), 'utf8')).toBe(theirs)
  // Taken in, they are what the next save goes by
  await drag(page, SIZE, [100, 100], [120, 100])
  await saved(page)
  await expect.poll(() => read(dir, 's.json').shapes[0].points[0].map(Math.round)).toEqual([70, 50])
  expect(read(dir, 's.json').shapes[0].label).toBe('dog')
  await finish(app, errors)
})

test("labels another app changed while the window passed them by can't be undone back", async () => {
  const dir = folder('passed-by')
  classes(dir, ['cat', 'dog', 'bird'])
  for (const name of ['a', 'b', 'c']) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'b.json'), labels('b', [square('cat', 50, 50, 150, 150), square('dog', 250, 50, 350, 150)]))
  const { app, page, errors } = await launch(dir)
  // Undo on b would bring back the dog
  await open(page, 'b.png', SIZE)
  await page.locator('.shape').nth(1).click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect.poll(() => read(dir, 'b.json').shapes.length).toBe(1)
  await open(page, 'a.png', SIZE)
  const theirs = labels('b', [square('cat', 50, 50, 150, 150), square('bird', 10, 200, 60, 250)], { by: 'Seamtag' })
  seamtagSaves(dir, 'b', theirs)
  // Passing b on the way to c, before b's labels arrive: the window doesn't take them
  await slowLoads(app, { 'b.png': 800 })
  await page.keyboard.press('d')
  await expect.poll(async () => (await loads(app)).asked).toEqual(['b.png'])
  await page.keyboard.press('d')
  await expect.poll(async () => [...(await loads(app)).answered].sort()).toEqual(['b.png', 'c.png'])
  await expect(page.locator('.status')).toContainText('c.png')
  await slowLoads(app, {})
  await page.keyboard.press('a')
  await expect(page.locator('.shape .name')).toHaveText(['cat', 'bird'])
  await page.keyboard.press('Control+z')
  await page.waitForTimeout(700)
  expect(readFileSync(path.join(dir, 'b.json'), 'utf8')).toBe(theirs)
  await expect(page.locator('.shape .name')).toHaveText(['cat', 'bird'])
  // What the window took then is what its saves go by
  await page.locator('.shape').nth(1).click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect.poll(() => read(dir, 'b.json').shapes.map((s: any) => s.label)).toEqual(['cat'])
  expect(read(dir, 'b.json').by).toBe('Seamtag')
  await finish(app, errors)
})

test("labels reloaded after another app changed them, once the window moved on, can't be undone back", async () => {
  const dir = folder('moved-on')
  classes(dir, ['cat', 'dog', 'bird'])
  for (const name of ['a', 'b']) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'a.json'), labels('a', [square('cat', 50, 50, 150, 150), square('dog', 250, 50, 350, 150)]))
  const { app, page, errors } = await launch(dir)
  await open(page, 'a.png', SIZE)
  await page.locator('.shape').nth(1).click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect.poll(() => read(dir, 'a.json').shapes.length).toBe(1)
  const theirs = labels('a', [square('cat', 50, 50, 150, 150), square('bird', 10, 200, 60, 250)], { by: 'Seamtag' })
  seamtagSaves(dir, 'a', theirs)
  // On the way to b, the window comes back into focus: a is reloaded, but b's labels, asked for
  // first, arrive first, as a big folder's refresh gives them time to
  await slowLoads(app, { 'a.png': 1500, 'b.png': 1500 })
  await page.keyboard.press('d')
  await expect.poll(async () => (await loads(app)).asked).toEqual(['b.png'])
  await focus(page)
  await expect.poll(async () => (await loads(app)).asked).toEqual(['b.png', 'a.png'])
  await expect.poll(async () => (await loads(app)).answered, { timeout: 10000 }).toEqual(['b.png', 'a.png'])
  await slowLoads(app, {})
  await page.keyboard.press('a')
  await expect(page.locator('.shape .name')).toHaveText(['cat', 'bird'])
  await page.keyboard.press('Control+z')
  await page.waitForTimeout(700)
  expect(readFileSync(path.join(dir, 'a.json'), 'utf8')).toBe(theirs)
  await expect(page.locator('.notice')).toHaveCount(0)
  await finish(app, errors)
})

test('shapes that differ only in a long id keep their own ids as other shapes change', async () => {
  const dir = folder('long-ids')
  classes(dir, ['cat', 'dog'])
  await image(dir, 'i.png', SIZE[0], SIZE[1])
  // Ids past 2^53, as some tools write them: as numbers, all three are one
  const shape = (label: string, x: number, id: string) => ({ ...square(label, x, 50, x + 100, 150), annotation_id: `<${id}>` })
  const text = labels('i', [shape('cat', 50, '1234567890123456789'), shape('cat', 50, '1234567890123456790'), shape('dog', 250, '1234567890123456800')])
  writeFileSync(path.join(dir, 'i.json'), text.replace(/"<(\d+)>"/g, '$1'))
  const ids = () => [...readFileSync(path.join(dir, 'i.json'), 'utf8').matchAll(/"annotation_id": (\d+)/g)].map((m) => m[1])
  const { app, page, errors } = await launch(dir)
  await open(page, 'i.png', SIZE)
  await page.locator('.shape').nth(2).click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect.poll(ids).toEqual(['1234567890123456789', '1234567890123456790'])
  // The first of the two cats goes, and the second keeps its own id
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect.poll(ids).toEqual(['1234567890123456790'])
  await finish(app, errors)
})

test("a label file that couldn't be read is read again once it can be", async () => {
  test.skip(process.platform === 'win32' || process.getuid?.() === 0, 'Here every file can be read')
  const dir = folder('unreadable')
  classes(dir, ['cat'])
  for (const name of ['a', 'b']) {
    await image(dir, `${name}.png`, SIZE[0], SIZE[1])
    writeFileSync(path.join(dir, `${name}.json`), labels(name, [square('cat', 10, 10, 90, 90)]))
  }
  // As a file whose permissions are wrong, or that another program holds
  chmodSync(path.join(dir, 'b.json'), 0o000)
  const { app, page, errors } = await launch(dir)
  await expect(file(page, 'b.png').locator('.dot')).toHaveClass(/broken/)
  await expect(page.locator('.topbar')).toContainText('1/2 labelled')
  // Put right, with its size and time as they were
  chmodSync(path.join(dir, 'b.json'), 0o644)
  await focus(page)
  await expect(file(page, 'b.png').locator('.dot')).toHaveClass(/done/)
  await expect(page.locator('.topbar')).toContainText('2/2 labelled')
  await finish(app, errors)
})

test('looking at the folder again picks up new images and labels, and keeps the chosen image', async () => {
  const dir = folder('refresh')
  classes(dir, ['cat', 'dog'])
  for (const name of ['a', 'b', 'c']) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'a.json'), labels('a', [square('cat', 10, 10, 90, 90)]))
  const { app, page, errors } = await launch(dir)
  await open(page, 'b.png', SIZE)
  await expect(page.locator('.topbar')).toContainText('1/3 labelled')

  // Seamtag adds an image before this one and one in a subfolder, labels another, and labels this one
  await image(dir, '0.png', SIZE[0], SIZE[1])
  await image(dir, 'sub/n.png', SIZE[0], SIZE[1])
  seamtagSaves(dir, 'c', labels('c', [square('bird', 20, 20, 60, 60)]))
  seamtagSaves(dir, 'b', labels('b', [square('dog', 200, 100, 300, 200), square('cat', 20, 20, 60, 60)]))
  // Coming back to Seamline
  await focus(page)
  await expect(page.locator('.file')).toHaveCount(5)
  await expect(page.locator('.file.on')).toHaveAttribute('title', 'b.png')
  await expect(file(page, 'sub/n.png')).toBeVisible()
  // c's bird isn't a class, so c doesn't count until it's added, as in training
  await expect(page.locator('.topbar')).toContainText('2/5 labelled')
  // Nothing was being changed here, so this image's new labels show
  await expect(page.locator('.shape')).toHaveCount(2)
  await expect(page.locator('.notice')).toHaveCount(0)
  // A label no class has stays out of the class list until someone adds it
  await expect(page.locator('.class .name')).toHaveText(['cat', 'dog'])

  // Back from the Train page, the folder is looked at again
  await page.keyboard.press('Control+2')
  await image(dir, 'd.png', SIZE[0], SIZE[1])
  await page.keyboard.press('Control+1')
  await expect(page.locator('.file')).toHaveCount(6)
  await expect(page.locator('.file.on')).toHaveAttribute('title', 'b.png')

  // An image another app removed leaves the list, and the one after it is chosen
  await open(page, 'c.png', SIZE)
  rmSync(path.join(dir, 'c.png'))
  rmSync(path.join(dir, 'c.json'))
  await focus(page)
  await expect(page.locator('.file')).toHaveCount(5)
  await expect(page.locator('.file.on')).toHaveAttribute('title', 'd.png')
  await expect(page.locator('.status')).toContainText('d.png')
  await expect(page.locator('.topbar')).toContainText('2/5 labelled')
  await finish(app, errors)
})

// Called as: python engine/studio.py predict PROJECT --run RUN, with the image names as JSON on stdin
const PREDICT = `
let input = ''
process.stdin.on('data', (d) => (input += d))
process.stdin.on('end', () => {
  console.log(JSON.stringify({ device: 'cpu' }))
  for (const name of JSON.parse(input)) {
    const shape = { label: 'cat', points: [[100, 100], [200, 100], [200, 200], [100, 200]], group_id: null, description: '', shape_type: 'polygon', flags: { ai: true }, score: 0.7 }
    console.log(JSON.stringify({ name, shapes: [shape], size: [400, 300] }))
  }
})
`

test("pre-labelling leaves alone LabelImg's formats, and labels changed since Seamline read them", async () => {
  const dir = folder('prelabel')
  classes(dir, ['cat'])
  for (const name of ['u', 'v', 'w', 'x', 'z', 'sub/y']) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'v.xml'), '<annotation><size><width>400</width><height>300</height></size></annotation>\n')
  writeFileSync(path.join(dir, 'w.json'), '[]')
  writeFileSync(path.join(dir, 'x.json'), labels('x', [square('cat', 10, 10, 50, 50, true)]))
  const z = labels('z', [square('cat', 10, 10, 50, 50, true)])
  writeFileSync(path.join(dir, 'z.json'), z)
  pin(path.join(dir, 'z.json'))
  const engine = fakeEngine(folder('prelabel-engine'), PREDICT)
  const { app, page, errors } = await launch(dir, { MYSEG_PYTHON: engine })
  // An empty Pascal VOC file and an empty CreateML list are labels too, kept in Seamtag's formats
  await expect(page.getByRole('button', { name: 'Pre-label 4 images' })).toBeVisible()
  // Another app saves its own proposals for x after Seamline read the folder, and for z within the
  // same tick, in as many bytes
  const theirs = labels('x', [square('cat', 20, 20, 60, 60, true)], { by: 'Seamtag' })
  seamtagSaves(dir, 'x', theirs)
  const zTheirs = z.replace('"cat"', '"cow"')
  seamtagSaves(dir, 'z', zTheirs)
  pin(path.join(dir, 'z.json'))
  // Every image asked for, to see the main process keep to it whatever the window sends
  const result = await page.evaluate(() => window.api.prelabelAll('fake', ['u.png', 'v.png', 'w.png', 'x.png', 'z.png', 'sub/y.png']))
  expect(result).toMatchObject({ done: 6, total: 6, failed: [] })
  expect(read(dir, 'u.json').shapes[0].flags.ai).toBe(true)
  expect(read(dir, 'sub/y.json').imagePath).toBe('y.png')
  expect(readFileSync(path.join(dir, 'v.xml'), 'utf8')).toContain('<annotation>')
  expect(readFileSync(path.join(dir, 'w.json'), 'utf8')).toBe('[]')
  expect(readFileSync(path.join(dir, 'x.json'), 'utf8')).toBe(theirs)
  expect(readFileSync(path.join(dir, 'z.json'), 'utf8')).toBe(zTheirs)
  await finish(app, errors)
})

// As PREDICT, from stdin for a batch or from --images for one image, but it finds nothing on nothing.png
// and stale.png, notes the names each batch asked for, and holds later.png's result back until `hold` is made
const predictHeld = (hold: string, asked: string) => `
const fs = require('fs')
const shape = { label: 'cat', points: [[100, 100], [200, 100], [200, 200], [100, 200]], group_id: null, description: '', shape_type: 'polygon', flags: { ai: true }, score: 0.7 }
async function answer(names) {
  console.log(JSON.stringify({ device: 'cpu' }))
  for (const name of names) {
    while (name === 'later.png' && !fs.existsSync(${JSON.stringify(hold)})) await new Promise((done) => setTimeout(done, 50))
    console.log(JSON.stringify({ name, shapes: ['nothing.png', 'stale.png'].includes(name) ? [] : [shape], size: [400, 300] }))
  }
}
const at = process.argv.indexOf('--images')
if (at >= 0) answer(process.argv.slice(at + 1))
else {
  let input = ''
  process.stdin.on('data', (d) => (input += d))
  process.stdin.on('end', () => {
    fs.appendFileSync(${JSON.stringify(asked)}, input + '\\n')
    answer(JSON.parse(input))
  })
}
`
const emptyLabels = (name: string) => labels(name, [])
const proposals = (name: string) => labels(name, [square('cat', 10, 10, 50, 50, true)])
// A folder with a trained run, and an engine that answers as predictHeld does
async function prelabelFolder(at: string, names: string[]) {
  const dir = folder(at)
  classes(dir, ['cat'])
  for (const name of names) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  const run = path.join(dir, '.myseg/runs/20260101_000000')
  mkdirSync(run, { recursive: true })
  writeFileSync(path.join(run, 'events.jsonl'), JSON.stringify({ type: 'epoch', epoch: 0, epochs: 1, train_loss: 0.5, val_loss: 0.5, iou: 0.5, f1: 0.5 }) + '\n')
  const tools = folder(`${at}-engine`)
  const hold = path.join(tools, 'go'), asked = path.join(tools, 'asked')
  return { dir, hold, asked, engine: fakeEngine(tools, predictHeld(hold, asked)) }
}

test('pre-labelling leaves alone a label file someone emptied, as Seamtag does, even once it has started', async () => {
  const { dir, hold, asked, engine } = await prelabelFolder('prelabel-emptied', ['emptied', 'later', 'new', 'nothing', 'proposed', 'rejected', 'stale'])
  // Every shape rejected, in either app: a LabelMe file with no shapes
  writeFileSync(path.join(dir, 'emptied.json'), emptyLabels('emptied'))
  for (const name of ['later', 'proposed', 'stale']) writeFileSync(path.join(dir, `${name}.json`), proposals(name))
  const { app, page, errors } = await launch(dir, { MYSEG_PYTHON: engine })
  const batch = (n: number) => page.getByRole('button', { name: `Pre-label ${n} images` })
  await expect(batch(6)).toBeVisible()

  // Emptied here too: proposals Seamtag saved after Seamline read the folder, deleted. The count follows at once.
  seamtagSaves(dir, 'rejected', proposals('rejected'))
  await open(page, 'rejected.png', SIZE)
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await expect(batch(5)).toBeVisible()
  await saved(page)
  await expect.poll(() => read(dir, 'rejected.json').shapes).toEqual([])
  const rejected = readFileSync(path.join(dir, 'rejected.json'), 'utf8')

  await batch(5).click()
  await expect.poll(() => existsSync(asked)).toBe(true)
  expect(JSON.parse(readFileSync(asked, 'utf8'))).toEqual(['later.png', 'new.png', 'nothing.png', 'proposed.png', 'stale.png'])
  // Seamtag empties later.png's file while its result is on the way, and Seamline reads the folder again
  seamtagSaves(dir, 'later', emptyLabels('later'))
  await focus(page)
  await expect(file(page, 'later.png').locator('.dot')).not.toHaveClass(/ai/)
  writeFileSync(hold, '')
  await expect(page.locator('.notice')).toHaveText('Pre-labelled 5 of 5 images.')
  expect(read(dir, 'new.json').shapes).toMatchObject([{ label: 'cat', flags: { ai: true }, score: 0.7 }])
  expect(read(dir, 'proposed.json').shapes).toMatchObject([{ points: [[100, 100], [200, 100], [200, 200], [100, 200]], flags: { ai: true } }])
  // Where the model finds nothing, nothing is written: no file, and earlier proposals stay
  expect(existsSync(path.join(dir, 'nothing.json'))).toBe(false)
  expect(readFileSync(path.join(dir, 'stale.json'), 'utf8')).toBe(proposals('stale'))
  for (const name of ['emptied', 'later']) expect(readFileSync(path.join(dir, `${name}.json`), 'utf8')).toBe(emptyLabels(name))
  expect(readFileSync(path.join(dir, 'rejected.json'), 'utf8')).toBe(rejected)
  // Left to pre-label: no label file, or AI shapes alone
  await expect(batch(4)).toBeVisible()

  // The main process keeps to it whatever the window sends
  const result = await page.evaluate(() => window.api.prelabelAll('fake', ['emptied.png', 'rejected.png', 'later.png']))
  expect(result).toMatchObject({ done: 3, total: 3, failed: [] })
  for (const name of ['emptied', 'later']) expect(readFileSync(path.join(dir, `${name}.json`), 'utf8')).toBe(emptyLabels(name))
  expect(readFileSync(path.join(dir, 'rejected.json'), 'utf8')).toBe(rejected)
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test('Pre-label N images comes back once the batch ends, even if its last progress comes after its answer', async () => {
  const { dir, engine } = await prelabelFolder('prelabel-late', ['one', 'two'])
  const { app, page, errors } = await launch(dir, { MYSEG_PYTHON: engine })
  const batch = page.getByRole('button', { name: 'Pre-label 2 images' })
  await batch.click()
  await expect(page.locator('.notice')).toHaveText('Pre-labelled 2 of 2 images.')
  await expect(batch).toBeVisible()
  // The main process's messages and its answers don't keep their order: progress can still come
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].webContents.send('prelabel:progress', { name: 'two.png', shapes: null, done: 2, total: 2 }))
  await page.waitForTimeout(300)
  await expect(batch).toBeVisible()
  await expect(page.getByRole('button', { name: /Stop ·/ })).toHaveCount(0)
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test('Pre-label on one image the model finds nothing on leaves its labels as they were', async () => {
  const { dir, engine } = await prelabelFolder('prelabel-one-nothing', ['nothing', 'stale'])
  writeFileSync(path.join(dir, 'stale.json'), proposals('stale'))
  const { app, page, errors } = await launch(dir, { MYSEG_PYTHON: engine })
  const one = page.getByRole('button', { name: 'Pre-label', exact: true })
  for (const [name, shapes] of [['nothing', 0], ['stale', 1]] as const) {
    await open(page, `${name}.png`, SIZE)
    await one.click()
    await expect(page.locator('.notice')).toHaveText(`The model found nothing on ${name}.png, so its labels stay as they were.`)
    await expect(one).toBeEnabled()
    await expect(page.locator('.shape')).toHaveCount(shapes)
  }
  await page.waitForTimeout(700)
  // No empty label file, which would read as one someone emptied
  expect(existsSync(path.join(dir, 'nothing.json'))).toBe(false)
  expect(readFileSync(path.join(dir, 'stale.json'), 'utf8')).toBe(proposals('stale'))
  await expect(page.getByRole('button', { name: 'Pre-label 2 images' })).toBeVisible()
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test('with no shapes left, an image that had no label file has none again, but a file pre-labelling or another app made stays', async () => {
  const { dir, engine } = await prelabelFolder('drawn-undone', ['adopted', 'again', 'drawn', 'one', 'quick', 'theirs'])
  const json = (name: string) => path.join(dir, `${name}.json`)
  const shapesIn = (name: string) => existsSync(json(name)) && read(dir, `${name}.json`).shapes.length
  // As Seamtag reads the folder meanwhile, every few ms: a label file with no shapes counts there as done, and
  // keeps the image out of both apps' Pre-label N images
  const emptied = new Set<string>()
  const reader = setInterval(() => {
    for (const name of ['drawn', 'one', 'quick', 'theirs']) {
      try {
        if (!read(dir, `${name}.json`).shapes.length) emptied.add(name)
      } catch {
        // No file
      }
    }
  }, 5)
  const { app, page, errors } = await launch(dir, { MYSEG_PYTHON: engine })
  const batch = (n: number) => page.getByRole('button', { name: `Pre-label ${n} images` })
  const toLabel = page.locator('.filters button', { hasText: 'Label' }).locator('b')
  const triangle = () => polygon(page, SIZE, [[100, 100], [200, 100], [150, 200]])
  const reloaded = "These labels changed in another app, so they were reloaded. Your last change wasn't saved."
  await expect(batch(6)).toBeVisible()

  // Drawn and undone within the 400 ms a save waits, with the folder read again between: nothing is written
  await open(page, 'quick.png', SIZE)
  await triangle()
  await focus(page)
  await page.keyboard.press('Control+z')
  await expect(page.locator('.shape')).toHaveCount(0)
  await saved(page)
  await page.waitForTimeout(600)
  expect(existsSync(json('quick'))).toBe(false)
  await expect(batch(6)).toBeVisible()

  // Saved, the folder read again as on coming back from Seamtag, then deleted: the file Seamline made goes.
  // The list, its Label filter and Pre-label N images follow at once, before the save is done.
  await open(page, 'drawn.png', SIZE)
  await triangle()
  await saved(page)
  await expect.poll(() => shapesIn('drawn')).toBe(1)
  await expect(batch(5)).toBeVisible()
  await expect(toLabel).toHaveText('5')
  await focus(page)
  await page.waitForTimeout(300)
  await expect(page.locator('.shape')).toHaveCount(1)
  await expect(page.locator('.notice')).toHaveCount(0)
  await holdSaves(app, true)
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await expect(batch(6)).toBeVisible()
  await expect(toLabel).toHaveText('6')
  await expect(file(page, 'drawn.png').locator('.muted')).toHaveCount(0)
  expect(shapesIn('drawn')).toBe(1)
  await holdSaves(app, false)
  await saved(page)
  await expect.poll(() => existsSync(json('drawn'))).toBe(false)
  await focus(page)
  await page.waitForTimeout(300)
  await expect(batch(6)).toBeVisible()
  await expect(page.locator('.notice')).toHaveCount(0)

  // Undo brings the shape back, and its file; undone again on a later visit, the file goes again
  await page.keyboard.press('Control+z')
  await expect(page.locator('.shape')).toHaveCount(1)
  await saved(page)
  await expect.poll(() => shapesIn('drawn')).toBe(1)
  await open(page, 'quick.png', SIZE)
  await open(page, 'drawn.png', SIZE)
  await page.keyboard.press('Control+z')
  await expect(page.locator('.shape')).toHaveCount(0)
  await saved(page)
  await expect.poll(() => existsSync(json('drawn'))).toBe(false)
  await expect(batch(6)).toBeVisible()

  // So with Pre-label on one image, its proposal rejected: the image had no labels
  await open(page, 'one.png', SIZE)
  await page.getByRole('button', { name: 'Pre-label', exact: true }).click()
  await expect(page.locator('.shape .tag')).toHaveText(['AI 70%'])
  await saved(page)
  await expect.poll(() => shapesIn('one')).toBe(1)
  await page.keyboard.press('w')
  await page.keyboard.press('s')
  await expect(page.locator('.shape')).toHaveCount(0)
  await saved(page)
  await expect.poll(() => existsSync(json('one'))).toBe(false)
  await expect(batch(6)).toBeVisible()

  // A pre-labelling result the window took came with its label file: rejected, the file stays, with no shapes,
  // as someone emptied it
  await open(page, 'adopted.png', SIZE)
  expect(await page.evaluate(() => window.api.prelabelAll('fake', ['adopted.png']))).toMatchObject({ done: 1, failed: [] })
  await expect(page.locator('.shape .tag')).toHaveText(['AI 70%'])
  await page.keyboard.press('w')
  await page.keyboard.press('s')
  await expect(page.locator('.shape')).toHaveCount(0)
  await saved(page)
  await expect.poll(() => existsSync(json('adopted')) && read(dir, 'adopted.json').shapes).toEqual([])
  await expect(batch(5)).toBeVisible()

  // Written over by Pre-label N images, even with the same proposals, the file is pre-labelling's too
  await open(page, 'again.png', SIZE)
  await page.getByRole('button', { name: 'Pre-label', exact: true }).click()
  await expect(page.locator('.shape .tag')).toHaveText(['AI 70%'])
  await saved(page)
  await expect.poll(() => shapesIn('again')).toBe(1)
  const single = readFileSync(json('again'), 'utf8')
  const inode = statSync(json('again')).ino
  await open(page, 'quick.png', SIZE)
  expect(await page.evaluate(() => window.api.prelabelAll('fake', ['again.png']))).toMatchObject({ done: 1, failed: [] })
  expect(statSync(json('again')).ino).not.toBe(inode)
  expect(readFileSync(json('again'), 'utf8')).toBe(single)
  await open(page, 'again.png', SIZE)
  await page.keyboard.press('w')
  await page.keyboard.press('s')
  await expect(page.locator('.shape')).toHaveCount(0)
  await saved(page)
  await expect.poll(() => existsSync(json('again')) && read(dir, 'again.json').shapes).toEqual([])
  await expect(batch(4)).toBeVisible()

  // A file another app made or changed meanwhile is never removed: the save is dropped, and the window shows theirs
  await open(page, 'theirs.png', SIZE)
  const made = labels('theirs', [square('cat', 20, 20, 80, 80)], { by: 'Seamtag' })
  seamtagSaves(dir, 'theirs', made)
  await triangle()
  await page.keyboard.press('Control+z')
  await expect(page.locator('.notice')).toHaveText(reloaded)
  await expect(page.locator('.shape')).toHaveCount(1)
  await saved(page)
  await page.waitForTimeout(600)
  expect(readFileSync(json('theirs'), 'utf8')).toBe(made)
  await expect(page.locator('.notice')).toHaveCount(0, { timeout: 8000 })
  await open(page, 'drawn.png', SIZE)
  await triangle()
  await saved(page)
  await expect.poll(() => shapesIn('drawn')).toBe(1)
  const changed = JSON.stringify({ ...read(dir, 'drawn.json'), by: 'Seamtag' }, null, 2)
  seamtagSaves(dir, 'drawn', changed)
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await expect(page.locator('.notice')).toHaveText(reloaded)
  await expect(page.locator('.shape')).toHaveCount(1)
  await saved(page)
  await page.waitForTimeout(600)
  expect(readFileSync(json('drawn'), 'utf8')).toBe(changed)

  clearInterval(reader)
  expect([...emptied]).toEqual([])
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test('an edit made while the image is reloaded after a refresh waits for the new labels', async () => {
  const dir = folder('reload-edit')
  classes(dir, ['cat', 'dog', 'bird'])
  await image(dir, 'g.png', SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'g.json'), labels('g', [square('cat', 50, 50, 150, 150), square('dog', 250, 50, 350, 150)]))
  const { app, page, errors } = await launch(dir)
  await open(page, 'g.png', SIZE)
  // Reading labels takes a while, as a big file on a slow disk would
  await app.evaluate(({ ipcMain }) => {
    const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: unknown[]) => Promise<unknown>>
    const load = handlers.get('labels:load')!
    handlers.set('labels:load', async (...args) => {
      (globalThis as any).loads = ((globalThis as any).loads ?? 0) + 1
      const result = await load(...args)
      await new Promise((done) => setTimeout(done, 1500))
      return result
    })
  })
  // Seamtag moves the dog and adds a bird; back in Seamline, the cat is deleted while they're on their way
  const theirs = labels('g', [square('cat', 50, 50, 150, 150), square('dog', 260, 60, 360, 160), square('bird', 100, 200, 150, 250)], { by: 'Seamtag' })
  seamtagSaves(dir, 'g', theirs)
  await focus(page)
  await expect.poll(() => app.evaluate(() => (globalThis as any).loads)).toBe(1)
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await expect(page.locator('.shape .name')).toHaveText(['cat', 'dog', 'bird'])
  await saved(page)
  await page.waitForTimeout(800)
  expect(readFileSync(path.join(dir, 'g.json'), 'utf8')).toBe(theirs)
  // Once they're in, the same edit is made on their labels
  await page.locator('.shape').first().click()
  await page.keyboard.press('Delete')
  await saved(page)
  await expect.poll(() => read(dir, 'g.json').shapes.map((s: any) => s.label)).toEqual(['dog', 'bird'])
  expect(read(dir, 'g.json').by).toBe('Seamtag')
  await finish(app, errors)
})

test('a refresh that only moves the image in the list keeps the view and the polygon being drawn', async () => {
  const dir = folder('refresh-view')
  classes(dir, ['cat'])
  for (const name of ['b', 'c']) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  const { app, page, errors } = await launch(dir)
  await open(page, 'c.png', SIZE)
  const zoom = () => page.locator('.status span').nth(2).textContent()
  const fitted = await zoom()
  await page.mouse.move(...await onImage(page, SIZE, 200, 150))
  await page.mouse.wheel(0, -300)
  await expect.poll(zoom).not.toBe(fitted)
  const zoomed = await zoom()
  await page.keyboard.press('p')
  for (const [x, y] of [[190, 140], [210, 140]]) await page.mouse.click(...await onImage(page, SIZE, x, y))
  // Another app adds an image that comes before this one
  await image(dir, 'a.png', SIZE[0], SIZE[1])
  await focus(page)
  await expect(page.locator('.file')).toHaveCount(3)
  await expect(page.locator('.file.on')).toHaveAttribute('title', 'c.png')
  await page.waitForTimeout(500)
  expect(await zoom()).toBe(zoomed)
  await page.mouse.click(...await onImage(page, SIZE, 200, 160))
  await page.keyboard.press('Enter')
  await expect(page.locator('.shape')).toHaveCount(1)
  await saved(page)
  await expect.poll(() => read(dir, 'c.json').shapes[0].points.length).toBe(3)
  await finish(app, errors)
})

test("labels that aren't classes don't make an image labelled, and the Train page names them", async () => {
  const dir = folder('not-classes')
  classes(dir, ['crack'])
  const line = { label: 'dent', points: [[10, 10], [90, 90]], group_id: null, description: '', shape_type: 'line', flags: {} }
  const made: Record<string, unknown[]> = {
    a: [square('crack', 10, 10, 90, 90)],
    b: [square('crack', 10, 10, 90, 90)],
    // Labels typed in Seamtag go into its label files, not onto the class list
    dent: [square('dent', 10, 10, 90, 90)],
    mixed: [square('dent', 10, 10, 90, 90), square('scratch', 100, 100, 190, 190)],
    both: [square('crack', 10, 10, 90, 90), square('dent', 100, 100, 190, 190)],
    // A line doesn't train whatever its label, nor a shape no one has accepted
    line: [line],
    proposed: [square('dent', 10, 10, 90, 90, true)]
  }
  for (const [name, shapes] of Object.entries(made)) {
    await image(dir, `${name}.png`, SIZE[0], SIZE[1])
    writeFileSync(path.join(dir, `${name}.json`), labels(name, shapes))
  }
  const { app, page, errors } = await launch(dir)
  const filter = (name: string) => page.locator('.filters button', { hasText: name })
  await expect(page.locator('.topbar')).toContainText('3/7 labelled')
  await expect(filter('Label')).toContainText('3')
  await expect(filter('Done')).toContainText('3')
  await expect(file(page, 'dent.png').locator('.dot')).not.toHaveClass(/done/)
  await expect(file(page, 'both.png').locator('.dot')).toHaveClass(/done/)
  // The shape list tells their shapes apart
  await open(page, 'both.png', SIZE)
  await expect(page.locator('.shape .not-class')).toHaveText(['not a class'])
  await expect(page.locator('.shape').nth(1)).toContainText('dent')
  await open(page, 'mixed.png', SIZE)
  await expect(page.locator('.shape .not-class')).toHaveCount(2)

  // Beside the Train button, before any dataset check
  await page.keyboard.press('Control+2')
  const train = page.getByRole('button', { name: /Train on/ })
  const named = page.getByTestId('not-classes')
  await expect(train).toHaveText('Train on 3 labelled images')
  await expect(named).toHaveText("3 images use labels that aren't classes yet, so they won't train: dent (3 shapes), scratch (1).")
  // In the smallest window and a large one, the note and the warning keep inside their panels
  const within = (selector: string) => page.evaluate((s) => {
    const all = [...document.querySelectorAll(s)].map((e) => [e.getBoundingClientRect(), e.closest('.panel')!.getBoundingClientRect()])
    return all.length > 0 && all.every(([r, panel]) => r.width > 0 && r.left >= panel.left && r.right <= panel.right) &&
      document.documentElement.scrollWidth <= innerWidth
  }, selector)
  for (const [width, height] of [[1000, 640], [1600, 1000]]) {
    await app.evaluate(({ BrowserWindow }, size) => BrowserWindow.getAllWindows()[0].setSize(size[0], size[1]), [width, height])
    await page.waitForTimeout(300)
    await page.locator('.trainer .add-classes').scrollIntoViewIfNeeded()
    await shot(page, `not-classes-train-${width}`)
    expect(await within('[data-testid="not-classes"], .trainer .add-classes'), `${width}`).toBe(true)
    await page.keyboard.press('Control+1')
    await page.waitForTimeout(300)
    await shot(page, `not-classes-label-${width}`)
    expect(await within('.labeler .shape .not-class'), `${width}`).toBe(true)
    await page.keyboard.press('Control+2')
  }

  // Added as classes, they count at once
  await page.getByRole('button', { name: 'Add as classes' }).click()
  await expect(named).toHaveCount(0)
  await expect(train).toHaveText('Train on 5 labelled images')
  await expect(page.locator('.topbar')).toContainText('5/7 labelled')
  expect(read(dir, '.myseg/project.json').classes.map((c: any) => c.name)).toEqual(['crack', 'dent', 'scratch'])
  // And so does a class another app took off the list, once Seamline's window is back
  writeFileSync(path.join(dir, '.myseg/project.json'), JSON.stringify({
    name: 'not-classes', classes: [{ name: 'crack', color: '#4f8cff' }, { name: 'scratch', color: '#3ecf8e' }]
  }))
  await focus(page)
  await expect(page.locator('.topbar')).toContainText('4/7 labelled')
  await expect(train).toHaveText('Train on 4 labelled images')
  await expect(named).toHaveText("3 images use a label that isn't a class yet, so it won't train: dent (3 shapes).")
  await page.keyboard.press('Control+1')
  await expect(page.locator('.shape .not-class')).toHaveCount(1)
  await expect(filter('Done')).toContainText('4')
  await finish(app, errors)
})

test('the class list keeps what another app changed in it, and shows it on coming back', async () => {
  const dir = folder('classes')
  classes(dir, ['cat', 'dog'])
  await image(dir, 'p.png', SIZE[0], SIZE[1])
  const project = (list: [string, string][], extra = {}) => writeFileSync(path.join(dir, '.myseg/project.json'),
    JSON.stringify({ name: 'classes', classes: list.map(([name, color]) => ({ name, color })), ...extra }, null, 2))
  const { app, page, errors } = await launch(dir)
  await open(page, 'p.png', SIZE)
  const names = page.locator('.class .name')
  await expect(names).toHaveText(['cat', 'dog'])

  // Seamtag recolours the dog and adds a bird, with a key of its own
  project([['cat', '#4f8cff'], ['dog', '#00ff00'], ['bird', '#123456']], { seamtag: 1 })
  await focus(page)
  await expect(names).toHaveText(['cat', 'dog', 'bird'])
  await expect(page.locator('.class i').nth(1)).toHaveCSS('background-color', 'rgb(0, 255, 0)')

  // While Seamline isn't looking, Seamtag deletes the cat and adds a fish; Seamline adds a cow
  project([['dog', '#00ff00'], ['bird', '#123456'], ['fish', '#654321']], { seamtag: 2 })
  await page.getByPlaceholder('New class').fill('cow')
  await page.getByPlaceholder('New class').press('Enter')
  await expect(names).toHaveText(['dog', 'bird', 'cow', 'fish'])
  expect(read(dir, '.myseg/project.json')).toEqual({
    name: 'classes', seamtag: 2,
    classes: [{ name: 'dog', color: '#00ff00' }, { name: 'bird', color: '#123456' }, { name: 'cow', color: '#f7b731' }, { name: 'fish', color: '#654321' }]
  })
  // The class just added stays chosen as the list around it changes
  await expect(page.locator('.class.on .name')).toHaveText('cow')
  await page.getByPlaceholder('New class').blur()
  await page.keyboard.press('p')
  for (const [x, y] of [[100, 100], [200, 100], [150, 200]]) await page.mouse.click(...await onImage(page, SIZE, x, y))
  await page.keyboard.press('Enter')
  await saved(page)
  await expect.poll(() => read(dir, 'p.json').shapes.map((s: any) => s.label)).toEqual(['cow'])

  // Back from the Train page, the list is read again too
  await page.keyboard.press('Control+2')
  project([['dog', '#00ff00'], ['bird', '#123456'], ['cow', '#f7b731'], ['fish', '#654321'], ['ant', '#ffffff']], { seamtag: 3 })
  await page.keyboard.press('Control+1')
  await expect(names).toHaveText(['dog', 'bird', 'cow', 'fish', 'ant'])
  await finish(app, errors)
})

test("a save made while the folder is read again goes through at once, and the refresh knows it's Seamline's", async () => {
  const dir = folder('slow-refresh')
  classes(dir, ['cat'])
  for (const name of ['a', 'b', 'slow']) await image(dir, `${name}.png`, SIZE[0], SIZE[1])
  writeFileSync(path.join(dir, 'slow.json'), labels('slow', [square('cat', 10, 10, 50, 50)]))
  const { app, page, errors } = await launch(dir)
  await open(page, 'a.png', SIZE)
  await page.keyboard.press('b')
  await drag(page, SIZE, [60, 60], [120, 60])
  await saved(page)
  await expect.poll(() => existsSync(path.join(dir, 'a.json'))).toBe(true)
  // Another app changes a label file that's slow to read, as a big folder is slow to read again
  await app.evaluate(() => {
    const fs = (process as any).getBuiltinModule('fs')
    const readFile = fs.promises.readFile
    fs.promises.readFile = async (...args: any[]) => {
      if (!String(args[0]).endsWith('slow.json')) return readFile(...args)
      ;(globalThis as any).slow = 'reading'
      const text = await readFile(...args)
      await new Promise((done) => setTimeout(done, 2500))
      ;(globalThis as any).slow = 'read'
      return text
    }
  })
  seamtagSaves(dir, 'slow', labels('slow', [square('cat', 20, 20, 60, 60)]))
  await focus(page)
  await expect.poll(() => app.evaluate(() => (globalThis as any).slow)).toBe('reading')
  // Meanwhile another edit, saved on the way to the next image
  await drag(page, SIZE, [250, 200], [320, 200])
  await page.keyboard.press('d')
  await expect.poll(() => read(dir, 'a.json').shapes.length, { timeout: 1500 }).toBe(2)
  expect(await app.evaluate(() => (globalThis as any).slow)).toBe('reading')
  await expect.poll(() => app.evaluate(() => (globalThis as any).slow), { timeout: 5000 }).toBe('read')
  await page.waitForTimeout(300)
  // The list has a.png as saved, and its undo history stayed: Seamline's own save isn't another app's change
  await expect(file(page, 'a.png')).toContainText('2')
  await expect(page.locator('.notice')).toHaveCount(0)
  await page.keyboard.press('a')
  await expect(page.locator('.file.on')).toHaveAttribute('title', 'a.png')
  await expect(page.locator('.shape')).toHaveCount(2)
  await page.keyboard.press('Control+z')
  await expect(page.locator('.shape')).toHaveCount(1)
  await saved(page)
  await expect.poll(() => read(dir, 'a.json').shapes.length).toBe(1)
  await finish(app, errors)
})
