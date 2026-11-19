import { expect, Page, test } from '@playwright/test'
import { existsSync, mkdirSync, readFileSync, statSync, symlinkSync, utimesSync, writeFileSync } from 'fs'
import path from 'path'
import sharp from 'sharp'
import { copy, drag, file, files, finish, folder, image, launch, onImage, open, pixel, polygon, read, saved, shot } from './seamtag_folder'

// A folder as Seamtag leaves it: labelme 5 and labelme 7 files with every shape type, LabelImg's
// three formats, subfolders, and every image format Seamtag lists
const ALL = [640, 480], L7 = [200, 120], CARS = [64, 48], GREY = [50, 40]
const EXTS = ['avif', 'bmp', 'gif', 'jfif', 'jpg', 'tif', 'webp']
const GREY_BLUE = [0x44, 0x55, 0x66]
// A 1×1 PNG, as labelme keeps the image in its files unless told not to
const EMBEDDED = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

async function seamtagFolder(name: string) {
  const dir = folder(name)
  await image(dir, 'all.png', ALL[0], ALL[1])
  copy(dir, 'labelme5_all_shapes.json', 'all.json')
  await image(dir, 'sub/deeper/l7.png', L7[0], L7[1])
  copy(dir, 'labelme7_shapes.json', 'sub/deeper/l7.json')
  await image(dir, 'cars.png', CARS[0], CARS[1])
  copy(dir, 'voc_cars.xml', 'cars.xml')
  await image(dir, 'empty.png', 32, 32)
  copy(dir, 'voc_empty.xml', 'empty.xml')
  await image(dir, 'sub/grey.png', GREY[0], GREY[1])
  copy(dir, 'yolo_grey.txt', 'sub/grey.txt')
  copy(dir, 'yolo_classes.txt', 'sub/classes.txt')
  await image(dir, 'sub/cars.png', CARS[0], CARS[1])
  copy(dir, 'createml_cars.json', 'sub/cars.json')
  for (const ext of EXTS) await image(dir, `p_${ext}.${ext}`, 48, 36)
  // Not images to label: a hidden folder, and a folder linked from elsewhere
  await image(dir, '.hidden/x.png', 20, 20)
  const elsewhere = folder(`${name}-elsewhere`)
  await image(elsewhere, 'far.png', 20, 20)
  symlinkSync(elsewhere, path.join(dir, 'linked'))
  return dir
}

const filter = (page: Page, name: string) => page.locator('.filters button', { hasText: name })
const away = (p: number[]) => Math.max(...p.map((v, i) => Math.abs(v - GREY_BLUE[i])))

test('a folder labelled in Seamtag opens with every format counted', async () => {
  const dir = await seamtagFolder('counts')
  const { app, page, errors } = await launch(dir)
  const names = [
    'all.png', 'cars.png', 'empty.png', ...EXTS.map((e) => `p_${e}.${e}`), 'sub/cars.png', 'sub/deeper/l7.png', 'sub/grey.png'
  ]
  await expect(page.locator('.file')).toHaveCount(names.length)
  await shot(page, 'formats')
  expect(await page.locator('.file').evaluateAll((all) => all.map((e) => e.getAttribute('title')))).toEqual(names)
  // labelme files, Pascal VOC, YOLO and CreateML all count; the empty VOC file and plain images don't
  await expect(page.locator('.topbar')).toContainText('5/13 labelled')
  await expect(filter(page, 'All')).toContainText('13')
  await expect(filter(page, 'Label')).toContainText('8')
  await expect(filter(page, 'Review')).toContainText('1')
  await expect(filter(page, 'Done')).toContainText('4')
  await expect(file(page, 'cars.png')).toContainText('VOC')
  await expect(file(page, 'sub/grey.png')).toContainText('YOLO')
  await expect(file(page, 'sub/cars.png')).toContainText('CreateML')
  await expect(file(page, 'sub/cars.png')).toContainText('4')
  // Images with no labels are pre-labelled; the empty Pascal VOC file is Seamtag's to change
  await expect(page.getByRole('button', { name: 'Pre-label 7 images' })).toBeVisible()
  // The folder had no class list: it starts with the labels of every format
  for (const name of ['blob', 'car', 'cat', 'd', '고양이']) {
    await expect(page.locator('.class .name').filter({ hasText: new RegExp(`^${name}$`) })).toHaveCount(1)
  }

  // Every image format Seamtag lists opens: the window shows JPEG, PNG and BMP itself, the rest come as tiles
  for (const ext of EXTS) {
    const opened = await page.evaluate((n) => window.api.openImage(n), `p_${ext}.${ext}`)
    expect([opened.width, opened.height, !!opened.pyramid], ext).toEqual([48, 36, !['jpg', 'jfif', 'bmp'].includes(ext)])
    await open(page, `p_${ext}.${ext}`, [48, 36])
    await expect.poll(async () => away(await pixel(page, [48, 36], 24, 18)), { message: ext }).toBeLessThan(12)
  }
  await expect(page.locator('.error')).toHaveCount(0)
  await finish(app, errors)
})

test('shapes only Seamtag reshapes stay exactly as they are, and saving keeps every key', async () => {
  const dir = await seamtagFolder('fixed')
  // An imagePath from the project folder, as Seamline once wrote: it's written from the label file's own folder.
  // The image labelme keeps in the file stays, and so do ids too long for a number, at the top and in a shape.
  const l7 = read(dir, 'sub/deeper/l7.json')
  l7.shapes[0].id = 'SHAPE_ID'
  writeFileSync(path.join(dir, 'sub/deeper/l7.json'), JSON.stringify({ ...l7, imagePath: 'sub/deeper/l7.png', imageData: EMBEDDED, annotation_id: 'FILE_ID' }, null, 2)
    .replace('"FILE_ID"', '1234567890123456789').replace('"SHAPE_ID"', '9007199254740993'))
  const before = { all: read(dir, 'all.json'), l7: read(dir, 'sub/deeper/l7.json') }
  const { app, page, errors } = await launch(dir)

  await open(page, 'all.png', ALL)
  expect(await page.locator('.shape .kind').allTextContents()).toEqual(['3 pts', 'box', 'circle', 'line', 'point', 'line strip', 'mask', '3 pts'])
  // The circle: picked by a click inside it, but it doesn't move
  await page.keyboard.press('v')
  await drag(page, ALL, [100, 100], [200, 160])
  await expect(page.locator('.shape.on')).toContainText('circle')
  await expect(page.getByTestId('fixed-hint')).toContainText('Seamtag reshapes this circle')
  await expect(page.locator('.notice')).toContainText('Seamtag reshapes this circle')
  await shot(page, 'circle')
  await page.waitForTimeout(700)
  expect(read(dir, 'all.json')).toEqual(before.all)
  // Painting continues its class beside it, and the eraser leaves it whole
  await page.keyboard.press('b')
  await drag(page, ALL, [90, 100], [110, 100])
  await expect(page.locator('.shape')).toHaveCount(9)
  await saved(page)
  await page.keyboard.press('e')
  await drag(page, ALL, [100, 80], [100, 120])
  await saved(page)
  await page.keyboard.press('Escape')
  await polygon(page, ALL, [[300, 300], [400, 300], [350, 400]])
  await saved(page)
  await expect.poll(() => read(dir, 'all.json').shapes.at(-1)?.points?.length).toBe(3)
  const all = read(dir, 'all.json')
  expect(all.shapes.slice(0, before.all.shapes.length)).toEqual(before.all.shapes)
  const { shapes: _a, imagePath, ...keys } = all
  const { shapes: _b, imagePath: _c, ...keysBefore } = before.all
  expect(keys).toEqual(keysBefore)
  expect(imagePath).toBe('all.png')

  await open(page, 'sub/deeper/l7.png', L7)
  expect(await page.locator('.shape .kind').allTextContents()).toEqual([
    'box', '4 pts', 'circle', 'rotated box', 'mask', 'line', 'mask', '4 pts', 'point', 'line strip'
  ])
  await expect(page.locator('.shape').nth(7).locator('[aria-label="locked"]')).toHaveCount(1)
  // A mask is its PNG's pixels where labelme 7 puts them: whole-number points give the box's size,
  // fractional ones the PNG's own, from the first point rounded
  const tinted = async (x: number, y: number) => away(await pixel(page, L7, x, y)) > 12
  await expect.poll(() => tinted(75.5, 75.5)).toBe(true)
  expect(await tinted(110.5, 80.5)).toBe(false)
  expect(await tinted(70.5, 70.5)).toBe(false)
  expect(await tinted(151.5, 61.5)).toBe(true)
  expect(await tinted(150.5, 61.5)).toBe(false)
  expect(await tinted(153.5, 62.5)).toBe(true)
  // Clicked on its pixels, the mask is picked; the eraser leaves it be
  await page.keyboard.press('v')
  await page.mouse.click(...await onImage(page, L7, 75.5, 75.5))
  await expect(page.locator('.shape.on')).toContainText('mask')
  await shot(page, 'mask')
  await page.keyboard.press('e')
  await drag(page, L7, [65, 75], [85, 75])
  await expect(page.locator('.notice')).toContainText('Seamtag reshapes this mask')
  // The locked polygon and the rotated box don't move, by a corner or whole
  await page.keyboard.press('v')
  await page.locator('.shape').nth(7).click()
  await expect(page.getByTestId('fixed-hint')).toContainText('locked')
  await drag(page, L7, [21, 101], [40, 90])
  await drag(page, L7, [40, 108], [80, 60])
  await drag(page, L7, [130, 25], [100, 60])
  await page.waitForTimeout(700)
  expect(read(dir, 'sub/deeper/l7.json')).toEqual(before.l7)
  // Painting over the locked polygon adds a shape of its own; erasing across the others takes nothing
  await page.keyboard.press('Escape')
  await page.locator('.class').filter({ has: page.locator('.name', { hasText: /^g$/ }) }).click()
  await page.keyboard.press('b')
  await drag(page, L7, [10, 108], [70, 108])
  await saved(page)
  await expect(page.locator('.shape')).toHaveCount(11)
  await page.keyboard.press('Escape')
  await page.keyboard.press('e')
  await drag(page, L7, [30, 72], [76, 76])
  await drag(page, L7, [118, 22], [146, 30])
  await expect(page.locator('.shape')).toHaveCount(11)
  await page.keyboard.press('Escape')
  await polygon(page, L7, [[100, 90], [140, 90], [120, 110]])
  await saved(page)
  await expect.poll(() => read(dir, 'sub/deeper/l7.json').shapes.length).toBe(12)
  const l7After = read(dir, 'sub/deeper/l7.json')
  expect(l7After.shapes.slice(0, before.l7.shapes.length)).toEqual(before.l7.shapes)
  expect(l7After.reviewed_by).toBe('labelme 7')
  expect(l7After.version).toBe('7.9.1')
  expect(l7After.imagePath).toBe('l7.png')
  expect(l7After.imageData).toBe(EMBEDDED)
  const l7Text = readFileSync(path.join(dir, 'sub/deeper/l7.json'), 'utf8')
  expect(l7Text).toContain('"annotation_id": 1234567890123456789')
  expect(l7Text).toContain('"id": 9007199254740993')
  await finish(app, errors)
})

test('only shapes with an area make an image labelled, and a file the engine would refuse is marked', async () => {
  const dir = folder('labelled')
  const size = [64, 48]
  const tri = [[5, 5], [30, 5], [15, 30]]
  const shape = (label: unknown, shape_type: string, points: number[][], extra = {}) =>
    ({ label, points, group_id: null, description: '', shape_type, flags: {}, ...extra })
  const labels: Record<string, unknown[]> = {
    area: [shape('cat', 'polygon', tri), shape('edge', 'line', [[1, 1], [9, 9]])],
    lines: [shape('edge', 'line', [[1, 1], [9, 9]]), shape('road', 'linestrip', tri), shape('tip', 'point', [[3, 3]]), shape('dots', 'points', tri)],
    thin: [shape('thin', 'polygon', [[1, 1], [9, 9]]), shape('box', 'rectangle', [[1, 1]])],
    proposed: [shape('cat', 'polygon', tri, { flags: { ai: true }, score: 0.5 }), shape('owl', 'polygon', tri, { flags: { ai: true }, score: 0.6 })],
    // What the engine and Seamtag can't read: no label, a shape type labelme hasn't, a flag that isn't true or false
    nolabel: [shape(undefined, 'polygon', tri)],
    ellipse: [shape('cat', 'ellipse', tri)],
    flagtext: [shape('cat', 'polygon', tri, { flags: { ai: 'true' } })]
  }
  for (const [name, shapes] of Object.entries(labels)) {
    await image(dir, `${name}.png`, size[0], size[1])
    writeFileSync(path.join(dir, `${name}.json`), JSON.stringify({
      version: '5.5.0', flags: {}, shapes, imagePath: `${name}.png`, imageData: null, imageHeight: size[1], imageWidth: size[0]
    }))
  }
  await image(dir, 'plain.png', size[0], size[1])
  const { app, page, errors } = await launch(dir)
  await expect(page.locator('.error')).toHaveText("3 label files can't be read and won't be changed: ellipse.png, flagtext.png, nolabel.png")
  for (const name of ['ellipse.png', 'flagtext.png', 'nolabel.png']) await expect(file(page, name).locator('.dot')).toHaveClass(/broken/)
  // As training counts them: lines, points and shapes short of points don't make an image labelled,
  // though pre-labelling leaves those images be
  await expect(page.locator('.topbar')).toContainText('1/8 labelled')
  await expect(filter(page, 'Label')).toContainText('3')
  await expect(filter(page, 'Review')).toContainText('1')
  await expect(filter(page, 'Done')).toContainText('1')
  await expect(file(page, 'lines.png').locator('.dot')).not.toHaveClass(/done/)
  await expect(file(page, 'area.png').locator('.dot')).toHaveClass(/done/)
  await expect(page.getByRole('button', { name: 'Pre-label 2 images' })).toBeVisible()
  // The folder had no class list: it starts with the labels that would train
  await expect(page.locator('.class .name')).toHaveText(['cat'])
  await page.locator('.error').click()
  await file(page, 'nolabel.png').click()
  await expect(page.locator('.error')).toHaveText("Could not open nolabel.png: nolabel.json can't be read, so it's left unchanged: Shape 1 needs a label")
  await page.getByRole('tab', { name: /Train/ }).click()
  await expect(page.getByRole('button', { name: /Train on/ })).toHaveText('Train on 1 labelled images')
  await finish(app, errors)
})

test("images whose names differ only where a folder name can't keep them get tiles of their own", async () => {
  const dir = folder('tile-names')
  // Same size and time, as a camera's files unpacked from one archive: only the names tell them apart
  const made = [['a/b.tif', '#ff0000'], ['a_b.tif', '#00ff00'], ['정상/0001.tif', '#0000ff'], ['불량/0001.tif', '#ffff00']]
  const when = new Date('2026-11-01T10:00:00Z')
  for (const [name, color] of made) {
    mkdirSync(path.dirname(path.join(dir, name)), { recursive: true })
    await sharp({ create: { width: 300, height: 200, channels: 3, background: color } }).tiff({ compression: 'none' }).toFile(path.join(dir, name))
    utimesSync(path.join(dir, name), when, when)
  }
  expect(new Set(made.map(([name]) => statSync(path.join(dir, name)).size)).size).toBe(1)
  const { app, page, errors } = await launch(dir)
  for (const [name, color] of made) {
    await open(page, name, [300, 200])
    const want = [1, 3, 5].map((i) => parseInt(color.slice(i, i + 2), 16))
    const off = async () => Math.max(...(await pixel(page, [300, 200], 150, 100)).map((v, i) => Math.abs(v - want[i])))
    await expect.poll(off, { message: name }).toBeLessThan(16)
  }
  await finish(app, errors)
})

test("an image whose own masks pass the budget for decoded masks keeps them, rather than decoding them over and over", async () => {
  const dir = folder('big-masks')
  const W = 4000, H = 3000
  await image(dir, 'a.png', 64, 48)
  await image(dir, 'big.png', W, H)
  // Five masks of 12 MP, each its own PNG: 60 MP, past the 50 MP of decoded masks the window keeps
  const masks = await Promise.all([0, 1, 2, 3, 4].map(async (n) => {
    const bits = Buffer.alloc(W * H, 255)
    bits[n] = 0
    return (await sharp(bits, { raw: { width: W, height: H, channels: 1 } }).png().toBuffer()).toString('base64')
  }))
  const shapes = masks.map((mask) => ({ label: 'a', points: [[0, 0], [W - 1, H - 1]], group_id: null, description: '', shape_type: 'mask', flags: {}, mask }))
  writeFileSync(path.join(dir, 'big.json'), JSON.stringify({ version: '5.5.0', flags: {}, shapes, imagePath: 'big.png', imageData: null, imageHeight: H, imageWidth: W }))
  const { app, page, errors } = await launch(dir)
  // A decode reads the mask's pixels back once
  await page.evaluate(() => {
    const read = CanvasRenderingContext2D.prototype.getImageData
    ;(window as any).decodes = 0
    CanvasRenderingContext2D.prototype.getImageData = function (...args: Parameters<typeof read>) {
      ;(window as any).decodes++
      return read.apply(this, args)
    }
  })
  await open(page, 'big.png', [W, H])
  const decodes = () => page.evaluate(() => (window as any).decodes as number)
  await expect.poll(decodes, { timeout: 30000 }).toBeGreaterThanOrEqual(5)
  await page.waitForTimeout(1500)
  const settled = await decodes()
  await page.waitForTimeout(2500)
  expect(await decodes()).toBe(settled)
  expect(settled).toBe(5)
  await finish(app, errors)
})

test('Pascal VOC, YOLO and CreateML images open read-only, and their files stay as they are', async () => {
  const dir = await seamtagFolder('read-only')
  const before = files(dir)
  const { app, page, errors } = await launch(dir)
  const bar = page.getByTestId('read-only-bar')
  for (const [name, size, kind, count] of [
    ['cars.png', CARS, 'Pascal VOC boxes', 4], ['sub/grey.png', GREY, 'YOLO boxes', 1], ['sub/cars.png', CARS, 'CreateML boxes', 4]
  ] as const) {
    await open(page, name, size)
    await expect(bar).toContainText(`These labels are ${kind}. Seamline trains on them; edit them in Seamtag, which keeps the format.`)
    await expect(bar.getByRole('button', { name: 'Label in Seamtag' })).toBeVisible()
    await expect(page.locator('.shape')).toHaveCount(count)
    await shot(page, `read-only-${kind.split(' ')[0]}`)
    for (const tool of ['Polygon (P)', 'Rectangle (R)']) await expect(page.locator(`.toolbar button[title="${tool}"]`)).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Pre-label', exact: true })).toBeDisabled()
    // Drawing, painting, erasing, relabelling and deleting all leave it as it is
    await page.keyboard.press('p')
    for (const [x, y] of [[5, 5], [30, 5], [30, 30]]) await page.mouse.click(...await onImage(page, size, x, y))
    await page.keyboard.press('Enter')
    await page.keyboard.press('b')
    await drag(page, size, [5, 20], [40, 20])
    await page.keyboard.press('e')
    await drag(page, size, [5, 20], [40, 20])
    await page.locator('.shape').first().click()
    await page.keyboard.press('2')
    await page.keyboard.press('Delete')
    await expect(page.locator('.shape')).toHaveCount(count)
    await page.waitForTimeout(600)
  }
  await open(page, 'empty.png', [32, 32])
  await expect(bar).toContainText('Pascal VOC')
  await expect(page.locator('.labeler .shapes')).toContainText('No boxes in this file.')
  await page.waitForTimeout(600)

  // The smallest window: the bar and the top bar keep inside it
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(1000, 640))
  await open(page, 'cars.png', CARS)
  await page.waitForTimeout(300)
  await shot(page, 'read-only-small')
  const spill = await page.evaluate(() => {
    const stage = document.querySelector('.stage')!.getBoundingClientRect()
    const inBar = [...document.querySelectorAll('.read-only, .read-only *')].map((e) => e.getBoundingClientRect())
    const top = [...document.querySelectorAll('.topbar > *')].map((e) => e.getBoundingClientRect())
    return {
      bar: inBar.every((r) => r.left >= stage.left && r.right <= stage.right && r.bottom <= stage.bottom),
      top: top.every((r) => r.right <= innerWidth && r.height <= 46),
      page: document.documentElement.scrollWidth <= innerWidth
    }
  })
  expect(spill).toEqual({ bar: true, top: true, page: true })
  await finish(app, errors)
  expect(files(dir)).toEqual(before)
  for (const name of ['cars.json', 'sub/grey.json', 'empty.json']) expect(existsSync(path.join(dir, name))).toBe(false)
  expect(readFileSync(path.join(dir, 'cars.xml'), 'utf8')).toContain('<annotation verified="yes">')
})
