type Labelled = { label: string; points?: unknown[]; shape_type?: string | null; mask?: string | null; flags?: Record<string, boolean>; score?: number }

// labelme 7's shape types. Lines and points have no area, so they don't train.
const AREA_TYPES = ['polygon', 'rectangle', 'oriented_rectangle', 'circle', 'mask']
const SHAPE_TYPES = [...AREA_TYPES, 'line', 'linestrip', 'point', 'points']

/** Whether a shape covers pixels, as the engine decides: lines and points don't, nor shapes short of points */
export function hasArea(s: Labelled) {
  const type = s.shape_type ?? 'polygon'
  const count = s.points?.length ?? 0
  if (type === 'polygon' || type === 'oriented_rectangle') return count >= 3
  if (type === 'rectangle' || type === 'circle') return count >= 2
  return type === 'mask' && count >= 2 && !!s.mask
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === 'object' && v !== null && !Array.isArray(v)
const given = (v: unknown) => v !== undefined && v !== null
const isFlags = (v: unknown) => isObject(v) && Object.values(v).every((f) => typeof f === 'boolean')
const isWhole = (v: unknown): v is number => typeof v === 'number' && Number.isInteger(v)

/** Refuses a shape the engine and Seamtag can't read either, in the engine's words and order, so all three agree */
function checkShape(s: unknown, index: number) {
  const where = `Shape ${index + 1}`
  if (!isObject(s)) throw new Error(`${where} isn't an object`)
  if (typeof s.label !== 'string') throw new Error(`${where} needs a label`)
  const points = s.points
  if (!Array.isArray(points) || !points.length) throw new Error(`${where} needs points`)
  if (!points.every((p) => Array.isArray(p) && p.length === 2 && p.every((v) => typeof v === 'number' && Number.isFinite(v)))) {
    throw new Error(`${where} points must be [x, y] pairs`)
  }
  const type = s.shape_type ?? 'polygon'
  if (!SHAPE_TYPES.includes(type as string)) throw new Error(`${where} has an unknown shape_type: ${String(type)}`)
  if (given(s.description) && typeof s.description !== 'string') throw new Error(`${where} description must be text`)
  if (given(s.group_id) && !isWhole(s.group_id)) throw new Error(`${where} group_id must be a whole number`)
  if (given(s.mask) && typeof s.mask !== 'string') throw new Error(`${where} mask must be a base64 PNG`)
  if (type === 'oriented_rectangle' && points.length !== 4) throw new Error(`${where} is an oriented rectangle, so it needs 4 points`)
  // Unlike the other keys, null isn't a lock's "not set"
  if ('locked' in s && typeof s.locked !== 'boolean') throw new Error(`${where} locked must be true or false`)
  if (given(s.flags) && !isFlags(s.flags)) throw new Error(`${where} flags must map names to true or false`)
}

/** A LabelMe file as JSON gives it, checked as the engine and Seamtag check it, in the same words and order */
function checkLabelMe(doc: unknown): asserts doc is Record<string, unknown> & { shapes: unknown[] } {
  if (!isObject(doc)) throw new Error("It isn't a LabelMe file: the top level must be an object")
  if (!Array.isArray(doc.shapes)) throw new Error("It isn't a LabelMe file: there is no shapes list")
  if (given(doc.imagePath) && typeof doc.imagePath !== 'string') throw new Error('imagePath must be text')
  if (given(doc.imageData) && typeof doc.imageData !== 'string') throw new Error('imageData must be base64 text')
  if (given(doc.flags) && !isFlags(doc.flags)) throw new Error('Image flags must map names to true or false')
  doc.shapes.forEach(checkShape)
  for (const key of ['imageHeight', 'imageWidth']) {
    const size = doc[key]
    if (given(size) && !(isWhole(size) && size > 0)) throw new Error(`${key} must be a positive whole number`)
  }
}

/**
 * A LabelMe file's text, read and checked as the engine and Seamtag read it, or refused in their words.
 * tests/fixtures/seamtag/labelme_checks.json holds the cases all three must agree on.
 */
export function readLabelMe(text: string) {
  let doc: unknown
  try {
    doc = JSON.parse(text)
  } catch (e) {
    throw new Error(`It isn't valid JSON (${(e as Error).message})`)
  }
  checkLabelMe(doc)
  return doc
}

// What the image list shows. AI shapes stay out of the counts until accepted, as in training. Accepted
// shapes with an area are counted by label: whether they make the image labelled is up to the classes.
export function summarize(shapes: Labelled[]) {
  const classes: Record<string, number> = {}
  const areas: Record<string, number> = {}
  let ai = 0
  let score: number | null = null
  for (const s of shapes) {
    if (s.flags?.ai) {
      ai++
      if (s.score !== undefined) score = score === null ? s.score : Math.min(score, s.score)
    } else {
      classes[s.label] = (classes[s.label] ?? 0) + 1
      if (hasArea(s)) areas[s.label] = (areas[s.label] ?? 0) + 1
    }
  }
  return { count: shapes.length - ai, ai, areas, score, classes }
}

/** Labelled, as training counts it: with an accepted shape that has an area and a class's label */
export const isLabelled = (areas: Record<string, number>, classes: ReadonlySet<string>) =>
  Object.keys(areas).some((label) => classes.has(label))

/**
 * Labels that aren't classes on accepted shapes with an area, as the dataset check names them: those
 * shapes don't train until the labels are added. How many images use them, and each label's shapes,
 * most first. An empty label can't be a class, so it isn't named.
 */
export function notClasses(images: { areas: Record<string, number> }[], classes: ReadonlySet<string>) {
  const shapes = new Map<string, number>()
  let count = 0
  for (const { areas } of images) {
    const labels = Object.keys(areas).filter((label) => label && !classes.has(label))
    if (labels.length) count++
    for (const label of labels) shapes.set(label, (shapes.get(label) ?? 0) + areas[label])
  }
  return { images: count, labels: [...shapes].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)) }
}
