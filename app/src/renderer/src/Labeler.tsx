import { Check, CheckCheck, Eraser, ExternalLink, Eye, EyeOff, Hexagon, Lock, Maximize, MousePointer2, Paintbrush, Plus, Redo2, SkipForward, Sparkles, Square, Trash2, Undo2, X } from 'lucide-react'
import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from 'react'
import type { LabelClass } from '../../shared/classes'
import { hasArea, isLabelled, summarize } from '../../shared/labels'
import { accepted, AREAS, contains, editable, eraseShapes, inside, maskPlacement, outline, paintShapes, segmentDistance, strokeRegion } from './geometry'
import { decodedMask, decodeMask, dropTints, tintedMask } from './masks'
import { fileUrl, ImageItem, LabelFormat, Labels, OpenImage, OpenProject, PALETTE, Shape } from './types'

type Tool = 'select' | 'polygon' | 'rectangle' | 'brush' | 'eraser'
type Filter = 'all' | 'label' | 'review' | 'done'
type Props = {
  data: OpenProject
  /** Change the class list, from the list as it is when the change's turn comes */
  changeClasses: (change: (classes: LabelClass[]) => LabelClass[]) => Promise<void>
  setImages: (fn: (images: ImageItem[]) => ImageItem[]) => void
  active: boolean
  /** A dialog is open over the window: keys are for it */
  modal: boolean
  onError: (m: string) => void
  /** Open the current image in Seamtag */
  onSeamtag: () => void
  /** Seamtag is starting, so it isn't started again */
  seamtagStarting: boolean
}
export type LabelerHandle = {
  /** Saves what's waiting to be saved */
  flush(): Promise<void>
  /** The image chosen in the list */
  current(): string | null
}
type Drag =
  | { kind: 'pan'; x: number; y: number }
  | { kind: 'vertex'; shape: number; vertex: number; press: number[]; active: boolean }
  | { kind: 'move'; shape: number; x: number; y: number; press: number[]; active: boolean }
  | { kind: 'hold'; shape: number; press: number[]; told: boolean }
  | { kind: 'rect'; start: number[]; end: number[] }
  | { kind: 'stroke'; points: number[][]; radius: number }

const HIT = 8
const CACHE_TILES = 400
const DRAG_START = 4
const HISTORY_IMAGES = 30

// Labelled as training counts it: with an accepted shape that has an area, not just lines or points,
// and a class's label
const FILTERS: { key: Filter; name: string; test: (i: ImageItem, labelled: boolean) => boolean }[] = [
  { key: 'all', name: 'All', test: () => true },
  { key: 'label', name: 'Label', test: (i, labelled) => !labelled && !i.ai && !i.broken },
  { key: 'review', name: 'Review', test: (i) => i.ai > 0 },
  { key: 'done', name: 'Done', test: (i, labelled) => labelled && !i.ai }
]
const percent = (score: number) => `${Math.round(score * 100)}%`
// Seamline writes LabelMe JSON only. Labels kept in LabelImg's formats are shown, and trained on, as they are.
const BOXES: Record<LabelFormat, string> = { labelme: 'LabelMe shapes', voc: 'Pascal VOC boxes', yolo: 'YOLO boxes', createml: 'CreateML boxes' }
const readOnlyFormat = (f: LabelFormat | null | undefined) => !!f && f !== 'labelme'

const KINDS: Record<string, string> = {
  rectangle: 'box', oriented_rectangle: 'rotated box', circle: 'circle', line: 'line', linestrip: 'line strip',
  point: 'point', points: 'points', mask: 'mask'
}
const kindOf = (s: Shape) => (!s.shape_type || s.shape_type === 'polygon' ? `${s.points.length} pts` : KINDS[s.shape_type] ?? s.shape_type)
// What a shape Seamline's tools leave alone says about itself
const fixedHint = (s: Shape) => s.locked
  ? 'This shape is locked. Seamtag can unlock it; here you can relabel it, or delete it.'
  : `Seamtag reshapes this ${KINDS[s.shape_type] ?? 'shape'}. Here you can relabel it, or delete it.`

// Least certain first; shapes without a score come last
function reviewOrder(list: Shape[]) {
  return list
    .map((s, i) => ({ s, i }))
    .filter(({ s }) => s.flags?.ai)
    .sort((a, b) => (a.s.score ?? 1) - (b.s.score ?? 1) || a.i - b.i)
    .map(({ i }) => i)
}

// A mask is hit on its pixels, or on its box until its PNG is decoded
function hits(s: Shape, p: number[], tolerance: number) {
  if (s.shape_type !== 'mask') return contains(s, p, tolerance)
  const m = s.mask ? decodedMask(s.mask) : undefined
  if (!m || s.points.length < 2) return inside(p, outline(s))
  const at = maskPlacement(s, m)
  const col = Math.floor(p[0] - at.x), row = Math.floor(p[1] - at.y)
  return col >= 0 && row >= 0 && col < at.width && row < at.height && m.bits[row * m.width + col] === 1
}

const Labeler = forwardRef<LabelerHandle, Props>(function Labeler({ data, changeClasses, setImages, active, modal, onError, onSeamtag, seamtagStarting }, ref) {
  const { project, images, root } = data
  const [index, setIndex] = useState(0)
  const [query, setQuery] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [sort, setSort] = useState<'name' | 'certainty'>('name')
  const [tool, setTool] = useState<Tool>('polygon')
  // By name, as another app may add, remove or reorder classes. The first one until one is chosen.
  const [activeName, setActiveName] = useState<string>()
  const chosenAt = project.classes.findIndex((c) => c.name === activeName)
  const activeClass = chosenAt >= 0 ? chosenAt : 0
  const setActiveClass = (i: number) => setActiveName(project.classes[i]?.name)
  const [shapes, setShapesState] = useState<Shape[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [meta, setMeta] = useState<OpenImage | null>(null)
  const [format, setFormat] = useState<LabelFormat | null>(null)
  const [showShapes, setShowShapes] = useState(true)
  const [fill, setFill] = useState(0.3)
  const [zoom, setZoom] = useState(1)
  const [cursor, setCursor] = useState<number[] | null>(null)
  const [saved, setSaved] = useState<'saved' | 'saving'>('saved')
  const [loading, setLoading] = useState(false)
  const [busyAi, setBusyAi] = useState(false)
  const [newClass, setNewClass] = useState('')
  const [brushSize, setBrushSize] = useState(28)
  const [batch, setBatch] = useState<{ done: number; total: number } | null>(null)
  // While this window's batch runs. Its last progress can come after its answer, as the main process's
  // messages and answers don't keep their order, and mustn't bring back the Stop button once it's done.
  const batchRunning = useRef(false)
  const [notice, setNotice] = useState('')

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const view = useRef({ scale: 1, x: 0, y: 0 })
  const shapesRef = useRef<Shape[]>([])
  // The open image's shapes as its label file has them, as the window got or sent them: a shape still
  // the same object is still that one in the file, which keeps its long ids' digits
  const onDisk = useRef<Shape[]>([])
  const selectedRef = useRef<number | null>(null)
  const metaRef = useRef<OpenImage | null>(null)
  const formatRef = useRef<LabelFormat | null>(null)
  // The open image's labels are fresh, as the main process says: with no shapes left, it has no label file
  const freshLabels = useRef(false)
  const loadedName = useRef('')
  const alive = useRef(true)
  const history = useRef(new Map<string, { undo: Shape[][]; redo: Shape[][] }>())
  const reviewOnLoad = useRef(false)
  const reviewBack = useRef<string | null>(null)
  const undoOnLoad = useRef(false)
  const overCanvas = useRef(false)
  const draft = useRef<number[][]>([])
  const drag = useRef<Drag | null>(null)
  const mouse = useRef<number[]>([0, 0])
  const space = useRef(false)
  const undo = useRef<Shape[][]>([])
  const redo = useRef<Shape[][]>([])
  const tiles = useRef(new Map<string, HTMLImageElement>())
  const frame = useRef(0)
  const saveTimer = useRef<ReturnType<typeof setTimeout>>()
  const pendingSave = useRef<(() => Promise<void>) | null>(null)
  const inFlight = useRef(0)
  // How many reloads of labels another app changed are on their way: nothing may be edited on the old ones meanwhile
  const reloading = useRef(0)
  const refreshing = useRef(false)
  // A refresh asked for while one runs follows it, so a change made meanwhile isn't missed
  const refreshAgain = useRef(false)
  const imagesRef = useRef(images)
  const indexRef = useRef(index)
  const opts = useRef({ showShapes, fill, classes: project.classes, tool, activeClass, brushSize })
  opts.current = { showShapes, fill, classes: project.classes, tool, activeClass, brushSize }
  selectedRef.current = selected
  imagesRef.current = images
  indexRef.current = index
  const readOnly = readOnlyFormat(format)

  const colorOf = useCallback(
    (label: string) => opts.current.classes.find((c) => c.name === label)?.color || '#999',
    []
  )

  const getTile = (url: string) => {
    const cache = tiles.current
    let img = cache.get(url)
    if (img) {
      cache.delete(url)
      cache.set(url, img) // keep most recently used last
      return img
    }
    img = new Image()
    img.onload = () => invalidate()
    img.src = url
    cache.set(url, img)
    if (cache.size > CACHE_TILES) cache.delete(cache.keys().next().value!)
    return img
  }

  const draw = useCallback(() => {
    frame.current = 0
    const canvas = canvasRef.current
    const m = metaRef.current
    if (!canvas || !canvas.clientWidth) return
    const dpr = devicePixelRatio
    const W = canvas.clientWidth, H = canvas.clientHeight
    if (canvas.width !== W * dpr || canvas.height !== H * dpr) {
      canvas.width = W * dpr
      canvas.height = H * dpr
    }
    const ctx = canvas.getContext('2d', { alpha: false })!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.fillStyle = '#0d0e11'
    ctx.fillRect(0, 0, W, H)
    if (!m) return
    const { scale, x, y } = view.current
    ctx.imageSmoothingEnabled = scale < 2

    const put = (img: HTMLImageElement, ix: number, iy: number, iw: number, ih: number) => {
      if (!img.complete || !img.naturalWidth) return
      const x0 = Math.round(ix * scale + x), y0 = Math.round(iy * scale + y)
      ctx.drawImage(img, x0, y0, Math.round((ix + iw) * scale + x) - x0, Math.round((iy + ih) * scale + y) - y0)
    }
    if (!m.pyramid) {
      put(getTile(fileUrl(m.file)), 0, 0, m.width, m.height)
    } else {
      const { dir, maxLevel, tile } = m.pyramid
      const overview = Math.max(0, Math.ceil(Math.log2(Math.max(m.width, m.height) / tile)))
      const drawLevel = (level: number) => {
        const f = 2 ** (maxLevel - level)
        const c0 = Math.max(0, Math.floor(-x / scale / f / tile))
        const r0 = Math.max(0, Math.floor(-y / scale / f / tile))
        const c1 = Math.min(Math.ceil(m.width / f / tile) - 1, Math.floor((W - x) / scale / f / tile))
        const r1 = Math.min(Math.ceil(m.height / f / tile) - 1, Math.floor((H - y) / scale / f / tile))
        for (let r = r0; r <= r1; r++) {
          for (let c = c0; c <= c1; c++) {
            const img = getTile(fileUrl(`${dir}/${level}/${c}_${r}.jpeg`))
            if (img.naturalWidth) {
              put(img, c * tile * f, r * tile * f, img.naturalWidth * f, img.naturalHeight * f)
            }
          }
        }
      }
      // The whole-image tile fills gaps until sharper tiles arrive
      const coarsest = maxLevel - overview
      const want = Math.min(maxLevel, Math.max(coarsest, maxLevel + Math.ceil(Math.log2(scale * dpr))))
      drawLevel(coarsest)
      if (want !== coarsest) drawLevel(want)
    }

    const toScreen = (p: number[]): [number, number] => [p[0] * scale + x, p[1] * scale + y]
    const { showShapes, fill: alpha } = opts.current
    if (showShapes) {
      shapesRef.current.forEach((s, i) => {
        const type = s.shape_type ?? 'polygon'
        const color = colorOf(s.label)
        const isSel = i === selectedRef.current
        const strength = isSel ? Math.min(1, alpha + 0.15) : alpha
        if (type === 'point' || type === 'points') {
          for (const p of s.points) {
            ctx.beginPath()
            ctx.arc(...toScreen(p), isSel ? 5.5 : 4, 0, 7)
            ctx.fillStyle = color
            ctx.fill()
            ctx.strokeStyle = isSel ? '#fff' : '#0d0e11'
            ctx.lineWidth = isSel ? 2 : 1
            ctx.stroke()
          }
          return
        }
        if (type === 'mask') {
          const decoded = s.mask ? decodeMask(s.mask, invalidate) : null
          if (decoded && s.points.length >= 2) {
            const at = maskPlacement(s, decoded)
            if (at.width > 0 && at.height > 0) {
              // Pixels that fall off the image aren't part of it, as in labelme
              ctx.save()
              ctx.beginPath()
              ctx.rect(...toScreen([0, 0]), m.width * scale, m.height * scale)
              ctx.clip()
              ctx.globalAlpha = strength
              ctx.imageSmoothingEnabled = false
              ctx.drawImage(tintedMask(decoded, color), 0, 0, at.width, at.height, ...toScreen([at.x, at.y]), at.width * scale, at.height * scale)
              ctx.restore()
            }
          }
          if (!isSel) return
        }
        ctx.beginPath()
        if (type === 'circle' && s.points.length >= 2) {
          const [c, e] = s.points
          ctx.arc(...toScreen(c), Math.hypot(e[0] - c[0], e[1] - c[1]) * scale, 0, 7)
        } else {
          outline(s).forEach((p, k) => (k ? ctx.lineTo(...toScreen(p)) : ctx.moveTo(...toScreen(p))))
          if (AREAS.has(type)) ctx.closePath()
        }
        // A mask's pixels are its fill; selected, its box is outlined
        if (AREAS.has(type) && type !== 'mask') {
          ctx.globalAlpha = strength
          ctx.fillStyle = color
          ctx.fill('evenodd')
          ctx.globalAlpha = 1
        }
        // Handles only where Seamline can move them; a shape it leaves alone shows it's picked by its outline
        const handles = isSel && editable(s) && !readOnlyFormat(formatRef.current)
        ctx.strokeStyle = isSel && !handles ? '#fff' : color
        ctx.lineWidth = isSel ? 2.5 : 1.5
        ctx.setLineDash(s.flags?.ai ? [6, 4] : type === 'mask' ? [3, 3] : [])
        ctx.stroke()
        ctx.setLineDash([])
        if (handles) {
          for (const p of outline(s)) {
            const [px, py] = toScreen(p)
            ctx.beginPath()
            ctx.arc(px, py, 4.5, 0, 7)
            ctx.fillStyle = '#fff'
            ctx.fill()
            ctx.strokeStyle = color
            ctx.lineWidth = 2
            ctx.stroke()
          }
        }
      })
    }
    const current = opts.current.classes[opts.current.activeClass]?.color || '#fff'
    // The brush continues a selected shape's class, so it previews in that colour
    const chosen = selectedRef.current === null ? undefined : shapesRef.current[selectedRef.current]
    const brushColor = chosen ? colorOf(chosen.label) : current
    const d = drag.current
    if (d?.kind === 'rect') {
      const [a, b] = toScreen(d.start), [c, e] = toScreen(d.end)
      ctx.strokeStyle = current
      ctx.lineWidth = 2
      ctx.strokeRect(a, b, c - a, e - b)
    }
    const erasing = opts.current.tool === 'eraser'
    if (d?.kind === 'stroke') {
      ctx.beginPath()
      d.points.forEach((p, k) => (k ? ctx.lineTo(...toScreen(p)) : ctx.moveTo(...toScreen(p))))
      if (d.points.length === 1) ctx.lineTo(...toScreen(d.points[0]))
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'
      ctx.lineWidth = d.radius * 2 * scale
      ctx.globalAlpha = 0.55
      ctx.strokeStyle = erasing ? '#fff' : brushColor
      ctx.stroke()
      ctx.globalAlpha = 1
      ctx.lineCap = 'butt'
      ctx.lineJoin = 'miter'
    }
    if ((erasing || opts.current.tool === 'brush') && overCanvas.current && !readOnlyFormat(formatRef.current)) {
      ctx.beginPath()
      ctx.arc(...toScreen(mouse.current), opts.current.brushSize / 2, 0, 7)
      ctx.strokeStyle = erasing ? '#fff' : brushColor
      ctx.lineWidth = 1.5
      ctx.stroke()
    }
    if (draft.current.length) {
      ctx.beginPath()
      draft.current.forEach((p, k) => (k ? ctx.lineTo(...toScreen(p)) : ctx.moveTo(...toScreen(p))))
      ctx.lineTo(...toScreen(mouse.current))
      ctx.strokeStyle = current
      ctx.lineWidth = 2
      ctx.stroke()
      draft.current.forEach((p, k) => {
        ctx.beginPath()
        ctx.arc(...toScreen(p), k === 0 ? 6 : 3.5, 0, 7)
        ctx.fillStyle = k === 0 ? current : '#fff'
        ctx.fill()
      })
    }
  }, [colorOf])

  const invalidate = useCallback(() => {
    if (!frame.current) frame.current = requestAnimationFrame(draw)
  }, [draw])

  const fit = useCallback(() => {
    const canvas = canvasRef.current, m = metaRef.current
    if (!canvas || !m) return
    const pad = 24
    const scale = Math.min((canvas.clientWidth - pad * 2) / m.width, (canvas.clientHeight - pad * 2) / m.height)
    view.current = {
      scale,
      x: (canvas.clientWidth - m.width * scale) / 2,
      y: (canvas.clientHeight - m.height * scale) / 2
    }
    setZoom(scale)
    invalidate()
  }, [invalidate])

  const flush = useCallback(async () => {
    clearTimeout(saveTimer.current)
    const job = pendingSave.current
    pendingSave.current = null
    inFlight.current++
    try {
      if (job) await job()
    } finally {
      inFlight.current--
    }
    setSaved('saved')
  }, [])

  useImperativeHandle(ref, () => ({
    flush,
    current: () => imagesRef.current[indexRef.current]?.name ?? null
  }), [flush])

  // Shown, not edited: labels kept in another format, and labels being read back from disk
  const frozen = () => readOnlyFormat(formatRef.current) || reloading.current > 0

  const setShapes = (next: Shape[], record = true) => {
    if (frozen()) return
    if (record) {
      undo.current.push(shapesRef.current)
      if (undo.current.length > 200) undo.current.shift()
      redo.current = []
    }
    apply(next)
  }

  const apply = (next: Shape[]) => {
    if (frozen()) return
    shapesRef.current = next
    setShapesState(next)
    invalidate()
    const m = metaRef.current
    // Not the selected image: while the next one loads, these shapes still belong to the last
    const name = loadedName.current
    if (!m || !name) return
    // Saved as LabelMe, as Seamline saves every image it can change: a file emptied here counts as one,
    // unless the image had none when the window took its labels, as it has none again then
    const none = !next.length && freshLabels.current
    setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize(next), format: none ? undefined : 'labelme' } : i)))
    setSaved('saving')
    pendingSave.current = () => {
      const at = new Map(onDisk.current.map((s, i) => [s, i]))
      onDisk.current = next
      return window.api.saveLabels(root, name, next, [m.width, m.height], next.map((s) => at.get(s) ?? -1))
        .then((result) => { if (result.changed) void changedElsewhere(name) })
        .catch((e) => onError(`Save failed: ${e.message}`))
    }
    clearTimeout(saveTimer.current)
    saveTimer.current = setTimeout(flush, 400)
  }

  // Anything still being changed: a refresh leaves it be, and its save decides what happens
  const busy = () => !!(pendingSave.current || inFlight.current || drag.current || draft.current.length)

  /**
   * The open image's labels as they are on disk now. Its undo history goes too, unless nothing changed.
   * Nothing can be edited until they're in: an edit on the old labels would be saved over the new ones.
   */
  const reloadLabels = async (name: string) => {
    reloading.current++
    try {
      const loaded = await window.api.loadLabels(root, name)
      // Moved on meanwhile: what this image had here stays out of date, and its next load says so
      if (!alive.current || loadedName.current !== name) return
      void window.api.haveLabels(root, name, loaded.key, loaded.fresh)
      freshLabels.current = loaded.fresh
      setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize(loaded.shapes), format: loaded.format ?? undefined, broken: false } : i)))
      if (loaded.format === formatRef.current && JSON.stringify(loaded.shapes) === JSON.stringify(shapesRef.current)) {
        onDisk.current = shapesRef.current
        return
      }
      clearTimeout(saveTimer.current)
      pendingSave.current = null
      setSaved('saved')
      shapesRef.current = loaded.shapes
      onDisk.current = loaded.shapes
      formatRef.current = loaded.format
      undo.current = []
      redo.current = []
      drag.current = null
      draft.current = []
      setFormat(loaded.format)
      setShapesState(loaded.shapes)
      setSelected(null)
      invalidate()
    } finally {
      reloading.current--
    }
  }

  // Another app, such as Seamtag, saved these labels since the window had them: theirs stay
  const changedElsewhere = async (name: string) => {
    history.current.delete(name)
    if (name !== loadedName.current) {
      setNotice(`The labels of ${name} changed in another app, so your last change to them wasn't saved.`)
      void refresh()
      return
    }
    // Edits made since are on the old labels too
    clearTimeout(saveTimer.current)
    pendingSave.current = null
    drag.current = null
    draft.current = []
    try {
      await reloadLabels(name)
      setNotice("These labels changed in another app, so they were reloaded. Your last change wasn't saved.")
    } catch (e: any) {
      onError(`Could not reload ${name}: ${e.message}`)
    } finally {
      setSaved('saved')
    }
  }

  // Seamtag or another app may have changed the folder: the list follows, the chosen image stays chosen
  const refresh = async () => {
    if (refreshing.current) {
      refreshAgain.current = true
      return
    }
    refreshing.current = true
    try {
      const { images: fresh, changed } = await window.api.refreshProject(root)
      if (!alive.current) return
      const open = loadedName.current
      changed.forEach((name) => name !== open && history.current.delete(name))
      const reload = !!open && changed.includes(open) && !busy()
      // While an edit is on its way, the list shows it rather than what the disk still has
      const keep = open && busy() ? imagesRef.current.find((i) => i.name === open) : undefined
      const next = keep ? fresh.map((i) => (i.name === open ? keep : i)) : fresh
      const wanted = imagesRef.current[indexRef.current]?.name
      const at = next.findIndex((i) => i.name === wanted)
      setImages(() => next)
      setIndex(at >= 0 ? at : Math.max(0, Math.min(indexRef.current, next.length - 1)))
      if (reload) await reloadLabels(open).catch((e) => onError(`Could not reload ${open}: ${e.message}`))
    } catch (e: any) {
      onError(`Could not look at the folder again: ${e.message}`)
    } finally {
      refreshing.current = false
      if (refreshAgain.current && alive.current) {
        refreshAgain.current = false
        void refreshRef.current()
      }
    }
  }
  const refreshRef = useRef(refresh)
  refreshRef.current = refresh

  // Load the chosen image; older requests are dropped if you move on quickly. By name, so a refresh
  // that only moves it in the list leaves it, and what's being done on it, as it is.
  const chosen = images[index]?.name
  useEffect(() => {
    const name = chosen
    if (!name) return
    let stale = false
    setLoading(true)
    // Undo stays with each image, so going back can still take back a decision
    const rememberHistory = () => {
      const last = loadedName.current
      if (!last) return
      history.current.delete(last)
      history.current.set(last, { undo: undo.current, redo: redo.current })
      if (history.current.size > HISTORY_IMAGES) history.current.delete(history.current.keys().next().value!)
    }
    ;(async () => {
      await flush()
      try {
        // Labels last: a big image can take a while, and pre-labelling may write meanwhile
        const m = await window.api.openImage(name)
        if (stale) return
        const loaded: Labels = await window.api.loadLabels(root, name)
        // Not taken: the next load of this image says whether it changed since the window last had it
        if (stale) return
        void window.api.haveLabels(root, name, loaded.key, loaded.fresh)
        rememberHistory()
        // Labels changed by another app since: undoing would bring back what was there before
        if (loaded.changed) {
          history.current.delete(name)
          setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize(loaded.shapes), format: loaded.format ?? undefined, broken: false } : i)))
        }
        const kept = history.current.get(name)
        const fromReview = reviewOnLoad.current
        const undoHere = undoOnLoad.current
        reviewOnLoad.current = false
        undoOnLoad.current = false
        reviewBack.current = fromReview ? loadedName.current : null
        if (loadedName.current !== name) dropTints()
        metaRef.current = m
        formatRef.current = loaded.format
        freshLabels.current = loaded.fresh
        loadedName.current = name
        shapesRef.current = loaded.shapes
        onDisk.current = loaded.shapes
        undo.current = kept?.undo ?? []
        redo.current = kept?.redo ?? []
        draft.current = []
        drag.current = null
        setMeta(m)
        setFormat(loaded.format)
        setShapesState(loaded.shapes)
        setSelected(null)
        fit()
        if (undoHere) stepHistory(undo, redo)
        const first = reviewOrder(shapesRef.current)[0]
        if ((fromReview || undoHere) && first !== undefined) selectForReview(first, shapesRef.current)
      } catch (e: any) {
        if (stale) return
        onError(`Could not open ${name}: ${e.message}`)
        // Nothing on screen, so nothing can be edited or saved under this image's name
        rememberHistory()
        reviewOnLoad.current = false
        undoOnLoad.current = false
        reviewBack.current = null
        metaRef.current = null
        formatRef.current = null
        freshLabels.current = false
        loadedName.current = ''
        shapesRef.current = []
        onDisk.current = []
        undo.current = []
        redo.current = []
        drag.current = null
        setMeta(null)
        setFormat(null)
        setShapesState([])
        setSelected(null)
        invalidate()
      } finally {
        if (!stale) setLoading(false)
      }
    })()
    return () => { stale = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chosen])

  useEffect(invalidate, [showShapes, fill, selected, tool, activeClass, brushSize, project.classes, format, invalidate])
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 5000)
    return () => clearTimeout(timer)
  }, [notice])
  useEffect(() => {
    document.querySelector('.shape.on')?.scrollIntoView({ block: 'nearest' })
  }, [selected])
  useEffect(() => {
    document.querySelector('.file.on')?.scrollIntoView({ block: 'nearest' })
  }, [index])
  useEffect(() =>
    window.api.onPrelabel(({ name, shapes: found, key, done, total }) => {
      if (batchRunning.current) setBatch({ done, total })
      if (!found) return
      history.current.delete(name)
      if (name === loadedName.current) {
        // Anything being edited here wins: it saves over this result
        if (pendingSave.current || drag.current || shapesRef.current.some((s) => !s.flags?.ai)) return
        shapesRef.current = found
        onDisk.current = found
        // Taken with its label file there: emptied, it stays, as when every shape is rejected
        freshLabels.current = false
        if (key) void window.api.haveLabels(root, name, key, false)
        undo.current = []
        redo.current = []
        setShapesState(found)
        setSelected(null)
        invalidate()
      }
      setImages((all) => all.map((i) => (i.name === name ? { ...i, ...summarize(found), format: 'labelme' } : i)))
    }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [invalidate]
  )
  useEffect(() => {
    const canvas = canvasRef.current!
    const observer = new ResizeObserver(() => invalidate())
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [invalidate])
  const shown = useRef(active)
  useEffect(() => {
    if (active) { fit(); canvasRef.current && invalidate() }
    // Back from the Train page: the folder may have changed meanwhile
    if (active && !shown.current) void refreshRef.current()
    shown.current = active
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active])
  // Back from another app, such as Seamtag
  useEffect(() => {
    const onFocus = () => void refreshRef.current()
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [])
  useEffect(() => () => void flush(), [flush])
  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])
  useEffect(() => window.api.onFlush(() => void flush().finally(() => window.api.flushed())), [flush])
  useEffect(() => {
    const broken = images.filter((i) => i.broken).map((i) => i.name)
    if (!broken.length) return
    const list = broken.slice(0, 3).join(', ') + (broken.length > 3 ? ', …' : '')
    onError(`${broken.length} ${broken.length === 1 ? 'label file' : 'label files'} can't be read and won't be changed: ${list}`)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const toImage = (e: { clientX: number; clientY: number }) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const { scale, x, y } = view.current
    return [(e.clientX - rect.left - x) / scale, (e.clientY - rect.top - y) / scale]
  }
  const clampPoint = (p: number[]) => {
    const m = metaRef.current!
    return [Math.max(0, Math.min(m.width, p[0])), Math.max(0, Math.min(m.height, p[1]))]
  }

  const onWheel = (e: React.WheelEvent) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const v = view.current
    const next = Math.max(0.005, Math.min(50, v.scale * Math.exp(-e.deltaY * 0.0022)))
    const px = e.clientX - rect.left, py = e.clientY - rect.top
    view.current = {
      scale: next,
      x: px - ((px - v.x) / v.scale) * next,
      y: py - ((py - v.y) / v.scale) * next
    }
    setZoom(next)
    invalidate()
  }

  // Shapes need a class: another app may have deleted every one
  const newLabel = () => {
    const label = project.classes[activeClass]?.name
    if (!label) setNotice('Add a class first, then draw.')
    return label
  }

  const finishPolygon = () => {
    if (draft.current.length >= 3) {
      const label = newLabel()
      if (!label) return
      const next = [...shapesRef.current, { label, points: draft.current, shape_type: 'polygon' as const, flags: {} }]
      setShapes(next)
      setSelected(next.length - 1)
    }
    draft.current = []
    invalidate()
  }

  // The top shape under the point
  const shapeAt = (p: number[]) => {
    const list = shapesRef.current
    const tolerance = HIT / view.current.scale
    for (let i = list.length - 1; i >= 0; i--) if (hits(list[i], p, tolerance)) return i
    return -1
  }

  const onPointerDown = (e: React.PointerEvent) => {
    // While the next image loads, what's on screen is about to go
    if (!metaRef.current || loadedName.current !== images[index]?.name || reloading.current > 0) return
    canvasRef.current!.setPointerCapture(e.pointerId)
    const p = toImage(e)
    const radius = HIT / view.current.scale
    if (e.button === 1 || e.button === 2 || space.current) {
      drag.current = { kind: 'pan', x: e.clientX, y: e.clientY }
      return
    }
    if (readOnlyFormat(formatRef.current)) {
      // Only looked at: a click picks a shape, a drag pans
      const at = shapeAt(p)
      setSelected(at >= 0 ? at : null)
      drag.current = { kind: 'pan', x: e.clientX, y: e.clientY }
      return
    }
    if (tool === 'polygon') {
      const first = draft.current[0]
      if (first && draft.current.length >= 3 && Math.hypot(p[0] - first[0], p[1] - first[1]) < radius * 1.5) {
        finishPolygon()
      } else {
        draft.current = [...draft.current, clampPoint(p)]
        invalidate()
      }
      return
    }
    if (tool === 'rectangle') {
      const start = clampPoint(p)
      drag.current = { kind: 'rect', start, end: start }
      return
    }
    if (tool === 'brush' || tool === 'eraser') {
      drag.current = { kind: 'stroke', points: [p], radius: brushSize / 2 / view.current.scale }
      invalidate()
      return
    }
    const list = shapesRef.current
    if (selected !== null && list[selected] && editable(list[selected])) {
      const pts = list[selected].points
      const v = pts.findIndex((q) => Math.hypot(p[0] - q[0], p[1] - q[1]) < radius)
      if (v >= 0) {
        if (e.altKey && list[selected].shape_type === 'polygon' && pts.length > 3) {
          setShapes(list.map((s, i) => (i === selected ? { ...accepted(s), points: pts.filter((_, k) => k !== v) } : s)))
          return
        }
        undo.current.push(list)
        redo.current = []
        drag.current = { kind: 'vertex', shape: selected, vertex: v, press: [e.clientX, e.clientY], active: false }
        return
      }
    }
    const at = shapeAt(p)
    if (at >= 0) {
      setSelected(at)
      if (!editable(list[at])) {
        drag.current = { kind: 'hold', shape: at, press: [e.clientX, e.clientY], told: false }
        return
      }
      undo.current.push(list)
      redo.current = []
      drag.current = { kind: 'move', shape: at, x: p[0], y: p[1], press: [e.clientX, e.clientY], active: false }
      return
    }
    setSelected(null)
    drag.current = { kind: 'pan', x: e.clientX, y: e.clientY }
  }

  const onPointerMove = (e: React.PointerEvent) => {
    const p = toImage(e)
    mouse.current = p
    setCursor(p)
    overCanvas.current = true
    const d = drag.current
    if (!d) {
      if (draft.current.length || tool === 'brush' || tool === 'eraser') invalidate()
      return
    }
    if (d.kind === 'hold') {
      // Seamline doesn't move this one: say where it can be moved, once
      if (!d.told && Math.hypot(e.clientX - d.press[0], e.clientY - d.press[1]) >= DRAG_START) {
        d.told = true
        const s = shapesRef.current[d.shape]
        if (s) setNotice(fixedHint(s))
      }
      return
    }
    if ((d.kind === 'vertex' || d.kind === 'move') && !d.active) {
      // A slip of the hand during a click isn't a drag
      if (Math.hypot(e.clientX - d.press[0], e.clientY - d.press[1]) < DRAG_START) return
      d.active = true
    }
    if (d.kind === 'pan') {
      view.current.x += e.clientX - d.x
      view.current.y += e.clientY - d.y
      d.x = e.clientX
      d.y = e.clientY
    } else if (d.kind === 'rect') {
      d.end = clampPoint(p)
    } else if (d.kind === 'stroke') {
      const last = d.points[d.points.length - 1]
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= d.radius * 0.4) d.points.push(p)
    } else if (d.kind === 'vertex') {
      shapesRef.current = shapesRef.current.map((s, i) =>
        i === d.shape ? { ...s, points: s.points.map((q, k) => (k === d.vertex ? clampPoint(p) : q)) } : s
      )
    } else {
      const dx = p[0] - d.x, dy = p[1] - d.y
      d.x = p[0]
      d.y = p[1]
      shapesRef.current = shapesRef.current.map((s, i) =>
        i === d.shape ? { ...s, points: s.points.map((q) => [q[0] + dx, q[1] + dy]) } : s
      )
    }
    invalidate()
  }

  const onPointerUp = () => {
    const d = drag.current
    drag.current = null
    if (!d) return
    if (d.kind === 'stroke') {
      commitStroke(d.points, d.radius)
      invalidate()
    } else if (d.kind === 'rect') {
      const label = Math.abs(d.start[0] - d.end[0]) > 1 && Math.abs(d.start[1] - d.end[1]) > 1 && newLabel()
      if (label) {
        const next = [...shapesRef.current, { label, points: [d.start, d.end], shape_type: 'rectangle' as const, flags: {} }]
        setShapes(next)
        setSelected(next.length - 1)
      }
      invalidate()
    } else if (d.kind === 'vertex' || d.kind === 'move') {
      // A click only selects: looking at an AI shape must not accept it
      if (shapesRef.current === undo.current[undo.current.length - 1]) {
        undo.current.pop()
        return
      }
      apply(shapesRef.current.map((s, i) => (i === d.shape && s.flags?.ai ? accepted(s) : s)))
    }
  }

  const commitStroke = (points: number[][], radius: number) => {
    const m = metaRef.current!
    const list = shapesRef.current
    try {
      const region = strokeRegion(points, radius, 0.5 / view.current.scale, [m.width, m.height])
      if (!region.length) return
      if (tool === 'eraser') {
        if (selected !== null && list[selected] && !editable(list[selected])) return setNotice(fixedHint(list[selected]))
        const result = eraseShapes(list, region, selected)
        if (result.shapes !== list) {
          setShapes(result.shapes)
          setSelected(result.selected)
        }
        return
      }
      // Painting with a shape selected continues that shape's class
      const label = (selected !== null && list[selected]?.label) || project.classes[activeClass]?.name
      if (!label) return
      const at = project.classes.findIndex((c) => c.name === label)
      if (at >= 0) setActiveClass(at)
      setSelected(null)
      setShapes(paintShapes(list, region, label))
    } catch (e: any) {
      onError(`That stroke couldn't be applied, so the shapes are unchanged: ${e.message}`)
    }
  }

  const onDoubleClick = (e: React.MouseEvent) => {
    if (readOnlyFormat(formatRef.current)) return
    if (tool === 'polygon') {
      // The second click of the double-click added a point already
      draft.current = draft.current.slice(0, -1)
      finishPolygon()
      return
    }
    if (selected === null) return
    const s = shapesRef.current[selected]
    if (s?.shape_type !== 'polygon' || !editable(s)) return
    const p = toImage(e)
    let best = -1, bestDist = HIT / view.current.scale
    s.points.forEach((a, i) => {
      const dist = segmentDistance(p, a, s.points[(i + 1) % s.points.length])
      if (dist < bestDist) { best = i; bestDist = dist }
    })
    if (best >= 0) {
      const points = [...s.points]
      points.splice(best + 1, 0, clampPoint(p))
      setShapes(shapesRef.current.map((q, i) => (i === selected ? { ...accepted(q), points } : q)))
    }
  }

  const stepHistory = (from: typeof undo, to: typeof undo) => {
    if (frozen()) return
    const prev = from.current.pop()
    if (!prev) {
      // Right after a review jump, the last thing done was on the image before
      const back = from === undo && reviewBack.current ? images.findIndex((i) => i.name === reviewBack.current) : -1
      if (back >= 0) {
        undoOnLoad.current = true
        setIndex(back)
      }
      return
    }
    to.current.push(shapesRef.current)
    setSelected(null)
    apply(prev)
  }
  const removeSelected = () => {
    if (selected === null || frozen()) return
    setShapes(shapesRef.current.filter((_, i) => i !== selected))
    setSelected(null)
  }
  const chooseClass = (i: number) => {
    setActiveClass(i)
    if (selected !== null) {
      setShapes(shapesRef.current.map((s, k) => (k === selected ? { ...accepted(s), label: project.classes[i].name } : s)))
    }
  }

  // Which images are labelled, by index: it changes with the class list too, such as when another app adds a class
  const classNames = useMemo(() => new Set(project.classes.map((c) => c.name)), [project.classes])
  const labelled = useMemo(() => images.map((i) => isLabelled(i.areas, classNames)), [images, classNames])
  const visible = useMemo(() => {
    const test = FILTERS.find((f) => f.key === filter)!.test
    const list = images
      .map((image, i) => ({ image, i }))
      .filter(({ image }) => image.name.toLowerCase().includes(query.toLowerCase()))
      .filter(({ image, i }) => test(image, labelled[i]))
    if (sort === 'certainty') {
      const rank = (i: ImageItem) => (i.ai ? i.score ?? 1 : 2)
      list.sort((a, b) => rank(a.image) - rank(b.image) || a.i - b.i)
    }
    return list
  }, [images, labelled, query, filter, sort])
  const go = useCallback(
    (delta: number) => {
      const at = visible.findIndex((v) => v.i === index)
      const next = visible[Math.max(0, Math.min(visible.length - 1, at + delta))]
      if (next) setIndex(next.i)
    },
    [visible, index]
  )

  // Bring a shape into view: closer in if it's off screen or too small to judge
  const focusShape = (s: Shape) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const pts = outline(s)
    const x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]))
    const y0 = Math.min(...pts.map((p) => p[1])), y1 = Math.max(...pts.map((p) => p[1]))
    const W = canvas.clientWidth, H = canvas.clientHeight
    const { scale, x, y } = view.current
    const seen = x0 * scale + x >= 0 && y0 * scale + y >= 0 && x1 * scale + x <= W && y1 * scale + y <= H
    const large = Math.max(x1 - x0, y1 - y0) * scale >= 80
    if (seen && large) return
    const fits = (x1 - x0) * scale <= W && (y1 - y0) * scale <= H
    const next = large && fits ? scale : Math.min(8, W / ((x1 - x0 || 1) * 3), H / ((y1 - y0 || 1) * 3))
    view.current = { scale: next, x: W / 2 - ((x0 + x1) / 2) * next, y: H / 2 - ((y0 + y1) / 2) * next }
    setZoom(next)
    invalidate()
  }
  const selectForReview = (i: number, list: Shape[]) => {
    setSelected(i)
    setTool('select')
    focusShape(list[i])
  }
  const nextReviewImage = () => {
    const at = visible.findIndex((v) => v.i === index)
    const order = [...visible.slice(at + 1), ...visible.slice(0, Math.max(at, 0))]
    const next = order.find((v) => v.i !== index && v.image.ai > 0)
    setSelected(null)
    if (!next) return setNotice('Nothing left to review in this list.')
    reviewOnLoad.current = true
    setIndex(next.i)
  }
  const reviewNext = (list: Shape[]) => {
    const first = reviewOrder(list)[0]
    if (first !== undefined) selectForReview(first, list)
    else nextReviewImage()
  }
  // Only once this image's shapes are in: keys can arrive faster than images load
  const reviewReady = () => !drag.current && !reloading.current && loadedName.current === images[index]?.name
  const acceptSelected = () => {
    if (!reviewReady()) return
    const list = shapesRef.current
    if (selected === null || !list[selected]?.flags?.ai) return reviewNext(list)
    const next = list.map((s, i) => (i === selected ? accepted(s) : s))
    setShapes(next)
    reviewNext(next)
  }
  const rejectSelected = () => {
    if (!reviewReady()) return
    const list = shapesRef.current
    if (selected === null || !list[selected]?.flags?.ai) return reviewNext(list)
    const next = list.filter((_, i) => i !== selected)
    setShapes(next)
    reviewNext(next)
  }
  const skipShape = () => {
    if (!reviewReady()) return
    const list = shapesRef.current
    const order = reviewOrder(list)
    if (!order.length) return nextReviewImage()
    selectForReview(order[(order.indexOf(selected ?? -1) + 1) % order.length], list)
  }
  const acceptAll = () => {
    if (!reviewReady()) return
    if (shapesRef.current.some((s) => s.flags?.ai)) {
      setShapes(shapesRef.current.map((s) => (s.flags?.ai ? accepted(s) : s)))
    }
    nextReviewImage()
  }

  useEffect(() => {
    if (!active || modal) return
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLInputElement
      if (target.tagName === 'SELECT' || (target.tagName === 'INPUT' && target.type !== 'range')) return
      if (e.code === 'Space') { space.current = e.type === 'keydown'; e.preventDefault(); return }
      if (e.type !== 'keydown') return
      // Nothing may change the image, tool or shapes under a drag in progress
      if (drag.current && e.key !== 'Escape') return
      if (e.ctrlKey && e.key.toLowerCase() === 'z') {
        e.shiftKey ? stepHistory(redo, undo) : stepHistory(undo, redo)
        return
      }
      if (e.ctrlKey) return
      const k = e.key.toLowerCase()
      if (k === 'v') setTool('select')
      else if (k === 'p') setTool('polygon')
      else if (k === 'r') setTool('rectangle')
      else if (k === 'b') setTool('brush')
      else if (k === 'e') setTool('eraser')
      else if (e.key === '[') setBrushSize((s) => Math.max(4, s - 4))
      else if (e.key === ']') setBrushSize((s) => Math.min(200, s + 4))
      else if (e.key === 'Tab') { e.preventDefault(); skipShape() }
      else if (k === 'w' && !e.repeat) e.shiftKey ? acceptAll() : acceptSelected()
      else if (k === 's' && !e.repeat) rejectSelected()
      else if (k === 'f') setShowShapes((s) => !s)
      else if (k === 'a') go(-1)
      else if (k === 'd') go(1)
      else if (k === '0') fit()
      else if (e.key === 'Enter') finishPolygon()
      else if (e.key === 'Escape') {
        draft.current = []
        if (drag.current?.kind === 'stroke') drag.current = null
        setSelected(null)
        invalidate()
      }
      else if (e.key === 'Backspace' && draft.current.length) { draft.current = draft.current.slice(0, -1); invalidate() }
      else if (e.key === 'Delete' || e.key === 'Backspace') removeSelected()
      else if (/^[1-9]$/.test(k) && project.classes[+k - 1]) chooseClass(+k - 1)
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keyup', onKey)
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('keyup', onKey) }
  })

  const addClass = () => {
    const name = newClass.trim()
    if (!name || project.classes.some((c) => c.name === name)) return
    setActiveName(name)
    setNewClass('')
    changeClasses((classes) => classes.some((c) => c.name === name)
      ? classes
      : [...classes, { name, color: PALETTE[classes.length % PALETTE.length] }])
      .catch((e) => onError(`The classes couldn't be saved: ${e.message}`))
  }

  const latestRun = async () => {
    const runs = await window.api.listRuns()
    const run = runs.find((r) => r.events.some((e) => e.type === 'epoch'))
    if (!run) onError('Train a model first, then it can pre-label images.')
    return run
  }
  const prelabel = async () => {
    if (readOnlyFormat(formatRef.current)) return
    const run = await latestRun()
    if (!run) return
    setBusyAi(true)
    try {
      const name = images[index].name
      const { results } = await window.api.predict(run.name, [name])
      if (!alive.current) return
      if (loadedName.current !== name || drag.current || reloading.current > 0) {
        return setNotice(`You were working on something else, so the result for ${name} was dropped.`)
      }
      // As in a batch, nothing found changes nothing: emptied, the label file would read as emptied by someone
      if (!results[name].length) return setNotice(`The model found nothing on ${name}, so its labels stay as they were.`)
      const kept = shapesRef.current.filter((s) => !s.flags?.ai)
      setSelected(null)
      setShapes([...kept, ...results[name]])
    } catch (e: any) {
      onError(`Pre-label failed: ${e.message}`)
    } finally {
      setBusyAi(false)
    }
  }
  // As in Seamtag, pre-labelling touches only images with no labels yet or AI shapes alone: not those
  // with accepted shapes, nor a LabelMe file someone emptied, as by rejecting every shape. Labels kept
  // in LabelImg's formats are Seamtag's to change.
  const toPrelabel = images.filter((i) => !i.count && !i.broken && (!i.format || (i.format === 'labelme' && i.ai > 0)))
  const prelabelAll = async () => {
    const run = await latestRun()
    if (!run) return
    await flush()
    batchRunning.current = true
    setBatch({ done: 0, total: toPrelabel.length })
    try {
      const result = await window.api.prelabelAll(run.name, toPrelabel.map((i) => i.name))
      setNotice(`${result.stopped ? 'Stopped after' : 'Pre-labelled'} ${result.done} of ${result.total} images.`)
      if (result.error) {
        onError(`Pre-labelling stopped early: ${result.error}`)
      } else if (result.failed.length) {
        onError(`${result.failed.length} ${result.failed.length === 1 ? 'image' : 'images'} failed. First: ${result.failed[0]}`)
      }
    } catch (e: any) {
      onError(`Pre-label failed: ${e.message}`)
    } finally {
      batchRunning.current = false
      setBatch(null)
    }
  }
  const aiCount = shapes.filter((s) => s.flags?.ai).length
  const counts = useMemo(() => {
    const out: Record<string, number> = {}
    shapes.forEach((s) => (out[s.label] = (out[s.label] ?? 0) + 1))
    return out
  }, [shapes])
  const totals = useMemo(() => {
    const out: Record<string, number> = {}
    images.forEach((i) => Object.entries(i.classes).forEach(([k, v]) => (out[k] = (out[k] ?? 0) + v)))
    return out
  }, [images])
  const largest = Math.max(1, ...Object.values(totals))
  const tally = (test: (i: ImageItem, labelled: boolean) => boolean) => images.filter((image, i) => test(image, labelled[i])).length
  const doneShare = images.length ? tally(FILTERS[3].test) / images.length : 0
  const reviewShare = images.length ? tally(FILTERS[2].test) / images.length : 0
  const current = selected !== null && shapes[selected]?.flags?.ai ? shapes[selected] : null
  const fixed = selected !== null && shapes[selected] && !readOnly && !editable(shapes[selected]) ? shapes[selected] : null
  const drawingTool = (t: Tool) => (readOnly ? 'select' : t)

  return (
    <div className="labeler">
      <aside className="panel left">
        <input className="search" placeholder={`Search ${images.length} images`} value={query}
          onChange={(e) => setQuery(e.target.value)} />
        <div className="progress" data-testid="progress">
          <i className="done" style={{ width: `${doneShare * 100}%` }} />
          <i className="review" style={{ width: `${reviewShare * 100}%` }} />
        </div>
        <div className="seg small filters">
          {FILTERS.map((f) => (
            <button key={f.key} className={filter === f.key ? 'on' : ''} onClick={() => setFilter(f.key)}>
              <b>{tally(f.test)}</b>{f.name}
            </button>
          ))}
        </div>
        <label className="sort">Order
          <select value={sort} onChange={(e) => { setSort(e.target.value as typeof sort); e.target.blur() }}>
            <option value="name">Name</option>
            <option value="certainty">Least certain first</option>
          </select>
        </label>
        <div className="files">
          {visible.map(({ image, i }) => (
            <button key={image.name} className={`file ${i === index ? 'on' : ''}`} onClick={() => setIndex(i)} title={image.name}>
              <i className={image.broken ? 'dot broken' : labelled[i] ? 'dot done' : image.ai ? 'dot ai' : 'dot'}
                title={image.broken ? "Its label file can't be read" : undefined} />
              <span className="name">{image.name}</span>
              {readOnlyFormat(image.format) && <span className="format" title="Kept in this format: edit it in Seamtag">{image.format === 'voc' ? 'VOC' : image.format === 'yolo' ? 'YOLO' : 'CreateML'}</span>}
              {image.ai > 0 && image.score !== null && <span className="certainty">{percent(image.score)}</span>}
              {image.count > 0 && <span className="muted">{image.count}</span>}
            </button>
          ))}
          {!visible.length && <p className="muted pad">No images match.</p>}
        </div>
        {batch ? (
          <button className="ghost batch" onClick={() => window.api.stopPrelabel()}>
            <i style={{ width: `${(batch.done / Math.max(1, batch.total)) * 100}%` }} />
            <X size={15} /> Stop · {batch.done} / {batch.total}
          </button>
        ) : (
          <button className="ghost batch" onClick={prelabelAll} disabled={!toPrelabel.length || busyAi}
            title="Run the newest model on every image with no label file or only AI shapes">
            <Sparkles size={15} /> Pre-label {toPrelabel.length} {toPrelabel.length === 1 ? 'image' : 'images'}
          </button>
        )}
      </aside>

      <main className="stage">
        <div className="toolbar">
          <div className="seg">
            <button className={tool === 'select' || readOnly ? 'on' : ''} onClick={() => setTool('select')} title="Select and edit (V)"><MousePointer2 size={16} /></button>
            <button className={drawingTool(tool) === 'polygon' ? 'on' : ''} onClick={() => setTool('polygon')} disabled={readOnly} title="Polygon (P)"><Hexagon size={16} /></button>
            <button className={drawingTool(tool) === 'rectangle' ? 'on' : ''} onClick={() => setTool('rectangle')} disabled={readOnly} title="Rectangle (R)"><Square size={16} /></button>
            <button className={drawingTool(tool) === 'brush' ? 'on' : ''} onClick={() => setTool('brush')} disabled={readOnly} title="Brush: paint a class, joining shapes it touches (B)"><Paintbrush size={16} /></button>
            <button className={drawingTool(tool) === 'eraser' ? 'on' : ''} onClick={() => setTool('eraser')} disabled={readOnly} title="Eraser: cut from the selected shape, or anything it touches (E)"><Eraser size={16} /></button>
          </div>
          <button className="icon" onClick={() => stepHistory(undo, redo)} disabled={readOnly} title="Undo (Ctrl+Z)"><Undo2 size={16} /></button>
          <button className="icon" onClick={() => stepHistory(redo, undo)} disabled={readOnly} title="Redo (Ctrl+Shift+Z)"><Redo2 size={16} /></button>
          <button className="icon" onClick={fit} title="Fit image (0)"><Maximize size={16} /></button>
          <button className="icon" onClick={() => setShowShapes(!showShapes)} title="Show or hide shapes (F)">
            {showShapes ? <Eye size={16} /> : <EyeOff size={16} />}
          </button>
          <label className="slider">Fill
            <input type="range" min={0} max={0.8} step={0.05} value={fill} onChange={(e) => setFill(+e.target.value)} />
          </label>
          {(tool === 'brush' || tool === 'eraser') && !readOnly && (
            <label className="slider" title="[ and ] change it">Size
              <input type="range" min={4} max={200} step={4} value={brushSize} onChange={(e) => setBrushSize(+e.target.value)} />
            </label>
          )}
          <div className="spacer" />
          <button className="ghost" onClick={prelabel} disabled={busyAi || !!batch || readOnly}>
            <Sparkles size={15} /> {busyAi ? 'Predicting…' : 'Pre-label'}
          </button>
        </div>
        <div className="canvas-wrap">
          <canvas ref={canvasRef} data-testid="canvas" className={`tool-${drawingTool(tool)}`} onWheel={onWheel}
            onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp}
            onPointerLeave={() => { overCanvas.current = false; invalidate() }}
            onDoubleClick={onDoubleClick} onContextMenu={(e) => e.preventDefault()} />
          {loading && <div className="loading">Preparing image…</div>}
          {notice && <div className="loading notice">{notice}</div>}
          {aiCount > 0 && !readOnly && (
            <div className="review-bar" data-testid="review-bar" onMouseDown={(e) => e.preventDefault()}>
              <span>
                <b>{aiCount}</b> AI {aiCount === 1 ? 'shape' : 'shapes'}
                {current?.score !== undefined && <> · this one <b className="certainty">{current ? percent(current.score!) : ''}</b></>}
              </span>
              <button className="ghost" onClick={acceptSelected}><Check size={15} /> {current ? 'Accept' : 'Review'} <kbd>W</kbd></button>
              <button className="ghost" onClick={rejectSelected} disabled={!current}><X size={15} /> Reject <kbd>S</kbd></button>
              <button className="ghost" onClick={skipShape} disabled={aiCount < 2 && !!current}><SkipForward size={15} /> Skip <kbd>Tab</kbd></button>
              <button className="ghost" onClick={acceptAll}><CheckCheck size={15} /> Accept all <kbd>⇧W</kbd></button>
            </div>
          )}
          {readOnly && format && (
            <div className="review-bar read-only" data-testid="read-only-bar" onMouseDown={(e) => e.preventDefault()}>
              <span><Lock size={14} /> These labels are {BOXES[format]}. Seamline trains on them; edit them in Seamtag, which keeps the format.</span>
              <button className="ghost" onClick={onSeamtag} disabled={seamtagStarting}>
                <ExternalLink size={15} /> {seamtagStarting ? 'Starting Seamtag…' : 'Label in Seamtag'}
              </button>
            </div>
          )}
          {!images.length && <div className="loading">This folder has no images.</div>}
        </div>
        <footer className="status">
          <span>{images[index]?.name}</span>
          {meta && <span>{meta.width} × {meta.height}</span>}
          <span>{Math.round(zoom * 1000) / 10}%</span>
          {cursor && meta && <span>{Math.round(cursor[0])}, {Math.round(cursor[1])}</span>}
          <div className="spacer" />
          <span data-testid="save-state">{saved === 'saved' ? 'Saved' : 'Saving…'}</span>
          <span className="muted">A/D image · 1–9 class · B brush · E eraser · Space pan</span>
        </footer>
      </main>

      <aside className="panel right">
        <h3>Classes <span title="Shapes on this image · accepted shapes in the whole project">here · all</span></h3>
        <div className="classes">
          {project.classes.map((c, i) => (
            <button key={c.name} className={`class ${i === activeClass ? 'on' : ''}`} onClick={() => chooseClass(i)}>
              <i style={{ background: c.color }} /><span className="name">{c.name}</span>
              <span className="muted">{counts[c.name] ?? 0} · {totals[c.name] ?? 0}</span><kbd>{i + 1}</kbd>
              <em style={{ width: `${((totals[c.name] ?? 0) / largest) * 100}%`, background: c.color }} />
            </button>
          ))}
        </div>
        <div className="add">
          <input placeholder="New class" value={newClass} onChange={(e) => setNewClass(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && addClass()} />
          <button className="icon" onClick={addClass} title="Add class"><Plus size={16} /></button>
        </div>
        <h3>Shapes on this image <span className="muted">{shapes.length}</span></h3>
        <div className="shapes">
          {shapes.map((s, i) => (
            <button key={i} className={`shape ${i === selected ? 'on' : ''}`} onClick={() => { setSelected(i); setTool('select') }}
              title={!readOnly && !editable(s) ? fixedHint(s) : undefined}>
              <i style={{ background: colorOf(s.label) }} />
              {/* A label that isn't a class says so under it, leaving the row's width to the label */}
              <span className="label">
                <span className="name">{s.label}</span>
                {!classNames.has(s.label) && (
                  <small className="not-class" title={hasArea(s) ? "Its label isn't one of the classes, so it won't train until the label is added as a class" : undefined}>
                    not a class
                  </small>
                )}
              </span>
              {s.flags?.ai && <span className="tag">AI{s.score !== undefined ? ` ${percent(s.score)}` : ''}</span>}
              {s.locked && <Lock size={12} className="muted" aria-label="locked" />}
              <span className="muted kind">{kindOf(s)}</span>
            </button>
          ))}
          {!shapes.length && <p className="muted pad">{readOnly ? 'No boxes in this file.' : 'Nothing yet. Click on the image to start a polygon.'}</p>}
        </div>
        {fixed && <p className="hint" data-testid="fixed-hint">{fixedHint(fixed)}</p>}
        {selected !== null && !readOnly && (
          <button className="ghost danger" onClick={removeSelected}><Trash2 size={15} /> Delete shape</button>
        )}
      </aside>
    </div>
  )
})

export default Labeler
