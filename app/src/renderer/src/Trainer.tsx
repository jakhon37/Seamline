import { Copy, ExternalLink, PackageOpen, Play, Plus, RefreshCw, Square, Table2, Timer } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { FORMAT_NAMES, type LabelFormat } from '../../shared/formats'
import { isLabelled, notClasses } from '../../shared/labels'
import {
  architecture, ARCHITECTURES, AUGMENTATIONS, DECODER_WIDTHS, ENCODERS, fitArchitecture, fromRun, LOSSES, PRESETS, problem,
  SCHEDULERS, type Size, sizeOf, takes, type TrainOptions, weightsOf
} from '../../shared/models'
import { type Estimate, type Exported, type LabelClass, type OpenProject, PALETTE, type Run, type TrainEvent } from './types'

const DEFAULTS: TrainOptions = {
  ...PRESETS.balanced.options,
  encoder_weights: 'imagenet', epochs: 30, batch_size: 8, image_size: 512, learning_rate: 0.0001, val_split: 0.2, tile: 1024,
  num_workers: 4
}
const SIZES: Size[] = ['fast', 'balanced', 'accurate', 'custom']
// Shapes with no area, which a segmentation mask can't hold
const NO_AREA: Record<string, [string, string]> = {
  line: ['line', 'lines'], linestrip: ['line strip', 'line strips'], point: ['point', 'points'], points: ['point set', 'point sets']
}

type Props = {
  data: OpenProject
  active: boolean
  onError: (m: string) => void
  onTraining: (on: boolean) => void
  /** Change the class list, from the list as it is when the change's turn comes */
  changeClasses: (change: (classes: LabelClass[]) => LabelClass[]) => Promise<void>
  /** Open the folder in Seamtag */
  onSeamtag: () => void
  /** Seamtag is starting, so it isn't started again */
  seamtagStarting: boolean
}
type Series = { name: string; color: string; values: number[] }

const leftOut = (counts: Record<string, number>) =>
  Object.entries(counts).filter(([, n]) => n > 0).map(([type, n]) => {
    const [one, many] = NO_AREA[type] ?? [`${type} shape`, `${type} shapes`]
    return `${n} ${n === 1 ? one : many}`
  }).join(', ')
// Only the formats the folder has
const formatList = (counts: Record<string, number>) =>
  Object.entries(counts).filter(([, n]) => n > 0).map(([f, n]) => `${FORMAT_NAMES[f as LabelFormat] ?? f} ${n}`).join(' · ')
// The labels named, most used first; the rest are counted
const NAMED = 5
const untrained = ({ images, labels }: ReturnType<typeof notClasses>) => {
  const one = labels.length === 1
  const named = labels.slice(0, NAMED).map(([label, n], i) => `${label} (${n}${i ? '' : n === 1 ? ' shape' : ' shapes'})`).join(', ')
  const more = labels.length > NAMED ? ` and ${labels.length - NAMED} more` : ''
  return `${images} ${images === 1 ? 'image uses' : 'images use'} ${one ? "a label that isn't a class" : "labels that aren't classes"} yet, ` +
    `so ${one ? 'it' : 'they'} won't train: ${named}${more}.`
}

function Chart({ title, series, max }: { title: string; series: Series[]; max?: number }) {
  const W = 600, H = 200, L = 40, B = 20, T = 8
  const all = series.flatMap((s) => s.values)
  const top = max ?? Math.max(...all, 1e-6) * 1.1
  const count = Math.max(...series.map((s) => s.values.length))
  const n = Math.max(count, 2)
  const px = (i: number) => L + (i / (n - 1)) * (W - L - 8)
  const py = (v: number) => T + (1 - Math.min(v, top) / top) * (H - T - B)
  return (
    <section className="card chart">
      <header>
        <h3>{title}</h3>
        <div className="legend">
          {series.map((s) => (
            <span key={s.name}><i style={{ background: s.color }} />{s.name}{' '}
              <b>{s.values.length ? s.values[s.values.length - 1].toFixed(3) : '–'}</b></span>
          ))}
        </div>
      </header>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={L} x2={W - 8} y1={py(top * f)} y2={py(top * f)} className="grid" />
            {count > 0 && <text x={L - 6} y={py(top * f) + 4} textAnchor="end">{(top * f).toFixed(2)}</text>}
          </g>
        ))}
        {count > 0
          ? <text x={W - 8} y={H - 4} textAnchor="end">epoch {count}</text>
          : <text x={(L + W - 8) / 2} y={py(top * 0.5) - 8} textAnchor="middle">No epochs yet</text>}
        {series.map((s) => (
          <g key={s.name}>
            <polyline fill="none" stroke={s.color} strokeWidth={2}
              points={s.values.map((v, i) => `${px(i)},${py(v)}`).join(' ')} />
            {s.values.length === 1 && <circle cx={px(0)} cy={py(s.values[0])} r={3.5} fill={s.color} />}
          </g>
        ))}
      </svg>
    </section>
  )
}

// Progress bars redraw with \r and would bury the useful lines
const cleanLog = (text: string) =>
  text.split(/[\r\n]+/).filter((l) => l.trim() && !/\d+%\|/.test(l)).map((l) => l + '\n').join('')

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="field"><span>{label}</span>{children}</label>
}

const millions = (n: number) => `${(n / 1e6).toFixed(1)} M`
const gigabytes = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`
const milliseconds = (ms: number) => (ms >= 100 ? `${Math.round(ms)} ms` : `${ms.toFixed(1)} ms`)
// Runs are named by when they started: 20261120_101500
const runDate = (name: string) => name.replace(/^(\d{4})(\d\d)(\d\d)_(\d\d)(\d\d)\d\d$/, '$1-$2-$3 $4:$5')
const widthName = (w: number) => (w === 1 ? '1× (its own)' : `${w}×`)

/** A run's network in a few words: what differs from the architecture's own is named */
function describe(options: any) {
  if (!options) return 'Unknown model'
  const parts = [options.architecture, options.encoder]
  const own = architecture(options.architecture)?.depth
  if (options.encoder_depth != null && options.encoder_depth !== own) parts.push(`depth ${options.encoder_depth}`)
  if (options.decoder_width != null && options.decoder_width !== 1) parts.push(`decoder ${options.decoder_width}×`)
  if (options.decoder_attention_type) parts.push(options.decoder_attention_type === 'scse' ? 'scSE' : options.decoder_attention_type)
  return parts.filter(Boolean).join(' · ')
}

// An error from the main process, without the wrapping Electron gives it on the way
const reason = (e: any) => String(e?.message ?? e).replace(/^Error invoking remote method '[^']*': (Error: )?/, '')

// Estimates already made, by what was asked: the machine doesn't change while the app runs
const estimated = new Map<string, Estimate>()
type Shown = { key: string; estimate?: Estimate; error?: string; timing?: boolean }

/**
 * The engine's estimate for the settings, kept current as they change: sizes first, as they take a moment, then the time
 * per tile when `timing`. One at a time, the last settings asked for next; `again` asks anew when it changes.
 */
function useEstimate(request: Record<string, unknown> | null, timing: boolean, again: unknown) {
  const key = request ? JSON.stringify(request) : null
  const [shown, setShown] = useState<Shown | null>(null)
  const wanted = useRef<{ key: string | null; timing: boolean }>({ key: null, timing })
  const busy = useRef(false)
  const pump = useRef<() => Promise<void>>()
  pump.current = async () => {
    const job = wanted.current
    if (busy.current || !job.key) return
    busy.current = true
    const k = job.key
    const show = (s: Shown) => { if (wanted.current.key === k) setShown(s) }
    const ask = async (time: boolean) => {
      const cached = estimated.get(`${time}${k}`)
      if (cached) return cached
      const answer = await window.api.estimateModel(JSON.parse(k), time, null)
      // A time taken while a run had the CPU isn't kept
      if (!answer.busy) estimated.set(`${time}${k}`, answer)
      return answer
    }
    try {
      const sizes = estimated.get(`true${k}`) ?? await ask(false)
      show({ key: k, estimate: sizes, timing: job.timing && sizes.ms_per_tile == null })
      if (job.timing && sizes.ms_per_tile == null && wanted.current.key === k) show({ key: k, estimate: await ask(true) })
    } catch (e: any) {
      show({ key: k, error: reason(e) })
    } finally {
      busy.current = false
      if (wanted.current !== job) pump.current?.()
    }
  }
  useEffect(() => {
    wanted.current = { key, timing }
    if (!key) return
    const timer = setTimeout(() => pump.current?.(), 300)
    return () => clearTimeout(timer)
  }, [key, timing, again])
  return { shown, current: !!shown && shown.key === key }
}

export default function Trainer({ data, active, onError, onTraining, changeClasses, onSeamtag, seamtagStarting }: Props) {
  const [options, setOptions] = useState<TrainOptions>(DEFAULTS)
  // Chosen, or Custom once an advanced setting is changed by hand
  const [size, setSize] = useState<Size>('balanced')
  const [advanced, setAdvanced] = useState(false)
  const [comparing, setComparing] = useState(false)
  const [runs, setRuns] = useState<Run[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [running, setRunning] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [log, setLog] = useState('')
  const [saved, setSaved] = useState('')
  // The run whose output is in `log`: any other run shows what it saved
  const streamed = useRef<string | null>(null)
  const [check, setCheck] = useState<any>(null)
  const [checking, setChecking] = useState(false)
  const stored = (key: string, fallback: string) => {
    try { return localStorage.getItem(key) ?? fallback } catch { return fallback }
  }
  const [where, setWhere] = useState(stored('trainWhere', 'local'))
  const [host, setHost] = useState(stored('trainHost', ''))
  const [dir, setDir] = useState(stored('trainDir', 'seamline'))
  const [remote, setRemote] = useState<string>('')
  const [there, setThere] = useState<Shown | null>(null)
  const [exporting, setExporting] = useState<string | null>(null)
  const [exported, setExported] = useState<Record<string, Exported>>({})
  const logRef = useRef<HTMLPreElement>(null)

  const set = (key: keyof TrainOptions, value: unknown) => setOptions((o) => ({ ...o, [key]: value }))
  // A setting a model size sets, changed by hand
  const tune = (key: keyof TrainOptions, value: unknown) => {
    setSize('custom')
    setOptions((o) => (key === 'architecture' ? fitArchitecture(o, value as string, o.architecture) : { ...o, [key]: value }))
  }
  const choose = (chosen: Size) => {
    if (chosen === 'custom') return setAdvanced(true)
    setSize(chosen)
    setOptions((o) => ({ ...o, ...PRESETS[chosen].options }))
  }
  // A run's settings back in the form, to train a variant of it, and where it trained
  const duplicate = (run: Run) => {
    const copied = fromRun(run.options, DEFAULTS)
    setOptions(copied)
    setSize(sizeOf(copied))
    const target = run.options?.target
    if (target?.host) {
      setWhere('ssh')
      setHost(target.host)
      setDir(target.dir ?? dir)
    } else {
      setWhere('local')
    }
  }
  const reload = async () => {
    const list = await window.api.listRuns()
    setRuns(list)
    setCurrent((c) => c ?? list[0]?.name ?? null)
  }
  // Asked rather than assumed: each folder mounts a new Trainer, and another run can outlast the one that ended
  const syncRunning = () => window.api.activeRun().then((r) => {
    if (r) streamed.current = r
    setRunning(r)
    onTraining(!!r)
  })

  useEffect(() => {
    reload()
    syncRunning()
    const offLog = window.api.onLog((t) => setLog((l) => (l + cleanLog(t)).slice(-20000)))
    const offEvents = window.api.onEvents(({ run, events }) =>
      setRuns((all) => all.map((r) => (r.name === run ? { ...r, events } : r)))
    )
    const offExit = window.api.onExit(() => { syncRunning(); reload() })
    return () => { offLog(); offEvents(); offExit() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useEffect(() => { if (active) reload() /* eslint-disable-next-line */ }, [active])
  useEffect(() => {
    setSaved('')
    if (!current || current === streamed.current) return
    let live = true
    window.api.runLog(current).then((text) => { if (live) setSaved(cleanLog(text).slice(-20000)) }).catch(() => {})
    return () => { live = false }
  }, [current, running])
  // A hidden panel can't scroll, so opening the tab counts as new text
  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight)
  }, [log, saved, active])

  const runCheck = async () => {
    setChecking(true)
    try {
      setCheck(await window.api.checkDataset(options.val_split, options.tile))
    } catch (e: any) {
      onError(`Dataset check failed: ${e.message}`)
    } finally {
      setChecking(false)
    }
  }
  // Labels left out of training go on the class list, so the next run trains on them. A dataset check
  // on screen is run again, as what it says has changed.
  const addClasses = async (labels: string[]) => {
    try {
      await changeClasses((classes) => {
        const names = labels.filter((l) => !classes.some((c) => c.name === l))
        const at = classes.length
        return names.length ? [...classes, ...names.map((name, i) => ({ name, color: PALETTE[(at + i) % PALETTE.length] }))] : classes
      })
    } catch (e: any) {
      return onError(`The classes couldn't be saved: ${e.message}`)
    }
    if (check) await runCheck()
  }
  // The model as ONNX beside the run, where Seamtag's pre-label menu finds it
  const exportRun = async (name: string) => {
    setExporting(name)
    try {
      const result = await window.api.exportModel(name)
      setExported((all) => ({ ...all, [name]: result }))
      await reload()
    } catch (e: any) {
      onError(`Export failed: ${e.message}`)
    } finally {
      setExporting(null)
    }
  }
  const testRemote = async () => {
    setRemote('Connecting…')
    try {
      const r = await window.api.checkRemote(host, dir)
      const [name, free, total] = r.gpu.split(', ').map((part) => part.trim())
      const memory = total ? ` · ${(parseInt(free) / 1024).toFixed(1)} of ${(parseInt(total) / 1024).toFixed(1)} GB memory free` : ''
      setRemote(`${name || 'No GPU found'}${memory} · ${r.gpuJobs} GPU jobs · ${(r.freeKb / 1048576).toFixed(1)} GB free on disk`)
    } catch (e: any) {
      setRemote(`Failed: ${e.message}`)
    }
  }
  const start = async () => {
    setStarting(true)
    setLog('')
    try {
      localStorage.setItem('trainWhere', where)
      localStorage.setItem('trainHost', host)
      localStorage.setItem('trainDir', dir)
      const run = await window.api.startTraining(options, where === 'ssh' ? { host, dir } : null)
      streamed.current = run
      setRunning(run)
      onTraining(true)
      setCurrent(run)
      await reload()
    } catch (e: any) {
      onError(`Could not start training: ${e.message}`)
    } finally {
      setStarting(false)
    }
  }

  // The model the form describes, as it would train here or on the GPU machine. Settings that can't train aren't estimated.
  const blocked = problem(options)
  const request = active && !blocked
    ? {
        architecture: options.architecture, encoder: options.encoder, encoder_depth: options.encoder_depth,
        decoder_width: options.decoder_width, decoder_attention_type: options.decoder_attention_type,
        image_size: options.image_size, batch_size: options.batch_size, num_classes: data.project.classes.length + 1
      }
    : null
  const { shown, current: fresh } = useEstimate(request, where === 'local', running)
  const estimate = shown?.estimate
  const thereKey = request ? JSON.stringify({ ...request, host, dir }) : null
  const timedThere = there?.key === thereKey ? there : null
  // Timed on the GPU machine itself, only when asked: it sends the code there first, as training does
  const timeThere = async () => {
    const key = thereKey!
    setThere({ key, timing: true })
    try {
      setThere({ key, estimate: await window.api.estimateModel(request, true, { host, dir }) })
    } catch (e: any) {
      setThere({ key, error: reason(e) })
    }
  }
  const info = architecture(options.architecture) ?? architecture('Unet')!
  const weights = weightsOf(options.encoder)
  const hereTime = estimate?.ms_per_tile != null
    ? `${milliseconds(estimate.ms_per_tile)} · ${estimate.device}`
    : estimate?.busy ? 'after the run' : shown?.timing ? 'timing…' : '–'
  const thereTime = timedThere?.estimate?.ms_per_tile != null
    ? `${milliseconds(timedThere.estimate.ms_per_tile)} · ${timedThere.estimate.device}`
    : timedThere?.timing ? 'timing…' : '–'

  // Each run's network and costs as measured when it started, and how well it did. Runs from before runs measured
  // themselves have no costs.
  const compared = useMemo(() => runs.map((r) => {
    const cost = r.events.find((e) => e.type === 'model')
    const done = r.events.filter((e) => e.type === 'epoch')
    return {
      name: r.name, network: describe(r.options), input: cost?.image_size ?? r.options?.image_size ?? null,
      params: cost?.params ?? null, ms: cost?.ms_per_tile ?? null, device: cost?.device ?? null,
      iou: done.length ? Math.max(...done.map((e) => e.iou)) : null
    }
  }), [runs])
  const topIou = Math.max(...compared.map((row) => row.iou ?? -1))

  const selected = runs.find((r) => r.name === current)
  const events: TrainEvent[] = selected?.events ?? []
  const epochs = useMemo(() => events.filter((e) => e.type === 'epoch'), [events])
  const failure = events.find((e) => e.type === 'error')
  const prepared = events.find((e) => e.type === 'prepared')
  const last = epochs[epochs.length - 1]
  const best = epochs.length ? epochs.reduce((a, b) => (b.iou > a.iou ? b : a)) : null
  // As training counts them: with an accepted shape that has an area and a class's label. Shapes with
  // other labels are named beside the Train button, from the image list, before any check is run.
  const classNames = useMemo(() => new Set(data.project.classes.map((c) => c.name)), [data.project.classes])
  const labelled = useMemo(() => data.images.filter((i) => isLabelled(i.areas, classNames)).length, [data.images, classNames])
  const missing = useMemo(() => notClasses(data.images, classNames), [data.images, classNames])
  const isRunning = !!running && running === current
  // Shown from the project folder, where it is
  const model = selected && (exported[selected.name]?.model ?? (selected.exported ? `${data.root}/.myseg/runs/${selected.name}/model.onnx` : null))
  const shownModel = model && model.startsWith(data.root) ? model.slice(data.root.length + 1) : model

  return (
    <div className="trainer">
      <aside className="panel left wide">
        <h3>New training run</h3>
        <div className="field">
          <span id="model-size">Model size</span>
          <div className="seg small sizes" role="radiogroup" aria-labelledby="model-size">
            {SIZES.map((s) => (
              <button key={s} role="radio" aria-checked={size === s} className={size === s ? 'on' : ''} onClick={() => choose(s)}
                title={s === 'custom' ? 'Your own settings, under Advanced' : `${PRESETS[s].options.architecture} on ${PRESETS[s].options.encoder}`}>
                {s === 'custom' ? 'Custom' : PRESETS[s].label}
              </button>
            ))}
          </div>
        </div>
        <div className={`estimate ${fresh || blocked ? '' : 'stale'}`} data-testid="estimate">
          <div><span>Network</span><b>{describe(options)}</b></div>
          {blocked ? (
            <p className="warn" data-testid="problem">{blocked}</p>
          ) : shown?.error && fresh ? (
            <p className="warn">It can't be estimated: {shown.error}</p>
          ) : (
            <>
              <div><span>Parameters</span><b data-testid="params">{estimate ? millions(estimate.params) : '…'}</b></div>
              <div><span>Work a tile</span><b>{estimate ? `${estimate.gflops.toFixed(1)} GFLOPs` : '…'}</b></div>
              <div title={where === 'ssh'
                ? 'Weights, gradients, optimiser state and the tensors kept for the backward pass, at 16 bits where mixed precision keeps them so'
                : 'Weights, gradients, optimiser state and the tensors kept for the backward pass, in 32-bit floats'}>
                <span>{where === 'ssh' ? 'GPU memory' : 'Memory'} at batch {options.batch_size}</span>
                <b>{estimate ? `≈ ${gigabytes(where === 'ssh' ? estimate.memory.mixed : estimate.memory.float32)}` : '…'}</b>
              </div>
              {where === 'local'
                ? <div><span>Time a tile here</span><b data-testid="time-here">{hereTime}</b></div>
                : <div><span>Time a tile there</span><b data-testid="time-there">{thereTime}</b></div>}
            </>
          )}
        </div>
        {where === 'ssh' && !blocked && (
          <>
            <button className="ghost" onClick={timeThere} disabled={!host || !request || !!timedThere?.timing}>
              <Timer size={15} /> {timedThere?.timing ? `Timing on ${host}…` : `Time a tile on ${host || 'the GPU machine'}`}
            </button>
            {timedThere?.error && <p className="warn">It couldn't be timed there: {timedThere.error}</p>}
          </>
        )}
        <div className="row">
          <Field label="Epochs"><input type="number" min={1} value={options.epochs} onChange={(e) => set('epochs', +e.target.value)} /></Field>
          <Field label="Batch"><input type="number" min={info.minBatch} value={options.batch_size} onChange={(e) => set('batch_size', +e.target.value)} /></Field>
        </div>
        <div className="row">
          <Field label="Input size"><input type="number" step={32} min={info.minSize} value={options.image_size} onChange={(e) => set('image_size', +e.target.value)} /></Field>
          <Field label="Learning rate"><input type="number" step={0.0001} min={0} value={options.learning_rate} onChange={(e) => set('learning_rate', +e.target.value)} /></Field>
        </div>
        <div className="row">
          <Field label="Validation share"><input type="number" step={0.05} min={0.05} max={0.5} value={options.val_split} onChange={(e) => set('val_split', +e.target.value)} /></Field>
          <Field label="Tile size"><input type="number" step={256} min={256} value={options.tile} onChange={(e) => set('tile', +e.target.value)} /></Field>
        </div>
        <label className="check">
          <input type="checkbox" checked={options.encoder_weights === 'imagenet'}
            onChange={(e) => set('encoder_weights', e.target.checked ? 'imagenet' : null)} />
          Start from ImageNet weights
        </label>
        {options.encoder_weights === 'imagenet' && weights && (
          <p className="muted note" data-testid="weights">{options.encoder}'s ImageNet weights: {weights.weights}, {weights.licence}</p>
        )}
        <details className="advanced" open={advanced} onToggle={(e) => setAdvanced(e.currentTarget.open)}>
          <summary>Advanced</summary>
          <Field label="Architecture">
            <select value={options.architecture} onChange={(e) => tune('architecture', e.target.value)}>
              {ARCHITECTURES.map((a) => <option key={a}>{a}</option>)}
            </select>
          </Field>
          <Field label="Encoder">
            <select value={options.encoder} onChange={(e) => tune('encoder', e.target.value)}>
              {ENCODERS.map((name) => (
                <option key={name} disabled={!takes(options.architecture, name)}>{name}</option>
              ))}
            </select>
          </Field>
          <div className="row">
            <Field label="Encoder depth">
              <select value={options.encoder_depth} onChange={(e) => tune('encoder_depth', +e.target.value)}>
                {info.depths.map((d) => <option key={d} value={d}>{d === info.depth ? `${d} (its own)` : d}</option>)}
              </select>
            </Field>
            <Field label="Decoder width">
              <select value={options.decoder_width} disabled={!info.decoderWidth} onChange={(e) => tune('decoder_width', +e.target.value)}>
                {DECODER_WIDTHS.map((w) => (
                  <option key={w} value={w} disabled={!!options.decoder_attention_type && w < (info.attentionMinWidth ?? 0)}>
                    {widthName(w)}
                  </option>
                ))}
              </select>
            </Field>
          </div>
          <Field label="Attention">
            <select value={options.decoder_attention_type ?? ''} disabled={!info.attention.length}
              onChange={(e) => tune('decoder_attention_type', e.target.value || null)}>
              <option value="">None</option>
              {info.attention.map((a) => (
                <option key={a} value={a} disabled={options.decoder_width < (info.attentionMinWidth ?? 0)}>
                  {a === 'scse' ? 'scSE, on channels and space' : a}
                </option>
              ))}
            </select>
          </Field>
          <div className="row">
            <Field label="Loss">
              <select value={options.loss} onChange={(e) => tune('loss', e.target.value)}>
                {LOSSES.map((a) => <option key={a}>{a}</option>)}
              </select>
            </Field>
            <Field label="Augmentation">
              <select value={options.augmentation} onChange={(e) => tune('augmentation', e.target.value)}>
                {AUGMENTATIONS.map((a) => <option key={a} value={a}>{a[0].toUpperCase() + a.slice(1)}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Learning-rate schedule">
            <select value={options.scheduler} onChange={(e) => tune('scheduler', e.target.value)}>
              {Object.entries(SCHEDULERS).map(([value, name]) => <option key={value} value={value}>{name}</option>)}
            </select>
          </Field>
          <div className="row">
            <Field label="Warm-up epochs">
              <input type="number" min={0} value={options.warmup_epochs} onChange={(e) => tune('warmup_epochs', +e.target.value)} />
            </Field>
            <Field label="Seed">
              <input type="number" min={0} value={options.seed} onChange={(e) => tune('seed', +e.target.value)} />
            </Field>
          </div>
          <Field label="Early-stopping patience, epochs">
            <input type="number" min={1} value={options.early_stopping_patience}
              onChange={(e) => tune('early_stopping_patience', +e.target.value)} />
          </Field>
        </details>
        <Field label="Run on">
          <select value={where} onChange={(e) => setWhere(e.target.value)}>
            <option value="local">This computer</option>
            <option value="ssh">GPU machine over SSH</option>
          </select>
        </Field>
        {where === 'ssh' && (
          <>
            <Field label="SSH host"><input placeholder="user@host" value={host} onChange={(e) => setHost(e.target.value)} /></Field>
            <Field label="Remote folder"><input value={dir} onChange={(e) => setDir(e.target.value)} /></Field>
            <button className="ghost" onClick={testRemote} disabled={!host}>Test connection</button>
            {remote && <p className="muted" data-testid="remote-status">{remote}</p>}
          </>
        )}
        {running ? (
          <button className="primary danger" onClick={() => window.api.stopTraining()}><Square size={15} /> Stop training</button>
        ) : (
          <button className="primary" onClick={start} disabled={starting || labelled < 2 || (where === 'ssh' && !host) || !!blocked}>
            <Play size={15} /> {starting ? 'Preparing dataset…' : `Train on ${labelled} labelled images`}
          </button>
        )}
        {labelled < 2 && <p className="muted">Label at least two images first.</p>}
        {missing.images > 0 && (
          <>
            <p className="warn" data-testid="not-classes">{untrained(missing)}</p>
            <button className="ghost add-classes" onClick={() => addClasses(missing.labels.map(([label]) => label))} disabled={checking}>
              <Plus size={15} /> Add as classes
            </button>
          </>
        )}

        <h3 className="gap">Dataset check
          <button className="icon" onClick={runCheck} disabled={checking} title="Check dataset"><RefreshCw size={14} /></button>
        </h3>
        {checking && <p className="muted">Checking…</p>}
        {check && (
          <div className="kv">
            <div><span>Labelled images</span><b>{check.labelled}</b></div>
            <div><span>Not labelled yet</span><b>{check.unlabelled}</b></div>
            <div><span>Train / val samples</span><b>{check.train} / {check.val}</b></div>
            {check.formats && formatList(check.formats) && (
              <div><span>Formats</span><b>{formatList(check.formats)}</b></div>
            )}
            {Object.entries(check.class_pixels as Record<string, number>).map(([k, v]) => (
              <div key={k}><span>{k} pixels</span><b>{v.toLocaleString()}</b></div>
            ))}
            {check.skipped_shapes && leftOut(check.skipped_shapes) && (
              <p className="muted note">Left out, no area: {leftOut(check.skipped_shapes)}</p>
            )}
            {check.unknown_labels.length > 0 && (
              <>
                <p className="warn">Ignored, not in class list: {check.unknown_labels.join(', ')}</p>
                <button className="ghost add-classes" onClick={() => addClasses(check.unknown_labels)} disabled={checking}><Plus size={15} /> Add as classes</button>
              </>
            )}
          </div>
        )}
      </aside>

      <main className="results">
        {comparing && runs.length > 0 && (
          <section className="card compare" data-testid="compare">
            <header><h3>Runs compared</h3></header>
            <table>
              <thead>
                <tr>
                  <th>Started</th><th>Network</th><th className="num">Input</th><th className="num">Parameters</th>
                  <th className="num">Time a tile</th><th className="num">Best IoU</th>
                </tr>
              </thead>
              <tbody>
                {compared.map((row) => (
                  <tr key={row.name} className={row.name === current ? 'on' : ''} onClick={() => setCurrent(row.name)}>
                    <td>{runDate(row.name)}</td>
                    <td>{row.network}</td>
                    <td className="num">{row.input ? `${row.input} px` : '–'}</td>
                    <td className="num">{row.params != null ? millions(row.params) : '–'}</td>
                    <td className="num" title={row.device ?? undefined}>
                      {row.ms != null ? <>{milliseconds(row.ms)} <span className="muted">{row.device}</span></> : '–'}
                    </td>
                    <td className={`num ${row.iou != null && row.iou === topIou ? 'top' : ''}`}>{row.iou != null ? row.iou.toFixed(3) : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
        {selected && (
          <div className="run-head">
            <span className="what"><b data-testid="run-network">{describe(selected.options)}</b>
              <span className="muted"> · {selected.options?.image_size ? `${selected.options.image_size} px · ` : ''}{runDate(selected.name)}</span>
            </span>
            <button className="ghost" onClick={() => duplicate(selected)} title="Put this run's settings in the form, to train a variant of it">
              <Copy size={15} /> Duplicate
            </button>
          </div>
        )}
        <div className="tiles">
          <div className="tile"><span>Status</span>
            <b>{failure ? 'Failed' : isRunning ? 'Training' : events.some((e) => e.type === 'done') ? 'Finished' : selected ? 'Stopped' : '–'}</b></div>
          <div className="tile"><span>Epoch</span><b>{last ? `${last.epoch + 1} / ${last.epochs}` : '–'}</b></div>
          <div className="tile"><span>Best IoU</span><b>{best ? best.iou.toFixed(3) : '–'}</b></div>
          <div className="tile"><span>Device</span><b className="fit" data-testid="device">{events.find((e) => e.type === 'start')?.device ?? '–'}</b></div>
          <div className="tile"><span>Samples</span><b>{prepared ? `${prepared.train} / ${prepared.val}` : '–'}</b></div>
        </div>
        {failure && <div className="error static">{failure.message}</div>}
        {selected?.model && !isRunning && (
          <section className="card export" data-testid="export">
            <p>
              {model
                ? <>
                  Saved as <code>{shownModel}</code>. Seamtag 3.3 pre-labels with the newest exported run unless another
                  model is chosen in AI → Model for pre-labelling.
                </>
                : <span className="muted">Export this run's model as ONNX to pre-label with it in Seamtag.</span>}
            </p>
            <button className="ghost" onClick={() => exportRun(selected.name)} disabled={!!exporting}>
              <PackageOpen size={15} /> {exporting === selected.name ? 'Exporting…' : 'Export for Seamtag'}
            </button>
            {model && (
              <button className="ghost" onClick={onSeamtag} disabled={seamtagStarting}>
                <ExternalLink size={15} /> {seamtagStarting ? 'Starting Seamtag…' : 'Open in Seamtag'}
              </button>
            )}
          </section>
        )}
        <div className="charts">
          <Chart title="Loss" series={[
            { name: 'train', color: '#4f8cff', values: epochs.map((e) => e.train_loss) },
            { name: 'validation', color: '#ff8f4d', values: epochs.map((e) => e.val_loss) }
          ]} />
          <Chart title="Validation quality" max={1} series={[
            { name: 'IoU', color: '#3ecf8e', values: epochs.map((e) => e.iou) },
            { name: 'F1', color: '#b57bff', values: epochs.map((e) => e.f1) }
          ]} />
        </div>
        <section className="card logs">
          <header><h3>Log</h3></header>
          <pre ref={logRef} data-testid="log">{saved || log || 'Output appears here once a run starts.'}</pre>
        </section>
      </main>

      <aside className="panel right">
        <h3>Runs
          <button className={`icon ${comparing ? 'on' : ''}`} onClick={() => setComparing((c) => !c)} disabled={!runs.length}
            aria-pressed={comparing} aria-label="Compare runs" title="Compare runs"><Table2 size={14} /></button>
        </h3>
        <div className="shapes">
          {runs.map((r) => {
            const done = r.events.filter((e) => e.type === 'epoch')
            const top = done.length ? Math.max(...done.map((e) => e.iou)) : null
            return (
              <button key={r.name} className={`run ${r.name === current ? 'on' : ''}`} onClick={() => setCurrent(r.name)}>
                <span className="name" title={describe(r.options)}>{describe(r.options)}</span>
                <span className="muted">{r.name.replace('_', ' ')}{r.options?.target ? ` · ${r.options.target.host}` : ''}</span>
                <span className="muted">{r.name === running ? 'running' : top !== null ? `IoU ${top.toFixed(3)}` : 'no result'}</span>
              </button>
            )
          })}
          {!runs.length && <p className="muted pad">Runs you start appear here.</p>}
        </div>
      </aside>
    </div>
  )
}
