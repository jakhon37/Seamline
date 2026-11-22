import { Cpu, Download, Square } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Backend, EngineInfo, EngineProgress, EngineStatus } from './types'

const reason = (e: any) => String(e?.message ?? e).replace(/^Error invoking remote method '[^']*': (Error: )?/, '')
const size = (bytes: number) => bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`

/** What the engine runs on, in a few words: "PyTorch 2.15.0, CUDA on NVIDIA GeForce RTX 3060" */
export function engineSummary(info: EngineInfo) {
  return `PyTorch ${info.torch.replace(/\+.*/, '')}, ${info.cuda ? `CUDA on ${info.device}` : 'CPU'}`
}

/**
 * Sets up the engine Seamline trains with: PyTorch and the packages it needs, downloaded once into
 * Seamline's own folder, in the build that fits the computer. Labelling works without it.
 */
export default function EngineSetup({ status, onClose, onDone }: {
  status: EngineStatus
  onClose: () => void
  onDone: (info: EngineInfo) => void
}) {
  const update = status.outdated && !!status.info
  const [backend, setBackend] = useState<Backend>(update ? status.info!.backend : status.suggested)
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<EngineProgress>({})
  const [step, setStep] = useState('')
  const [error, setError] = useState('')
  const [done, setDone] = useState<EngineInfo | null>(null)

  useEffect(() => window.api.onEngineProgress((p) => {
    if (p.step) setStep(p.step)
    setProgress((was) => ({ ...was, ...p }))
  }), [])

  const start = async () => {
    setRunning(true)
    setError('')
    setProgress({})
    try {
      const info = await window.api.setupEngine(backend, update)
      setDone(info)
      onDone(info)
    } catch (e) {
      setError(reason(e))
    } finally {
      setRunning(false)
    }
  }

  const choice = status.choices.find((c) => c.backend === backend)
  const share = progress.total ? Math.min(1, (progress.downloaded ?? 0) / progress.total) : 0
  return (
    <div className="overlay">
      <div className="dialog engine" role="dialog" aria-labelledby="engine-title">
        <h2 id="engine-title">{done ? 'The engine is ready' : update ? 'Update the engine' : 'Set up the engine'}</h2>
        {done ? (
          <>
            <p>{engineSummary(done)}.</p>
            {done.cuda && !done.usable && (
              <p className="failed">
                This PyTorch build can't run on {done.device}.{done.backend === 'cu126'
                  ? ' No build Seamline offers does, so training runs on the CPU here, or on a GPU machine over SSH.'
                  : ' Set the engine up again with Older NVIDIA GPU.'}
              </p>
            )}
            {!done.cuda && backend !== 'cpu' && (
              <p className="failed">PyTorch can't reach an NVIDIA GPU here, so training runs on the CPU. An older driver can be the cause: nvidia-smi shows the newest CUDA it runs.</p>
            )}
            <div className="actions"><button className="primary" onClick={onClose} autoFocus>Close</button></div>
          </>
        ) : running ? (
          <>
            <p>{step || 'Starting'}…</p>
            <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(share * 100)}>
              <i style={{ width: `${share * 100}%` }} />
            </div>
            <p className="muted line" data-testid="engine-line">
              {progress.total ? `${size(progress.downloaded ?? 0)} of ${size(progress.total)} downloaded · ` : ''}{progress.line ?? ''}
            </p>
            <div className="actions">
              <button className="ghost" onClick={() => window.api.cancelEngineSetup()}><Square size={14} /> Stop</button>
            </div>
          </>
        ) : (
          <>
            {update ? (
              <p>Seamline was updated, and its engine needs the packages this version uses. It keeps PyTorch as it is.</p>
            ) : (
              <p>Training, pre-labelling and export run in Python with PyTorch, which Seamline downloads once into its own
                folder. Labelling works without it.</p>
            )}
            {!update && (
              <>
                <p className="muted found">
                  <Cpu size={14} /> {status.gpu ? `Found ${status.gpu.name}, driver ${status.gpu.driver}` : 'No NVIDIA GPU found'}
                </p>
                <div className="choices" role="radiogroup" aria-label="PyTorch build">
                  {status.choices.map((c) => (
                    <label key={c.backend} className={c.backend === backend ? 'choice on' : 'choice'}>
                      <input type="radio" name="backend" value={c.backend} checked={c.backend === backend}
                        onChange={() => setBackend(c.backend)} />
                      <span>
                        <b>{c.title}</b>
                        <span className="muted">{c.detail}</span>
                        <span className="muted size">{c.download} to download, {c.disk} on disk</span>
                      </span>
                    </label>
                  ))}
                </div>
              </>
            )}
            {!status.canSetUp && <p className="failed">uv, which sets up the engine, isn't part of this copy of Seamline.</p>}
            {status.runtimeMissing && (
              <p className="failed">
                PyTorch needs Microsoft's Visual C++ Redistributable, which this computer doesn't have yet.{' '}
                <button className="link" onClick={() => window.api.openRuntimeDownload()}>Download it</button>, install it, then set up the engine.
              </p>
            )}
            {error && <p className="failed" data-testid="engine-error">{error}</p>}
            <div className="actions">
              <button className="ghost" onClick={onClose}>Later</button>
              <button className="primary" onClick={start} disabled={!status.canSetUp} autoFocus>
                <Download size={15} /> {error ? 'Try again' : update ? 'Update' : `Set up${choice ? ` (${choice.download})` : ''}`}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
