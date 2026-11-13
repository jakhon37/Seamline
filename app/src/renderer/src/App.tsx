import { Brain, Download, ExternalLink, FolderOpen, PenTool, Search } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { isLabelled } from '../../shared/labels'
import Labeler, { type LabelerHandle } from './Labeler'
import Logo from './Logo'
import Trainer from './Trainer'
import type { ImageItem, LabelClass, OpenProject } from './types'

type Mode = 'label' | 'train'
type ClassChange = (classes: LabelClass[]) => LabelClass[]

export default function App() {
  const [data, setData] = useState<OpenProject | null>(null)
  const [mode, setMode] = useState<Mode>('label')
  const [error, setError] = useState('')
  const [training, setTraining] = useState(false)
  // Seamtag wasn't found, or didn't start: what it was to open, for when the user shows where it is
  const [seamtag, setSeamtag] = useState<{ name: string | null; failed?: string } | null>(null)
  // Seamtag is starting: Seamline waits a moment to see it doesn't close at once, and starts no other meanwhile
  const [starting, setStarting] = useState(false)
  const labeler = useRef<LabelerHandle>(null)
  // The class list the window shows: as the main process last read or wrote it, which it merges the
  // next change with. Changes and reads go one at a time, so each starts from where the last left it.
  const classes = useRef<{ root: string; list: LabelClass[] } | null>(null)
  const classWork = useRef<Promise<unknown>>(Promise.resolve())
  // The words Electron puts before errors from the main process go, wherever a message has them
  const showError = (message: string) =>
    setError(message.replace(/Error invoking remote method '[^']+': (\w*Error: )?/, ''))

  const show = (next: OpenProject) => {
    classes.current = { root: next.root, list: next.project.classes }
    setData(next)
  }
  const open = async (loader: () => Promise<OpenProject | null>) => {
    try {
      const next = await loader()
      if (next) {
        show(next)
        localStorage.setItem('lastProject', next.root)
      }
    } catch (e: any) {
      showError(e.message)
    }
  }

  useEffect(() => {
    const initial = new URLSearchParams(location.search).get('project')
    const last = initial ?? localStorage.getItem('lastProject')
    if (last) window.api.openProject(last).then(show).catch(() => {})
    const key = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setSeamtag(null)
      if (!e.ctrlKey) return
      if (e.key === '1') setMode('label')
      if (e.key === '2') setMode('train')
    }
    // Back from another app, such as Seamtag
    const onFocus = () => void reloadClasses()
    window.addEventListener('keydown', key)
    window.addEventListener('focus', onFocus)
    return () => {
      window.removeEventListener('keydown', key)
      window.removeEventListener('focus', onFocus)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const inTurn = (work: () => Promise<void>) => {
    const done = classWork.current.then(work)
    classWork.current = done.catch(() => {})
    return done
  }
  const showClasses = (root: string, list: LabelClass[]) => {
    if (classes.current?.root !== root) return
    classes.current = { root, list }
    setData((d) => d && (d.root === root ? { ...d, project: { ...d.project, classes: list } } : d))
  }
  const changeClasses = (root: string, change: ClassChange) => inTurn(async () => {
    const before = classes.current
    if (before?.root !== root) return
    const list = change(before.list)
    if (list === before.list) return
    showClasses(root, list)
    showClasses(root, await window.api.saveClasses(root, list))
  })
  // Another app, such as Seamtag, may have added, recoloured or deleted classes meanwhile
  const reloadClasses = () => inTurn(async () => {
    const at = classes.current
    // One that can't be read now is read again next time
    const list = at && (await window.api.readClasses(at.root).catch(() => null))
    if (at && list) showClasses(at.root, list)
  })
  // Back on the Label page, as the folder is looked at again there
  const shownMode = useRef(mode)
  useEffect(() => {
    if (mode === 'label' && shownMode.current !== 'label') void reloadClasses()
    shownMode.current = mode
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])
  const setImages = (fn: (images: ImageItem[]) => ImageItem[]) =>
    setData((d) => d && { ...d, images: fn(d.images) })

  // Seamtag on the image chosen here, or on the whole folder. What's waiting to be saved goes first.
  // Not found, or stopped as it started: the user can download it, or show where another copy is.
  const openSeamtag = async (image: boolean) => {
    if (!data) return
    setStarting(true)
    try {
      await labeler.current?.flush()
      const name = image ? labeler.current?.current() ?? null : null
      const result = await window.api.openSeamtag(data.root, name)
      if (result.missing || result.failed) setSeamtag({ name, failed: result.failed })
    } catch (e: any) {
      showError(`Seamtag couldn't start: ${e.message}`)
    } finally {
      setStarting(false)
    }
  }
  const locateSeamtag = async () => {
    if (!data || !seamtag) return
    setStarting(true)
    try {
      const result = await window.api.locateSeamtag(data.root, seamtag.name)
      if (result.launched) setSeamtag(null)
      else if (result.failed) setSeamtag({ ...seamtag, failed: result.failed })
    } catch (e: any) {
      setSeamtag(null)
      showError(`Seamtag couldn't start: ${e.message}`)
    } finally {
      setStarting(false)
    }
  }
  const downloadSeamtag = () => {
    setSeamtag(null)
    window.api.downloadSeamtag().catch((e) => showError(e.message))
  }

  // Labelled as training counts it: with an accepted shape that has an area and a class's label
  const classNames = new Set(data?.project.classes.map((c) => c.name))
  const done = data?.images.filter((i) => isLabelled(i.areas, classNames)).length ?? 0
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><Logo size={22} /> Seamline</div>
        {data && (
          <div className="switch" role="tablist">
            <button role="tab" aria-selected={mode === 'label'} className={mode === 'label' ? 'on' : ''}
              onClick={() => setMode('label')} title="Ctrl+1">
              <PenTool size={15} /> Label
            </button>
            <button role="tab" aria-selected={mode === 'train'} className={mode === 'train' ? 'on' : ''}
              onClick={() => setMode('train')} title="Ctrl+2">
              <Brain size={15} /> Train {training && <i className="pulse" />}
            </button>
          </div>
        )}
        <div className="spacer" />
        {data && <span className="muted">{data.project.name} · {done}/{data.images.length} labelled</span>}
        {data && (
          <button className="ghost" onClick={() => openSeamtag(true)} disabled={starting}
            title="Seamtag, the label-only app for these folders, reshapes every labelme shape and keeps LabelImg's formats">
            <ExternalLink size={15} /> {starting ? 'Starting Seamtag…' : 'Label in Seamtag'}
          </button>
        )}
        <button className="ghost" onClick={() => open(window.api.pickProject)}>
          <FolderOpen size={15} /> Open folder
        </button>
      </header>
      {error && <div className="error" onClick={() => setError('')}>{error}</div>}
      {!data ? (
        <div className="empty">
          <Logo size={72} />
          <h1>Open a folder of images</h1>
          <p>Labels are saved beside each image as LabelMe JSON. Folders labelled in labelme, LabelImg or Seamtag open as they are.</p>
          <button className="primary" onClick={() => open(window.api.pickProject)}>
            <FolderOpen size={16} /> Open folder
          </button>
        </div>
      ) : (
        <>
          {/* Both stay mounted, so switching keeps your place and view */}
          <div className="page" hidden={mode !== 'label'}>
            <Labeler key={data.root} ref={labeler} data={data} changeClasses={(change) => changeClasses(data.root, change)}
              setImages={setImages} active={mode === 'label'} modal={!!seamtag} onError={showError}
              onSeamtag={() => openSeamtag(true)} seamtagStarting={starting} />
          </div>
          <div className="page" hidden={mode !== 'train'}>
            <Trainer key={data.root} data={data} active={mode === 'train'} onError={showError}
              onTraining={setTraining} changeClasses={(change) => changeClasses(data.root, change)}
              onSeamtag={() => openSeamtag(false)} seamtagStarting={starting} />
          </div>
        </>
      )}
      {seamtag && (
        <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && setSeamtag(null)}>
          <div className="dialog" role="dialog" aria-labelledby="seamtag-help">
            <h2 id="seamtag-help">{seamtag.failed ? "Seamtag didn't start" : "Seamline can't find Seamtag"}</h2>
            {seamtag.failed ? (
              <>
                <p className="failed">{seamtag.failed}</p>
                <p>Download Seamtag, or show Seamline another copy of it.</p>
              </>
            ) : (
              <p>Seamtag is the label-only app made for these folders: it reshapes every labelme shape and keeps
                LabelImg's formats. Download it, or show Seamline where it is.</p>
            )}
            <div className="actions">
              {/* Keys go to the dialog, not to the image behind it */}
              <button className="ghost" onClick={() => setSeamtag(null)} autoFocus>Cancel</button>
              <button className="ghost" onClick={locateSeamtag} disabled={starting}>
                <Search size={15} /> {starting ? 'Starting Seamtag…' : 'Locate Seamtag…'}
              </button>
              <button className="primary" onClick={downloadSeamtag}><Download size={15} /> Download Seamtag</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
