import type { LabelClass } from '../../shared/classes'
import type { LabelFormat } from '../../shared/formats'

export type { LabelClass, LabelFormat }
// labelme 7's shape types. Seamline draws polygons and rectangles; Seamtag draws and reshapes the rest.
export type ShapeType = 'polygon' | 'rectangle' | 'oriented_rectangle' | 'circle' | 'line' | 'point' | 'linestrip' | 'points' | 'mask'
export type Shape = {
  label: string
  points: number[][]
  shape_type: ShapeType
  group_id?: number | null
  description?: string
  flags?: Record<string, boolean>
  score?: number
  /** A mask shape's pixels inside its box, as a base64 PNG */
  mask?: string | null
  /** labelme 7's lock: Seamtag won't move or reshape it, and neither will Seamline */
  locked?: boolean
  /** Keys other apps write are kept as they were */
  [key: string]: unknown
}
export type Project = { name: string; classes: LabelClass[] }
export type ImageItem = {
  name: string; count: number; ai: number; score: number | null; classes: Record<string, number>
  /** Accepted shapes with an area, by label: with one whose label is a class, the image is labelled, as training counts it */
  areas: Record<string, number>
  broken?: boolean
  /** Where its labels are kept; undefined when it has none yet */
  format?: LabelFormat
}
export type OpenProject = { root: string; project: Project; images: ImageItem[] }
export type OpenImage = {
  width: number; height: number; file: string; ms: number
  pyramid: { dir: string; maxLevel: number; tile: number } | null
}
export type Stamp = { size: number; mtime: number }
/**
 * `key` names these labels for haveLabels; `changed`, that they changed since the window last had them;
 * `fresh`, that the image has no label file, or only the one Seamline made for the labels the window took
 * where it had none: saved with no shapes, it has no file again
 */
export type Labels = { shapes: Shape[]; format: LabelFormat | null; stamp: Stamp | null; key: string; changed: boolean; fresh: boolean }
export type TrainEvent = { type: string; [key: string]: any }
export type Run = { name: string; events: TrainEvent[]; options: any; running: boolean; model: boolean; exported: boolean }
export type Exported = { model: string; sidecar: string; classes: string[]; input_size: number; tile: number }
/**
 * A model's costs before training, from the engine's estimate: training memory in bytes, at the batch size, in float32 as
 * on a CPU and with mixed precision as on a GPU; time per tile on `device`, or null when not timed (`busy`: a run here
 * had the CPU)
 */
export type Estimate = {
  params: number; gflops: number; memory: { float32: number; mixed: number }; image_size: number; batch_size: number
  ms_per_tile: number | null; device: string | null; busy?: boolean
}
/** `failed` says why Seamtag didn't start */
export type SeamtagResult = { launched?: boolean; missing?: boolean; canceled?: boolean; failed?: string }

/** The engine's Python environment, as the main process finds it (see src/main/engine.ts) */
export type Backend = 'auto' | 'cu126' | 'cpu'
export type EngineChoice = { backend: Backend; title: string; detail: string; download: string; disk: string }
export type EngineInfo = { backend: Backend; torch: string; cuda: boolean; device: string; usable: boolean; requirements: string; at: string }
export type EngineStatus = {
  ready: boolean
  managed: boolean
  python: string
  info: EngineInfo | null
  outdated: boolean
  gpu: { name: string; capability: number; driver: string } | null
  suggested: Backend
  choices: EngineChoice[]
  canSetUp: boolean
  /** Windows lacks the Visual C++ runtime PyTorch and ONNX Runtime load */
  runtimeMissing: boolean
  platform: string
}
export type EngineProgress = { line?: string; step?: string; downloaded?: number; total?: number }

declare global {
  interface Window {
    api: {
      pickProject(): Promise<OpenProject | null>
      openProject(dir: string): Promise<OpenProject>
      refreshProject(root: string): Promise<{ images: ImageItem[]; changed: string[] }>
      /** Writes the class list, merged with what another app changed there, and gives the list written */
      saveClasses(root: string, classes: LabelClass[]): Promise<LabelClass[]>
      /** The class list as .myseg/project.json has it now, or null when it can't be read */
      readClasses(root: string): Promise<LabelClass[] | null>
      openImage(name: string): Promise<OpenImage>
      loadLabels(root: string, name: string): Promise<Labels>
      /** The window shows the labels it loaded: its saves are checked against them from now on, and go by `fresh` */
      haveLabels(root: string, name: string, key: string, fresh: boolean): Promise<void>
      /** `from` gives each shape's place in the label file as the window last had it, or -1 for one it made */
      saveLabels(root: string, name: string, shapes: Shape[], size: number[], from?: number[]): Promise<{ changed: boolean; stamp?: Stamp | null }>
      openSeamtag(root: string, name: string | null): Promise<SeamtagResult>
      locateSeamtag(root: string, name: string | null): Promise<SeamtagResult>
      downloadSeamtag(): Promise<void>
      listRuns(): Promise<Run[]>
      runLog(name: string): Promise<string>
      exportModel(run: string): Promise<Exported>
      startTraining(options: unknown, target: { host: string; dir: string } | null): Promise<string>
      checkRemote(host: string, dir: string): Promise<{ freeKb: number; gpu: string; gpuJobs: number; torch: string }>
      /** On this computer, or on the GPU machine when `target` names it */
      estimateModel(options: unknown, time: boolean, target: { host: string; dir: string } | null): Promise<Estimate>
      stopTraining(): Promise<void>
      activeRun(): Promise<string | null>
      checkDataset(val: number, tile: number): Promise<any>
      predict(run: string, images: string[]): Promise<{ device: string; results: Record<string, Shape[]> }>
      prelabelAll(run: string, names: string[]): Promise<{ done: number; total: number; stopped: boolean; failed: string[]; error?: string }>
      stopPrelabel(): Promise<void>
      /** `key` names the labels written, for haveLabels, when there are `shapes` */
      onPrelabel(fn: (p: { name: string; shapes: Shape[] | null; key?: string; done: number; total: number }) => void): () => void
      onFlush(fn: () => void): () => void
      flushed(): void
      onLog(fn: (text: string) => void): () => void
      onEvents(fn: (p: { run: string; events: TrainEvent[] }) => void): () => void
      onExit(fn: (p: { run: string; code: number }) => void): () => void
      /** process.platform of the main process */
      platform: string
      engineStatus(): Promise<EngineStatus>
      /** Makes Seamline's own engine environment with this PyTorch build, or brings it up to date (`update`) */
      setupEngine(backend: Backend, update: boolean): Promise<EngineInfo>
      cancelEngineSetup(): Promise<void>
      /** Microsoft's Visual C++ Redistributable, in the browser */
      openRuntimeDownload(): Promise<void>
      onEngineProgress(fn: (progress: EngineProgress) => void): () => void
    }
  }
}
export const fileUrl = (p: string) => 'myseg://f/' + encodeURIComponent(p)
export const PALETTE = ['#4f8cff', '#ff6b6b', '#3ecf8e', '#f7b731', '#b57bff', '#22c7d6', '#ff8f4d', '#ec6fb3']
