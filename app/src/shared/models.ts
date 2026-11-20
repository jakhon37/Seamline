import catalogue from './models.json'

/** What a training run is asked to do: the form on the Train page, and a run's options.json */
export type TrainOptions = {
  architecture: string
  encoder: string
  /** Encoder stages used; the architecture's own by default */
  encoder_depth: number
  /** Multiplies the decoder's default widths */
  decoder_width: number
  decoder_attention_type: string | null
  loss: string
  scheduler: string
  warmup_epochs: number
  augmentation: string
  seed: number
  early_stopping_patience: number
  encoder_weights: string | null
  epochs: number
  batch_size: number
  image_size: number
  learning_rate: number
  val_split: number
  tile: number
  num_workers: number
}

type Architecture = {
  depth: number; depths: number[]; decoderWidth: boolean; attention: string[]; attentionMinWidth?: number; minBatch: number
  minSize: number; without: string[]
}
const ARCHITECTURE_INFO = catalogue.architectures as Record<string, Architecture>
const ENCODER_INFO = catalogue.encoders as Record<string, { family: string; weights: string; licence: string }>

export const ARCHITECTURES = Object.keys(ARCHITECTURE_INFO)
export const ENCODERS = Object.keys(ENCODER_INFO)
export const DECODER_WIDTHS: number[] = catalogue.decoderWidths
export const LOSSES = ['combined', 'focal', 'dice', 'cross_entropy']
export const SCHEDULERS: Record<string, string> = {
  reduce_on_plateau: 'Lower on plateau', cosine: 'Cosine', constant: 'Constant'
}
export const AUGMENTATIONS = ['none', 'light', 'standard', 'strong']
export const architecture = (name: string): Architecture | undefined => ARCHITECTURE_INFO[name]
export const weightsOf = (encoder: string) => ENCODER_INFO[encoder]
export const takes = (arch: string, encoder: string) => !!architecture(arch) && !architecture(arch)!.without.includes(encoder)

/** The settings a model size sets: the network, and how it trains */
export const ADVANCED = [
  'architecture', 'encoder', 'encoder_depth', 'decoder_width', 'decoder_attention_type',
  'loss', 'scheduler', 'warmup_epochs', 'augmentation', 'seed', 'early_stopping_patience'
] as const
type Advanced = Pick<TrainOptions, (typeof ADVANCED)[number]>
export type Size = 'fast' | 'balanced' | 'accurate' | 'custom'
const RECIPE = catalogue.recipe as Omit<Advanced, 'encoder'>
export const PRESETS = Object.fromEntries(
  Object.entries(catalogue.presets).map(([size, { label, ...preset }]) => [size, { label, options: { ...RECIPE, ...preset } as Advanced }])
) as Record<Exclude<Size, 'custom'>, { label: string; options: Advanced }>

/** The size whose settings these are, or custom */
export function sizeOf(options: TrainOptions): Size {
  const found = Object.entries(PRESETS).find(([, preset]) => ADVANCED.every((key) => options[key] === preset.options[key]))
  return (found?.[0] as Size) ?? 'custom'
}

/**
 * Settings that fit the architecture, kept where they do: an encoder it can't take becomes the one closest in the list,
 * and the encoder's depth, decoder width and attention go back to the architecture's own where it has no such choice
 */
export function fitArchitecture(options: TrainOptions, arch: string, previous?: string): TrainOptions {
  const info = architecture(arch)!
  const encoder = takes(arch, options.encoder) ? options.encoder : ENCODERS.find((e) => takes(arch, e))!
  // The previous architecture's own depth becomes this one's own
  const own = previous && options.encoder_depth === architecture(previous)?.depth
  return {
    ...options,
    architecture: arch,
    encoder,
    encoder_depth: own || !info.depths.includes(options.encoder_depth) ? info.depth : options.encoder_depth,
    decoder_width: info.decoderWidth ? options.decoder_width : 1,
    decoder_attention_type: info.attention.includes(options.decoder_attention_type ?? '') ? options.decoder_attention_type : null
  }
}

/** Why these options can't train, the first reason, or null when they can */
export function problem(options: TrainOptions): string | null {
  const info = architecture(options.architecture)
  if (!info) return `Seamline doesn't offer ${options.architecture}.`
  if (!weightsOf(options.encoder)) return `Seamline doesn't offer ${options.encoder}.`
  if (!takes(options.architecture, options.encoder)) return `${options.architecture} can't be built on ${options.encoder}.`
  if (!info.depths.includes(options.encoder_depth)) {
    return `${options.architecture} takes an encoder depth of ${info.depths.join(' or ')}.`
  }
  if (options.decoder_width !== 1 && !info.decoderWidth) return `${options.architecture}'s decoder has no width to change.`
  if (options.decoder_attention_type && !info.attention.includes(options.decoder_attention_type)) {
    return `${options.architecture}'s decoder has no ${options.decoder_attention_type} attention.`
  }
  if (options.decoder_attention_type && options.decoder_width < (info.attentionMinWidth ?? 0)) {
    return `Attention needs a decoder width of ${info.attentionMinWidth}× or more.`
  }
  if (!(options.batch_size >= info.minBatch)) return `${options.architecture} trains on batches of ${info.minBatch} or more.`
  if (!(options.image_size >= info.minSize)) return `${options.architecture} needs an input size of ${info.minSize} or more.`
  if (options.image_size % 32) return 'The input size must be a multiple of 32.'
  return null
}

/**
 * A run's options as the form takes them. Runs from before a setting existed trained as it now is by default, so
 * those are filled in; a run that trained elsewhere also says where.
 */
export function fromRun(saved: any, defaults: TrainOptions): TrainOptions {
  const known = Object.fromEntries(Object.entries(saved ?? {}).filter(([key, value]) => key in defaults && value !== undefined))
  const arch = typeof known.architecture === 'string' ? known.architecture : defaults.architecture
  const filled = { ...defaults, ...RECIPE, encoder_depth: architecture(arch)?.depth ?? RECIPE.encoder_depth, ...known } as TrainOptions
  // Saved as null: the architecture's own depth, and no change of width
  if (filled.encoder_depth == null) filled.encoder_depth = architecture(arch)?.depth ?? RECIPE.encoder_depth
  if (filled.decoder_width == null) filled.decoder_width = 1
  return filled
}
