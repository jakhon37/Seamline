// electron-builder's beforePack (electron-builder.yml), before it packs each system's app: the
// third-party notices from this system's packages, Chromium's licences for a Mac app, and uv for
// the system and processor being built. It stops the build when a file extraResources names is
// missing, which electron-builder would only warn about.
import { ARCHES, checkResources, chromiumLicences, write } from './notices.mjs'
import { fetchUv } from './fetch-uv.mjs'

export default async function beforePack({ packager, arch, electronPlatformName }) {
  const processor = ARCHES[arch] ?? 'x64'
  write()
  if (electronPlatformName === 'darwin') await chromiumLicences('darwin', processor)
  await fetchUv(electronPlatformName, processor)
  checkResources(packager)
}
