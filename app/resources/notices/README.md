# Licence texts node_modules lacks

`scripts/notices.mjs` writes these into THIRD-PARTY-NOTICES.txt, which the installers carry, beside the texts it takes from node_modules. They are the same texts Seamtag's installers carry for the same packages.

- `LGPL-3.0.txt`: the GNU Lesser General Public License version 3, as the FSF publishes it (https://www.gnu.org/licenses/lgpl-3.0.txt). sharp's libvips packages name the licence but don't carry it.
- `sharp-libvips-1.2.4/`: the licence files of libvips and of each library built into it, from their sources at the versions sharp-libvips 1.2.4 builds. Its README says where each came from and what it leaves out. When sharp moves to another sharp-libvips version, the script stops until a folder for that version is there.
- `clipper-lib/BSL-1.0.txt`: the Boost Software License 1.0 (https://www.boost.org/LICENSE_1_0.txt), which clipper.js names.
- `clipper-lib/jsbn-LICENSE`: the licence of Tom Wu's JSBN, part of which is in clipper.js, as the jsbn 1.1.0 package (https://github.com/andyperlitch/jsbn) carries it.

`../legal/uv-LICENSE-MIT.txt` is uv's licence, from https://github.com/astral-sh/uv at tag 0.13.7, the version `scripts/fetch-uv.mjs` puts in the installers. uv is under the MIT or the Apache 2.0 licence; Seamline takes it under MIT.
