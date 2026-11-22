# Installing Seamline

Seamline runs from this repository. It has two parts: the app, an Electron program you build with Node.js, and the engine, a Python program that prepares datasets, trains and pre-labels with PyTorch. Most of the download is PyTorch, which comes in a different build for each kind of GPU, so the one step that needs a choice is which build to install.

Seamline is developed and tested on Ubuntu 24.04, with Python 3.12 and Node.js 22. The same steps should work on other Linux systems and on Macs with Apple silicon, though Seamline hasn't been tested there. It doesn't run from source on Windows: see [Other systems](#other-systems).

- [What you need](#what-you-need)
- [Install](#install)
- [Choose the PyTorch build](#choose-the-pytorch-build)
- [Check that it works](#check-that-it-works)
- [Other systems](#other-systems)
- [Seamtag](#seamtag)
- [Set up a GPU machine for training over SSH](#set-up-a-gpu-machine-for-training-over-ssh)
- [Disk space](#disk-space)
- [Update or remove](#update-or-remove)
- [Troubleshooting](#troubleshooting)

## What you need

- **Git**
- **Node.js 22 or later**, from [nodejs.org](https://nodejs.org/en/download)
- **uv 0.13.7 or later**, Astral's Python package manager, which also fetches Python 3.12 if your computer doesn't have it. To install it:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  ```

  The second line puts it on the path in the shell you're in; new shells find it anyway. `uv self update` updates an older uv, and [uv's installation page](https://docs.astral.sh/uv/getting-started/installation/) has other ways to install it, such as Homebrew.
- **1.5 to 8 GB of disk**, most of it for PyTorch: about 2 GB with its CPU build, and 7 to 8 GB with its CUDA build. See [Disk space](#disk-space).
- **An NVIDIA GPU to train at a useful speed**, in this computer or in a Linux machine you can reach over SSH. Labelling, reviewing and pre-labelling work without one, and small models train on the CPU.

## Install

On an NVIDIA GPU from before the RTX 20 and GTX 16 series, write `--torch-backend=cu126` in place of `auto` (see [the next section](#choose-the-pytorch-build)).

```bash
git clone https://github.com/jakhon37/Seamline.git
cd Seamline
uv venv --python 3.12
uv pip install --torch-backend=auto -r requirements.txt
cd app
npm ci
npm run build
npm start
```

- `uv venv` makes the engine's Python environment in `.venv`, at the top of the repository, where the app looks for it.
- `uv pip install` installs the engine's packages, with the PyTorch build that `--torch-backend` names, or that `auto` picks for this computer.
- `npm ci` installs the app's packages as `package-lock.json` lists them, Electron among them: about 0.4 GB.
- `npm run build` builds the app and `npm start` starts it. Until you update, `npm start` in `app/` is all it takes.

Open a folder of images: [the app's guide](../app/README.md) takes it from there. If `npm start` stops with an error about `chrome-sandbox`, as it can on Ubuntu 24.04 and later, see [Troubleshooting](#troubleshooting).

To use a Python environment of your own instead of `.venv`, name its interpreter in `MYSEG_PYTHON` before starting the app. It needs the packages in `requirements.txt`, with Pillow 12 or later: with an older one, 16-bit grey PNGs train as white and AVIF images can't be read.

## Choose the PyTorch build

With `--torch-backend=auto`, uv looks at the computer's GPU driver. With an NVIDIA driver it picks the newest CUDA build the driver can run, and with none it picks the CPU build, which is a fraction of the size. To choose yourself, give one of these instead of `auto`:

| Your computer | `--torch-backend=` | Download, then on disk |
|---|---|---|
| Linux, no NVIDIA GPU | `cpu` | 0.5 GB, 1.7 GB |
| Linux, NVIDIA RTX 20 or GTX 16 series or newer | `cu132` | 3.3 GB, 6.4 GB |
| Linux, older NVIDIA GPU | `cu126` | 3.9 GB, 7.4 GB |
| Mac with Apple silicon | `cpu` | 0.3 GB, 1.0 GB |

`cpu` and `cu132` install PyTorch 2.15, the second with CUDA 13.2. The sizes are for all the engine's packages, with Python 3.12, measured from the package files on 22 November 2026: PyTorch and the CUDA libraries it brings are most of it.

**Older NVIDIA GPUs need PyTorch 2.14.** PyTorch 2.15's CUDA builds run on Turing GPUs and newer (compute capability 7.5). For the GTX 10 and 900 series, the Titan V, and the Tesla P100 and V100, `cu126` installs PyTorch 2.14 with CUDA 12.6, the last release with builds for them. `auto` goes by the driver, not the GPU, so with a recent driver it picks a build that can't use an older card, though `torch.cuda.is_available()` still says `True`.

**Plain pip downloads the CUDA build.** On Linux, `pip install -r requirements.txt` takes PyTorch from PyPI, whose Linux build carries CUDA: 3.3 GB, with a GPU or without. With pip, add the index of the build you want, such as the CPU build's:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**AMD and Intel GPUs.** With ROCm installed, `auto` picks PyTorch's ROCm build for an AMD GPU on Linux, which the engine uses as it uses CUDA. For an Intel GPU it picks the XPU build, which the engine doesn't use, so give `cpu` there. Seamline hasn't been tried on either.

**To change the build later**, delete `.venv` and install again with the other build. uv keeps what it downloaded, so the packages you had already come from its cache.

## Check that it works

From the top of the repository:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

It prints the PyTorch build and whether it can train on an NVIDIA GPU here: for example `2.15.0+cu132 True`, `2.15.0+cpu False`, or on a Mac `2.15.0 False`.

Then ask the engine what a small model costs, as the Train page does:

```bash
.venv/bin/python engine/studio.py estimate --options '{"architecture": "Unet", "encoder": "resnet18", "image_size": 256, "batch_size": 2}'
```

Within about ten seconds it prints one line of JSON: the network's 14,328,354 parameters, its GFLOPs and training memory, and the time one tile took, with `"device"` set to `"CPU"` or the GPU's name.

## Other systems

**Macs** need Apple silicon: PyTorch's last build for Intel Macs was 2.2.2, from March 2024, and Seamline needs 2.3 or later. The engine trains on NVIDIA GPUs through CUDA, so on a Mac it trains on the CPU. Training on a GPU machine over SSH fails from a Mac: SSH can't make the socket the app asks for to share one connection between commands, as its path is longer than macOS allows.

**Windows** isn't supported from source. The app looks for the engine's Python at `.venv/bin/python`, where uv puts it at `.venv\Scripts\python.exe`; it stops runs by signalling their Unix process group; and it trains over SSH with `rsync`, `du` and shared connections, none of which Windows has.

## Seamtag

[Seamtag](https://github.com/jakhon37/Seamtag), the labelling app that pairs with Seamline, has installers on [its releases page](https://github.com/jakhon37/Seamtag/releases/latest). **Label in Seamtag** opens it, and it pre-labels with the models **Export for Seamtag** writes. Working on one folder in both apps at once needs Seamtag 3.3 or later. On Ubuntu 24.04 and later, install its `.deb`: its AppImage needs `--no-sandbox` there, which Seamline doesn't add. Seamline looks for Seamtag where its installers put it and on your `PATH`, and if it can't find it, lets you pick the program.

## Set up a GPU machine for training over SSH

On the Train page, **Run on: GPU machine over SSH** trains on another computer while you label on yours. The app sends the engine's code and the prepared dataset there, starts training detached, and brings the best model back when the run ends. [The app's guide](../app/README.md#train-on-a-gpu-machine) says how runs behave. The machine needs:

- **Linux and an NVIDIA GPU**, with a driver that `nvidia-smi` shows.
- **SSH with a key.** The app runs `ssh` in batch mode, so it can't ask for a password or a key's passphrase: a key with a passphrase has to be loaded in `ssh-agent`. A host from your `~/.ssh/config` works in place of `user@host`.
- **rsync**, on the machine and on your computer.
- **A `python3` with PyTorch 2.3 or later, built for its GPU, as commands run over SSH find it.** That may not be the `python3` you get when you log in: see below.
- **Python 3.10 to 3.12**, with its `venv` module. The engine's environment there keeps numpy below 2, for older PyTorch builds, and numpy 1 has no builds for newer Pythons.
- **Access to PyPI**, as the first run installs the engine's other packages.
- **Room on its disk** for the prepared dataset, plus 3 GB for the environment, checkpoints and logs. The app checks before each run.
- **No other program on its GPUs.** The app won't start a run while any process is using one of the machine's GPUs.

### PyTorch on the machine

Check from your computer:

```bash
ssh user@host "python3 -c 'import sys, torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())'"
```

If it prints a Python from 3.10 to 3.12, a PyTorch from 2.3 on, and `True`, the machine is ready. If not, give PyTorch an environment of its own on the machine, with uv and a Python of uv's own:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv ~/seamline-python --python 3.12 --managed-python
uv pip install --python ~/seamline-python --torch-backend=auto torch torchvision
```

On a GPU older than Turing, write `cu126` in place of `auto`, as above. Then put the environment first on the path for commands run over SSH: add this line to the top of `~/.bashrc` on the machine, above any line that stops early for non-interactive shells (with zsh, add it to `~/.zshenv`):

```bash
export PATH="$HOME/seamline-python/bin:$PATH"
```

It becomes the `python3` of your account on that machine. Run the check again.

### What the first run sets up

In the remote folder, `seamline` in your home folder unless you give another, the first run makes:

- `code/`: the engine as it is on your computer, sent again before each run.
- `venv/`: the engine's environment, made by [`engine/remote_setup.py`](../engine/remote_setup.py). It links the machine's PyTorch rather than copying it, and installs the rest: about 0.5 GB on disk, or 1.6 GB with PyTorch 2.11 or later, part of whose CUDA libraries it installs again.
- `projects/`: each run's dataset and results. When a run ends, its best model comes back and its files there are removed.

## Disk space

- **The repository:** 12 MB, where you clone it.
- **The app's packages:** 0.4 GB, in `app/node_modules`.
- **The engine's environment:** 1.0 to 7.4 GB in `.venv`, by the PyTorch build: see [the table](#choose-the-pytorch-build).
- **uv's cache:** about the environment's size, in the folder `uv cache dir` prints. When it's on the same disk as `.venv`, uv links files from it rather than copying them, so it adds little. `uv cache clean` empties it.
- **ImageNet weights:** 14 to 103 MB for each encoder, in `~/.cache/huggingface/hub`, downloaded the first time a run starts from them: 47, 87 and 103 MB for Fast, Balanced and Accurate.
- **Each project:** its `.myseg/` folder holds the prepared dataset, which can be many times the images' size (311 MB for the demo's 24 MB of JPEG photos), each run's best model (56 MB for Fast, 95 MB for Balanced), and a tile cache for large images.
- **A GPU machine:** the environment, 0.5 to 1.6 GB, and each run's dataset while it trains, in the remote folder; outside it, PyTorch if you installed it as above, and the ImageNet weights.

## Update or remove

To update, from the top of the repository, giving the same `--torch-backend` as when you installed:

```bash
git pull
uv pip install --torch-backend=auto -r requirements.txt
cd app
npm ci
npm run build
```

To remove Seamline, delete the repository's folder: that takes the app, `.venv` and `app/node_modules` with it. Each project keeps its runs and its settings in its `.myseg/` folder, which you can delete too. The rest is outside the repository:

- the app's own settings: `~/.config/Seamline` on Linux, `~/Library/Application Support/Seamline` on macOS
- uv's cache, which `uv cache clean` empties, and the Pythons uv fetched, in the folder `uv python dir` prints
- Electron's download cache: `~/.cache/electron` on Linux, `~/Library/Caches/electron` on macOS
- the ImageNet weights: `~/.cache/huggingface/hub/models--smp-hub--*`
- on a GPU machine, the remote folder

## Troubleshooting

**`npm start` stops with "The SUID sandbox helper binary was found, but is not configured correctly".** Ubuntu 24.04 and later can keep Chromium's sandbox from the user namespaces it uses, when the program has no AppArmor profile, as an Electron run from `node_modules` has none. Give the sandbox helper what it asks for, from `app/`:

```bash
sudo chown root node_modules/electron/dist/chrome-sandbox
sudo chmod 4755 node_modules/electron/dist/chrome-sandbox
```

`npm ci` puts back a new copy of the file, so do this again after each update.

**The install downloads gigabytes of `nvidia-` packages.** That is PyTorch's CUDA build. On a computer without an NVIDIA GPU, delete `.venv` and install again with `--torch-backend=cpu`.

**`torch.cuda.is_available()` prints `False` on a computer with an NVIDIA GPU.** Either the CPU build is installed (the version ends in `+cpu`), or the driver can't run the build. `nvidia-smi` shows the newest CUDA version the driver runs: install a build no newer than that, or update the driver.

**Training stops with "no kernel image is available for execution on the device".** PyTorch 2.15 is on a GPU older than Turing. Install with `--torch-backend=cu126`, here or on the GPU machine.

**The Train page says "No GPU PyTorch on host".** The `python3` that commands over SSH find can't import a CUDA build of PyTorch, and the message ends with what it printed. See [PyTorch on the machine](#pytorch-on-the-machine).

**Starting a run fails with "Permission denied (publickey)" or "Host key verification failed".** SSH needs a key the machine accepts, loaded in `ssh-agent` if it has a passphrase, and the machine's host key in your `known_hosts`. Make a key with `ssh-keygen` and copy it with `ssh-copy-id user@host`, then connect once by hand to accept the host key. `ssh -o BatchMode=yes user@host true` should then return without a word.

**"The GPU on host is busy with N other job(s)".** Another program is using one of the machine's GPUs. `nvidia-smi` on the machine shows which.

**"Not enough disk on host".** Free some space there, or give a remote folder on a disk with more room.

**Setting up the environment fails with "ensurepip is not available".** The machine's `python3` lacks Python's `venv` module: on Debian and Ubuntu, `sudo apt install python3-venv`. A Python that uv fetched, as above, has it.

**Setting up the environment fails while installing numpy.** The machine's `python3` is 3.13 or later, for which numpy 1 has no builds. Use Python 3.12, as above.
