"""What engine/remote_setup.py links into the GPU machine's environment, rather than letting pip
download it again."""
import importlib.util
from importlib import metadata
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "remote_setup", Path(__file__).resolve().parents[2] / "engine" / "remote_setup.py"
)
remote_setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(remote_setup)


class Dist:
    def __init__(self, name, version="1.0", requires=()):
        self.name, self.version, self.requires = name, version, list(requires)


def installed(monkeypatch, dists):
    by_name = {d.name: d for d in dists}

    def distribution(name):
        if name not in by_name:
            raise metadata.PackageNotFoundError(name)
        return by_name[name]

    def version(name):
        return distribution(name).version

    monkeypatch.setattr(remote_setup.metadata, "distribution", distribution)
    monkeypatch.setattr(remote_setup.metadata, "version", version)


LINUX = '; platform_system == "Linux"'

# As PyTorch 2.15 is built: most of CUDA comes through cuda-toolkit's extras
TORCH_215 = [
    Dist(
        "torch",
        "2.15.0",
        [
            "filelock",
            "sympy>=1.13.3",
            f"cuda-toolkit[cublas,cufft,cudart]==13.2.2{LINUX}",
            f"cuda-bindings<14,>=13.0.3{LINUX}",
            f"nvidia-cudnn-cu13==9.26.0.51{LINUX}",
            f"triton~=3.9.0{LINUX}",
            'opt-einsum>=3.3; extra == "opt-einsum"',
        ],
    ),
    Dist("torchvision", "0.29.2", ["numpy", "torch>=2.14.0", "pillow!=8.3.*,>=5.3.0"]),
    Dist(
        "cuda-toolkit",
        "13.2.2",
        [
            'nvidia-cublas==13.4.1.3; extra == "cublas"',
            'nvidia-cufft==12.2.0.57; extra == "cufft"',
            'nvidia-cuda-runtime==13.2.75; extra == "cudart"',
            'nvidia-curand==10.4.2; extra == "curand"',
        ],
    ),
    Dist("cuda-bindings", "13.1.0", ["cuda-pathfinder~=1.1"]),
    Dist("cuda-pathfinder", "1.3.2"),
    Dist("nvidia-cudnn-cu13", "9.26.0.51", ["nvidia-cublas"]),
    Dist("nvidia-cublas", "13.4.1.3"),
    Dist("nvidia-cufft", "12.2.0.57", ['nvidia-nvjitlink; sys_platform == "linux"']),
    Dist("nvidia-nvjitlink", "13.2.78"),
    Dist("nvidia-cuda-runtime", "13.2.75"),
    Dist("nvidia-curand", "10.4.2"),
    Dist("triton", "3.9.0"),
    Dist("filelock", "3.20.0"),
    Dist("sympy", "1.14.0"),
    Dist("opt-einsum", "3.4.0"),
    Dist("numpy", "2.5.4"),
]


def names(dists):
    return sorted(d.name for d in dists)


def test_cuda_libraries_that_come_through_cuda_toolkits_extras_are_linked(monkeypatch):
    installed(monkeypatch, TORCH_215)
    assert names(remote_setup.linked_distributions()) == [
        "cuda-bindings",
        "cuda-pathfinder",
        "cuda-toolkit",
        "nvidia-cublas",
        "nvidia-cuda-runtime",
        "nvidia-cudnn-cu13",
        "nvidia-cufft",
        "nvidia-nvjitlink",
        "torch",
        "torchvision",
        "triton",
    ]


def test_extras_nobody_asked_for_and_packages_pip_may_upgrade_stay_out(monkeypatch):
    installed(monkeypatch, TORCH_215)
    linked = names(remote_setup.linked_distributions())
    # cuda-toolkit's curand extra isn't asked for; numpy, sympy and filelock get copies of their own
    for name in ("nvidia-curand", "numpy", "sympy", "filelock", "opt-einsum"):
        assert name not in linked


def test_an_extra_asked_for_later_is_walked_as_well(monkeypatch):
    dists = [
        Dist("torch", "2.15.0", ["cuda-toolkit[cublas]==13.2.2", "nvidia-cudnn-cu13"]),
        Dist("torchvision", "0.29.2", ["torch"]),
        Dist("nvidia-cudnn-cu13", "9.26.0.51", ["cuda-toolkit[cufft]"]),
        Dist(
            "cuda-toolkit",
            "13.2.2",
            ['nvidia-cublas; extra == "cublas"', 'nvidia-cufft; extra == "cufft"'],
        ),
        Dist("nvidia-cublas"),
        Dist("nvidia-cufft"),
    ]
    installed(monkeypatch, dists)
    assert "nvidia-cufft" in names(remote_setup.linked_distributions())


def test_older_builds_that_name_nvidia_packages_themselves_link_as_before(monkeypatch):
    installed(
        monkeypatch,
        [
            Dist(
                "torch",
                "2.9.0",
                [
                    "nvidia-cuda-runtime-cu12==12.8.90",
                    "nvidia-cublas-cu12==12.8.4.1",
                    "triton==3.5.0",
                    "jinja2",
                ],
            ),
            Dist("torchvision", "0.24.0", ["torch==2.9.0", "numpy"]),
            Dist("nvidia-cuda-runtime-cu12"),
            Dist("nvidia-cublas-cu12"),
            Dist("triton"),
            Dist("jinja2"),
            Dist("numpy"),
        ],
    )
    assert names(remote_setup.linked_distributions()) == [
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "torch",
        "torchvision",
        "triton",
    ]


@pytest.mark.parametrize(
    "torch_version, numpy",
    [
        ("2.1.0", "numpy<2"),
        ("2.2.2", "numpy<2"),
        ("2.3.0", "numpy"),
        ("2.15.0+cu132", "numpy"),
        (None, "numpy"),
    ],
)
def test_numpy_2_only_with_pytorch_built_for_it(monkeypatch, torch_version, numpy):
    installed(monkeypatch, [Dist("torch", torch_version)] if torch_version else [])
    assert remote_setup.numpy_requirement() == numpy


def test_parse_reads_names_extras_and_the_extra_marker():
    assert remote_setup.parse(
        'cuda-toolkit[cublas, cufft]==13.2.2; platform_system == "Linux"'
    ) == ("cuda-toolkit", {"cublas", "cufft"}, None)
    assert remote_setup.parse("nvidia_cublas==13.4.1.3; extra == 'cublas'") == (
        "nvidia-cublas",
        set(),
        "cublas",
    )
    assert remote_setup.parse(
        'opt-einsum>=3.3; python_version >= "3.9" and extra == "opt-einsum"'
    ) == ("opt-einsum", set(), "opt-einsum")


class Installed:
    """A distribution as installed in a site folder: its files, found where they are"""

    def __init__(self, site, files):
        self.site, self.files = site, [Path(f) for f in files]
        # A file outside site-packages, such as a script in bin/, is listed but never linked
        for f in self.files:
            if f.parts[0] != "..":
                (site / f).parent.mkdir(parents=True, exist_ok=True)
                (site / f).write_text("")

    def locate_file(self, path):
        return self.site / path


def test_shared_namespace_folders_are_linked_one_package_at_a_time(tmp_path):
    cublas = Installed(
        tmp_path,
        [
            "nvidia/cublas/lib/libcublas.so.13",
            "nvidia/cublas/include/cublas.h",
            "nvidia_cublas-13.4.1.3.dist-info/METADATA",
        ],
    )
    bindings = Installed(
        tmp_path,
        [
            "cuda/bindings/__init__.py",
            "cuda/bindings/runtime.so",
            "cuda_bindings-13.1.0.dist-info/RECORD",
        ],
    )
    torch = Installed(
        tmp_path,
        [
            "torch/__init__.py",
            "torch/lib/libtorch.so",
            "torch-2.15.0.dist-info/METADATA",
            "../../../bin/torchrun",
        ],
    )
    assert remote_setup.link_roots(cublas) == {
        ("nvidia", "cublas"),
        ("nvidia_cublas-13.4.1.3.dist-info",),
    }
    assert remote_setup.link_roots(bindings) == {
        ("cuda", "bindings"),
        ("cuda_bindings-13.1.0.dist-info",),
    }
    assert remote_setup.link_roots(torch) == {("torch",), ("torch-2.15.0.dist-info",)}
