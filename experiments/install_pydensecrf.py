"""Install a Python-3.11+ compatible build of pydensecrf.

The upstream sdist ships Cython-0.29 generated C++ and includes the private
``longintrepr.h`` header, which no longer exists at the top level in Python
3.11+.  On top of that, Cython 3 rejects the unqualified ``MatrixXf`` names in
``densecrf.pyx``.  This helper applies the minimal patches needed to build the
package on modern Python/Colab, then installs it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def _run(cmd, **kwargs):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def _pypi_sdist_url() -> str:
    with urllib.request.urlopen("https://pypi.org/pypi/pydensecrf/json", timeout=60) as f:
        data = json.load(f)
    for item in data["urls"]:
        if item["packagetype"] == "sdist":
            return item["url"]
    raise RuntimeError("No pydensecrf sdist found on PyPI")


def install() -> None:
    try:
        import pydensecrf.densecrf  # noqa: F401

        print("pydensecrf already works; nothing to do")
        return
    except Exception:
        pass

    _run([sys.executable, "-m", "pip", "install", "-q", "cython>=3.1"])

    workdir = Path(tempfile.mkdtemp(prefix="pydensecrf_build_"))
    archive = workdir / "pydensecrf.tar.gz"
    print(f"Downloading {_pypi_sdist_url()}")
    urllib.request.urlretrieve(_pypi_sdist_url(), archive)
    with tarfile.open(archive) as tar:
        tar.extractall(workdir)
    src = next(p for p in workdir.iterdir() if p.is_dir() and (p / "setup.py").exists())

    if sys.version_info >= (3, 11):
        (src / "pydensecrf" / "longintrepr.h").write_text(
            "#include <cpython/longintrepr.h>\n"
        )

    pxd = src / "pydensecrf" / "densecrf.pxd"
    pxd.write_text(pxd.read_text().replace("from eigen cimport *", "from .eigen cimport *"))

    pyx = src / "pydensecrf" / "densecrf.pyx"
    text = pyx.read_text()
    text = text.replace("import eigen\ncimport eigen", "from . import eigen\nfrom . cimport eigen")
    text = re.sub(r"(?<!\.)\bMatrixXf\b", "eigen.MatrixXf", text)
    pyx.write_text(text)

    env = dict(os.environ)
    # Never use /tmp for the large build artefacts on Colab.
    env.setdefault("TMPDIR", str(workdir))
    _run([sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"], cwd=str(src), env=env)
    import pydensecrf.densecrf  # noqa: F401
    print("pydensecrf installed successfully")


if __name__ == "__main__":
    install()
