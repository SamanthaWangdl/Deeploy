# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import coloredlogs
import pytest
from testUtils.pytestRunner import PERF_MARKER, get_worker_id

from Deeploy.Logging import DEFAULT_FMT
from Deeploy.Logging import DEFAULT_LOGGER as log


def pytest_addoption(parser: pytest.Parser) -> None:
    """Native PyTest hook: add custom command-line options for Deeploy tests."""
    parser.addoption(
        "--skipgen",
        action = "store_true",
        default = False,
        help = "Skip network generation step",
    )
    parser.addoption(
        "--skipsim",
        action = "store_true",
        default = False,
        help = "Skip simulation step (only generate and build)",
    )
    parser.addoption(
        "--profile-untiled",
        action = "store_true",
        default = False,
        help = "Enable profiling for untiled Siracusa runs",
    )
    parser.addoption(
        "--toolchain",
        action = "store",
        default = "LLVM",
        help = "Compiler toolchain to use (LLVM or GCC)",
    )
    parser.addoption(
        "--toolchain-install-dir",
        action = "store",
        default = os.environ.get("LLVM_INSTALL_DIR"),
        help = "Path to toolchain installation directory",
    )
    parser.addoption(
        "--cmake-args",
        action = "append",
        default = [],
        help = "Additional CMake arguments (can be used multiple times)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Native PyTest hook: configure pytest for Deeploy tests."""
    # Register custom markers
    config.addinivalue_line("markers", "generic: mark test as a Generic platform test")
    config.addinivalue_line("markers", "cortexm: mark test as a Cortex-M (QEMU-ARM) platform test")
    config.addinivalue_line("markers", "mempool: mark test as a MemPool platform test")
    config.addinivalue_line("markers", "chimera: mark test as a Chimera platform test")
    config.addinivalue_line("markers", "softhier: mark test as a SoftHier platform test")
    config.addinivalue_line("markers", "snitch: mark test as a Snitch platform test")
    config.addinivalue_line("markers", "snitch_tiled: mark test as a Snitch platform test (tiled)")
    config.addinivalue_line("markers", "siracusa: mark test as a Siracusa platform test (untiled)")
    config.addinivalue_line("markers", "siracusa_tiled: mark test as a Siracusa platform test (tiled)")
    config.addinivalue_line("markers",
                            "siracusa_neureka_tiled: mark test as a Siracusa + Neureka platform test (tiled)")
    config.addinivalue_line("markers", "gap9: mark test as a GAP9 platform test")
    config.addinivalue_line("markers", "gap9_tiled: mark test as a GAP9 platform test (tiled)")
    config.addinivalue_line("markers", "xdna2: mark test as an XDNA2 (AIE2p) platform test")
    config.addinivalue_line("markers", "kernels: mark test as a kernel test (individual operators)")
    config.addinivalue_line("markers", "models: mark test as a model test (full networks)")
    config.addinivalue_line("markers", "singlebuffer: mark test as single-buffer configuration")
    config.addinivalue_line("markers", "doublebuffer: mark test as double-buffer configuration")
    config.addinivalue_line("markers", "l2: mark test as L2 default memory level")
    config.addinivalue_line("markers", "l3: mark test as L3 default memory level")
    config.addinivalue_line("markers", "wmem: mark test as using Neureka weight memory")
    config.addinivalue_line("markers", "dma: mark test as DMA test")
    config.addinivalue_line(
        "markers",
        "deeploy_internal: mark test as internal Deeploy test (state serialization, extensions, transformations)")

    # Configure logging based on verbosity
    verbosity = config.option.verbose
    if verbosity >= 3:
        coloredlogs.install(level = 'DEBUG', logger = log, fmt = DEFAULT_FMT)
    elif verbosity >= 2:
        coloredlogs.install(level = 'INFO', logger = log, fmt = DEFAULT_FMT)
    else:
        coloredlogs.install(level = 'WARNING', logger = log, fmt = DEFAULT_FMT)


@pytest.fixture(scope = "session")
def deeploy_test_dir():
    """Return the DeeployTest directory path."""
    return Path(__file__).parent


@pytest.fixture(scope = "session")
def tests_dir(deeploy_test_dir):
    """Return the Tests directory path."""
    return deeploy_test_dir / "Tests"


@pytest.fixture(scope = "session")
def toolchain_dir(request):
    """Return the toolchain installation directory."""
    toolchain_install = request.config.getoption("--toolchain-install-dir")
    if toolchain_install is None:
        pytest.skip(reason = "LLVM_INSTALL_DIR not set")
    return toolchain_install


@pytest.fixture(scope = "session", autouse = True)
def ccache_dir():
    """Setup and return ccache directory."""
    # Use existing CCACHE_DIR if already set
    if "CCACHE_DIR" in os.environ:
        return Path(os.environ["CCACHE_DIR"])

    # Fall back to /app/.ccache if it exists (for CI containers)
    ccache_path = Path("/app/.ccache")
    if ccache_path.exists():
        os.environ["CCACHE_DIR"] = str(ccache_path)
        return ccache_path

    return None


@pytest.fixture
def skipgen(request):
    """Return whether to skip network generation."""
    return request.config.getoption("--skipgen")


@pytest.fixture
def skipsim(request):
    """Return whether to skip simulation."""
    return request.config.getoption("--skipsim")


@pytest.fixture
def profile_untiled(request):
    """Return whether untiled profiling is enabled."""
    return request.config.getoption("--profile-untiled")


@pytest.fixture
def toolchain(request):
    """Return the toolchain to use."""
    return request.config.getoption("--toolchain")


@pytest.fixture
def cmake_args(request):
    """Return additional CMake arguments."""
    return request.config.getoption("--cmake-args")


# ---------------------------------------------------------------------------
# Performance summary
#
# Every test funnels through run_and_assert_test, which prints one PERF_MARKER
# line per simulation. pytest_runtest_logreport runs on the xdist master for
# every worker's report, so scraping the captured stdout there is what makes the
# numbers from all four workers land in one process. Under GitHub Actions the
# same table is appended to the job summary, so the cycles show up on the check
# page without opening the log.
# ---------------------------------------------------------------------------

_PERF_RESULTS: List[Dict[str, Any]] = []

_CYCLES_RE = re.compile(re.escape(PERF_MARKER) + r"\s+runtime_cycles=(\d+)")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Collect the runtime cycles each test reported."""
    if report.when != "call" or report.outcome not in ("passed", "failed"):
        return

    blob = "\n".join(filter(None, [getattr(report, "capstdout", None), getattr(report, "capstderr", None)]))
    matches = _CYCLES_RE.findall(blob)
    if not matches:
        return

    # A test that runs several simulations reports the last one, matching what
    # its assertions were made against.
    _PERF_RESULTS.append({
        "nodeid": report.nodeid,
        "outcome": report.outcome,
        "cycles": int(matches[-1]),
    })


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    """Print the cycle table and append it to the GitHub Actions job summary."""
    # This hook also fires in every xdist worker, each holding only the tests it
    # ran. The master's logreport hook sees all of them, so it is the only one
    # with a complete table -- and the only one that may append to the job
    # summary, which is a shared file.
    if get_worker_id() != "master" or not _PERF_RESULTS:
        return

    results = sorted(_PERF_RESULTS, key = lambda r: r["nodeid"])

    terminalreporter.write_sep("=", "Performance Summary")
    width = max(len(r["nodeid"]) for r in results)
    for r in results:
        mark = "PASS" if r["outcome"] == "passed" else "FAIL"
        terminalreporter.write_line(f"  [{mark}] {r['nodeid']:<{width}}  {r['cycles']:>12,} cycles")

    summaryPath = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summaryPath:
        return

    lines = ["## Performance Summary", "", "| Test | Status | Runtime (cycles) |", "|---|:---:|---:|"]
    for r in results:
        status = ":white_check_mark:" if r["outcome"] == "passed" else ":x:"
        lines.append(f"| `{r['nodeid']}` | {status} | {r['cycles']:,} |")
    lines.append("")

    try:
        with open(summaryPath, "a") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        terminalreporter.write_line(f"[perf-summary] could not write GITHUB_STEP_SUMMARY: {e}")
