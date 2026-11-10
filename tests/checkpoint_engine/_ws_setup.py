# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch-free Ray worker bootstrap for the weight-sync benchmark.

Why this module exists
----------------------
On Ray-on-Arnold (GB200 / verl-base image) the entrypoint script only runs on
the head pod; worker pods have a separate filesystem with only the image's
built-in packages. Worse, Ray worker processes are spawned by raylet with a
different HOME/user than the job driver, so Python user-site directories such
as ``/home/tiger/.local/lib/python3.12/site-packages`` (where torch lives in
the image) are NOT on ``sys.path`` -- actors whose module does
``import torch`` at top level die with ``No module named 'torch'``, even when
``py_executable`` points at the right interpreter.

This module is shipped to every worker via ``runtime_env["py_modules"]`` and
its :func:`worker_process_setup_hook` runs in each worker process BEFORE any
actor class is imported. It therefore must never import torch (or any heavy
dep) itself -- stdlib + ray only.

It provides:

* :func:`worker_process_setup_hook` -- fix ``sys.path`` (user-site, system
  site, deps target dir) so torch becomes importable; never raises.
* :func:`probe` -- a Ray remote task reporting the real interpreter /
  ``sys.path`` / HOME / package locations of one node (ground truth).
* :func:`ensure_deps` -- a Ray remote task run once per node that pip-installs
  the RDMA-engine dependencies (cupy / pyzmq / nixl / mooncake / ...) into a
  fixed target dir on that node, guarded by a file lock + sentinel.
"""
import getpass
import glob
import importlib
import json
import os
import re
import site
import socket
import subprocess
import sys
import time

_HINT = "[ws_setup]"


def _log(msg: str) -> None:
    print(f"{_HINT} {msg}", file=sys.stderr, flush=True)


# The setup hook runs in EVERY Ray worker process, so its per-process chatter is
# emitted dozens of times. On Ray-on-Arnold the only failure output that reaches
# streamlog is a 20,000-char tail, and that chatter was crowding the actual
# Python traceback out of it. Keep it behind WS_VERBOSE; warnings still log.
_VERBOSE = os.environ.get("WS_VERBOSE", "0").lower() not in ("", "0", "false", "no")


def _vlog(msg: str) -> None:
    if _VERBOSE:
        _log(msg)


# Pip packages -> import module name. Order matters: cheap/common first.
# `import verl.checkpoint_engine` runs verl/__init__ -> protocol (torch_functional
# needs transformers) and checkpoint_engine.base -> verl.workers.config ->
# profiler (codetiming) / omegaconf. The rollout serving stack (fastapi/uvicorn/
# pydantic/yaml/orjson) is pulled in unguarded on some import paths; install the
# whole verl-core lightweight set so no deep import chain blows up at runtime.
_PKG_JOBS = [
    ("pyzmq", "zmq"),
    ("safetensors", "safetensors"),
    ("tensordict", "tensordict"),
    # transformers + its tightly-coupled deps MUST come from the same ws_deps
    # target so they load as one consistent set. If huggingface_hub resolves to
    # the (newer/older) system copy we get `cannot import name 'is_offline_mode'`.
    ("huggingface_hub", "huggingface_hub"),
    ("tokenizers", "tokenizers"),
    ("transformers", "transformers"),
    ("codetiming", "codetiming"),
    ("omegaconf", "omegaconf"),
    ("hydra-core", "hydra"),
    ("pyyaml", "yaml"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("orjson", "orjson"),
    ("pytest", "pytest"),
]
# Lib-only wheels (no importable top-level module): provide libcudart.so.12 for
# the cuda12-built nixl/mooncake engines on a cuda13 (cu130) image.
_LIB_PKGS = ["nvidia-cuda-runtime-cu12"]
_CUPY_CANDIDATES = [("cupy-cuda13x", "cupy"), ("cupy-cuda12x", "cupy")]
_RDMA_PKGS = [("nixl", "nixl"), ("mooncake-transfer-engine", "mooncake")]

# Nodes ship a torch build (2.14+cu130) that is sometimes missing its companion
# nvidia-* CUDA lib wheels -- e.g. worker user-site torch without libcusparseLt.
# Map the dlopen soname fragment in the ImportError to the pip wheel that ships
# it (cu13 variants). Order matters: more specific keys first (cusparselt before
# cublasLt/cublas; cublasLt before cublas).
_TORCH_SO_PKGS = [
    ("libcusparselt", "nvidia-cusparselt-cu13"),
    ("libnvjitlink", "nvidia-nvjitlink-cu13"),
    ("libcublaslt", "nvidia-cublas-cu13"),
    ("libcublas", "nvidia-cublas-cu13"),
    ("libcusparse.", "nvidia-cusparse-cu13"),
    ("libcudnn", "nvidia-cudnn-cu13"),
    ("libnccl", "nvidia-nccl-cu13"),
    ("libcufft", "nvidia-cufft-cu13"),
    ("libcurand", "nvidia-curand-cu13"),
    ("libcusolver", "nvidia-cusolver-cu13"),
    ("libcupti", "nvidia-cuda-cupti-cu13"),
    ("libcudart.so.13", "nvidia-cuda-runtime-cu13"),
    ("libnpp", "nvidia-npp-cu13"),
    ("libnvtx", "nvidia-nvtx-cu13"),
]
# Installed in one shot when the missing soname is not in the table above.
_TORCH_FALLBACK_BUNDLE = [
    "nvidia-cusparselt-cu13",
    "nvidia-cuda-runtime-cu13",
    "nvidia-nvjitlink-cu13",
    "nvidia-cublas-cu13",
    "nvidia-cusparse-cu13",
]
_TORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"

# Pinned rather than bare "torch": on aarch64 (GB200 / Grace) plain PyPI serves a
# build that is not the one this stack is validated against, and an unpinned
# install silently changes the CUDA build. 2.11.0+cu130 from the cu130 index is
# what seed/verl-wrapper's uv.lock resolves for BOTH x86_64 and aarch64 cp312,
# so it is the spec to reproduce. Override with WS_TORCH_SPEC if that moves.
_TORCH_SPEC = os.environ.get("WS_TORCH_SPEC", "torch==2.11.0")

_PROBE_MODS = (
    "torch", "numpy", "pytest", "ray", "cupy", "zmq", "safetensors",
    "tensordict", "huggingface_hub", "tokenizers", "transformers",
    "codetiming", "omegaconf", "hydra", "yaml", "fastapi", "uvicorn",
    "pydantic", "orjson", "nixl", "mooncake", "verl",
)


def deps_target() -> "str | None":
    """Return a writable dir used as pip --target for node-local deps."""
    candidates = []
    env_dir = os.environ.get("WS_DEPS_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates += ["/opt/tiger/ws_deps", "/tmp/ws_deps"]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            probe_file = os.path.join(d, ".ws_write_probe")
            with open(probe_file, "w") as f:
                f.write("ok")
            os.remove(probe_file)
            return d
        except Exception as e:  # noqa: BLE001
            _log(f"deps target {d} not usable: {type(e).__name__}: {e}")
    return None


def candidate_site_dirs() -> list:
    """All directories that could hold site-packages on these images."""
    dirs = []
    t = deps_target()
    if t:
        dirs.append(t)
    try:
        u = site.getusersitepackages()
        if u:
            dirs.append(u)
    except Exception:  # noqa: BLE001
        pass
    try:
        dirs.extend(site.getsitepackages())
    except Exception:  # noqa: BLE001
        pass
    vtag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    dirs.append(os.path.join(sys.prefix, "lib", vtag, "site-packages"))
    dirs += [
        f"/usr/local/lib/{vtag}/dist-packages",
        f"/usr/local/lib/{vtag}/site-packages",
        f"/usr/lib/{vtag}/dist-packages",
        f"/usr/lib/{vtag}/site-packages",
        f"/home/tiger/.local/lib/{vtag}/site-packages",
        f"/root/.local/lib/{vtag}/site-packages",
    ]
    patterns = [
        "/home/*/.local/lib/python*/site-packages",
        "/home/*/lib/python*/site-packages",
        "/root/.local/lib/python*/site-packages",
        "/usr/local/lib/python*/dist-packages",
        "/usr/local/lib/python*/site-packages",
        "/usr/lib/python*/dist-packages",
        "/opt/conda/lib/python*/site-packages",
        "/opt/tiger/*/lib/python*/site-packages",
        "/opt/tiger/*/*/lib/python*/site-packages",
        "/opt/*/lib/python*/site-packages",
    ]
    for p in patterns:
        dirs.extend(glob.glob(p))
    seen = set()
    out = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def add_site_dir(d: str) -> bool:
    """Append a site dir to sys.path and process its .pth files."""
    if not d or not os.path.isdir(d):
        return False
    added = False
    if d not in sys.path:
        sys.path.append(d)
        added = True
    try:
        site.addsitedir(d)
    except Exception:  # noqa: BLE001
        pass
    return added


def prefer_dir_front(d: str) -> bool:
    """Move deps target to the FRONT of sys.path.

    pip installs a self-consistent set into this target (transformers + a matching
    huggingface_hub + tokenizers, cu13 torch, ...). Appending it lets the system
    dist-packages shadow individual members and splits matched sets (e.g. ws_deps
    transformers vs system huggingface_hub -> `cannot import name is_offline_mode`).
    Ray core is already imported (and cached in sys.modules) before the hook runs,
    so fronting the target only affects user-code imports (verl/torch/transformers).
    """
    if not d or not os.path.isdir(d):
        return False
    try:
        while d in sys.path:
            sys.path.remove(d)
    except Exception:  # noqa: BLE001
        pass
    sys.path.insert(0, d)
    try:
        site.addsitedir(d)
    except Exception:  # noqa: BLE001
        pass
    return True


def _preload_cudart12(target: "str | None" = None) -> list:
    """RTLD_GLOBAL-preload libcudart.so.12 so cuda12-built nixl/mooncake resolve it.

    On a cu13 (cuda13) image only libcudart.so.13 ships; mooncake/nixl wheels are
    built against cuda12 and dlopen libcudart.so.12 as a NEEDED entry. Loading the
    .12 runtime into the global namespace makes that symbol resolvable; .12 and .13
    have different sonames so torch's .13 is unaffected.
    """
    loaded = []
    try:
        import ctypes

        bases = [target, deps_target(), "/opt/tiger/ws_deps", "/tmp/ws_deps",
                 "/usr/local/cuda/lib64", "/usr/local/cuda-12/lib64"]
        patterns = [
            "nvidia/cuda_runtime/lib/libcudart.so.12*",
            "**/libcudart.so.12*",
            "libcudart.so.12*",
        ]
        cands = []
        seen = set()
        for b in bases:
            if not b:
                continue
            for pat in patterns:
                for so in glob.glob(os.path.join(b, pat), recursive=True):
                    if so not in seen and os.path.exists(os.path.realpath(so)):
                        seen.add(so)
                        cands.append(so)
        for so in cands:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                loaded.append(so)
            except Exception as e:  # noqa: BLE001
                _log(f"preload cudart12 failed {so}: {type(e).__name__}: {e}")
        if loaded:
            os.environ["WS_CUDART12_PRELOADED"] = "1"
            _log(f"preloaded libcudart.so.12 (RTLD_GLOBAL): {loaded[:2]}")
    except Exception as e:  # noqa: BLE001
        _log(f"_preload_cudart12 error (non-fatal): {type(e).__name__}: {e}")
    return loaded


def _preload_target_nvidia(target: "str | None" = None) -> list:
    """RTLD_GLOBAL-preload every nvidia/*/lib/*.so* installed into the deps target.

    The node's torch often lives in user-site while we install its companion CUDA
    libs into a separate --target root. torch's $ORIGIN-relative RPATH therefore
    cannot see them, but a globally-preloaded soname resolves regardless of root.
    Load runtimes/cores first so other libs' NEEDED entries resolve; failures are
    tolerated (a lib whose deps are absent simply stays unloaded).
    """
    import ctypes

    roots = []
    for r in (target, deps_target(), "/opt/tiger/ws_deps", "/tmp/ws_deps"):
        if r and r not in roots and os.path.isdir(r):
            roots.append(r)
    sos = []
    seen = set()
    for r in roots:
        for so in glob.glob(os.path.join(r, "nvidia", "*", "lib", "**", "*.so*"), recursive=True):
            rp = os.path.realpath(so)
            if rp not in seen:
                seen.add(rp)
                sos.append(so)

    def _rank(p: str) -> int:
        b = os.path.basename(p)
        for i, key in enumerate(
            ("libcudart", "libnvjitlink", "libcublas", "libcusparse", "libcudnn")
        ):
            if b.startswith(key):
                return i
        return 9

    sos.sort(key=lambda p: (_rank(p), p))
    loaded = []
    for so in sos:
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            loaded.append(os.path.basename(so))
        except Exception as e:  # noqa: BLE001
            _vlog(f"preload nvidia lib failed {os.path.basename(so)}: {type(e).__name__}")
    if loaded:
        os.environ["WS_NVIDIA_LIBS_PRELOADED"] = str(len(loaded))
        _vlog(f"preloaded nvidia libs (RTLD_GLOBAL) x{len(loaded)}: {loaded[:6]}")
    return loaded


def _purge_torch_modules() -> None:
    """Drop a partially-imported/broken torch so a repair is re-evaluated."""
    for k in list(sys.modules):
        if k == "torch" or k.startswith("torch."):
            sys.modules.pop(k, None)


def _torch_reinit_error(msg: str) -> bool:
    """True if msg is torch's C extension refusing to initialize a second time.

    Dropping torch from sys.modules and re-importing it in a process where the
    extension module is already loaded re-runs its docstring registration and
    raises RuntimeError("function '_has_torch_function' already has a
    docstring"). That says torch is already up, not that it is broken.
    """
    return "already has a docstring" in msg


def _torch_health() -> "tuple[bool, str, str | None]":
    """Return (ok, error, missing_soname).

    A torch that is already imported is reported healthy WITHOUT touching
    sys.modules: this runs after the deps pass, which imports tensordict and
    transformers, and both pull torch in. Purging it here and re-importing hits
    the re-init RuntimeError above and makes a working torch look broken.
    """
    mod = sys.modules.get("torch")
    if mod is not None and getattr(mod, "__version__", None):
        return True, "", None
    _purge_torch_modules()
    try:
        importlib.import_module("torch")
        return True, "", None
    except BaseException as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if _torch_reinit_error(msg):
            return True, "", None
        soname = None
        m = re.search(r"([A-Za-z0-9_.+\-]+\.so(?:\.[0-9A-Za-z]+)*)", msg)
        if m:
            soname = os.path.basename(m.group(1)).lower()
        return False, msg, soname


def _wheel_for_soname(soname: "str | None") -> "str | None":
    if not soname:
        return None
    s = soname.lower()
    for key, wheel in _TORCH_SO_PKGS:
        if key in s:
            return wheel
    return None


def _pip_retry_indexes(target: str, spec: str, timeout: int = 1800) -> "tuple[int, list]":
    """pip install into target, retrying the cu130 and NVIDIA indexes.

    cu130 goes first: it is the index the stack is pinned to, and for torch on
    aarch64 it is the only one that serves the validated build.
    """
    attempts = [
        ["--index-url", _TORCH_CU130_INDEX],
        None,
        ["--extra-index-url", "https://pypi.nvidia.com"],
    ]
    traces = []
    rc = -1
    for extra in attempts:
        rc, out = _pip_install(target, spec, timeout=timeout, extra_args=extra)
        traces.append({"extra": extra, "rc": rc, "tail": out[-500:] if rc else ""})
        if rc == 0:
            break
    return rc, traces


def _ensure_torch(target: str, report: dict) -> None:
    """Guarantee a usable torch on this node.

    If torch imports, do nothing. Otherwise read the missing soname from the
    ImportError, install the matching nvidia-cu13 lib wheel into the deps target
    and RTLD_GLOBAL-preload it; loop to cover several missing libs. Only when the
    torch package itself is absent or unrepairable do we fall back to a full
    self-consistent cu130 torch install. Idempotent and node-local-locked.
    """
    import fcntl

    res = {"action": None, "ok": False, "version": None, "file": None, "rounds": []}
    os.makedirs(target, exist_ok=True)
    lock_path = os.path.join(target, ".torch.lock")
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        add_site_dir(target)
        prefer_dir_front(target)
        _preload_cudart12(target)
        _preload_target_nvidia(target)

        ok, err, soname = _torch_health()
        if ok:
            res["action"] = "present"
        else:
            res["initial_error"] = err[:300]
            _log(f"torch not healthy, will repair: {err[:200]}")
            full_tried = False
            for rnd in range(6):
                missing_torch = "no module named 'torch'" in err.lower()
                wheel = None if missing_torch else _wheel_for_soname(soname)
                if missing_torch:
                    spec = _TORCH_SPEC
                    rc, traces = _pip_retry_indexes(target, spec)
                    action = "install-torch"
                    full_tried = True
                elif wheel:
                    rc, traces = _pip_retry_indexes(target, wheel)
                    action = f"install-{wheel}"
                else:
                    spec = " ".join(_TORCH_FALLBACK_BUNDLE)
                    rc, traces = _pip_retry_indexes(target, spec)
                    action = "install-fallback-bundle"
                add_site_dir(target)
                prefer_dir_front(target)
                _preload_cudart12(target)
                _preload_target_nvidia(target)
                ok, err, soname = _torch_health()
                res["rounds"].append({
                    "round": rnd,
                    "action": action,
                    "soname": soname,
                    "rc": rc,
                    "ok": ok,
                    "attempts": traces,
                    "error": None if ok else err[:200],
                })
                _log(f"torch repair round {rnd} {action} rc={rc} ok={ok} so={soname}")
                if ok:
                    res["action"] = action
                    break
                if full_tried:
                    break
            if not ok and not full_tried:
                rc, traces = _pip_retry_indexes(target, "torch")
                add_site_dir(target)
                prefer_dir_front(target)
                _preload_target_nvidia(target)
                ok, err, _ = _torch_health()
                res["rounds"].append({
                    "round": -1, "action": "install-torch-last-resort",
                    "rc": rc, "ok": ok, "attempts": traces,
                    "error": None if ok else err[:200],
                })
                full_tried = True
                if ok:
                    res["action"] = "install-torch-last-resort"

        if ok:
            try:
                t = importlib.import_module("torch")
                res["ok"] = True
                res["version"] = getattr(t, "__version__", "ok")
                res["file"] = getattr(t, "__file__", None)
                try:
                    res["cuda_available"] = bool(t.cuda.is_available())
                except Exception:  # noqa: BLE001
                    res["cuda_available"] = None
            except Exception as e:  # noqa: BLE001
                ok = False
                res["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        else:
            res["error"] = err[:300]

    report["torch_repair"] = res
    _log(
        f"torch guarantee: ok={res['ok']} action={res['action']} "
        f"version={res.get('version')} rounds={len(res['rounds'])}"
    )


def _torch_importable() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except BaseException:  # noqa: BLE001
        return False


def fix_sys_path(verbose: bool = False) -> tuple:
    """Add every plausible site dir; if torch still missing, scan for it."""
    added = []
    for d in candidate_site_dirs():
        if add_site_dir(d):
            added.append(d)
    torch_ok = _torch_importable()
    if not torch_ok:
        for d in candidate_site_dirs():
            if os.path.exists(os.path.join(d, "torch", "__init__.py")):
                add_site_dir(d)
        torch_ok = _torch_importable()
        if not torch_ok:
            _log("torch STILL not importable after filesystem scan")
    if verbose:
        _vlog(
            f"fix_sys_path added {len(added)} dirs; torch_ok={torch_ok}; "
            f"user_site={site.getusersitepackages()}; HOME={os.environ.get('HOME')}"
        )
    return added, torch_ok


def worker_process_setup_hook(worker_id=None) -> None:
    """Runs in every Ray worker process before any actor/task class loads.

    Must be defensive: an exception here can kill worker startup.
    """
    try:
        os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
        added, torch_ok = fix_sys_path(verbose=True)
        t = deps_target()
        if t:
            add_site_dir(t)
            prefer_dir_front(t)
        preloaded = _preload_cudart12(t)
        nv_preloaded = _preload_target_nvidia(t)
        torch_ok_after = _torch_importable()
        _vlog(
            f"hook done worker_id={worker_id} pid={os.getpid()} "
            f"user={getpass.getuser()} HOME={os.environ.get('HOME')} "
            f"torch_ok={torch_ok} torch_ok_after_preload={torch_ok_after} "
            f"deps_target={t} cudart12_preloaded={len(preloaded)} "
            f"nvidia_preloaded={len(nv_preloaded)}"
        )
    except Exception as e:  # noqa: BLE001
        _log(f"hook ERROR (non-fatal): {type(e).__name__}: {e}")


def _probe_impl() -> dict:
    fix_sys_path()
    t = deps_target()
    if t:
        add_site_dir(t)
        prefer_dir_front(t)
    _preload_cudart12(t)
    info = {
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "version": sys.version.split()[0],
        "machine": os.uname().machine,
        "pid": os.getpid(),
        "user": getpass.getuser(),
        "HOME": os.environ.get("HOME"),
        "USER": os.environ.get("USER"),
        "cwd": os.getcwd(),
        "sys_path": list(sys.path),
        "user_site": site.getusersitepackages(),
        "deps_target": t,
        "candidate_dirs": [d for d in candidate_site_dirs() if os.path.isdir(d)],
    }
    deps = {}
    for m in _PROBE_MODS:
        try:
            mod = importlib.import_module(m)
            deps[m] = {
                "ok": True,
                "version": str(getattr(mod, "__version__", "ok")),
                "file": getattr(mod, "__file__", None),
            }
        except Exception as e:  # noqa: BLE001
            deps[m] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    info["deps"] = deps
    try:
        import torch  # noqa: F401

        info["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "count": torch.cuda.device_count(),
            "name0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_file": torch.__file__,
        }
    except Exception as e:  # noqa: BLE001
        info["torch_cuda"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    return info


_RD_RELAY = os.environ.get("WS_PIP_PROXY", "http://sys-proxy-rd-relay.byted.org:8118")


def _pip_install(target: str, pkg: str, timeout: int = 1200, extra_args=None) -> tuple:
    """pip install into target; retry through the RD relay proxy if it fails.

    Some nodes reach the wheel indexes only through the relay, which is why the
    job entrypoint retries the same way. Without the retry a worker pod silently
    ends up missing a package the head pod installed fine.
    """
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--no-input", "--disable-pip-version-check", "--no-cache-dir",
        "--target", target, pkg,
    ]
    if extra_args:
        cmd[5:5] = list(extra_args)
    base_env = dict(os.environ)
    base_env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    proxy_env = dict(base_env)
    proxy_env["https_proxy"] = proxy_env["http_proxy"] = _RD_RELAY

    rc, out = -3, ""
    for env in (base_env, proxy_env):
        try:
            p = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout, env=env,
            )
            rc, out = p.returncode, (p.stdout or "")[-1200:]
        except subprocess.TimeoutExpired:
            rc, out = -1, f"TIMEOUT after {timeout}s installing {pkg}"
        except Exception as e:  # noqa: BLE001
            rc, out = -2, f"{type(e).__name__}: {str(e)[:300]}"
        if rc == 0:
            break
    return rc, out


def _module_ok(mod: str) -> "tuple[bool, str]":
    try:
        m = importlib.import_module(mod)
        return True, str(getattr(m, "__version__", "ok"))
    except Exception as e:  # noqa: BLE001
        # An already-initialized torch extension is present, not missing.
        if _torch_reinit_error(f"{type(e).__name__}: {e}"):
            m = sys.modules.get(mod)
            return True, str(getattr(m, "__version__", "ok"))
        return False, ""


def _ensure_deps_impl() -> dict:
    fix_sys_path()
    target = deps_target()
    report = {"target": target, "results": {}, "cached": False}
    if target is None:
        report["fatal"] = "no writable deps target dir"
        return report

    import fcntl

    lock_path = os.path.join(target, ".install.lock")
    sentinel = os.path.join(target, ".install.done")
    os.makedirs(target, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if os.path.exists(sentinel):
            report["cached"] = True
        else:
            jobs = list(_PKG_JOBS)
            # lib-only wheels (libcudart.so.12) — install, no import check
            for libpkg in _LIB_PKGS:
                rc, out = _pip_install(target, libpkg)
                report["results"][libpkg] = {
                    "status": "ok" if rc == 0 else "fail",
                    "rc": rc,
                    "tail": out if rc != 0 else "",
                }
            # cupy: try cu13 (Blackwell) then cu12
            jobs.append(("__CUPY__", "cupy"))
            jobs.extend(_RDMA_PKGS)
            for pkg, mod in jobs:
                ok, ver = _module_ok(mod)
                if ok:
                    report["results"][pkg] = {"status": "already", "version": ver}
                    continue
                if pkg == "__CUPY__":
                    done = False
                    for cpkg, _ in _CUPY_CANDIDATES:
                        rc, out = _pip_install(target, cpkg)
                        add_site_dir(target)
                        ok, ver = _module_ok("cupy")
                        report["results"][cpkg] = {
                            "status": "ok" if ok else ("fail" if rc != 0 else "installed-but-import-failed"),
                            "rc": rc,
                            "version": ver,
                            "tail": out if not ok else "",
                        }
                        if ok:
                            done = True
                            break
                    if not done:
                        _log("all cupy wheel candidates failed")
                    continue
                rc, out = _pip_install(target, pkg)
                add_site_dir(target)
                ok, ver = _module_ok(mod)
                entry = {
                    "status": "ok" if ok else ("fail" if rc != 0 else "installed-but-import-failed"),
                    "rc": rc,
                    "version": ver,
                }
                if not ok and pkg == "nixl":
                    # retry nixl against the NVIDIA index as a fallback
                    rc2, out2 = _pip_install(
                        target, pkg, extra_args=["--extra-index-url", "https://pypi.nvidia.com"]
                    )
                    add_site_dir(target)
                    ok2, ver2 = _module_ok(mod)
                    entry = {
                        "status": "ok" if ok2 else "fail",
                        "rc": rc2,
                        "version": ver2,
                        "tail": (out + "\n--retry--\n" + out2) if not ok2 else "",
                    }
                elif not ok:
                    entry["tail"] = out
                report["results"][pkg] = entry
            with open(sentinel, "w") as f:
                f.write(str(time.time()))

    add_site_dir(target)
    prefer_dir_front(target)
    _preload_cudart12(target)
    # Deterministic torch guarantee -- runs every node even when the deps
    # sentinel is cached, so a broken node-preinstalled torch gets repaired.
    _ensure_torch(target, report)
    _preload_target_nvidia(target)
    final = {}
    for mod in ("torch", "numpy", "cupy", "zmq", "safetensors", "tensordict",
                "huggingface_hub", "tokenizers", "transformers",
                "codetiming", "omegaconf", "hydra", "yaml",
                "fastapi", "uvicorn", "pydantic", "orjson",
                "nixl", "mooncake", "pytest"):
        ok, ver = _module_ok(mod)
        if ok:
            try:
                m = importlib.import_module(mod)
                final[mod] = {"ok": True, "version": ver, "file": getattr(m, "__file__", None)}
            except Exception:  # noqa: BLE001
                final[mod] = {"ok": True, "version": ver}
        else:
            final[mod] = {"ok": False}
    report["final"] = final

    # Whole-chain smoke test: this is the exact import the benchmark driver/actors
    # need. Report success or the precise missing module so a failure is visible in
    # [deps] instead of crashing later during actor engine_create.
    ce = {"ok": False, "error": None}
    for p in ("/opt/tiger/open_verl", os.getcwd()):
        if p and os.path.isdir(os.path.join(p, "verl")) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        import importlib as _il
        _il.import_module("verl.checkpoint_engine")
        ce = {"ok": True, "error": None}
    except Exception as _e:  # noqa: BLE001
        ce = {"ok": False, "error": f"{type(_e).__name__}: {_e}"}
    report["verl_checkpoint_engine"] = ce

    report["probe"] = _probe_impl()
    return report


try:
    import ray

    probe = ray.remote(_probe_impl)
    ensure_deps = ray.remote(_ensure_deps_impl)
except Exception as _e:  # noqa: BLE001
    ray = None
    probe = None
    ensure_deps = None

# Wired below its impl on purpose: doing it in the block above would reference a
# name defined later in this file, and the resulting NameError was swallowed as
# "ray unavailable", silently disabling every remote in this module.
ensure_uv_env = None

_UV_PROJECT = os.environ.get("WS_UV_PROJECT", "/opt/tiger/verl-wrapper")
# `fsdp` is the lightest extra carrying the engine stack the checkpoint engine
# needs (torch / tensordict / cupy); vllm and megatron cost many minutes and add
# nothing the benchmark uses.
_UV_EXTRAS = os.environ.get("WS_UV_EXTRAS", "fsdp").split()
# Resolved but NOT installed:
#   verl -- supplied via PYTHONPATH ("verl from anywhere", per the wrapper's
#           pyproject) so the benchmark measures whichever checkout we publish;
#   ray  -- the cluster already runs the platform's ray build and a client from a
#           different build cannot start workers against it ("worker startup
#           repeatedly failed"), so the venv is created with
#           --system-site-packages and inherits the platform ray instead.
_UV_SKIP_PACKAGES = os.environ.get("WS_UV_SKIP_PACKAGES", "verl ray").split()
# verl[fsdp] does not carry transformers (only the vllm/sglang extras do), yet
# verl/protocol.py imports it at module scope. The wrapper's own
# override-dependencies pin transformers==5.3.0 for every extra that is not
# vllm/sglang/cpu, so that is the version to add -- not a guess.
_UV_EXTRA_PKGS = os.environ.get("WS_UV_EXTRA_PKGS", "transformers==5.3.0").split()
_UV_REPAIR_ROUNDS = int(os.environ.get("WS_UV_REPAIR_ROUNDS", "6"))


def _uv_bin() -> "str | None":
    for cand in (os.path.expanduser("~/.local/bin/uv"), "/usr/local/bin/uv", "uv"):
        try:
            if subprocess.run([cand, "--version"], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=60).returncode == 0:
                return cand
        except Exception:  # noqa: BLE001
            continue
    return None


def _venv_install(uv: "str | None", project: str, spec: str, timeout: int = 1800) -> tuple:
    """Install one spec into the project's venv and prove it imports.

    Plain pip inside the venv goes FIRST deliberately. `uv pip install` applies
    the project's [tool.uv] override-dependencies, whose transformers pins are
    guarded by `extra == ...` markers that mean nothing outside a project sync --
    so the install resolves to nothing, exits 0, and installs nothing. That is
    exactly how transformers stayed missing while the step reported success,
    hence the import check below rather than trusting the return code.
    """
    venv_py = os.path.join(project, ".venv", "bin", "python")
    ppi = os.environ.get("WS_PPI", "https://bytedpypi.byted.org/simple")
    env = dict(os.environ)
    env.setdefault("UV_HTTP_TIMEOUT", "600")
    proxy_env = dict(env)
    proxy_env["https_proxy"] = proxy_env["http_proxy"] = _RD_RELAY
    pip_base = [venv_py, "-m", "pip", "install", "--no-input",
                "--disable-pip-version-check", spec]

    attempts = [(pip_base, env), (pip_base + ["-i", ppi], env),
                (pip_base + ["-i", ppi], proxy_env)]
    if uv:
        attempts.append(([uv, "pip", "install", spec], env))

    mod = re.split(r"[=<>!\[]", spec)[0].replace("-", "_")
    rc, out = -1, ""
    for cmd, cmd_env in attempts:
        try:
            p = subprocess.run(cmd, cwd=project, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               timeout=timeout, env=cmd_env)
            rc, out = p.returncode, (p.stdout or "")[-500:]
        except Exception as e:  # noqa: BLE001
            rc, out = -1, f"{type(e).__name__}: {str(e)[:200]}"
            continue
        if rc != 0:
            continue
        try:
            chk = subprocess.run([venv_py, "-c", f"import {mod}"], cwd=project,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, timeout=600)
        except Exception as e:  # noqa: BLE001
            rc, out = 1, f"import check error: {type(e).__name__}: {e}"
            continue
        if chk.returncode == 0:
            return 0, out
        rc, out = 1, f"exit 0 but not importable: {(chk.stdout or '')[-300:]}"
    return rc, out


def _repair_verl_imports(uv: "str | None", project: str, venv_py: str,
                         verl_src: "str | None") -> list:
    """Import verl.checkpoint_engine, installing whatever it names as missing.

    The wrapper's lock does not cover this benchmark's exact import surface, so
    rather than guessing the closure we let each ImportError name the next
    missing module and add it, bounded. Every round is recorded so a genuinely
    broken chain shows up in the report instead of looping silently.
    """
    trail = []
    env = dict(os.environ)
    if verl_src:
        env["PYTHONPATH"] = verl_src + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for _ in range(_UV_REPAIR_ROUNDS):
        try:
            p = subprocess.run([venv_py, "-c", "import verl.checkpoint_engine; print('OK')"],
                               cwd=project, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, timeout=900, env=env)
        except Exception as e:  # noqa: BLE001
            trail.append({"probe_error": f"{type(e).__name__}: {str(e)[:200]}"})
            break
        out = (p.stdout or "")[-800:]
        if p.returncode == 0:
            trail.append({"ok": True})
            break
        m = re.search(r"No module named '([A-Za-z0-9_.]+)'", out)
        if not m:
            trail.append({"rc": p.returncode, "unparsed": out[-400:]})
            break
        missing = m.group(1).split(".")[0]
        rc, tail = _venv_install(uv, project, missing)
        trail.append({"missing": missing, "install_rc": rc, "tail": tail[-200:]})
        if rc != 0:
            break
    return trail


def _ensure_uv_env_impl(verl_src: "str | None" = None) -> dict:
    """Build the verl-wrapper uv env on THIS node and report its interpreter.

    The lock is the single source of truth for the whole stack (torch
    2.11.0+cu130 on aarch64, tensordict, cupy), which is why this replaced
    ad-hoc per-package pip installs: head and worker pods run the same command
    against the same lock, so they cannot drift apart.
    """
    rep = {"hostname": socket.gethostname(), "project": _UV_PROJECT,
           "extras": _UV_EXTRAS, "skipped": _UV_SKIP_PACKAGES, "uv": None,
           "rc": None, "venv_python": None, "tail": "", "torch": None,
           "ray": None, "verl_src": None, "extra_pkgs": {}, "verl_import": None}
    uv = _uv_bin()
    rep["uv"] = uv
    if uv is None:
        rep["tail"] = "uv not found on PATH or ~/.local/bin"
        return rep
    if not os.path.isdir(_UV_PROJECT):
        rep["tail"] = f"{_UV_PROJECT} does not exist on this node"
        return rep

    # verl's code lives in the open_verl submodule; initialise it if the
    # platform's clone did not.
    local_src = os.path.join(_UV_PROJECT, "open_verl")
    if not os.path.isdir(os.path.join(local_src, "verl")):
        try:
            subprocess.run(["git", "submodule", "update", "--init", "--recursive"],
                           cwd=_UV_PROJECT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=1800)
        except Exception as e:  # noqa: BLE001
            rep["tail"] += f"submodule init failed: {type(e).__name__}: {e}; "
    rep["verl_src"] = local_src if os.path.isdir(os.path.join(local_src, "verl")) else None

    # Pre-create the venv so the platform ray stays importable; uv reuses it.
    if not os.path.isdir(os.path.join(_UV_PROJECT, ".venv")):
        try:
            vp = subprocess.run([uv, "venv", "--system-site-packages"], cwd=_UV_PROJECT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=900)
            rep["venv_create"] = {"rc": vp.returncode, "tail": (vp.stdout or "")[-300:]}
        except Exception as e:  # noqa: BLE001
            rep["venv_create"] = {"rc": -1, "tail": f"{type(e).__name__}: {e}"}

    cmd = [uv, "sync", "--frozen"]
    for pkg in _UV_SKIP_PACKAGES:
        cmd += ["--no-install-package", pkg]
    for extra in _UV_EXTRAS:
        cmd += ["--extra", extra]
    env = dict(os.environ)
    env.setdefault("UV_HTTP_TIMEOUT", "600")
    try:
        p = subprocess.run(cmd, cwd=_UV_PROJECT, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, timeout=5400, env=env)
        rep["rc"], rep["tail"] = p.returncode, (p.stdout or "")[-1200:]
    except subprocess.TimeoutExpired:
        rep["rc"], rep["tail"] = -1, "uv sync TIMEOUT"
        return rep
    except Exception as e:  # noqa: BLE001
        rep["rc"], rep["tail"] = -2, f"{type(e).__name__}: {e}"
        return rep

    venv_py = os.path.join(_UV_PROJECT, ".venv", "bin", "python")
    if not os.path.exists(venv_py):
        return rep
    rep["venv_python"] = venv_py
    for spec in _UV_EXTRA_PKGS:
        rc, tail = _venv_install(uv, _UV_PROJECT, spec)
        rep["extra_pkgs"][spec] = {"rc": rc, "tail": tail[-250:] if rc else ""}
    for probe_name, code in (
            ("torch", "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"),
            ("ray", "import ray;print(ray.__version__, ray.__file__)")):
        try:
            pr = subprocess.run([venv_py, "-c", code], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=600)
            rep[probe_name] = (pr.stdout or "").strip()[-200:]
        except Exception as e:  # noqa: BLE001
            rep[probe_name] = f"probe failed: {type(e).__name__}: {e}"
    rep["verl_import"] = _repair_verl_imports(uv, _UV_PROJECT, venv_py,
                                              verl_src or rep.get("verl_src"))
    return rep


if ray is not None:
    ensure_uv_env = ray.remote(_ensure_uv_env_impl)


def bootstrap_uv_env(node_ids, verl_src: "str | None" = None, verbose: bool = True) -> dict:
    """Run the uv sync above once per node; return {node_id: report}.

    Raises if a node did not end up with a venv interpreter, because the actors
    are launched with that exact path and a node missing it can only fail later
    as an unschedulable actor or a bare "No module named 'torch'".
    """
    if ray is None or ensure_uv_env is None:
        raise RuntimeError("ray is not importable; cannot bootstrap the uv env")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_ids = sorted(set(node_ids))
    if verbose:
        print(f"[uv] syncing {_UV_PROJECT} (extras={_UV_EXTRAS}, skip={_UV_SKIP_PACKAGES}) "
              f"on {len(node_ids)} node(s); the first run downloads the whole locked "
              "stack and can take a while ...")
    refs = {
        nid: ensure_uv_env.options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
        ).remote(verl_src)
        for nid in node_ids
    }
    reports, bad = {}, {}
    for nid, rep in ((n, _uv_get(r)) for n, r in refs.items()):
        reports[nid] = rep
        if verbose:
            print(f"[uv] node={rep.get('hostname')} sync_rc={rep.get('rc')} "
                  f"python={rep.get('venv_python')}")
            print(f"[uv]   torch={rep.get('torch')}")
            print(f"[uv]   ray={rep.get('ray')}")
            print(f"[uv]   extra_pkgs={rep.get('extra_pkgs')}")
            print(f"[uv]   verl import trail={rep.get('verl_import')}")
            if rep.get("rc"):
                print(f"[uv]   sync tail: {str(rep.get('tail'))[-700:]}")
        if not rep.get("venv_python"):
            bad[rep.get("hostname", nid[:12])] = str(rep.get("tail"))[-300:]
        else:
            trail = rep.get("verl_import") or []
            if not any(x.get("ok") for x in trail if isinstance(x, dict)):
                bad[rep.get("hostname", nid[:12])] = f"verl import not repaired: {trail}"
    if bad:
        raise RuntimeError(f"uv env not usable on node(s): {bad}")
    return reports


def _uv_get(ref):
    try:
        return ray.get(ref, timeout=5700)
    except Exception as e:  # noqa: BLE001
        return {"hostname": "?", "rc": -3, "venv_python": None,
                "tail": f"{type(e).__name__}: {str(e)[:400]}"}


def bootstrap_nodes(node_ids, require=(), verbose: bool = True) -> dict:
    """Probe and dependency-install every node in ``node_ids``, once per node.

    On Ray-on-Arnold the entrypoint only runs on the head pod, so this is the
    only way a worker pod gets anything the image does not already ship. Returns
    ``{node_id: report}``. Never raises for install failures -- the per-node
    report carries them so a benchmark can print them and SKIP -- but DOES raise
    if a module named in ``require`` is missing somewhere, since a benchmark that
    proceeds without it would fail later with a far less legible error.
    """
    if ray is None or ensure_deps is None:
        raise RuntimeError("ray is not importable; cannot bootstrap nodes")
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_ids = sorted(set(node_ids))
    reports = {}
    if verbose:
        print(f"[bootstrap] probing {len(node_ids)} nodes before actor creation ...")
    for fn, timeout, label in ((probe, 300, "probe"), (ensure_deps, 3000, "deps")):
        refs = {
            nid: fn.options(
                num_cpus=0,
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nid, soft=False),
            ).remote()
            for nid in node_ids
        }
        for nid, ref in refs.items():
            try:
                rep = ray.get(ref, timeout=timeout)
            except Exception as e:  # noqa: BLE001
                rep = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
            if label == "deps":
                reports[nid] = rep
            if verbose:
                host = (rep.get("probe") or rep).get("hostname", "?")
                print(f"[{label}] node={host} {json.dumps({k: v for k, v in rep.items() if k in ('target', 'cached', 'error')})}")

    missing = {}
    for nid, rep in reports.items():
        final = rep.get("final") or {}
        host = (rep.get("probe") or {}).get("hostname", nid[:8])
        absent = [m for m in require if not (final.get(m) or {}).get("ok")]
        if verbose:
            have = {m: (final.get(m) or {}).get("version") for m in require}
            print(f"[deps]   node={host} required={have}")
        if absent:
            missing[host] = absent
    if missing:
        raise RuntimeError(f"required modules missing after bootstrap: {missing}")
    return reports
