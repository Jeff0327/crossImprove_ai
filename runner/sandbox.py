"""
runner/sandbox.py  —  THE JUDGE. The loop must NOT edit this file.

Runs a verifier's check() against a solution in a SEPARATE process. Phase 3
hardening addresses review findings C1/C2/C3:

  * C3 — the verdict is returned over a DEDICATED result file, not stdout, so a
         verifier that prints fake JSON cannot spoof the result.
  * C2 — the child runs in its own session (start_new_session) so the whole
         process group can be cleaned up; grandchildren can't leak silently.
  * C1 — POSIX rlimits (CPU, address space, file size, NPROC) + a stripped
         environment + isolated mode (-I) are applied before exec. This is real
         but PARTIAL: it does NOT sandbox the filesystem or network. For untrusted
         self-generated code you STILL must run inside a container with no host
         mounts and egress disabled. require_hardening=True fails CLOSED where
         rlimits are unavailable (e.g. Windows) instead of pretending.

A subprocess + rlimits bounds runaway resource use; it is NOT a complete boundary
against a determined adversary. Containerize for untrusted code.
"""
from __future__ import annotations
import subprocess
import sys
import os
import json
import tempfile
import textwrap

try:
    import resource  # POSIX only
    _HAVE_RLIMIT = True
except ImportError:  # pragma: no cover - Windows
    resource = None
    _HAVE_RLIMIT = False

# harness writes its verdict to RESULT_FD_PATH, never to stdout.
_HARNESS = textwrap.dedent('''
    import json, os
    _verifier_src = {verifier_src!r}
    _solution = {solution!r}
    _out = os.environ["RESULT_FD_PATH"]
    _ns = {{}}
    try:
        exec(compile(_verifier_src, "<verifier>", "exec"), _ns)
        ok = bool(_ns["check"](_solution))
        res = {{"ok": ok}}
    except Exception as e:
        res = {{"ok": False, "error": repr(e)}}
    with open(_out, "w") as _f:
        _f.write(json.dumps(res))
''')


def _limits(cpu_seconds: int, mem_bytes: int):
    def _apply():  # runs in the child after fork, before exec
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass
    return _apply


def verify(verifier_src: str, solution: str, timeout: float = 5.0,
           mem_bytes: int = 512 * 1024 * 1024,
           require_hardening: bool = False) -> dict:
    """Execute check(solution) in a hardened child. Returns {'ok': bool, ...}."""
    if require_hardening and not _HAVE_RLIMIT:
        return {"ok": False, "error": "hardening unavailable (no rlimits); refusing"}

    workdir = tempfile.mkdtemp(prefix="sbx_")
    src_path = os.path.join(workdir, "harness.py")
    res_path = os.path.join(workdir, "result.json")
    with open(src_path, "w") as f:
        f.write(_HARNESS.format(verifier_src=verifier_src, solution=solution))

    env = {"PATH": "/usr/bin:/bin", "RESULT_FD_PATH": res_path}  # stripped env
    kwargs = dict(capture_output=True, text=True, timeout=timeout, env=env, cwd=workdir)
    if _HAVE_RLIMIT:
        kwargs["preexec_fn"] = _limits(int(timeout) + 1, mem_bytes)
        kwargs["start_new_session"] = True

    proc = None
    try:
        proc = subprocess.run([sys.executable, "-I", src_path], **kwargs)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}
    finally:
        try:
            if os.path.exists(src_path):
                os.unlink(src_path)
        except OSError:
            pass

    try:
        with open(res_path) as f:
            verdict = json.loads(f.read())
    except (OSError, json.JSONDecodeError):
        rc = getattr(proc, "returncode", "?")
        err = (getattr(proc, "stderr", "") or "")[:200]
        verdict = {"ok": False, "error": f"no result (rc={rc}); stderr={err}"}
    finally:
        try:
            if os.path.exists(res_path):
                os.unlink(res_path)
            os.rmdir(workdir)
        except OSError:
            pass
    return verdict
