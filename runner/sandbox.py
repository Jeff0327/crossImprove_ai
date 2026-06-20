"""
runner/sandbox.py  —  THE JUDGE. The loop must NOT edit this file.

Runs a verifier's check() against a solution in a SEPARATE process. Hardening
addresses review findings C1/C2/C3:

  * C3 — the verdict is returned over a DEDICATED result file, not stdout, so a
         verifier that prints fake JSON cannot spoof the result.
  * C2 — the child runs in its own session/process group; on timeout the ENTIRE
         group is SIGKILLed (os.killpg), so grandchildren the verifier spawned do
         not leak. (Previously start_new_session created the group but nothing
         killed it — fixed.)
  * C1 — POSIX rlimits (CPU, address space, file size, NPROC) + a stripped env +
         isolated mode (-I) are applied before exec. Real but PARTIAL: it does NOT
         sandbox the filesystem or network. For untrusted self-generated code you
         STILL must run inside a container with no host mounts and egress off.
         require_hardening=True fails CLOSED where rlimits are unavailable.

A subprocess + rlimits bounds runaway resource use; it is NOT a complete boundary
against a determined adversary. Containerize for untrusted code.
"""
from __future__ import annotations
import subprocess
import sys
import os
import json
import signal
import tempfile
import textwrap

try:
    import resource  # POSIX only
    _HAVE_RLIMIT = True
except ImportError:  # pragma: no cover - Windows
    resource = None
    _HAVE_RLIMIT = False

_POSIX = os.name == "posix"

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
    def _apply():  # child, after fork, before exec
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024,) * 2)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
        except (ValueError, OSError):
            pass
    return _apply


def _kill_group(proc) -> None:
    """SIGKILL the child's whole process group so grandchildren don't leak (C2)."""
    try:
        if _POSIX:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:  # pragma: no cover
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


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
    popen_kw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    env=env, cwd=workdir)
    if _HAVE_RLIMIT:
        popen_kw["preexec_fn"] = _limits(int(timeout) + 1, mem_bytes)
    if _POSIX:
        popen_kw["start_new_session"] = True  # own process group

    proc = None
    try:
        proc = subprocess.Popen([sys.executable, "-I", src_path], **popen_kw)
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)                # C2: reap the whole group
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return {"ok": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        if proc is not None:
            _kill_group(proc)
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
        verdict = {"ok": False, "error": f"no result (rc={rc}); stderr={(stderr or '')[:200]}"}
    finally:
        try:
            if os.path.exists(res_path):
                os.unlink(res_path)
            os.rmdir(workdir)
        except OSError:
            pass
    return verdict
