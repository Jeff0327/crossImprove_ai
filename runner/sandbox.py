"""
runner/sandbox.py  —  THE JUDGE. The loop must NOT edit this file.

Runs a verifier's check() against a solution in a SEPARATE process. Hardening
addresses review findings C1/C2/C3:

  * C3 — the verdict is returned over a DEDICATED result file, not stdout, so a
         verifier that prints fake JSON cannot spoof the result. The result also
         carries a per-run NONCE the harness stamps and the parent checks, so the
         next-cheapest spoof — check() writing a fake {"ok": true} to the result
         file then os._exit() to skip the harness's own write — is rejected
         (wrong/absent nonce). The harness unlinks its own source first so the
         nonce can't be read back from __file__. This is still BEST-EFFORT: code
         that climbs stack frames in-process can reach the nonce; defeating that
         needs the container below. It closes the cheap, common exploits.
  * C2 — the child runs in its own session/process group; the ENTIRE group is
         SIGKILLed (os.killpg) on every exit path, so grandchildren the verifier
         spawned are reaped. This is BEST-EFFORT: a verifier that spawns AND exits
         faster than we can signal can still race a grandchild out before the
         kill lands. A hard guarantee needs a cgroup / PID namespace — i.e. the
         container you must use for untrusted code anyway. Established
         grandchildren (alive when verify() returns) are reliably reaped.
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
import secrets
import shutil
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
    # Best-effort anti-spoof: remove our own source immediately so a hostile
    # check() cannot read the integrity nonce out of this file via __file__ /
    # sys.argv[0]. On POSIX the already-loaded process keeps running after the
    # unlink. (A determined adversary can still climb stack frames to reach the
    # nonce in-process — that is exactly why untrusted code needs the container
    # this module's docstring requires. This only raises the bar against the
    # cheap, common exploit: write a fake verdict file then os._exit().)
    try:
        os.unlink(__file__)
    except OSError:
        pass
    _verifier_src = {verifier_src!r}
    _solution = {solution!r}
    _nonce = {nonce!r}
    _out = os.environ["RESULT_FD_PATH"]
    _ns = {{}}
    try:
        exec(compile(_verifier_src, "<verifier>", "exec"), _ns)
        ok = bool(_ns["check"](_solution))
        res = {{"_n": _nonce, "ok": ok}}
    except Exception as e:
        res = {{"_n": _nonce, "ok": False, "error": repr(e)}}
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


def _kill_pgid(pgid, proc) -> None:
    """SIGKILL the child's whole process group so grandchildren don't leak (C2)."""
    try:
        if _POSIX and pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:  # pragma: no cover
            if proc is not None:
                proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def verify(verifier_src: str, solution: str, timeout: float = 5.0,
           mem_bytes: int = 512 * 1024 * 1024,
           require_hardening: bool = False) -> dict:
    """Execute check(solution) in a hardened child. Returns {'ok': bool, ...}."""
    if require_hardening and not _HAVE_RLIMIT:
        return {"ok": False, "error": "hardening unavailable (no rlimits); refusing"}

    nonce = secrets.token_hex(16)          # per-run verdict integrity token
    workdir = tempfile.mkdtemp(prefix="sbx_")
    # Outer guard: whatever happens below (normal, timeout, exception), the temp
    # workdir is ALWAYS removed — the old code only cleaned up on the normal
    # path, leaking a dir per timeout/error over a long run.
    try:
        src_path = os.path.join(workdir, "harness.py")
        res_path = os.path.join(workdir, "result.json")
        with open(src_path, "w") as f:
            f.write(_HARNESS.format(verifier_src=verifier_src, solution=solution,
                                    nonce=nonce))

        err_path = os.path.join(workdir, "stderr.txt")
        env = {"PATH": "/usr/bin:/bin", "RESULT_FD_PATH": res_path}  # stripped env
        # Verdict comes from the result FILE, so we DON'T pipe stdout. Crucially,
        # using PIPE + communicate() would BLOCK until every inherited fd is
        # closed — a grandchild holding the pipe open would stall us until it
        # exits, defeating both the timeout and the group-kill. We redirect
        # stdout to /dev/null and stderr to a file, and use wait() (which only
        # tracks the direct child), so a lingering grandchild can neither stall
        # us nor escape the prompt killpg.
        errf = open(err_path, "w")
        popen_kw = dict(stdout=subprocess.DEVNULL, stderr=errf, env=env, cwd=workdir)
        if _HAVE_RLIMIT:
            popen_kw["preexec_fn"] = _limits(int(timeout) + 1, mem_bytes)
        if _POSIX:
            popen_kw["start_new_session"] = True  # own process group

        proc = None
        pgid = None
        try:
            proc = subprocess.Popen([sys.executable, "-I", src_path], **popen_kw)
            if _POSIX:
                try:
                    pgid = os.getpgid(proc.pid)   # capture now; reaped after wait
                except OSError:
                    pgid = None
            try:
                proc.wait(timeout=timeout)        # wait() does NOT block on inherited fds
            except subprocess.TimeoutExpired:
                _kill_pgid(pgid, proc)            # reap the whole group on timeout
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                return {"ok": False, "error": "timeout"}
        except Exception as e:  # noqa: BLE001
            _kill_pgid(pgid, proc)
            return {"ok": False, "error": repr(e)}
        finally:
            # reap stragglers on EVERY path (best-effort; see module docstring)
            _kill_pgid(pgid, proc)
            try:
                errf.close()
            except OSError:
                pass

        try:
            with open(res_path) as f:
                verdict = json.loads(f.read())
        except (OSError, json.JSONDecodeError):
            rc = getattr(proc, "returncode", "?")
            try:
                stderr = open(err_path).read()
            except OSError:
                stderr = ""
            return {"ok": False, "error": f"no result (rc={rc}); stderr={(stderr or '')[:200]}"}

        # Integrity check: the verdict must carry the nonce the harness wrote.
        # A spoofed result file (e.g. check() writing {"ok": true} then os._exit)
        # won't have it, so it can't flip the verdict.
        if verdict.get("_n") != nonce:
            return {"ok": False, "error": "verdict integrity check failed (nonce mismatch)"}
        verdict.pop("_n", None)
        return verdict
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
