"""
runner/sandbox.py  —  THE JUDGE. The loop must NOT edit this file.

Runs a verifier's check() against a solution in a SEPARATE process so that
self-generated code cannot monkeypatch or overwrite the scoring result in-process.

WARNING — this is a minimal isolator, adequate for trusted/templated verifiers
only. Before running untrusted self-generated code, harden it:
  * run inside a container (no host mounts)
  * disable network egress
  * apply CPU/mem/time rlimits and seccomp
  * drop privileges; read-only filesystem
A subprocess alone is NOT a security boundary against adversarial code.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
import textwrap
import os
import json

_HARNESS = textwrap.dedent('''
    import json, sys
    _verifier_src = {verifier_src!r}
    _solution = {solution!r}
    _ns = {{}}
    try:
        exec(compile(_verifier_src, "<verifier>", "exec"), _ns)
        ok = bool(_ns["check"](_solution))
        print(json.dumps({{"ok": ok}}))
    except Exception as e:
        print(json.dumps({{"ok": False, "error": repr(e)}}))
''')


def verify(verifier_src: str, solution: str, timeout: float = 5.0) -> dict:
    """Execute check(solution) in a child process. Returns {'ok': bool, ...}."""
    code = _HARNESS.format(verifier_src=verifier_src, solution=solution)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
            # TODO(harden): add preexec_fn for rlimits, env={} to strip env,
            # and run this whole call inside a network-disabled container.
        )
        out = (proc.stdout or "").strip().splitlines()
        if not out:
            return {"ok": False, "error": f"no output; stderr={proc.stderr[:200]}"}
        return json.loads(out[-1])
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": repr(e)}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
