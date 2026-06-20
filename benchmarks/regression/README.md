# regression/ — the non-regression gate

Capabilities that must NEVER break. Every promotion must pass all of these
(`regression_ok` in the gate). This is the monotonic gate: add capabilities here
as the agent acquires them, but prefer storing a *representative sample per
capability* over an ever-growing fixed list, so cost does not explode.

Pair this with procedural generation: store the *capability* (e.g. "solves linear
equations"), and test it each run with a freshly generated instance, so passing
requires the method, not a memorized answer.
