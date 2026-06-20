# anchors/ — ground-truth anchor set (1–10%)

A small set of problems with **known, human-verified answers**. The loop mixes
these into evaluation every generation and watches the score on them.

Purpose: detect Goodhart / late-stage collapse. If the proxy score keeps rising
while the anchor score falls, the optimizer is hacking the metric — the gate
(`runner/gate.py`, `anchor_ok`) refuses promotion. Research shows even ~1% of
fully-verified anchors is enough to substantially suppress reward hacking.

Keep these:
- truly held out (the loop never trains/searches on them),
- diverse across the capabilities you care about,
- refreshed occasionally so the anchor itself does not get overfit.

Format is up to you; a simple option is one JSON per item:
`{"prompt": "...", "verifier": "def check(solution): ...", "source": "human"}`
