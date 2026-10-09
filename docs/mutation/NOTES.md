# Mutation testing notes

Tool: testteeth 0.2.0, 2 workers, run from a clean clone of the repo, targets
`scorer.py`, `schema.py`, `data/clean.py`, `pairs.py`, `formatting.py`, tests run with
`-m "not smoke and not parity"` (no torch).

| Run | Commit of tests | Killed | Timeouts | Survived | Score |
|---|---|---|---|---|---|
| `baseline.*` | tests before bcff652 | 411 | 1 | 26 | 94.1% (412/438) |
| `after.*` | bcff652 | 432 | 1 | 5 | 98.9% (433/438) |

Score = (killed + timeouts) / mutants.

## The 5 survivors in `after.txt` (all in `pairs.py`)

| Line | Mutant | Verdict |
|---|---|---|
| L56 | `reason.split(":", 1)` -> `split(":", 2)` | equivalent: only element `[0]` is used |
| L147 | `len(synth) > 1` -> `>= 1` | equivalent: shuffling an empty tail consumes no randomness |
| L147 | `len(synth) > 1` -> `> 2` | equivalent: shuffling a one-element tail consumes no randomness |
| L156 | `seen.add(comp)` -> `pass` | equivalent: the synthetic kinds always render distinct completions, so nothing later in the loop can hit the dedup |
| L148 | `synth[1:]` -> `synth[2:]` | **real gap** (drops one synthetic kind). Killed by `test_every_synthetic_kind_is_used_when_room_allows`, added after this run and checked by applying the mutant by hand; the full mutation run was not repeated, so the recorded score stays 98.9%. |
