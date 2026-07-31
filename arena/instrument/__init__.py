"""
Arena-owned instrumentation for bots. Never bundled, never hashed.

Everything here observes a bot from the outside. It is deliberately not under
`bots/`: a bot's source closure is its rating identity and its submission
bundle, so per-turn introspection living inside it would ship to the judge and
would fork every bot's hash on every instrumentation edit.

The one file that does live beside a bot is `bots/<name>/probe.py`, which
`arena.records.fingerprint` excludes from the closure by rule — and refuses to
let any closure module import.

See docs/arena/trajectories.md.
"""
