# CLAUDE.md — gabor-tica

Guidance for working in this subfolder. It is a self-contained PyTorch project
(package `gtv`), independent of the TensorFlow code elsewhere in the repo.

- `docs/PLAN.md` is the source of truth. Implement one phase at a time and do
  not start a phase until the previous phase's exit criteria pass.
- Prototype-driven: land the simplest fixed/closed-form version first, with a
  test and an `experiments/phaseN_*.py` script that checks the exit criteria,
  then add learning.
- New stage blocks subclass `gtv.stages.Stage`, register in `STAGE_KINDS`, and
  are selected from YAML via `kind:`.
- Every new component gets a probe-based test with known ground truth
  (`gtv.probes`), not only natural-image checks.
- Keep the knobs separate (gain, expectation, neighborhoods, routing): each is
  its own module and can be toggled from config.
- Update the phase-status table in `README.md` when a phase's exit criteria pass.
- Run from this folder: `pip install -e ".[dev]"`, `pytest`, `python experiments/phaseN_*.py`
  (each writes figures and a `report.json` with pass/fail per exit check).
  Outputs go to `runs/` (git-ignored).
