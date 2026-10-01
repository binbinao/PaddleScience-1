# Reproduce a Case from Scratch: New-User Playbook

This page targets **first-time PaddleScience users**. It uses the [EMS power plane DC IR-drop simulation](examples/ems_ir_drop.md) as the worked example and lays out the full path from environment setup to submitting a PR. The same route works for reproducing any other case in this repository.

## 0. The Big Picture: What a PINN Case Is Made Of

PaddleScience is config-driven (Hydra + OmegaConf + pydantic). A standard case = one script + one config:

```
examples/<case>/
├── <case>.py          # @hydra.main entry, dispatches train/eval/export/infer via cfg.mode
└── conf/<case>.yaml   # geometry, sampling, model, training, evaluation, inference — all declared here
```

Runtime data flow (each stage maps to a code section of the [EMS case](examples/ems_ir_drop.md)):

```
cfg (yaml) ──► ppsci.arch.<Model>(**cfg.MODEL)        ← model
          ──► ppsci.equation.<PDE>                    ← sympy-expression equations
          ──► ppsci.geometry.* + sampling              ← computational domain
          ──► ppsci.constraint.<Constraint>            ← equations + BCs → losses
          ──► ppsci.optimizer.<Opt>(model)             ← optimizer
          ──► ppsci.validate.<Validator>               ← validator
          ──► ppsci.solver.Solver(...).train()         ← training loop
```

Two factory patterns must be told apart (the most common confusion for newcomers):

| Pattern | How it works | Used for |
| :-- | :-- | :-- |
| `name:`-eval | `cfg.pop("name")` → `eval(name)(**cfg)` | `MODEL` / `loss` / `dataset` — single-choice slots |
| dict-key dispatch | the config key IS the class name; an inner `name:` is only an instance label | `constraint` / `validator` / `EQUATION` — multi-instance slots |

## 1. Environment Setup (10 minutes)

``` sh
# 1. Install PaddlePaddle first, separately (CPU or GPU per your machine)
pip install paddlepaddle        # or: pip install paddlepaddle-gpu

# 2. Clone and install PaddleScience
git clone https://github.com/PaddlePaddle/PaddleScience.git
cd PaddleScience
pip install -r requirements.txt
python -m pip install -e .

# 3. Install the quality-gate tools (pytest and pre-commit are NOT in requirements)
pip install pytest pre-commit

# 4. Verify the installation
python -c "import ppsci; ppsci.utils.run_check()"
```

## 2. Run Your First Case (30 minutes)

Start with [examples/ems_ir_drop](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/ems_ir_drop) — it needs no downloaded dataset (the reference solution is computed on the fly by a built-in FDM solver), and a full training run takes about 22 minutes on CPU:

``` sh
cd examples/ems_ir_drop
python ems_ir_drop.py                      # trains 800 epochs + evaluates automatically
```

Three experiments to try immediately (this is how you learn Hydra override syntax — any config key can be overridden from the command line):

``` sh
# a. shorten training to watch the convergence trend
python ems_ir_drop.py TRAIN.epochs=200

# b. tune the anchor supervision strength and observe its effect on accuracy
python ems_ir_drop.py +TRAIN.anchor_points=1000 TRAIN.anchor_weight=5.0

# c. after training, run eval / export / inference separately
python ems_ir_drop.py mode=eval EVAL.pretrained_model_path=<checkpoint path>
python ems_ir_drop.py mode=export INFER.pretrained_model_path=<checkpoint path>
```

The training log shows two layers of numbers — **learn to tell them apart**:

- `[Train]` per-constraint losses (EQ / VRM / PAD_* / EDGE / HOLE / ANCHOR) — these only reflect constraint satisfaction;
- `[Eval][Avg] FDM_ref/L2Rel.u_ref` and the pad-voltage table — **the real acceptance metrics**.

On boundary-driven problems the two can decouple badly (falling training losses do not imply approaching the true solution). This is a known PINN pathology — and the reason the EMS case adopts sparse anchor supervision (see section 2.3 of the case doc).

## 3. Understand the Case Design (1-2 hours)

Studying one case thoroughly beats skimming ten. Recommended order:

1. **Config first**: read `conf/ems_ir_drop.yaml` end to end; each section maps to one build step (geometry → sampling sizes → model → training). Changing the config IS running a new experiment — no code edits needed.
2. **Read the script via its function map** (`ems_ir_drop.py`):

| Function | Line | Responsibility |
| :-- | :-- | :-- |
| `LaplaceV` | L65 | custom sympy PDE (inherits `ppsci.equation.pde.PDE`) |
| `_u_singularity_terms` / `_u_s` | L86 / L104 | analytic singularity-split term $u_s$ |
| `solve_ir_drop_fd` | L156 | built-in FDM reference (self-developed validation baseline) |
| `build_geometry` | L308 | CSG geometry assembly |
| `build_anchor_points` | L348 | sparse anchor sampling from the FDM grid |
| `build_constraints` | L395 | five constraint groups (PDE / boundary / anchors) |
| `build_validator` | L520 | validator against FDM |
| `train` / `evaluate` / `export` / `inference` | L614+ | four mode entry points |

3. **Cross-check the API docs**: for an unfamiliar class (e.g. `InteriorConstraint`), look up its parameter semantics in the [API documentation](api/arch.md); for an unfamiliar yaml key, find its pydantic schema in `ppsci/utils/config.py`.
4. **Read the tests**: `test/` mirrors the `ppsci` package layout and is the most authoritative usage reference (e.g. `test/equation/test_laplace.py` shows how to verify an equation implementation against hand-rolled gradients).

## 4. Practice on Another Case (Pick by Interest)

| Interest | Recommended case | Why |
| :-- | :-- | :-- |
| Minimal PINN starter | `examples/laplace` | purely equation-driven, no data dependency, simplest structure |
| Data-driven surrogate | `examples/chip_heat` | PI-DeepONet multi-branch architecture |
| Industrial deployment template | `examples/ems_ir_drop` | dual-baseline validation + deployment chain + engineering narrative |
| CFD / fluids | `examples/cylinder` | classic external-flow case, transformer variant included |

## 5. Next Step: Build Your Own Case

To fit a new problem into PaddleScience, advance through this checklist (every step has a counterpart in the EMS case):

1. **Mathematical modeling**: write down the governing PDE + boundary/initial conditions; decide whether singularity splitting is needed (mandatory when the field has $1/r$ or $\ln(1/r)$ singularities near sources — otherwise the network collapses).
2. **Baseline before training**: build a self-developed reference first (FDM / analytic / literature data). Without a baseline there is no acceptance — this is the single most important methodological lesson of the EMS case.
3. **Geometry and sampling**: express the domain via CSG; give each boundary class its own constraint and **keep loss keys unique**.
4. **Config assembly**: write the yaml in MODEL → constraints → optimizer → validator order; new keys need pydantic fields in `ppsci/utils/config.py`.
5. **Training diagnostics**: when training losses and acceptance metrics decouple, run small controlled experiments (pure-regression upper bound, single-constraint ablation) to locate the failure before choosing a fix (anchor supervision / curriculum training / geometry simplification).
6. **Deliver completely**: train/eval/export/infer four modes + bilingual docs + mkdocs nav + README row, then run `pre-commit run --all-files`.

## 6. Common Pitfalls Quick Reference

| Symptom | Cause & fix |
| :-- | :-- |
| Hydra raises `Key not in struct` | the overridden yaml key is new and lacks the `+` prefix: use `+TRAIN.xxx=...` |
| Training losses drop but metrics stall | PINN deceptive solution / gradient pathology; build an independent acceptance baseline, never trust training losses |
| Second derivatives explode or NaN | unsplit source singularities, or the EQ penalty crushing a physically real quasi-discontinuous jump |
| `pytest test/` under-collects | some files under `test/loss/` lack the `test_` prefix; use `pytest -o python_files=*.py test/loss/` |
| Checkpoint loading fails | URL auto-download needs network; use a local path via `EVAL.pretrained_model_path=<abs path>` |
| `outputs*/` shows up in git status | Hydra run artifacts — **never commit them** |
| Afraid of breaking the config | Hydra overrides are all command-line; `git checkout conf/` restores everything |

## 7. Getting Help

- the "References" section at the end of each case doc;
- [GitHub Issues](https://github.com/PaddlePaddle/PaddleScience/issues);
- [Development & Reproduction Guide](development.md) — required reading before opening a PR.

!!! tip "One Rule of Thumb"

    Config-driven means: **run experiments by editing the yaml first; only touch code for what the yaml cannot express**. 90% of newcomer confusion comes from treating a configuration problem as a code problem.
