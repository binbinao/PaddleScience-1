# Repository Guidelines

## Project Overview

PaddleScience (`ppsci`) is a PaddlePaddle-based scientific computing / physics-informed machine learning library: PINNs, data-driven surrogate models, and equation solvers for fluid dynamics, solid mechanics, weather, materials, etc. It is config-driven (Hydra + OmegaConf + pydantic ≥2.5). `examples/` holds 88 top-level cases — 80 follow the hydra pattern, 8 are outliers (see Architecture). README is Chinese-first; docs are bilingual (`docs/zh` default, `docs/en` mirror).

## Architecture & Data Flow

**Config-driven assembly.** The canonical example is a Hydra app (`examples/allen_cahn/allen_cahn_causal.py`):

```python
@hydra.main(version_base=None, config_path="./conf", config_name="<case>.yaml")
def main(cfg):
    if cfg.mode == "train": train(cfg)      # mode ∈ {train, eval, export, infer}
    elif cfg.mode == "eval": evaluate(cfg)
    ...
```

YAML `defaults:` pull ConfigStore nodes (`ppsci_default`, `TRAIN: train_default`, `TRAIN/ema: ema_default`, `TRAIN/swa: swa_default`, `EVAL: eval_default`, `INFER: infer_default`, ...) registered **programmatically** from pydantic models in `ppsci/utils/config.py` — there are no physical default YAML files.

**PINN training flow:**

```
ppsci.arch.<Model>(**cfg.MODEL)      # ~63 archs (MLP, DeepONet, TFNO*dNet, Transolver, ...); base Arch
→ ppsci.equation.<PDE>               # sympy exprs (ppsci/equation/pde/base.py)
→ ppsci.geometry.* + sampling
→ ppsci.constraint.<Constraint>      # dataset + dataloader + loss
→ ppsci.optimizer.<Opt>(lr_scheduler)(model)
→ ppsci.validate.<Validator>
→ ppsci.solver.Solver(model, ..., cfg=cfg)
→ solver.train() / eval() / predict() / export()
```

- `Solver` (`ppsci/solver/solver.py`): `model` positional + ~34 kwargs; `cfg` is an **optional trailing kwarg** (not required yet). Methods: `train`, `finetune(pretrained_model_path)`, `eval`, `visualize`, `predict`, `export(input_spec, export_path, with_onnx=False, ...)`, `plot_loss_history`, `register_callback_on_{epoch,iter}_{begin,end}`.
- Forward passes run through `ExpressionSolver` (`ppsci/utils/expression.py`): `train_forward/eval_forward/visu_forward` (base `forward()` raises NotImplementedError). It runs model forward + lambdified sympy equations (`ppsci.utils.symbolic.lambdify`) with `ppsci.autodiff` cached Jacobians/Hessians (module singletons `ppsci.autodiff.jacobian`/`hessian`; `ppsci.autodiff.clear()` runs after every forward). `train_forward` returns `(losses_all, losses_constraint)`: the flat dict feeds the MTL aggregator (default `Sum`; alternatives GradNorm/NTK/PCGrad/Relobralo/AGDA in `ppsci/loss/mtl/`), the grouped dict is log-only.
- Train loop (`ppsci/solver/train.py`): per-constraint iters → loss aggregation → backward (optional AMP `GradScaler`) → `optimizer.step` gated by `TRAIN.update_freq`; separate LBFGS closure path.
- Export: `paddle.jit.to_static` + `jit.save` (PIR mode → `*.json`/`*.pdiparams`), optional ONNX (paddle2onnx). Inference: `deploy/python_infer/` — `Predictor` base + `PINNPredictor(cfg)`; `GeneralPredictor` is a trivial subclass alias of `PINNPredictor`.
- Hydra `InitCallback` (`ppsci/utils/callbacks.py`): pydantic-validates cfg (exit 2 = ValidationError, 1 = runtime), seeds, inits logger, sets device, pins devices in sweeps, enables `prim: true`, dumps git diff to `output_dir/code_snapshot/` when `trace: true`.
- Weights: URL paths in `EVAL/INFER.pretrained_model_path` auto-download (md5-checked) to `~/.paddlesci/weights` via `ppsci/utils/save_load.py:load_pretrain` → `ppsci/utils/download.py`.

**Two factory patterns** (each core module exposes `build_*` in its `__init__.py`):
1. **`name:`-eval** — `cfg.pop("name")` → `eval(name)(**cfg)`: `build_model`, `build_loss`, `build_dataset`, `build_mtl_aggregator`; `build_lr_scheduler`/`build_optimizer` additionally CALL the constructed factory with `(model_list)`. YAML: `MODEL: {name: MLP, ...}`.
2. **Dict-key dispatch** — the item key IS the class; an inner `name:` is only an instance label: `build_constraint`, `build_validator`, `build_visualizer`, `build_equation`, `build_geometry`, `build_metric`, `build_transforms`. YAML: `- InteriorConstraint: {name: EQ, ...}`.
- `build_dataloader` (`ppsci/data/__init__.py`) is different: sampler via `getattr(paddle.io, cfg.name)`, auto-swapped to `DistributedBatchSampler` when world_size>1.
- **Custom dataset seam:** `ppsci.data.dataset.register_to_dataset` — the only registry mechanism in ppsci (setattr into the dataset module namespace consumed by `build_dataset`'s eval; NameError prints the registration recipe).

**Non-hydra example outliers** (don't assume `@hydra.main`): `operator_learning/sedan_aero_ai`, `unetformer`, `UTAE`, `xrdmatch` (argparse + `yaml.safe_load`); `iops`, `MLP_LI`, `quick_start` (plain scripts); `adr`, `ML2DDB` (doc-only placeholders). Config dirs are `conf/` in most hydra examples but `config/` in `smc_reac`/`LatentNO`; `cylinder` nests under `2d_unsteady/`.

## Key Directories

| Path | Purpose |
|---|---|
| `ppsci/` | Core library: `arch` (~63 models), `equation` (sympy PDEs), `geometry`, `constraint`, `data`, `loss` (+ `mtl/`), `metric`, `optimizer`, `validate`, `visualize`, `autodiff`, `solver`, `experimental` (unstable math APIs), `probability` (HMC — needs explicit `import ppsci.probability`; not loaded by `ppsci/__init__.py`), `utils`, `externals/` (vendored forks: deepali, neuraloperator, Open3D, paddle_* — excluded from the wheel) |
| `examples/<case>/` | Runnable cases: `<case>.py` + `conf/<case>.yaml` pairs (+ optional `requirements.txt` for extra deps like `pgl`, `einops`) |
| `test/` | pytest suite, mirrors `ppsci` subpackages (24 collected files + 3 uncollected — see Testing) |
| `test_tipc/` | Paddle TIPC benchmark harness (shell, not pytest); 2 case configs |
| `docs/{zh,en}/` | mkdocs static-i18n site (folder mode, zh default); API via mkdocstrings |
| `deploy/python_infer/` | `Predictor` base + `PINNPredictor`/`GeneralPredictor` |
| `jointContribution/` | Community reproductions — **excluded from pre-commit; treat as read-mostly** |
| `competition/` | Git submodule (`git submodule update --init --recursive competition/IJCAI_2024_CAR`) |
| `docker/`, `recipe/` | **Stale** legacy envs (CUDA 11.6, py3.9, pre-3.0 paddle pins; Dockerfile references a missing `pymesh.tar.xz`) — not current install guidance |

## Development Commands

```bash
# Install (editable, recommended). PaddlePaddle itself is installed separately first
# (README: pip install paddlepaddle / -gpu from paddlepaddle.org.cn indexes).
pip install -r requirements.txt
python -m pip install -e .
python -c "import ppsci; ppsci.utils.run_check()"   # verify (README:272; docs use ppsci.run_check() — both valid)

# Run an example (from example dir; hydra overrides via CLI)
cd examples/allen_cahn
python allen_cahn_piratenet.py                                   # train
python allen_cahn_piratenet.py mode=eval EVAL.pretrained_model_path=<url>
python allen_cahn_piratenet.py TRAIN.lr_scheduler.warmup_epoch=5  # override any cfg key

# Tests (pytest NOT in requirements.txt — install it yourself)
pytest test/                            # whole suite
pytest test/equation/test_laplace.py -v # single file
python test/equation/test_laplace.py    # standalone (files have pytest.main() footers)
pytest -o python_files=*.py test/loss/  # force-collect (see Testing gotchas)

# Lint/format — the effective quality gate (no real CI in-repo)
pre-commit run --all-files              # isort + black + ruff (+ clang-format for C++/CUDA)
pre-commit install                      # run automatically on commit

# Docs
pip install -r docs/requirements.txt
mkdocs serve                            # preview; new doc pages MUST be added to mkdocs.yml nav

# Complex-geometry support (optional): builds PyMesh from source (Ubuntu/Debian, root)
bash install_mesh.sh
```

## Code Conventions & Common Patterns

- **Formatting (pre-commit):** isort 5.11.5 (`profile=black`), black 22.3.0 (88 cols), ruff v0.0.272 (line-length 88, ignores `E501,E741,E731`; **extend-excludes** `ppsci/geometry/inflation.py`, `ppsci/autodiff/__init__.py`), plus hygiene hooks (`requirements-txt-fixer`, `check-yaml` excluding `mkdocs.yml`/`recipe/meta.yaml`, md CRLF/tab bans). C/C++/CUDA/proto: clang-format 3.8 — `.clang_format.hook` requires "3.8" in `clang-format -version` output. Global exclude `^jointContribution/`.
- **Docstrings:** Google style (`Args:`/`Returns:`/`Examples:`) with `>>>` doctests (some `# doctest: +SKIP`); Apache-2.0 license headers on every file.
- **Logging:** `ppsci/utils/logger.py` — rank-aware: rank≠0 gets ERROR level; all five log fns are `@misc.run_at_rank0`-gated (even `logger.error`); callers raise exceptions themselves after logging.
- **Optional heavy deps** (pymesh, open3d, pysdf, pgl, rdkit, xarray...) are never hard imports. Canonical guard: `TYPE_CHECKING` import for hints + `ppsci.utils.checker.dynamic_import_to_globals(["pymesh"])` at runtime + lazy `import pymesh` inside methods (see `ppsci/geometry/mesh.py`); missing deps raise `ModuleNotFoundError` with a pip-install hint.
- **Config validation:** all cfg schemas are pydantic models in `ppsci/utils/config.py` (`EMAConfig`/`SWAConfig` drive `ppsci/utils/ema.py`); add new `Solver`/`TRAIN`/`EVAL`/`INFER` keys there.
- **OmegaConf resolvers** `${numpy:...}`, `${sum:[...]}` registered in `ppsci/__init__.py`.
- **Naming:** snake_case modules/dirs; examples mostly snake_case (legacy CamelCase outliers: `velocityGAN`, `UTAE`, `NLS-MB`, `RegAE`, `CNN_UTS`, `LatentNO`, `MLP_LI`, `ML2DDB` — not the pattern to copy).
- **Error handling:** explicit `ValueError` with message on unsupported `cfg.mode` etc.; `build_visualizer` raises on duplicate names.
- **Determinism:** `seed: 42` in example YAMLs; tests call `paddle.seed()` at module level.

## Important Files

| File | Role |
|---|---|
| `ppsci/solver/solver.py` | `Solver` lifecycle (train/finetune/eval/visualize/predict/export), callback registration |
| `ppsci/utils/config.py` | pydantic cfg schemas + Hydra ConfigStore node registration (~L379-447) |
| `ppsci/utils/expression.py` | `ExpressionSolver` — forward + equation exprs + loss plumbing |
| `ppsci/equation/pde/base.py` | `PDE` base: sympy equations dict, `add_equation`, detach |
| `ppsci/arch/base.py` | `Arch` base: dict-in/dict-out `forward`, input/output transforms, freeze |
| `ppsci/data/dataset/__init__.py` | `build_dataset` + `register_to_dataset` custom-dataset seam |
| `ppsci/autodiff/ad.py` | `jacobian`/`hessian` cached singletons + `clear()` |
| `ppsci/utils/save_load.py` / `download.py` | checkpoint load/save; URL weight download to `~/.paddlesci/weights` |
| `ppsci/_version.py` | **GENERATED** by setuptools_scm from git tags `v(X.Y.Z)` (fallback 1.4.0); gitignored — never edit |
| `pyproject.toml` | Authoritative packaging; `setup.py` is a legacy mirror reading the same `requirements.txt` — keep both in sync on dep changes |
| `examples/allen_cahn/` | Canonical hydra example template (script + conf pairs) |
| `docs/en/development.md` | Contribution workflow (fork → branch → pre-commit → docs → PR) |
| `.github/PULL_REQUEST_TEMPLATE.md` | PR form: `### PR types` / `### PR changes` / `### Describe` |

## Runtime/Tooling Preferences

- **Python ≥ 3.8** (classifiers list 3.8–3.10); **PaddlePaddle ≥ 3.0**, installed separately — note: upstream `requirements.txt` contains **no paddle line**; if you see `paddlepaddle-gpu>=3.0.0` there, it's an uncommitted local edit. CPU builds also work for tests.
- `requirements.txt` is a single flat list (read into `pyproject` dynamically) — no extras groups. Heavy scientific deps stay out and are import-guarded at usage sites. Wheel excludes `docs*`, `examples*`, `test*`, `test_tipc*`, `tools*`, `jointContribution*`, `ppsci/externals*`.
- No Node/Bun toolchain. Docs build: Python 3.10 on ReadTheDocs, installing BOTH `docs/requirements.txt` and root `requirements.txt`.
- Outputs land in `outputs_<example>/<date>/<time>/<override_dirname>` (hydra `run.dir`) — **`outputs*/` is NOT in the root `.gitignore`**: never commit these; expect them as untracked noise in `git status`.
- Example data is downloaded manually (wget/curl to `./dataset/`) or via per-example `download_dataset.py`; weights auto-download from `paddle-org.bj.bcebos.com` URLs.

## Testing & QA

- **Framework:** pytest only. No `conftest.py`, no custom fixtures (built-ins like `tmpdir`/`caplog` are used), no unittest classes. Files `test_*.py`, functions `test_*`, heavy `@pytest.mark.parametrize`, `if __name__ == "__main__": pytest.main()` footer on every file (exception: `test/data/test_register_dataset.py`).
- **Assertions:** `paddle.allclose` for tensors, `np.testing.assert_allclose` with explicit tolerances for numerics. Equation tests compare lambdified sympy results against hand-rolled `paddle.grad` Jacobians/Hessians (exemplar: `test/equation/test_laplace.py`).
- **Determinism:** module-level `paddle.seed(1024)` (some files 42/2023); tests are device-agnostic (no `paddle.set_device`).
- **Config tests:** write YAML to tmpdir, `hydra.initialize` + `hydra.compose`, assert `SystemExit` codes from `InitCallback` (exemplar: `test/utils/test_config.py`).
- **Gotchas:**
  - `test/loss/{aggregator,chamfer,func}.py` lack the `test_` prefix → **not collected** by `pytest test/`; use `pytest -o python_files=*.py test/loss/`.
  - `.github/workflows/` contains only a non-functional demo — **there is no CI test/lint gate**. Run `pre-commit run --all-files` and the relevant pytest subset yourself before handing off; passing pre-commit is the de facto PR gate per `docs/en/development.md`.
- **TIPC** (`test_tipc/`) is Paddle's shell-driven train/infer benchmark harness, unrelated to pytest. Exactly two directive configs: `configs/train_2d_unsteady_continuous/` (cylinder2d_unsteady_Re100) and `configs/train_eular_beam/`. `test_train_inference_python.sh` positionally parses lines 1–51 of the txt (load-bearing); `benchmark_train.sh` reads up to line 65 and needs an external `BENCHMARK_ROOT` (tools.tar.gz — `test_tipc/tools` is not in-repo) and python3.10. Adding a benchmark-worthy example ⇒ add `test_tipc/configs/<case>/<case>_train_infer_python.txt` (copy an existing one).
- **New example checklist:** script + `conf/*.yaml` (+ optional `requirements.txt`), doc page in **both** `docs/zh/examples/` and `docs/en/examples/` (follow `allen_cahn.md` template), added to `mkdocs.yml` nav under `经典案例` (5 discipline subtrees; `strict: false` so it's convention, not a build error), README model-zoo row. Note: `examples/operator_learning/sedan_aero_ai` (argparse-style, no ppsci import) currently lacks the docs/nav wiring.
