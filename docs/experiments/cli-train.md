English | [简体中文](./cli-train.zh-CN.md)

# `experiments/train.py` — CLI reference

`train.py` is the main training CLI. Run it with `uv run --group experiments`.

## Modes

- **Single config** (default): `--config PATH` runs one YAML.
- **Batch**: `--config-dir DIR` runs every `*.yaml`/`*.yml` in `DIR` (sorted),
  each producing its own run dir + logs + npz. CLI overrides (`--dataset`,
  `--n-splits`, `--output-dir`, ...) apply to **every** config in the batch.

`--config` and `--config-dir` are mutually exclusive (you give one or the
other).

## Options

### `--config PATH`
- **What**: YAML config path.
- **Default**: `experiments/configs/default.yaml`.
- **Conflict**: with `--config-dir`.
- **See**: [config-reference.md](config-reference.md) for the YAML schema.

### `--config-dir DIR`
- **What**: Run every `*.yaml`/`*.yml` in `DIR` sequentially.
- **Default**: unset (single-config mode).
- **Conflict**: with `--config`. Raises `FileNotFoundError` if no configs found.
- **Example**:
  ```bash
  uv run --group experiments python experiments/train.py \
      --config-dir examples/configs --n-splits 2
  ```

### `--dataset NAME`
- **What**: Overrides the config's `dataset:` (loads `{data_path}/{NAME}.pkl`).
- **Default**: none (uses config).

### `--model NAME`
- **What**: Overrides `model.name`. Accepts the names in
  [models.md](models.md); `tbls`/`bls` and every supported baseline.
- **Default**: none (uses config).

### `--map-num N`
- **What**: TBLS-only legacy alias for `TBLS(n_map_trees=N)`. Overriding the
  `model.map_num` YAML key.
- **Default**: none (uses config).
- **Note**: ignored for any non-TBLS model.

### `--n-splits N`
- **What**: Overrides `cv.n_splits`. In batch mode, applies to **every**
  config.
- **Default**: none (uses config).

### `--output-dir DIR`
- **What**: Overrides `output_dir` (run dir + per-cohort Excel root). In batch
  mode, applies to every config — so all runs in the batch land under the same
  parent dir (one child dir each).
- **Default**: none (uses config).

### `--fusion {cca,gfcca}`
- **What**: Overrides `fusion.method` for **multi-view** cohorts only. **Has
  no effect** on single-view pkl cohorts.
- **Default**: none (uses config — defaults to `gfcca` if a multi-view cohort
  is loaded and the config has a `fusion` block).
- **See**: [../usage-multiview-fusion.md](../usage-multiview-fusion.md).

### `--grid`
- **What**: Switches from a single k-fold CV to a hyperparameter sweep using
  the resolved grid (default `TBLS_GRID`/`BLS_GRID`, overridable by YAML
  `grid:`). Writes `Grid_{i:03d}`, `GridSummary` (ranked), and
  `GridSearchLog` Excel sheets plus per-grid-point JSONL events.
- **Default**: off (single CV run).
- **Conflict**: incompatible with baselines that have no YAML `grid:` — in
  that case it falls back to a single k-fold run and logs a warning (does not
  silently sweep a fixed-args baseline).
- **See**: [grid-search.md](grid-search.md).

## Outputs

After a run, under `{output_dir}`:
- `{run_name}/{timestamp}/logs/{dataset}_{timestamp}.jsonl` — structured log
  ([outputs.md](outputs.md)).
- `{run_name}/{timestamp}/logs/{dataset}_{timestamp}_{cohort}_predictions.npz`
  — populated for non-grid runs only (raw per-fold `y_true`/`y_pred`/
  `y_score`).
- `{run_name}/{cohort}/{timestamp}/{cohort}_{model_name}_FS-..._RS-..._xlsx` —
  per-cohort Excel.

Full layout + sheet/event/npz schema: [outputs.md](outputs.md).

## Typical invocations

```bash
# One config
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml

# Everything in examples/configs, with a 2-fold override for speed
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2

# Grid-search the default TBLS axes
uv run --group experiments python experiments/train.py \
    --config examples/configs/default.yaml --grid

# Use LR instead of TBLS for the same dataset/preprocess/CV
uv run --group experiments python experiments/train.py \
    --config examples/configs/default.yaml --model lr
```

## Common errors

- **`Dataset pkl not found: ...`** — the pkl named by `--dataset` (or YAML
  `dataset`) is not under `data_path`. Confirm `ls {data_path}/{dataset}.pkl`.
- **`Unknown classifier '...'`** — `model.name` (or `--model`) is not one of
  the supported names in [models.md](models.md). The error message lists the
  supported options.
- **`ImportError: lightgbm is not installed...`** (or `catboost`, ...) — you
  asked for a baseline whose optional dependency isn't installed. Either `pip
  install lightgbm` (outside the `experiments` group) or use a different
  estimator.
- **`No default grid for model.name='...'`** — you passed `--grid` for a
  baseline without a YAML `grid:`. Add a `grid:` block or drop `--grid`.