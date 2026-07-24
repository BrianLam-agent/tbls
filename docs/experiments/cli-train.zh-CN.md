[English](./cli-train.md) | 简体中文

# `experiments/train.py` — CLI 参考

`train.py` 是主训练 CLI。用 `uv run --group experiments` 运行。

## 模式

- **单配置**(默认):`--config PATH` 跑一个 YAML。
- **批处理**:`--config-dir DIR` 顺序跑 `DIR` 下每个 `*.yaml`/`*.yml`(已
  排序),每个产出自己的运行目录 + 日志 + npz。CLI 覆盖项(`--dataset`、
  `--n-splits`、`--output-dir`、...)对批里**每个**配置都生效。

`--config` 与 `--config-dir` 互斥(二选一)。

## 选项

### `--config PATH`
- **干啥**:YAML 配置路径。
- **默认**:`experiments/configs/default.yaml`。
- **冲突**:与 `--config-dir`。
- **见**:[config-reference.md](config-reference.md) 的 YAML 模式。

### `--config-dir DIR`
- **干啥**:顺序跑 `DIR` 下每个 `*.yaml`/`*.yml`。
- **默认**:未设(单配置模式)。
- **冲突**:与 `--config`。若无配置会抛 `FileNotFoundError`。
- **示例**:
  ```bash
  uv run --group experiments python experiments/train.py \
      --config-dir examples/configs --n-splits 2
  ```

### `--dataset NAME`
- **干啥**:覆盖配置的 `dataset:`(加载 `{data_path}/{NAME}.pkl`)。
- **默认**:无(用配置)。

### `--model NAME`
- **干啥**:覆盖 `model.name`。接受 [models.md](models.md) 里的名字;
  `tbls`/`bls` 和所有支持的基线。
- **默认**:无(用配置)。

### `--map-num N`
- **干啥**:仅 TBLS 用,旧别名,等于 `TBLS(n_map_trees=N)`。覆盖 YAML 的
  `model.map_num` 键。
- **默认**:无(用配置)。
- **注**:对非 TBLS 模型忽略。

### `--n-splits N`
- **干啥**:覆盖 `cv.n_splits`。批模式下,**每个**配置都覆盖。
- **默认**:无(用配置)。

### `--output-dir DIR`
- **干啥**:覆盖 `output_dir`(运行目录 + 各 cohort Excel 的根)。批模式下
  每个配置都覆盖 — 所以批里所有运行都落在同一父目录下(各自一个子目录)。
- **默认**:无(用配置)。

### `--fusion {cca,gfcca}`
- **干啥**:覆盖 `fusion.method`,仅对**多视图** cohort。**对单视图 pkl
  cohort 无效**。
- **默认**:无(用配置 — 多视图 cohort 加载且配置有 `fusion` 块时默认
  `gfcca`)。
- **见**:[../usage-multiview-fusion.md](../usage-multiview-fusion.md)。

### `--grid`
- **干啥**:从单次 k-fold CV 切到超参扫描,用解析后的网格(默认
  `TBLS_GRID`/`BLS_GRID`,可由 YAML `grid:` 覆盖)。输出 `Grid_{i:03d}`、
  `GridSummary`(排名)、`GridSearchLog` Excel sheet,以及每个网格点的
  JSONL 事件。
- **默认**:关(单次 CV 运行)。
- **冲突**:对无 YAML `grid:` 的基线不兼容 — 此时退化为单次 k-fold 运行
  并打告警(不静默扫固定参基线)。
- **见**:[grid-search.md](grid-search.md)。

## 输出

运行后在 `{output_dir}` 下:
- `{run_name}/{timestamp}/logs/{dataset}_{timestamp}.jsonl` — 结构化日志
  ([outputs.md](outputs.md))。
- `{run_name}/{timestamp}/logs/{dataset}_{timestamp}_{cohort}_predictions.npz`
  — 仅非网格运行时写(原始各折 `y_true`/`y_pred`/`y_score`)。
- `{run_name}/{cohort}/{timestamp}/{cohort}_{model_name}_FS-..._RS-..._xlsx`
  — 各 cohort Excel。

完整布局 + sheet/事件/npz schema:[outputs.md](outputs.md)。

## 典型调用

```bash
# 一份配置
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml

# examples/configs 下全部,加个 2 折覆盖(快)
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2

# 用 --grid 扫默认 TBLS 轴
uv run --group experiments python experiments/train.py \
    --config experiments/configs/default.yaml --grid

# 同一份 dataset/preprocess/CV,换 LR 而不是 TBLS
uv run --group experiments python experiments/train.py \
    --config experiments/configs/default.yaml --model lr
```

## 常见错误

- **`Dataset pkl not found: ...`** — `--dataset`(或 YAML `dataset`)指明的
  pkl 不在 `data_path` 下。确认 `ls {data_path}/{dataset}.pkl`。
- **`Unknown classifier '...'`** — `model.name`(或 `--model`)不是
  [models.md](models.md) 里的支持名。错误信息会列出支持集。
- **`ImportError: lightgbm is not installed...`**(或 `catboost`、...) —
  你要的基线其可选依赖没装。要么 `pip install lightgbm`(在 `experiments`
  组外),要么换一个估计器。
- **`No default grid for model.name='...'`** — 你对没 YAML `grid:` 的基线
  传了 `--grid`。加 `grid:` 段或去掉 `--grid`。
