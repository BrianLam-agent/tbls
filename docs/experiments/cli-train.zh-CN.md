English | [简体中文](./cli-train.zh-CN.md)

# `experiments/train.py` — 命令行参考

`train.py` 是主训练脚本，用 `uv run --group experiments` 运行。

## 运行模式

- **单配置**（默认）：`--config PATH` 跑一个 YAML。
- **批量**：`--config-dir DIR` 依次跑 `DIR` 下所有 `*.yaml`/`*.yml`（按
  文件名排序），每个配置产出各自的运行目录、日志和 npz。命令行覆盖项
  （`--dataset`、`--n-splits`、`--output-dir` 等）对批内**每个**配置都
  生效。

`--config` 与 `--config-dir` 互斥，二选一。

## 选项

### `--config PATH`
- **作用**：YAML 配置文件路径。
- **默认**：`experiments/configs/default.yaml`。
- **冲突**：与 `--config-dir` 互斥。
- **详见**：[config-reference.md](config-reference.md)。

### `--config-dir DIR`
- **作用**：依次跑 `DIR` 下所有 YAML 配置。
- **默认**：未设置（单配置模式）。
- **冲突**：与 `--config` 互斥。目录下无 YAML 会抛 `FileNotFoundError`。
- **示例**：
  ```bash
  uv run --group experiments python experiments/train.py \
      --config-dir examples/configs --n-splits 2
  ```

### `--dataset NAME`
- **作用**：覆盖 YAML 中的 `dataset:`，加载 `{data_path}/{NAME}.pkl`。
- **默认**：不覆盖，使用 YAML 中的值。

### `--model NAME`
- **作用**：覆盖 `model.name`。接受 [models.md](models.md) 中列出的
  名字——`tbls`/`bls` 以及所有支持的基线。
- **默认**：不覆盖。

### `--map-num N`
- **作用**：仅 TBLS 有效，等价于 `TBLS(n_map_trees=N)`，覆盖 YAML 中
  的 `model.map_num`。
- **默认**：不覆盖。
- **注意**：非 TBLS 模型忽略此选项。

### `--n-splits N`
- **作用**：覆盖 `cv.n_splits`。批量模式下对**每个**配置都覆盖。
- **默认**：不覆盖。

### `--output-dir DIR`
- **作用**：覆盖 `output_dir`（运行目录和各 cohort Excel 的输出根目录）。
  批量模式下所有配置的运行结果都落在同一个父目录下，各自一个子目录。
- **默认**：不覆盖。

### `--fusion {cca,gfcca}`
- **作用**：覆盖 `fusion.method`，**仅对多视图 cohort 有效**。单视图
  pkl cohort 不受影响。
- **默认**：不覆盖（多视图 cohort 加载且配置有 `fusion` 块时默认用
  `gfcca`）。
- **详见**：[../usage-multiview-fusion.md](../usage-multiview-fusion.md)。

### `--grid`
- **作用**：从单次 k-fold CV 切换为超参搜索，使用解析后的网格（默认
  `TBLS_GRID`/`BLS_GRID`，可由 YAML `grid:` 覆盖）。输出
  `Grid_{i:03d}`、`GridSummary`（排名）和 `GridSearchLog` 三个 Excel
  sheet，以及每个网格点的 JSONL 事件。
- **默认**：关闭（单次 CV）。
- **冲突**：对没有 YAML `grid:` 的基线无效——此时退化为单次 k-fold 并
  打告警（不会静默扫固定参数的基线）。
- **详见**：[grid-search.md](grid-search.md)。

## 运行输出

运行结束后在 `{output_dir}` 下生成：

- `{run_name}/{timestamp}/logs/{dataset}_{timestamp}.jsonl` — 结构化日志
  （详见 [outputs.md](outputs.md)）。
- `{run_name}/{timestamp}/logs/{dataset}_{timestamp}_{cohort}_predictions.npz`
  — 原始各折预测（仅非网格运行）。
- `{run_name}/{cohort}/{timestamp}/{cohort}_{model_name}_FS-..._RS-..._xlsx`
  — 各 cohort Excel。

完整布局和各文件格式见 [outputs.md](outputs.md)。

## 常用命令

```bash
# 跑单个配置
uv run --group experiments python experiments/train.py \
    --config examples/configs/tbls_full.yaml

# 批量跑，统一用 2 折（快）
uv run --group experiments python experiments/train.py \
    --config-dir examples/configs --n-splits 2

# 网格搜索默认 TBLS 超参
uv run --group experiments python experiments/train.py \
    --config experiments/configs/default.yaml --grid

# 同一份配置换个模型（LR 代替 TBLS）
uv run --group experiments python experiments/train.py \
    --config experiments/configs/default.yaml --model lr
```

## 常见报错

- **`Dataset pkl not found: ...`** — YAML 或 `--dataset` 指定的 pkl 不在
  `data_path` 下。用 `ls {data_path}/{dataset}.pkl` 确认。
- **`Unknown classifier '...'`** — `model.name` 或 `--model` 的值不在
  [models.md](models.md) 的支持列表中。错误信息会列出所有支持的名字。
- **`ImportError: lightgbm is not installed...`**（或 `catboost` 等）——
  该基线的可选依赖没装。单独 `pip install` 即可，或换个不需要额外依赖
  的模型。
- **`No default grid for model.name='...'`** — 对没有 YAML `grid:` 的
  基线传了 `--grid`。加 `grid:` 段或去掉 `--grid`。
