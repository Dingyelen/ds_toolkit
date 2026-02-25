# ds_toolkit

面向数据分析场景的标准化工具库：工程结构极简、模块解耦、配置与逻辑分离。

---

## 配置说明（configs/）

### 全局配置思路

- 所有任务的 **输入路径、编码列表、指标定义、样本量参数** 等，都建议写在 YAML 中。
- 常见用法：
  1. 用 `yaml.safe_load` 读取对应 YAML；
  2. 传入各模块的 `load_xxx_config`（如 `load_abtest_config`）得到强类型配置对象；
  3. 调用 `modules/` 下的业务函数输出结果。

### 默认配置：`configs/default_settings.yaml`

- 用于放一些通用默认：
  - 文件编码列表、分隔符、日期格式等；
  - 默认输出路径、日志级别；
  - 各模块的默认参数（如留存窗口、报表样式等）。
- 典型用法：在自己的任务配置中用“拷贝 + 局部覆盖”的方式继承/修改。

### AB 实验配置：`configs/abtest_demo.yaml`

`abtest_demo.yaml` 是 AB 实验模块的参考配置，结构大致如下（**均已落地实现**）：

```yaml
data:
  group_col: ab_group    # 实验组/对照组列名
  phase_col: phase       # 实验阶段列名，如 "before"/"after"

metrics:
  - name: arpu           # 指标名（自定义别名）
    type: continuous     # "continuous" 连续型 / "proportion" 比例型
    column: arpu         # df 中对应的列名（列内是样本列表）
    test:
      alternative: larger  # 备择假设：实验组是否“更大”
      prefer: auto         # "auto"/"mean"/"log_mean"/"mann_whitney"
      sided: one-sided     # "one-sided"/"two-sided"
    sample_size:           # 可选：配置则自动算样本量规划
      mde: 0.1             # 最小可检测效应（连续型为标准化效应大小 d）
      alpha: 0.05
      power: 0.8
      method: means        # "means" 或 "nonparametric"
```

**关键字段说明：**

- **`data.group_col` / `data.phase_col`**
  - run_ab_test 依赖这两个字段来识别“组别”和“实验阶段”；
  - df 中每行对应一个 `(group, phase[, 分层])` 组合。

- **`metrics[].type`**
  - `"continuous"`：连续型（如 ARPU、ARPPU、在线时长、人均次数）；
  - `"proportion"`：比例型（如付费率、留存率、活跃率、转化率）。

- **`metrics[].column`**
  - 对应 df 中的列名，该列的每个单元格是一个“样本列表”，可以来自 SQL 的 `array_agg` 等聚合。

- **`metrics[].test`**
  - `alternative`：
    - `"two-sided"`：只关心“是否不同”；
    - `"larger"`：关心“实验组是否更大”；
    - `"smaller"`：关心“实验组是否更小”。
  - `prefer`（连续型）：
    - `"auto"`：结合分布诊断自动在 `mean` / `log_mean` / `mann_whitney` 中选择；
    - `"mean"`：均值差检验；
    - `"log_mean"`：对数均值检验（适合 ARPPU 等右偏指标）；
    - `"mann_whitney"`：Mann–Whitney U 秩和检验。
  - `sided`：单双侧检验，影响 alpha 与样本量公式。

- **`metrics[].sample_size`（可选）**
  - 若配置，则 run_ab_test 会调用 `SampleSizeCalculation` 计算每组所需最小样本量 `n_per_group`：
    - 连续型：基于 Cohen's d 的均值检验样本量；
    - 比例型：基于两比例差的样本量公式。

---

## 核心能力概览（core/ 已实现）

### 文件读取（core.utils）

- **模块**：`core.utils.file_io.DataLoader`
- **功能**：
  - 按后缀自动选择 CSV/Excel 读取；
  - 对 CSV 支持编码回退（如 `utf-8 → gbk → gb18030`）；
  - 支持从配置注入编码列表。

示例：

```python
from core.utils import DataLoader

loader = DataLoader()
df = loader.read_data("data.csv")

# 或从 YAML 配置中注入编码列表
# loader = DataLoader(encoding_list=config["file_io"]["encodings"])
# df = loader.read_data(path, sep="\t", sheet_name=0)
```

### 统计模块（core.stats）

#### 分布诊断（连续型）

- **位置**：`core.stats.distribution_diagnostics.diagnose_continuous_distribution`
- **适用对象**：ARPU、ARPPU、在线时长、人均次数等连续型指标。
- **输出**：
  - 基本统计：`n`、均值、标准差、中位数、最小值、最大值；
  - 形态特征：偏度、超额峰度、零占比；
  - 标记与推荐：
    - `is_approximately_normal`：是否近似正态；
    - `is_heavy_tailed`：是否重尾/极端偏态；
    - `is_zero_inflated`：是否零膨胀（典型为 ARPU）；
    - `recommended_tests`：推荐的检验方法列表（如 `"mean_z_test"`, `"log_mean_test"`, `"mann_whitney_u"`）。

#### 假设检验（HypothesisTest）

- **位置**：`core.stats.HypothesisTest`
- **支持场景**：
  - 连续型：均值检验、对数均值检验、Mann–Whitney U 检验；
  - 比率型：两比例 z 检验；
  - ABN：对照组 vs 多实验组的批量比较。

主要方法（已实现）：

- `mean_test`：两独立样本均值差 z 检验（大样本近似 t 检验）；
- `log_mean_test`：log 空间做均值检验，并输出原始空间倍数变化；
- `mann_whitney_u_test`：Mann–Whitney U 非参数检验，更关注整体分布/中位数；
- `proportion_test`：两比例 z 检验（0/1 指标）。

#### 样本量计算（SampleSizeCalculation）

- **位置**：`core.stats.SampleSizeCalculation`
- **功能**：
  - 连续型均值检验样本量（含单双侧）；
  - 比率型两比例检验样本量；
  - 连续型非参数检验样本量近似（基于 Mann–Whitney U 渐近效率）。

示例（比率型样本量）：

```python
from core.stats import SampleSizeCalculation

ssc = SampleSizeCalculation()
n = ssc.calculate_for_proportions(
    baseline_rate=0.1,
    mde=0.02,
    alpha=0.05,
    power=0.8,
    sided="two-sided",
)
print("每组至少需要样本数:", n)
```

---

## AB 实验模块（modules.abtest）

### 数据与配置前提

- **数据 df**：
  - 每行对应一个 `(group_col, phase_col[, stratify_by...])` 组合；
  - 各指标列（如 `arppu`, `pay_rate`）为该组合下的**样本列表**。
- **配置 config**：
  - 使用 `load_abtest_config(raw_cfg)` 从 YAML 构建 `ABTestConfig`；
  - `data.group_col`、`data.phase_col`、`metrics` 等字段含义见上文。

---

### `run_ab_test`：用法与输入输出

#### 使用示例

```python
import yaml
from modules.abtest import load_abtest_config, run_ab_test

with open("configs/abtest_demo.yaml", "r", encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

config = load_abtest_config(raw_cfg)
result_df = run_ab_test(
    df=df,
    config=config,
    base_group="control",
    target_phases=["after"],    # ["before"] 可用于 AA 检验
    stratify_by=["country"],    # 可选：按国家/等级等分层
)
```

#### 核心统计逻辑

对于每个 `(metric, phase[, 分层组合])`：

1. 从 df 中抽取 base_group 与各 variant_group 的样本列表。
2. 根据 `metric_type` 与 `test.prefer` 选择检验方法：
   - 连续型：
     - 近似正态且右偏不严重：倾向 `mean`；
     - 明显右偏且值>0：倾向 `log_mean`；
     - 重尾/异常值多：倾向 `mann_whitney`。
   - 比例型：使用两比例 z 检验。
3. 使用 `HypothesisTest` 得到：
   - 统计量（z 或 U）；
   - p 值 `p_value`，显著性标记 `is_significant`；
   - 效应 `effect`：
     - 连续型 mean：均值差（实验组 − 对照组）；
     - 连续型 log_mean：报告中更关注 `effect_ratio`（倍数）；
     - 比例型：比例差（实验组 − 对照组）。
4. 若配置了 `sample_size`，调用 `SampleSizeCalculation` 得到样本量规划 `sample_size_plan`。

#### 结果表中的主要列（统计视角）

| 列名                | 含义                                                                                       |
|---------------------|--------------------------------------------------------------------------------------------|
| `metric`            | 指标名称，如 `arppu`、`pay_rate` 等                                                       |
| `metric_type`       | 指标类型：`"continuous"` 连续型 / `"proportion"` 比例型                                    |
| `phase`             | 实验阶段，如 `"before"` / `"after"`                                                       |
| `base_group`        | 对照组名称（与 `base_group` 入参一致）                                                    |
| `variant_group`     | 实验组名称                                                                                |
| 分层列（可选）      | 若传入 `stratify_by`，在 `variant_group` 后增加对应的分层字段（如 `country`、`level`）    |
| `used_method`       | 实际使用的检验方法：`"mean"` / `"log_mean"` / `"mann_whitney"` / `"proportion_z_test"` 等  |
| `p_value`           | p 值                                                                                       |
| `is_significant`    | 是否在给定 `alternative` + 置信水平下显著                                                 |
| `effect`            | 效应大小：连续型 mean 为均值差；log_mean 场景多用 `effect_ratio`（倍数）；比例型为比例差  |
| `diagnosis`         | 分布诊断与样本汇总信息（字典，结构见上文）                                                |
| `stat_detail`       | 统计检验的完整结果（字典，含统计量、置信区间、`pooled_std`/`pooled_std_log` 等）          |
| `sample_size_plan`  | 样本量规划结果（字典，配置了 `sample_size` 时存在，结构见下文）                           |

#### 业务解读建议

- **是否显著**：用 `p_value` + `is_significant` 判断“是否有统计学证据表明实验组≠对照组”。
- **效应有多大**：
  - 连续型：看 `effect`（差了多少单位）；
  - log_mean：看 `effect_ratio`（提升了多少百分比），及其倍数置信区间；
  - 比例型：看 `effect`（绝对百分点变化）。
- **样本量是否够**：
  - 对比实际样本量与 `sample_size_plan["n_per_group"]`：
    - 远高于规划值 + 显著：结论相对稳健；
    - 远低于规划值：不显著更可能是“功效不足”，显著也要谨慎“偶然显著”。

---

### `compute_did`：AA/AB + 双重差分（DID）

#### 使用示例

```python
from modules.abtest import compute_did

aa_df = run_ab_test(df, config, base_group="control", target_phases=["before"])
ab_df = run_ab_test(df, config, base_group="control", target_phases=["after"])

did_df = compute_did(
    aa_result=aa_df,
    ab_result=ab_df,
    stratify_by=["country"],  # 若 run_ab_test 做了分层，这里一并传
)
```

#### 统计学解释

对每个 `(metric, variant_group[, 分层])`，`compute_did` 会输出以下关键字段：

| 列名               | 含义                                                                                                   |
|--------------------|--------------------------------------------------------------------------------------------------------|
| `metric`           | 指标名称                                                                                               |
| `variant_group`    | 实验组名称                                                                                             |
| 分层列（可选）     | 若传入 `stratify_by`，以这些列作为分层键                                                               |
| `base_value`       | AB 阶段对照组的均值/比例（after）                                                                      |
| `variant_value`    | AB 阶段实验组的均值/比例（after）                                                                      |
| `effect_aa`        | AA 阶段组间差（before），连续型为均值差/倍数，比例型为比例差                                          |
| `effect_ab`        | AB 阶段组间差（after）                                                                                 |
| `did_effect`       | DID 点估计：`effect_ab - effect_aa`，表示“上线后差异 − 上线前差异”的净效应                            |
| `p_value`          | AB 阶段检验的 p 值                                                                                    |
| `aa_significant`   | AA 阶段是否显著                                                                                       |
| `ab_significant`   | AB 阶段是否显著                                                                                       |
| `n_base`           | AB 阶段对照组样本量                                                                                   |
| `n_variant`        | AB 阶段实验组样本量                                                                                   |
| `n_required`       | 继承自 AB 报告的 `sample_size_plan_ab["n_per_group"]`，表示设计期期望的每组最少样本量                 |
| `effect_n_required`| 基于当前效应（`effect_ab`/`effect_log`）重新计算的“检测该效应所需每组最小样本量”，alpha/power 与 run 对齐 |
| `sample_sufficient`| 当前定义为 `n_variant >= n_required`，表示实验组样本量是否已达到设计期要求                             |

> 注意：当前 DID 仅给出 **`did_effect` 点估计**，未对 DID 本身做显著性检验；  
> 用于“剃掉 AA 基线差异看 AB 净效应”，不直接提供 DID 的 p 值。

> 注意：当前 DID 仅给出 **`did_effect` 点估计**，未对 DID 本身做显著性检验；  
> 用于“剃掉 AA 基线差异看 AB 净效应”，不直接提供 DID 的 p 值。

#### 业务解读建议

- **AA 检查（随机性/口径）**
  - `aa_significant=True` 且 `effect_aa` 明显偏离 0：
    - 说明实验前随机分组或埋点/口径存在系统偏差；
    - AB 结果需结合 DID（`did_effect`）谨慎解读。

- **AB 结果**
  - `ab_significant=True` 且 `effect_ab>0`（或 `effect_ratio>1`）：
    - 在当前样本量与 alpha/power 下，有统计学证据支持“实验组优于对照组”；
  - 若不显著且 `n_variant << n_required`：
    - 更偏向“功效不足”，而非“验证了无效果”。

- **DID（did_effect）**
  - 用于“上线后差异减去上线前差异”，更接近因果效应：
    - 若 AA 不显著、AB 显著，则 DID 通常接近 AB，可视为实验净效应；
    - 若 AA 已显著、AB 也显著，但 DID 接近 0，则实验可能只是延续了原差异。

- **样本量双视角**
  - `n_required`：设计期规划的“应有样本量”；
  - `effect_n_required`：按当前观测效应反推“需要多少样本才能稳妥检测到这种效应”；
  - 与 `n_variant` 联合看：
    - `n_variant >= n_required`：基本满足设计期功效；
    - `n_variant` 远大于 `effect_n_required`：对当前效应来说样本偏富余；
    - `n_variant` 明显小于 `effect_n_required`：当前样本量对该效应仍偏紧。

---

## 环境与依赖

- Python 3.10+
- 主要依赖：
  - `pandas`, `numpy`, `PyYAML`
  - 统计/建模：`scikit-learn`（在部分 modules 中使用）
  - 报表/可视化：`Plotly`, `Matplotlib`（在 `core/visualizer` 与 `modules/reporter` 中使用）

建议在虚拟环境中按需安装依赖，并根据实际使用的模块选择性引入。 
