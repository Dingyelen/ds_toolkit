# ds_toolkit

面向数据分析场景的标准化工具库：工程结构极简、模块解耦、配置与逻辑分离。

---

## 项目介绍

- 为数据分析工作提供可复用的底层能力与业务模块。
- 配置统一放在 `configs/`，业务参数从 YAML 读取，不在代码中硬编码。
- 代码以可读性优先，函数式或简单类为主，关键函数带中文注释与类型标注。

---

## 项目结构

```
ds_toolkit/
├── configs/          # 纯配置（YAML），无 Python 逻辑
├── core/             # 底层技术工具，不引用 modules
│   ├── stats/        # 统计学底层（T 检验、卡方、正态性等）
│   ├── visualizer/   # 绘图模板（Plotly/Matplotlib，支持 Mac 中文与负号）
│   └── utils/        # 通用工具（文件读写等）
├── modules/          # 业务分析逻辑，子模块之间不互相 import
│   ├── ab_test/      # AB 实验
│   ├── cleaner/      # 数据清洗
│   ├── retention/    # 留存拟合与预测
│   ├── revenue/      # 收入预估 / LTV
│   ├── modeling/     # 预测模型
│   └── reporter/     # 报表输出
└── scripts/          # 入口脚本：读 config → 调 modules → 出结果
```

---

## 使用方法

### 环境

- Python 3.10+
- 依赖：pandas, numpy, PyYAML；建模用 scikit-learn，报表用 Plotly/Matplotlib。

### 配置

- 默认配置见 `configs/default_settings.yaml`，可按任务复制或覆盖其中节点（如路径、编码列表、各模块参数）。

### 文件读取（已实现）

通过 `core.utils.file_io.DataLoader` 按后缀自动选择 CSV/Excel，并对 CSV 做编码回退（如 utf-8 → gbk → gb18030）：

```python
from core.utils import DataLoader

loader = DataLoader()
df = loader.read_data("data.csv")

# 或从配置注入编码列表
# loader = DataLoader(encoding_list=config["file_io"]["encodings"])
# df = loader.read_data(path, sep="\t", sheet_name=0)
```

---

## 统计模块（core.stats，已实现部分）

### 分布诊断（连续型指标）

- 功能位置：`core.stats.distribution_diagnostics.diagnose_continuous_distribution`
- 适用对象：ARPU、ARPPU、在线时长、人均次数等连续型业务指标。
- 输出信息：
  - 基本统计量：样本量 `n`、均值、标准差、中位数、最小值、最大值；
  - 分布形态：偏度 `skewness`、超额峰度 `kurtosis_excess`、零占比 `zero_ratio`；
  - 标签与建议：
    - `is_approximately_normal`：是否“大致接近正态”；
    - `is_heavy_tailed`：是否重尾 / 极端偏态；
    - `is_zero_inflated`：是否存在大量 0（典型为 ARPU）；
    - `recommended_tests`：推荐检验类型列表（如 `"mean_z_test"`、`"log_mean_test"`、`"mann_whitney_u"`）。

示例：

```python
from core.stats import diagnose_continuous_distribution

diagnosis = diagnose_continuous_distribution(values)  # values 为一组连续型样本
print(diagnosis["recommended_tests"])
```

### 假设检验（HypothesisTest）

- 功能位置：`core.stats.HypothesisTest`
- 支持场景：
  - 连续型指标：人均收入、时长、人均次数等；
  - 比率型指标：付费率、留存率、活跃率、转化率等；
  - 多实验组 ABN：对照组 vs 多个实验组的批量比较。

主要方法：

- `mean_test`：两独立样本均值差 z 检验（大样本近似 t 检验）
  - 适用：分布不算太极端、关注“均值差”的场景。
- `log_mean_test`：对数变换后的均值检验
  - 适用：指标明显右偏且全部为正（如 ARPPU），更关注“相对变化（倍数变化）”；
  - 同时给出 log 空间的均值差和原始空间的倍数变化及置信区间。
- `mann_whitney_u_test`：Mann–Whitney U 非参数检验
  - 适用：分布严重偏态、重尾或异常值较多的连续指标；
  - 更关注“整体分布/中位数是否系统性偏移”。
- `proportion_test`：两比例 z 检验
  - 适用：付费率、留存率、活跃率、转化率等 0/1 指标；
  - 输入为成功数和总人数。
- `batch_mean_test` / `batch_proportion_test`
  - 适用：ABN 场景，将一个对照组与多个实验组批量比较，返回按实验组名称索引的结果字典。

简单示例（连续型指标）：

```python
from core.stats import HypothesisTest

ht = HypothesisTest()
result = ht.mean_test(control_values, variant_values, alternative="two-sided", confidence_level=0.95)

if result["is_significant"]:
    print("实验组均值显著不同，effect =", result["effect"])
```

### 样本量计算（SampleSizeCalculation）

- 功能位置：`core.stats.SampleSizeCalculation`
- 支持场景：
  - 连续型指标均值检验（含单双侧）；
  - 比率型指标两比例检验（含单双侧）；
  - 连续型指标基于非参数检验（如 Mann–Whitney U）的样本量近似。

主要方法：

- `calculate_for_means` / `summarize_for_means`
  - 输入：标准化效应大小 `mde`（均值差 ÷ 标准差）、显著性水平 `alpha`、统计功效 `power`、单双侧 `sided`；
  - 输出：每组最小样本量 `n_per_group`；
  - 适用：t/z 检验类的均值比较（包括对数变换后做均值检验）。
- `calculate_for_proportions` / `summarize_for_proportions`
  - 输入：基线比例 `baseline_rate`、希望检测的绝对差值 `mde`、`alpha`、`power`、`sided`；
  - 输出：每组最小样本量 `n_per_group`；
  - 适用：付费率、留存率、活跃率、转化率等指标。
- `calculate_for_means_nonparametric` / `summarize_for_means_nonparametric`
  - 基于 Mann–Whitney U 相对 t 检验的渐近效率，对连续型非参数检验给出一个“在 t 检验样本量基础上适度放大”的实用样本量估计；
  - 适用：已经决定使用秩和类非参数检验的连续指标。

示例（比率型指标样本量）：

```python
from core.stats import SampleSizeCalculation

ssc = SampleSizeCalculation()
n = ssc.calculate_for_proportions(
    baseline_rate=0.1,  # 当前付费率 10%
    mde=0.02,           # 希望至少能检测到 +2pct 的提升
    alpha=0.05,
    power=0.8,
    sided="two-sided",
)
print("每组至少需要样本数:", n)
```


---

## AB 实验模块（modules.abtest）

### run_ab_test 使用示例（快速上手）

- 配置：在 `configs/abtest_demo.yaml` 中声明要分析的指标（示例已内置 ARPU/ARPPU/付费率/留存/在线时长等）。
- 数据：准备一张宽表 `df`，**每行 = 一个 (ab_group, phase)**，各指标列为该组合下的样本列表。

```python
import yaml
from modules.abtest import load_abtest_config, run_ab_test

with open("configs/abtest_demo.yaml", "r", encoding="utf-8") as f:
    raw_cfg = yaml.safe_load(f)

config = load_abtest_config(raw_cfg)
result_df = run_ab_test(
    df=df,                 # 你聚合好的宽表
    config=config,
    base_group="control",  # 对照组名称
    target_phases=["after"]
)
```

### run_ab_test 结果字段说明（按列）

| 列名             | 含义                                                                 |
|------------------|----------------------------------------------------------------------|
| `metric`         | 指标名称（如 `arppu`, `pay_rate`, `rr1` 等）                         |
| `metric_type`    | 指标类型：`"continuous"` 连续型 / `"proportion"` 比例型              |
| `phase`          | 实验阶段（如 `"before"` / `"after"`）                                |
| `base_group`     | 对照组名称（与 `base_group` 入参一致）                              |
| `variant_group`  | 与对照组对比的实验组名称                                            |
| `used_method`    | 实际使用的检验方法（如 `"mean"`, `"log_mean"`, `"mann_whitney"` 等） |
| `p_value`        | p 值                                                                 |
| `is_significant` | 是否在给定 `alternative` + 置信水平下显著                           |
| `effect`         | 效应大小：连续型为均值差，比例型为比例差（实验组 − 对照组）         |
| `diagnosis`      | 分布诊断/样本汇总信息（字典，见下文）                               |
| `stat_detail`    | 统计检验的完整结果（字典，见下文）                                  |
| `sample_size_plan` | 样本量规划结果（字典，见下文，未配置 sample_size 时为 None）     |

### diagnosis 字段结构

- **连续型指标（`metric_type == "continuous"`）**

| 键名                   | 含义                                              |
|------------------------|---------------------------------------------------|
| `n`                    | 样本数（对照组样本数量）                          |
| `mean` / `std`         | 均值 / 标准差                                     |
| `min` / `max` / `median` | 最小值 / 最大值 / 中位数                        |
| `skewness`             | 偏度，>0 右偏，<0 左偏，绝对值越大越偏           |
| `kurtosis_excess`      | 超额峰度，>0 尖峰重尾，<0 平坦                   |
| `zero_ratio`           | 样本为 0 的占比（识别 ARPU 这类零膨胀分布）      |
| `is_approximately_normal` | 是否“大致正态”                               |
| `is_heavy_tailed`      | 是否重尾 / 极端偏态                              |
| `is_zero_inflated`     | 是否存在大量 0                                   |
| `recommended_tests`    | 推荐检验方法列表，如 `["mean_z_test"]` 等        |

- **比例型指标（`metric_type == "proportion"`）**

| 键名              | 含义                              |
|-------------------|-----------------------------------|
| `n_control`       | 对照组样本量                      |
| `n_variant`       | 实验组样本量                      |
| `control_success` | 对照组成功数（如付费人数）        |
| `variant_success` | 实验组成功数                      |

### stat_detail 字段结构（按 used_method 解读）

- **`used_method = "mean"`（连续型均值检验）**
  - 主要键：
    - `control_mean` / `variant_mean`：两组均值  
    - `effect`：均值差（实验组 − 对照组）  
    - `statistic`：z 统计量  
    - `p_value` / `is_significant` / `alpha` / `alternative`  
    - `confidence_interval`：均值差的置信区间 `(low, high)`

- **`used_method = "log_mean"`（对数均值检验，适合 ARPPU 等）**
  - 在 `mean` 的基础上增加：
    - `control_mean_log` / `variant_mean_log`：log 空间均值  
    - `effect_log`：对数均值差  
    - `effect_ratio`：`exp(effect_log)`，约等于“实验组/对照组”的倍数变化  
    - `confidence_interval_log`：log 空间区间  
    - `confidence_interval_ratio`：原始空间的倍数区间  
  - 业务解读示例：`effect_ratio = 1.10` 可读作“实验组 ARPPU 比对照组高约 10%”。

- **`used_method = "mann_whitney"`（Mann–Whitney U 非参数检验）**
  - 主要键：
    - `u_statistic`：U 统计量  
    - `z_statistic`：近似正态下的 z 值  
    - `p_value` / `is_significant` / `alternative`  
    - `n_control` / `n_variant`：样本量  
  - 更关注整体分布/中位数是否有系统性偏移，而不是直接给出均值差。

- **`used_method = "proportion_z_test"`（比例检验）**
  - 主要键：
    - `control_rate` / `variant_rate`：两组比例  
    - `effect`：比例差（实验组 − 对照组）  
    - `statistic`：z 统计量  
    - `p_value` / `is_significant` / `alpha` / `alternative`  
    - `confidence_interval`：比例差的置信区间

### sample_size_plan 字段结构

- **连续型指标**

| 键名          | 含义                                       |
|---------------|--------------------------------------------|
| `type`        | `"continuous"`                             |
| `n_per_group` | 每组推荐最小样本量（向上取整后的数值）     |
| `alpha`       | 显著性水平                                 |
| `power`       | 统计功效                                   |
| `mde`         | 最小可检测效应（标准化效应大小）           |
| `method`      | `"means"`（t/z 检验）或 `"nonparametric"`（非参数检验） |

- **比例型指标**

| 键名          | 含义                                       |
|---------------|--------------------------------------------|
| `type`        | `"proportion"`                             |
| `n_per_group` | 每组推荐最小样本量                         |
| `alpha`       | 显著性水平                                 |
| `power`       | 统计功效                                   |
| `mde`         | 绝对差值形式的最小可检测效应（如 +0.02）   |
| `baseline_rate` | 基线比例（如当前付费率 0.1）            |

结合 `diagnosis` 中的实际样本量（如 `n` 或 `n_control/n_variant`）对比 `n_per_group`，可以快速判断本次实验是“样本充足的显著性结果”，还是“样本偏小、只能当作参考”。 
