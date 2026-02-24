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

