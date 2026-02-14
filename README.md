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

## 依赖关系

- **单向依赖**：`scripts` → `modules` → `core`，禁止 core 引用 modules。
- **模块隔离**：`modules` 下各子目录不互相 import；组合使用在 `scripts` 中编排。
- **配置驱动**：业务参数（路径、阈值、列名等）从 `configs/*.yaml` 读取。

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

兼容旧写法：`from core.file_io import DataLoader` 同样可用。

---

## 作者与维护

| 项目 | 说明 |
|------|------|
| **作者 / Author** | （在此填写你的名字或 GitHub 用户名） |
| **仓库 / Repo** | 可在 GitHub 仓库 About 中填写简介与链接 |

如需按 GitHub 社区规范补充作者信息，建议：

- 在 **README 本表**中填写「作者」一行；
- 在仓库 **About** 中设置 Description 和 Website（个人主页或文档）；
- 多人协作时可另建 `CONTRIBUTORS.md` 列出贡献者。
