"""
衰减 / 生存曲线拟合工具（Weibull / PowerLaw / Exponential 等）。

设计目标：
- 仅实现通用统计学层面的曲线拟合与评估逻辑；
- 不包含任何与具体业务领域相关的概念；
- 不依赖 SciPy，仅使用标准库 math / statistics 即可完成最小二乘拟合。

约定：
- x: 自变量序列，可以从 0 开始（如时间、天数等）；
- y: 对应的比例 / 概率（0~1 之间的浮点数），内部会做简单裁剪以避免数值问题；
- 所有 fit_* 函数仅基于「观测区间」拟合参数与评估拟合优度，
  不做任何未来点的外推；外推统一通过 predict_* 完成。
"""

import math
from statistics import mean
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


def _to_float_list(values: Iterable[float], name: str) -> List[float]:
    """
    将任意可迭代对象转换为 float 列表，并做基础校验。

    参数：
    - values: 输入序列；
    - name: 参数名称，用于报错信息。
    """
    try:
        data = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} 必须是可迭代的数值序列。") from exc
    except ValueError as exc:
        raise ValueError(f"{name} 中存在无法转换为浮点数的元素。") from exc

    if not data:
        raise ValueError(f"{name} 不能为空。")
    return data


def _linear_regression(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """
    简单一元线性回归 y = a + b * x 的最小二乘解。

    输入：
    - x: 自变量序列；
    - y: 因变量序列；
    输出：
    - (a, b): 截距 a 与斜率 b。
    """
    if len(x) != len(y):
        raise ValueError("x 与 y 的长度必须一致。")
    if len(x) < 2:
        raise ValueError("线性回归至少需要 2 个样本点。")

    x_list = _to_float_list(x, "x")
    y_list = _to_float_list(y, "y")

    mx = mean(x_list)
    my = mean(y_list)

    sxx = 0.0
    sxy = 0.0
    for xi, yi in zip(x_list, y_list):
        dx = xi - mx
        sxx += dx * dx
        sxy += dx * (yi - my)

    if sxx == 0.0:
        raise ValueError("所有 x 值完全相同，无法进行线性回归。")

    b = sxy / sxx
    a = my - b * mx
    return a, b


def _compute_fit_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> Dict[str, float]:
    """
    计算拟合优度指标，用于模型选型与诊断。

    当前实现：
    - mae: 平均绝对误差；
    - rmse: 均方根误差；
    - mape: 平均绝对百分比误差（仅对 actual != 0 的点有效）。
    """
    y_true = _to_float_list(actual, "actual")
    y_pred = _to_float_list(predicted, "predicted")
    if len(y_true) != len(y_pred):
        raise ValueError("actual 与 predicted 的长度必须一致。")

    n = len(y_true)
    if n == 0:
        raise ValueError("计算拟合指标时，样本量不能为空。")

    abs_errors: List[float] = []
    sq_errors: List[float] = []
    abs_pct_errors: List[float] = []

    for yt, yp in zip(y_true, y_pred):
        diff = yp - yt
        abs_errors.append(abs(diff))
        sq_errors.append(diff * diff)
        if yt != 0.0:
            abs_pct_errors.append(abs(diff) / abs(yt))

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    mape = sum(abs_pct_errors) / len(abs_pct_errors) if abs_pct_errors else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n": float(n),
    }


def _clip_rate(rate: float, eps: float = 1e-8) -> float:
    """
    对比例 / 概率进行简单裁剪，避免出现 0 或 1 导致对数运算报错。
    """
    if math.isnan(rate):
        raise ValueError("输入序列中存在 NaN，无法进行拟合。")
    # 理论上在 [0, 1]，这里做一点宽松保护
    r = max(min(rate, 1.0 - eps), eps)
    return r


def fit_exponential_decay(
    x: Sequence[float],
    y: Sequence[float],
) -> Dict[str, float]:
    """
    拟合指数衰减曲线：y ≈ a * exp(-b * x)。

    输入：
    - x: 自变量序列（可包含 0）；
    - y: 对应的比例 / 概率（0~1）。

    输出：
    - 字典，包含：
      * "model": 模型名称 "exponential"；
      * "a": 截距参数（大致接近 y(0)）；
      * "b": 衰减强度参数；
      * 拟合优度指标（mae / rmse / mape / n）。
    """
    x_list = _to_float_list(x, "x")
    y_list = [_clip_rate(v, eps=1e-8) for v in _to_float_list(y, "y")]

    if len(x_list) != len(y_list):
        raise ValueError("x 与 y 的长度必须一致。")

    # 对数变换后线性回归：ln y = ln a - b * x
    ln_y = [math.log(v) for v in y_list]
    intercept, slope = _linear_regression(x_list, ln_y)

    a = math.exp(intercept)
    b = -slope

    # 基于观测区间上的预测，计算拟合指标
    y_pred = [a * math.exp(-b * t) for t in x_list]
    metrics = _compute_fit_metrics(y_list, y_pred)

    result: Dict[str, float] = {
        "model": "exponential",
        "a": a,
        "b": b,
    }
    result.update(metrics)
    return result


def predict_exponential_decay(
    x: Sequence[float],
    params: Mapping[str, float],
) -> List[float]:
    """
    使用指数衰减模型参数进行曲线预测。

    输入：
    - x: 需要预测的自变量序列；
    - params: 拟合得到的参数字典，至少包含 "a" 与 "b"。

    输出：
    - 对应位置的预测 y 值列表。
    """
    a = float(params.get("a", 0.0))
    b = float(params.get("b", 0.0))
    x_list = _to_float_list(x, "x")
    return [a * math.exp(-b * t) for t in x_list]


def fit_weibull_curve(
    x: Sequence[float],
    y: Sequence[float],
) -> Dict[str, float]:
    """
    拟合 Weibull 生存 / 衰减曲线：y ≈ exp(-(x / λ)^k)。

    说明：
    - 经典做法是对生存函数 S(x) 做对数变换：
      ln(-ln y(x)) = k * ln x - k * ln λ；
    - 为避免 ln(0) 与 ln(1)，内部会对 x > 0 的采样点，
      且对 y 做轻微裁剪（如 [1e-8, 1-1e-8] 区间）。

    输入：
    - x: 自变量序列，要求至少包含一个 > 0 的值；
    - y: 对应的比例 / 概率（0~1）。

    输出：
    - 字典，包含：
      * "model": "weibull"；
      * "k": 形状参数；
      * "lambda": 尺度参数 λ；
      * 拟合优度指标（基于参与拟合的样本点）。
    """
    raw_x = _to_float_list(x, "x")
    raw_y = [_clip_rate(v, eps=1e-8) for v in _to_float_list(y, "y")]

    if len(raw_x) != len(raw_y):
        raise ValueError("x 与 y 的长度必须一致。")

    # 仅使用 x > 0 的点参与拟合（避免 ln 0）
    x_log: List[float] = []
    y_loglog: List[float] = []
    used_rates: List[float] = []
    used_days: List[float] = []

    for t, v in zip(raw_x, raw_y):
        if t <= 0.0:
            continue
        vv = _clip_rate(v, eps=1e-8)
        # ln(-ln y)
        inner = -math.log(vv)
        if inner <= 0.0:
            # 理论上 inner>0，这里做保护
            continue
        x_log.append(math.log(t))
        y_loglog.append(math.log(inner))
        used_days.append(t)
        used_rates.append(vv)

    if len(x_log) < 2:
        raise ValueError("Weibull 拟合至少需要 2 个 x>0 的样本点。")

    intercept, slope = _linear_regression(x_log, y_loglog)
    k = slope
    if k == 0.0:
        raise ValueError("Weibull 拟合得到的形状参数 k 为 0，结果无意义。")
    # 截距 c = -k * ln λ -> ln λ = -c / k
    ln_lambda = -intercept / k
    lambda_param = math.exp(ln_lambda)

    # 基于用于拟合的样本点计算拟合指标
    y_pred = [math.exp(-((d / lambda_param) ** k)) for d in used_days]
    metrics = _compute_fit_metrics(used_rates, y_pred)

    result: Dict[str, float] = {
        "model": "weibull",
        "k": k,
        "lambda": lambda_param,
    }
    result.update(metrics)
    return result


def predict_weibull_curve(
    x: Sequence[float],
    params: Mapping[str, float],
) -> List[float]:
    """
    使用 Weibull 模型参数进行曲线预测。

    输入：
    - x: 需要预测的自变量序列；
    - params: 包含 "k" 与 "lambda" 的参数字典。

    输出：
    - 对应位置的预测 y 值列表。
    """
    k = float(params.get("k", 0.0))
    lambda_param = float(params.get("lambda", 0.0))
    if k <= 0.0 or lambda_param <= 0.0:
        raise ValueError("Weibull 模型参数 k 与 lambda 必须为正数。")

    x_list = _to_float_list(x, "x")
    return [math.exp(-((t / lambda_param) ** k)) if t >= 0.0 else 1.0 for t in x_list]


def fit_powerlaw_curve(
    x: Sequence[float],
    y: Sequence[float],
) -> Dict[str, float]:
    """
    拟合幂律衰减模型：y ≈ c * x^{-alpha}。

    说明：
    - 对数变换后为：ln y = ln c - alpha * ln x；
    - 仅使用 x > 0 的点参与拟合，且对 y 做裁剪避免 log(0)。

    输入：
    - x: 自变量序列（至少包含一个 >0 的值）；
    - y: 对应比例 / 概率（0~1）。

    输出：
    - 字典，包含：
      * "model": "powerlaw"；
      * "alpha": 幂指数；
      * "c": 系数；
      * 拟合优度指标。
    """
    raw_x = _to_float_list(x, "x")
    raw_y = [_clip_rate(v, eps=1e-8) for v in _to_float_list(y, "y")]

    if len(raw_x) != len(raw_y):
        raise ValueError("x 与 y 的长度必须一致。")

    x_log: List[float] = []
    y_log: List[float] = []
    used_days: List[float] = []
    used_rates: List[float] = []

    for t, v in zip(raw_x, raw_y):
        if t <= 0.0:
            continue
        vv = _clip_rate(v, eps=1e-8)
        x_log.append(math.log(t))
        y_log.append(math.log(vv))
        used_days.append(t)
        used_rates.append(vv)

    if len(x_log) < 2:
        raise ValueError("PowerLaw 拟合至少需要 2 个 x>0 的样本点。")

    intercept, slope = _linear_regression(x_log, y_log)
    c = math.exp(intercept)
    alpha = -slope

    # 基于用于拟合的样本点计算拟合指标
    y_pred = [c * (d ** (-alpha)) for d in used_days]
    metrics = _compute_fit_metrics(used_rates, y_pred)

    result: Dict[str, float] = {
        "model": "powerlaw",
        "alpha": alpha,
        "c": c,
    }
    result.update(metrics)
    return result


def predict_powerlaw_curve(
    x: Sequence[float],
    params: Mapping[str, float],
) -> List[float]:
    """
    使用幂律模型参数进行曲线预测。

    输入：
    - x: 需要预测的自变量序列；
    - params: 包含 "alpha" 与 "c" 的参数字典。

    输出：
    - 对应位置的预测 y 值列表。
    """
    alpha = float(params.get("alpha", 0.0))
    c = float(params.get("c", 0.0))
    if alpha < 0.0:
        raise ValueError("PowerLaw 模型的 alpha 不应为负数（通常为正衰减系数）。")

    x_list = _to_float_list(x, "x")
    result: List[float] = []
    for t in x_list:
        if t <= 0.0:
            # x<=0 时视为尚未发生衰减，直接返回基准值 c
            result.append(c)
        else:
            result.append(c * (t ** (-alpha)))
    return result

