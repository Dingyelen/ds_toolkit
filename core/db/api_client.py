"""
core.db.api_client
-------------------

基于 HTTP 查询 API 的 SQL 执行工具。

设计原则：
- 仅依赖标准库、pandas 与 requests；
- 不直接读取 configs/*.yaml，也不关心具体目录结构；
- 通过 ApiQueryConfig 接收调用方注入的所有配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os

import pandas as pd
import requests

try:
    # 若项目中已实现 core.logger，则优先使用统一日志
    from core.logger import get_logger  # type: ignore[import]

    _logger = get_logger(__name__)
except Exception:  # pragma: no cover - 日志模块缺失时静默降级
    _logger = None


@dataclass
class ApiQueryConfig:
    """
    HTTP 查询 API 配置对象。

    输入：
        query_url: 查询接口地址（完整 URL）。
        token_header: 承载 token 的请求头字段名，例如 "X-Token"。
        token_env_var: 从哪个环境变量中读取 token，例如 "TD_API_TOKEN"。
        timeout: 请求超时时间（秒）。
        extra_headers: 额外固定请求头（除 token 以外），可为空。
        extra_body: POST body 中除 SQL 以外的固定字段（如 project_id、db 等），可为空。
        sql_key: SQL 文本在 body 中对应的字段名，例如 "sql" 或 "query"。

    输出：
        无，作为 run_sql 的参数对象使用。
    """

    query_url: str
    token_header: str
    token_env_var: str
    timeout: int = 300
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    sql_key: str = "sql"


def _get_token_from_env(env_var: str) -> str:
    """
    从环境变量中读取 token。

    输入：
        env_var: 环境变量名称。
    输出：
        token 字符串。
    异常：
        RuntimeError: 未在环境变量中找到对应值时抛出。
    """
    token = os.getenv(env_var)
    if not token:
        msg = (
            f"未在环境变量中找到访问令牌：{env_var}。"
            f"请在终端中设置，例如：export {env_var}='你的真实token'，"
            f"或在 .env 文件中配置并确保已被加载。"
        )
        if _logger is not None:
            _logger.error(msg)
        raise RuntimeError(msg)
    return token


def _parse_response_data(payload: Dict[str, Any]) -> Any:
    """
    从响应 JSON 中解析出数据部分。

    兼容两类常见结构：
    1) {"data": [...]}              -> 直接返回 list/dict；
    2) {"data": {"rows": [...]}}    -> 返回 rows 部分。
    """
    if "data" not in payload:
        raise ValueError("响应 JSON 中不包含 'data' 字段，无法解析结果。")

    data = payload["data"]
    # 若 data 是 dict 且包含 rows，则优先取 rows 字段
    if isinstance(data, dict) and "rows" in data:
        return data["rows"]
    return data


def run_sql(sql: str, cfg: ApiQueryConfig) -> pd.DataFrame:
    """
    通过 HTTP 查询 API 执行一段 SQL，并返回结果 DataFrame。

    输入：
        sql: SQL 文本字符串；
        cfg: ApiQueryConfig 配置对象，由调用方从 configs/db_local.yaml 等处构造。
    输出：
        pd.DataFrame：查询结果表；若无数据则返回空 DataFrame。
    逻辑：
        1. 校验 SQL 非空；
        2. 从环境变量中读取 token 并构造 headers；
        3. 按 cfg 构造 POST body（extra_body + {sql_key: sql}）；
        4. 调用 requests.post 发送请求，检查状态码与 JSON 结构；
        5. 将解析出的 data 转为 DataFrame 返回。
    """
    sql_text = (sql or "").strip()
    if not sql_text:
        raise ValueError("传入的 SQL 文本为空，请检查 .sql 文件内容。")

    token = _get_token_from_env(cfg.token_env_var)

    headers: Dict[str, str] = {}
    if cfg.extra_headers:
        headers.update(cfg.extra_headers)
    headers[cfg.token_header] = token

    body: Dict[str, Any] = {}
    if cfg.extra_body:
        body.update(cfg.extra_body)
    body[cfg.sql_key] = sql_text

    try:
        resp = requests.post(cfg.query_url, headers=headers, json=body, timeout=cfg.timeout)
    except requests.RequestException as exc:  # pragma: no cover - 网络异常路径
        msg = f"请求查询接口失败：{cfg.query_url}，错误：{exc!s}"
        if _logger is not None:
            _logger.error(msg)
        raise RuntimeError(msg) from exc

    if not resp.ok:
        msg = (
            f"查询接口返回异常状态码：{resp.status_code}。"
            f"响应内容：{resp.text[:500]}"
        )
        if _logger is not None:
            _logger.error(msg)
        raise RuntimeError(msg)

    try:
        payload: Dict[str, Any] = resp.json()
    except ValueError as exc:
        msg = "查询接口返回内容不是合法 JSON，无法解析。"
        if _logger is not None:
            _logger.error(msg)
        raise RuntimeError(msg) from exc

    data = _parse_response_data(payload)

    # 若 data 为空或为 None，返回空表
    if data is None:
        if _logger is not None:
            _logger.warning("查询结果 data 字段为 None，返回空 DataFrame。")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(data)
    except Exception as exc:
        msg = f"无法将查询结果转换为 DataFrame，data 类型为：{type(data)!r}。"
        if _logger is not None:
            _logger.error(msg)
        raise RuntimeError(msg) from exc

    return df

