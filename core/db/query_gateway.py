"""
core.db.query_gateway
---------------------

统一查询网关：按配置选择 API 或 Trino 执行 SQL。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import pandas as pd

from .api_client import ApiQueryConfig, run_sql as run_sql_api
from .trino_client import TrinoQueryConfig, export_sql_trino_to_csv, run_sql_trino

QueryMode = Literal["api", "trino"]


@dataclass
class QueryRuntimeConfig:
    """
    查询网关运行时配置。

    输入：
        mode: 查询模式，支持 "api" 与 "trino"。
        api_config: API 模式配置对象。
        trino_config: Trino 模式配置对象。
    输出：
        无，作为网关执行参数。
    """

    mode: QueryMode
    api_config: Optional[ApiQueryConfig] = None
    trino_config: Optional[TrinoQueryConfig] = None


def build_query_runtime_config(db_cfg: Dict[str, Any], mode: Optional[str] = None) -> QueryRuntimeConfig:
    """
    从 db_local.yaml 解析出的字典构建网关运行时配置。

    输入：
        db_cfg: 配置字典。
        mode: 外部显式指定的查询模式（"api"/"trino"）。为 None 时回退到配置文件。
    输出：
        QueryRuntimeConfig。
    异常：
        KeyError: 缺少必要字段时抛出。
        ValueError: mode 非法时抛出。
    """
    selected_mode = str(mode if mode is not None else db_cfg.get("mode", "api")).strip().lower()
    if selected_mode not in {"api", "trino"}:
        raise ValueError(f"不支持的 mode：{selected_mode}。仅支持 api/trino。")

    api_cfg_dict = db_cfg.get("api") or {}
    sql_cfg = db_cfg.get("sql") or {}
    trino_cfg_dict = db_cfg.get("trino") or {}

    api_config: Optional[ApiQueryConfig] = None
    trino_config: Optional[TrinoQueryConfig] = None

    if selected_mode == "api":
        query_url = api_cfg_dict.get("query_url")
        token_env_var = api_cfg_dict.get("token_env_var")
        if not query_url:
            raise KeyError("db_local.yaml 中缺少必填字段：api.query_url")
        if not token_env_var:
            raise KeyError("db_local.yaml 中缺少必填字段：api.token_env_var")
        api_config = ApiQueryConfig(
            query_url=query_url,
            token_header=api_cfg_dict.get("token_header", "X-Token"),
            token_env_var=token_env_var,
            timeout=int(api_cfg_dict.get("timeout", 600)),
            retry_count=int(api_cfg_dict.get("retry_count", 2)),
            retry_interval=float(api_cfg_dict.get("retry_interval", 2)),
            extra_headers=api_cfg_dict.get("extra_headers") or {},
            extra_body=api_cfg_dict.get("extra_body") or {},
            sql_key=sql_cfg.get("sql_key", "sql"),
        )
    else:
        host = trino_cfg_dict.get("host")
        user = trino_cfg_dict.get("user")
        catalog = trino_cfg_dict.get("catalog")
        schema = trino_cfg_dict.get("schema")

        if not host:
            raise KeyError("db_local.yaml 中缺少必填字段：trino.host")
        if not user:
            raise KeyError("db_local.yaml 中缺少必填字段：trino.user")
        if not catalog:
            raise KeyError("db_local.yaml 中缺少必填字段：trino.catalog")
        if not schema:
            raise KeyError("db_local.yaml 中缺少必填字段：trino.schema")

        trino_config = TrinoQueryConfig(
            host=host,
            port=int(trino_cfg_dict.get("port", 8080)),
            user=user,
            catalog=catalog,
            schema=schema,
            timeout=int(trino_cfg_dict.get("timeout", 600)),
            retry_count=int(trino_cfg_dict.get("retry_count", 2)),
            retry_interval=float(trino_cfg_dict.get("retry_interval", 2)),
            http_scheme=str(trino_cfg_dict.get("http_scheme", "http")),
            fetch_size=int(trino_cfg_dict.get("fetch_size", 500000)),
            progress_log_every_batches=int(trino_cfg_dict.get("progress_log_every_batches", 10)),
        )

    return QueryRuntimeConfig(mode=selected_mode, api_config=api_config, trino_config=trino_config)


def run_sql(sql: str, runtime_cfg: QueryRuntimeConfig) -> pd.DataFrame:
    """
    统一 SQL 执行入口，根据 mode 路由到底层执行器。

    输入：
        sql: SQL 文本。
        runtime_cfg: QueryRuntimeConfig。
    输出：
        pd.DataFrame：查询结果。
    """
    if runtime_cfg.mode == "api":
        if runtime_cfg.api_config is None:
            raise ValueError("当前 mode=api，但 api_config 为空。")
        return run_sql_api(sql, runtime_cfg.api_config)

    if runtime_cfg.mode == "trino":
        if runtime_cfg.trino_config is None:
            raise ValueError("当前 mode=trino，但 trino_config 为空。")
        return run_sql_trino(sql, runtime_cfg.trino_config)

    # 理论上已在 build_query_runtime_config 中校验，此处做兜底
    raise ValueError(f"不支持的 mode：{runtime_cfg.mode}")


def export_sql_to_csv(
    sql: str,
    output_csv_path: str,
    runtime_cfg: QueryRuntimeConfig,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, int]:
    """
    统一 SQL 导出入口：根据 mode 执行并写入 CSV。

    输入：
        sql: SQL 文本。
        output_csv_path: 输出 CSV 路径。
        runtime_cfg: QueryRuntimeConfig。
        on_progress: 进度回调，参数为(累计行数, 已完成批次数)。
    输出：
        (row_count, col_count)：导出行数与列数。
    """
    if runtime_cfg.mode == "trino":
        if runtime_cfg.trino_config is None:
            raise ValueError("当前 mode=trino，但 trino_config 为空。")
        return export_sql_trino_to_csv(
            sql,
            runtime_cfg.trino_config,
            output_csv_path,
            on_progress=on_progress,
        )

    if runtime_cfg.mode == "api":
        if runtime_cfg.api_config is None:
            raise ValueError("当前 mode=api，但 api_config 为空。")
        df = run_sql_api(sql, runtime_cfg.api_config)
        df.to_csv(output_csv_path, index=False)
        return len(df), len(df.columns)

    raise ValueError(f"不支持的 mode：{runtime_cfg.mode}")
