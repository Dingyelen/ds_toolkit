"""
core.db.trino_client
--------------------

Trino 查询客户端。

设计原则：
- 不直接读取 configs/*.yaml，由调用方注入 TrinoQueryConfig；
- 对齐现有 api_client 的调用体验：输入 SQL，输出 DataFrame；
- 提供基础重试能力，降低瞬时网络抖动影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import csv
import time

import pandas as pd
from trino.dbapi import connect

try:
    # 若项目中已实现 core.logger，则优先使用统一日志
    from core.logger import get_logger  # type: ignore[import]

    _logger = get_logger(__name__)
except Exception:  # pragma: no cover - 日志模块缺失时静默降级
    _logger = None


@dataclass
class TrinoQueryConfig:
    """
    Trino 查询配置对象。

    输入：
        host: Trino 服务地址。
        port: Trino 服务端口。
        user: 登录用户名。
        catalog: 查询使用的 catalog。
        schema: 查询使用的 schema。
        timeout: 查询超时时间（秒）。
        retry_count: 失败后重试次数，0 表示不重试。
        retry_interval: 重试间隔（秒）。
        http_scheme: 协议，通常为 "http" 或 "https"。
        fetch_size: 流式抓取时每批行数。
        progress_log_every_batches: 每抓取多少批回调一次进度日志。
    输出：
        无，作为 run_sql_trino 的参数对象使用。
    """

    host: str
    port: int
    user: str
    catalog: str
    schema: str
    timeout: int = 600
    retry_count: int = 2
    retry_interval: float = 2.0
    http_scheme: str = "http"
    fetch_size: int = 500000
    progress_log_every_batches: int = 10


def _execute_cursor(cur: object, sql_text: str, timeout: int) -> None:
    """
    兼容不同版本 Trino DBAPI 的 execute 调用差异。

    输入：
        cur: Trino cursor 对象。
        sql_text: SQL 文本。
        timeout: 超时时间（秒）。
    输出：
        无。
    """
    try:
        cur.execute(sql_text, timeout=timeout)
    except TypeError:
        cur.execute(sql_text)


def _query_once(sql_text: str, cfg: TrinoQueryConfig) -> pd.DataFrame:
    """
    执行一次 Trino 查询并返回 DataFrame。

    输入：
        sql_text: 已校验的 SQL 文本。
        cfg: TrinoQueryConfig。
    输出：
        pd.DataFrame：查询结果。
    """
    conn = connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        catalog=cfg.catalog,
        schema=cfg.schema,
        http_scheme=cfg.http_scheme,
    )
    try:
        cur = conn.cursor()
        try:
            _execute_cursor(cur, sql_text, cfg.timeout)
            rows = cur.fetchall()
            columns = [desc[0] for desc in (cur.description or [])]
            return pd.DataFrame(rows, columns=columns)
        finally:
            cur.close()
    finally:
        conn.close()


def run_sql_trino(sql: str, cfg: TrinoQueryConfig) -> pd.DataFrame:
    """
    通过 Trino 执行 SQL，并返回结果 DataFrame。

    输入：
        sql: SQL 文本字符串。
        cfg: TrinoQueryConfig 配置对象。
    输出：
        pd.DataFrame：查询结果表；若无数据则返回空 DataFrame。
    逻辑：
        1. 校验 SQL 非空；
        2. 按 cfg 连接 Trino；
        3. 执行 SQL 并读取结果；
        4. 失败时按重试策略重试；
        5. 返回 DataFrame。
    """
    sql_text = (sql or "").strip()
    if not sql_text:
        raise ValueError("传入的 SQL 文本为空，请检查 .sql 文件内容。")

    try_max = 1 + max(0, cfg.retry_count)
    last_exc: Optional[Exception] = None

    for attempt in range(try_max):
        try:
            return _query_once(sql_text, cfg)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < try_max - 1:
                if _logger is not None:
                    _logger.warning(
                        f"Trino 查询失败，第 {attempt + 1}/{try_max} 次，"
                        f"{exc!s}，{cfg.retry_interval}s 后重试。"
                    )
                time.sleep(cfg.retry_interval)
            else:
                msg = (
                    f"Trino 查询失败（已重试 {cfg.retry_count} 次）："
                    f"{cfg.host}:{cfg.port}/{cfg.catalog}.{cfg.schema}，错误：{exc!s}"
                )
                if _logger is not None:
                    _logger.error(msg)
                raise RuntimeError(msg) from exc

    # 理论上不会走到此处，仅用于类型与防御性兜底
    raise RuntimeError(f"Trino 查询失败：{last_exc!s}")


def _stream_to_csv_once(
    sql_text: str,
    cfg: TrinoQueryConfig,
    output_csv_path: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, int]:
    """
    执行一次 Trino 流式查询，并将结果分批写入 CSV。

    输入：
        sql_text: 已校验 SQL。
        cfg: TrinoQueryConfig。
        output_csv_path: 结果 CSV 路径。
        on_progress: 进度回调，参数为(累计行数, 已完成批次数)。
    输出：
        (row_count, col_count)：总行数与列数。
    """
    fetch_size = max(1, int(cfg.fetch_size))
    log_every = max(1, int(cfg.progress_log_every_batches))
    row_count = 0
    batch_count = 0

    conn = connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.user,
        catalog=cfg.catalog,
        schema=cfg.schema,
        http_scheme=cfg.http_scheme,
    )
    try:
        cur = conn.cursor()
        try:
            _execute_cursor(cur, sql_text, cfg.timeout)
            columns = [desc[0] for desc in (cur.description or [])]
            col_count = len(columns)

            with open(output_csv_path, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(columns)

                while True:
                    batch_rows = cur.fetchmany(fetch_size)
                    if not batch_rows:
                        break
                    writer.writerows(batch_rows)
                    row_count += len(batch_rows)
                    batch_count += 1
                    if on_progress is not None and (batch_count % log_every == 0):
                        on_progress(row_count, batch_count)

            return row_count, col_count
        finally:
            cur.close()
    finally:
        conn.close()


def export_sql_trino_to_csv(
    sql: str,
    cfg: TrinoQueryConfig,
    output_csv_path: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[int, int]:
    """
    通过 Trino 流式执行 SQL 并直接写入 CSV，避免一次性 fetchall 导致大结果集不稳定。

    输入：
        sql: SQL 文本。
        cfg: TrinoQueryConfig 配置对象。
        output_csv_path: 输出 CSV 文件路径。
        on_progress: 进度回调，参数为(累计行数, 已完成批次数)。
    输出：
        (row_count, col_count)：结果总行数与列数。
    逻辑：
        1. 校验 SQL 非空；
        2. 使用 fetchmany 分批抓取；
        3. 分批写入 CSV；
        4. 失败后按重试策略重试。
    """
    sql_text = (sql or "").strip()
    if not sql_text:
        raise ValueError("传入的 SQL 文本为空，请检查 .sql 文件内容。")

    try_max = 1 + max(0, cfg.retry_count)
    last_exc: Optional[Exception] = None

    for attempt in range(try_max):
        try:
            return _stream_to_csv_once(sql_text, cfg, output_csv_path, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < try_max - 1:
                if _logger is not None:
                    _logger.warning(
                        f"Trino 流式导出失败，第 {attempt + 1}/{try_max} 次，"
                        f"{exc!s}，{cfg.retry_interval}s 后重试。"
                    )
                time.sleep(cfg.retry_interval)
            else:
                msg = (
                    f"Trino 流式导出失败（已重试 {cfg.retry_count} 次）："
                    f"{cfg.host}:{cfg.port}/{cfg.catalog}.{cfg.schema}，错误：{exc!s}"
                )
                if _logger is not None:
                    _logger.error(msg)
                raise RuntimeError(msg) from exc

    raise RuntimeError(f"Trino 流式导出失败：{last_exc!s}")
