"""
core.db: 数据库与查询相关的底层工具。

当前实现：
- HTTP 查询 API 客户端（基于 requests），通过 run_sql 执行 SQL 并返回 DataFrame。

注意：
- 禁止引用 modules 下的任何内容；
- 不直接读取 configs/*.yaml，由 scripts 或 modules 负责配置注入。
"""

from __future__ import annotations

from .api_client import ApiQueryConfig, run_sql

__all__ = ["ApiQueryConfig", "run_sql"]

