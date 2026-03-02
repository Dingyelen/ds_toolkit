"""
scripts/run_sql_export.py
-------------------------

根据配置与指定的需求目录名称，读取该目录下 sql/*.sql 中的 SQL，
通过 HTTP 查询 API 执行并将结果保存为 {filename}_res.csv 到需求目录根目录。

使用方式（示例）：
1. 确保已在 configs/default_settings.yaml 中配置 workspace_settings.base_path；
2. 在 configs/db_local.yaml 中配置 HTTP 查询 API 相关信息；
3. 在项目根目录 .env 中设置访问令牌（如 TD_API_TOKEN=你的真实token）；
4. 修改本脚本中的 REQ_DIR_NAME 为当前要处理的需求目录名；
5. 在项目根目录下运行：
   python -m scripts.run_sql_export
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any
import os

try:
    from dotenv import load_dotenv  # type: ignore[import]
except Exception:  # pragma: no cover - 若未安装 python-dotenv，则静默忽略
    load_dotenv = None

# 将项目根目录加入 sys.path，保证 from core.xxx 可被解析（无论从何处执行脚本）
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from core.utils import load_yaml, list_sql_files, read_sql_file  # noqa: E402
from core.db import ApiQueryConfig, run_sql  # noqa: E402


def _ensure_env_loaded() -> None:
    """
    确保 .env 中的环境变量被加载（若安装了 python-dotenv）。

    输入：
        无。
    输出：
        无。若存在 .env 且已安装 python-dotenv，则自动加载。
    """
    if load_dotenv is None:
        return
    env_path = _project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)  # type: ignore[call-arg]


def _load_default_config() -> Dict[str, Any]:
    """
    从 configs/default_settings.yaml 读取全局默认配置。

    输入：
        无。
    输出：
        dict：解析后的配置字典。
    异常：
        FileNotFoundError: 配置文件不存在时抛出。
    """
    config_path = _project_root / "configs" / "default_settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    return load_yaml(config_path)


def _load_db_config() -> Dict[str, Any]:
    """
    从 configs/db_local.yaml 读取本地数据库/API 配置。

    输入：
        无。
    输出：
        dict：解析后的配置字典。
    异常：
        FileNotFoundError: 配置文件不存在时抛出。
    """
    config_path = _project_root / "configs" / "db_local.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"本地数据库配置不存在：{config_path}。"
            f"请根据示例创建并填写 HTTP 查询 API 信息。"
        )
    return load_yaml(config_path)


def _build_api_config(db_cfg: Dict[str, Any]) -> ApiQueryConfig:
    """
    根据 db_local.yaml 的配置字典构造 ApiQueryConfig 对象。

    输入：
        db_cfg: 从 configs/db_local.yaml 解析得到的字典。
    输出：
        ApiQueryConfig: 供 run_sql 使用的配置对象。
    异常：
        KeyError: 缺少必要字段时抛出。
    """
    api_cfg = db_cfg.get("api") or {}
    sql_cfg = db_cfg.get("sql") or {}

    query_url = api_cfg.get("query_url")
    token_header = api_cfg.get("token_header", "X-Token")
    token_env_var = api_cfg.get("token_env_var")
    timeout = int(api_cfg.get("timeout", 60))
    extra_headers = api_cfg.get("extra_headers") or {}
    extra_body = api_cfg.get("extra_body") or {}
    sql_key = sql_cfg.get("sql_key", "sql")

    if not query_url:
        raise KeyError("db_local.yaml 中缺少必填字段：api.query_url")
    if not token_env_var:
        raise KeyError("db_local.yaml 中缺少必填字段：api.token_env_var")

    return ApiQueryConfig(
        query_url=query_url,
        token_header=token_header,
        token_env_var=token_env_var,
        timeout=timeout,
        extra_headers=extra_headers,
        extra_body=extra_body,
        sql_key=sql_key,
    )


def run_for_workspace(requre_dir_name: str) -> None:
    """
    针对单个需求工作目录，执行 sql/*.sql 并将结果导出为 *_res.csv。

    输入：
        requre_dir_name: 需求目录名称（不含 base_path），
                         如 "2025-02-28_RE001_运营部_留存分析"。
    输出：
        无。结果以 CSV 文件形式写入需求目录根目录。
    异常：
        各类文件不存在、配置缺失或 HTTP 请求异常会直接抛出，便于在命令行中排查。
    """
    _ensure_env_loaded()

    default_cfg = _load_default_config()
    ws = default_cfg.get("workspace_settings") or {}
    base_path = ws.get("base_path")
    if not base_path:
        raise ValueError(
            "未在 configs/default_settings.yaml 中配置 workspace_settings.base_path，"
            "无法定位需求工作目录。"
        )

    db_cfg = _load_db_config()
    api_cfg = _build_api_config(db_cfg)

    workspace_dir = Path(base_path) / requre_dir_name
    if not workspace_dir.is_dir():
        raise FileNotFoundError(f"需求工作目录不存在或不是目录：{workspace_dir}")

    sql_dir = workspace_dir / "sql"
    if not sql_dir.is_dir():
        raise FileNotFoundError(f"需求工作目录下未找到 sql 子目录：{sql_dir}")

    sql_files = list_sql_files(sql_dir)
    if not sql_files:
        print(f"[提示] 目录中未找到任何 .sql 文件：{sql_dir}")
        return

    total = len(sql_files)
    print(f"[开始] 工作目录：{workspace_dir}")
    print(f"[信息] SQL 目录：{sql_dir}，待运行 SQL 文件数：{total}")

    success_files: list[str] = []
    failed_files: list[str] = []

    for idx, sql_path in enumerate(sql_files, start=1):
        sql_text = read_sql_file(sql_path)
        print(f"[执行] 第 {idx}/{total} 个：{sql_path.name}")
        try:
            df = run_sql(sql_text, api_cfg)
            out_name = f"{sql_path.stem}_res.csv"
            out_path = workspace_dir / out_name
            df.to_csv(out_path, index=False)
            print(f"[完成] 第 {idx} 个：{sql_path.name}")
            success_files.append(sql_path.name)
        except Exception as exc:
            print(f"[失败] 第 {idx} 个：{sql_path.name}，错误：{exc!s}")
            failed_files.append(sql_path.name)

    print(f"[总结] 共 {total} 个 SQL 文件，成功 {len(success_files)} 个，失败 {len(failed_files)} 个。")
    if success_files:
        print(f"[成功列表] {', '.join(success_files)}")
    if failed_files:
        print(f"[失败列表] {', '.join(failed_files)}")


if __name__ == "__main__":
    # ---------- 使用前请修改下面变量 ----------
    # 需求目录名称：与 new_requre 创建的主目录名一致。
    # 例如：2025-02-28_RE001_运营部_留存分析
    REQURE_DIR_NAME = "2026-02-26_TP1-20260210-014_SJ_新版赛车活动效果分析"

    if not REQURE_DIR_NAME:
        raise ValueError(
            "请先在 scripts/run_sql_export.py 中设置 REQURE_DIR_NAME，"
            "然后再运行本脚本。"
        )

    run_for_workspace(REQURE_DIR_NAME)

