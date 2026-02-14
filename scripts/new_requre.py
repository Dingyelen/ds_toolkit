# scripts/new_requre.py
# 快捷入口：通过修改变量后运行，在配置的 base_path 下创建需求工作目录

from pathlib import Path

import yaml

from core.utils import ProjectInitializer


def _load_config() -> dict:
    """从 configs/default_settings.yaml 读取配置。"""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # ---------- 只需修改下面三个变量，即可快速创建需求目录 ----------
    REQURE_ID = "TP1-20260210-014"       # 需求 ID
    REQUESTER = "SJ"      # 项目/部门
    REQ_NAME = "新版赛车活动效果分析"      # 需求名称
    # 不填则使用 configs 中 workspace_settings.base_path
    BASE_PATH: str | None = None

    config = _load_config()
    ws = config.get("workspace_settings") or {}
    base_path = BASE_PATH or ws.get("base_path")
    if not base_path:
        raise ValueError(
            "未设置工作路径。请在 configs/default_settings.yaml 的 workspace_settings.base_path 中配置，或在本脚本中设置 BASE_PATH。"
        )

    initializer = ProjectInitializer()
    initializer.create_workspace(
        requre_id=REQURE_ID,
        requester=REQUESTER,
        req_name=REQ_NAME,
        base_path=base_path,
    )
