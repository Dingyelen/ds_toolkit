# core/utils/file_made.py
# 自动化创建日常分析需求的工作目录（主文件夹 + py/sql 子目录 + 空 Excel）

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional
import shutil

from openpyxl import Workbook
from openpyxl.styles import Font


def _sanitize_folder_part(text: str) -> str:
    """
    将字符串中不适合作为文件夹名的字符替换为下划线，避免跨平台路径错误。

    Input:
        text: 原始字符串（如 req_id、requester、req_name）。
    Output:
        替换后的字符串。
    """
    return re.sub(r'[\\/:*?"<>|]', "_", str(text).strip())


class ProjectInitializer:
    """
    日常分析需求工作目录初始化器：按规范创建主文件夹、py/sql 子目录及同名空 Excel。

    Input:
        无（构造器）。
    Output:
        无。实际产出通过 create_workspace() 返回路径并在控制台打印。
    """

    def create_workspace(
        self,
        requre_id: str,
        requester: str,
        req_name: str,
        base_path: Optional[str | Path] = None,
        excel_style: Optional[dict] = None,
        excel_template: Optional[str | Path] = None,
    ) -> Path:
        """
        在 base_path 下创建以「日期_req_id_requester_req_name」命名的工作目录及空 Excel。

        Input:
            requre_id: 需求 ID（如 REQ001）。
            requester: 项目/部门 ID（如 运营部）。
            req_name: 需求名称（如 留存分析）。
            base_path: 工作根路径；为 None 时由调用方从 config 读取后传入，此处不读配置以保持 core 与 config 解耦。
            excel_style: Excel 初始化样式配置，可从 configs/default_settings.yaml 读取后透传，
                支持字段：
                    - font_name: 字体名称（如 微软雅黑）
                    - font_size: 字号（如 10）
                    - show_gridlines: 是否显示网格线（bool）
            excel_template: Excel 模板文件路径；如存在则优先复制该模板并更名，
                若为空或路径无效，则退回到 excel_style 创建空工作簿的逻辑。
        Output:
            Path: 最终创建的主文件夹绝对路径（若已存在则带后缀的路径）。
        逻辑:
            1. 主文件夹名：YYYY-MM-DD_{requre_id}_{requester}_{req_name}，并对各段做文件名安全替换。
            2. 若该路径已存在，则自动加后缀 _1、_2… 直至不冲突，并在控制台提示。
            3. 创建子目录 py、sql。
            4. 创建与主文件夹同名的空 Excel（.xlsx），并按 excel_style 设置默认字体/字号/网格线。
            5. 在控制台打印主文件夹绝对路径，便于点击跳转。
        """
        if base_path is None:
            raise ValueError(
                "未传入 base_path。请从 configs/default_settings.yaml 的 workspace_settings.base_path 读取并传入，或在脚本中指定。"
            )
        root = Path(base_path)
        if not root.is_dir():
            raise FileNotFoundError(f"工作根路径不存在或不是目录：{root.absolute()}")

        today = date.today().strftime("%Y-%m-%d")
        safe_req_id = _sanitize_folder_part(requre_id)
        safe_proj = _sanitize_folder_part(requester)
        safe_name = _sanitize_folder_part(req_name)
        base_name = f"{today}_{safe_req_id}_{safe_proj}_{safe_name}"

        target_dir = root / base_name
        suffix = 0
        while target_dir.exists():
            suffix += 1
            target_dir = root / f"{base_name}_{suffix}"
            print(f"[提示] 路径已存在，已使用带后缀的目录：{target_dir.name}")

        target_dir.mkdir(parents=True)
        (target_dir / "py").mkdir()
        (target_dir / "sql").mkdir()

        excel_name = f"{target_dir.name}.xlsx"
        excel_path = target_dir / excel_name

        # 若配置了模板且存在文件，则优先复制模板再更名
        if excel_template is not None:
            tpl_path = Path(excel_template)
            if tpl_path.is_file():
                shutil.copy(tpl_path, excel_path)
            else:
                print(f"[提示] Excel 模板不存在或不是文件，已退回默认样式创建：{tpl_path}")

        # 如未使用模板（或模板无效），则按样式配置创建空工作簿
        if not excel_path.exists():
            style = excel_style or {}
            font_name = style.get("font_name", "微软雅黑")
            font_size = style.get("font_size", 10)
            show_gridlines = bool(style.get("show_gridlines", False))

            wb = Workbook()
            ws = wb.active

            ws.sheet_view.showGridLines = show_gridlines

            default_font = Font(name=font_name, size=font_size)
            for row in range(1, 201):
                for col in range(1, 21):
                    cell = ws.cell(row=row, column=col)
                    cell.font = default_font

            wb.save(excel_path)

        abs_path = target_dir.resolve()
        print(f"[完成] 工作目录已创建，可点击跳转：\n{abs_path}")
        return abs_path
