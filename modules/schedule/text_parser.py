from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Tuple


@dataclass
class ParsedActivityText:
    """
    单元格文本解析结果。

    输入：
    - raw_text: 原始单元格文本内容。

    输出：
    - owner: 负责人姓名，若无法解析则为空字符串或配置中的默认值。
    - activity_name: 活动名称，去除负责人等前缀后的主体名称。
    - start_date: 起始日期（date），通常来自列头日期。
    - end_date: 结束日期（date），通常来自列头日期。
    - time_note: 单元格内用于描述时间段的原始文本（如 "1/4 12:00 - 1/9 12:00"），若不存在则为 None。
    """

    raw_text: str
    owner: str
    activity_name: str
    start_date: Optional[date]
    end_date: Optional[date]
    time_note: Optional[str]


def _parse_owner_and_name(first_line: str, text_rules: Dict[str, Any]) -> Tuple[str, str]:
    """
    从第一行文本中解析负责人与活动名称。

    输入：
    - first_line: 第一行文本，例如 "（张三）618大促预热"
    - text_rules: text_rules 配置字典，至少包含：
        - owner_pattern: 用于匹配负责人的正则表达式，需包含命名分组 owner
        - owner_missing_default: 当未解析出负责人时的默认值

    输出：
    - (owner, activity_name): 二元组

    逻辑：
    - 先用 owner_pattern 匹配括号内负责人；
    - 若匹配成功，去掉匹配子串后剩余部分作为活动名称；
    - 若匹配失败，则 owner 取默认值，活动名称为原始 first_line 去除首尾空白。
    """
    owner_pattern = text_rules.get("owner_pattern") or ""
    owner_default = text_rules.get("owner_missing_default", "")

    if not owner_pattern:
        return owner_default, first_line.strip()

    try:
        regex = re.compile(owner_pattern)
    except re.error:
        # 配置错误时退化为简单截断
        return owner_default, first_line.strip()

    match = regex.search(first_line)
    if not match:
        return owner_default, first_line.strip()

    owner = match.groupdict().get("owner", "").strip() or owner_default
    # 去掉匹配到的括号部分，剩余为活动名称
    activity_name = (first_line[: match.start()] + first_line[match.end() :]).strip()
    return owner, activity_name


def parse_activity_text(
    raw_value: Any,
    default_start_date: date,
    default_end_date: date,
    datetime_cfg: Dict[str, Any],
    text_rules: Dict[str, Any],
) -> ParsedActivityText:
    """
    解析单个活动单元格的文本内容。

    输入：
    - raw_value: 单元格原始值，通常为字符串，可包含换行符。
    - default_start_date: 从列头推断的起始日期（仅日期部分）
    - default_end_date: 从列头推断的结束日期（仅日期部分）
    - datetime_cfg: datetime 配置字典（当前仅用于兼容参数，不参与具体逻辑）
    - text_rules: 文本解析规则配置字典

    输出：
    - ParsedActivityText 对象，包含负责人、活动名与起止日期及时间备注。

    逻辑：
    - 按行拆分文本，第一行解析负责人与活动名称；
    - 若有第二行及以上，则将第二行（首个非空行）视为时间备注 time_note；
    - 起止日期始终使用 default_start_date / default_end_date，不从文本中覆盖。
    """
    text = "" if raw_value is None else str(raw_value).strip()
    if not text:
        return ParsedActivityText(
            raw_text="",
            owner=text_rules.get("owner_missing_default", ""),
            activity_name="",
            start_date=None,
            end_date=None,
            time_note=None,
        )

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ParsedActivityText(
            raw_text=text,
            owner=text_rules.get("owner_missing_default", ""),
            activity_name="",
            start_date=None,
            end_date=None,
            time_note=None,
        )

    first_line = lines[0].strip()
    owner, activity_name = _parse_owner_and_name(first_line, text_rules)

    # 若存在第二行，则将其视为时间备注
    time_note: Optional[str] = None
    if len(lines) >= 2:
        time_note = lines[1].strip()

    return ParsedActivityText(
        raw_text=text,
        owner=owner,
        activity_name=activity_name,
        start_date=default_start_date,
        end_date=default_end_date,
        time_note=time_note,
    )

