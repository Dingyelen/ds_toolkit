# core/file_io.py
# 兼容层：将 file_io 功能保留在此处重导出，便于外部继续使用 from core.file_io import DataLoader

from core.utils.file_io import DataLoader

__all__ = ["DataLoader"]
