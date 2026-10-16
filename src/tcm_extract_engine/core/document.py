from enum import Enum


class DocumentType(str, Enum):
    """整本书的文档类型。"""

    STANDARD = "STANDARD"  # 规范/标准（如《中药饮片临床应用规范》）
    HANDBOOK = "HANDBOOK"  # 手册（禁忌手册/毒性手册）
    MONOGRAPH = "MONOGRAPH"  # 专著
    TEXTBOOK = "TEXTBOOK"  # 教材
    CASE_COLLECTION = "CASE_COLLECTION"  # 医案集
    UNKNOWN = "UNKNOWN"
