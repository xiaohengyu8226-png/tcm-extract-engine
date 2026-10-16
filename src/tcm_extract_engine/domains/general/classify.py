import re


def route_section(title: str, **context: object) -> str:
    # Exact structural headings; do not suppress a chapter merely mentioning references.
    cleaned = re.sub(r"^[\s\d一二三四五六七八九十、.．]+", "", title).strip()
    if cleaned in {
        "目录",
        "参考文献",
        "索引",
        "前言",
        "序言",
        "版权页",
        "Contents",
        "References",
        "Index",
    }:
        return "IRRELEVANT"
    return "KNOWLEDGE"


def infer_document_type(book_name: str) -> str:
    return "UNKNOWN"
