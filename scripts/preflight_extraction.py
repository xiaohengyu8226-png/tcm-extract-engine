"""Read-only service checks; never prints credentials or document content."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from tcm_extract_engine.common.config import get_settings


async def main() -> None:
    settings = get_settings()
    rows = []
    for role in ("prompt", "extract", "verify", "vision"):
        config = getattr(settings, role + "_llm")
        entry = {
            "role": role,
            "model": config.model,
            "configured": bool(config.api_key),
            "reachable": False,
        }
        try:
            async with httpx.AsyncClient(
                timeout=15,
                trust_env=urlsplit(config.base_url).hostname
                not in {"localhost", "127.0.0.1", "::1"},
            ) as client:
                response = await client.get(
                    config.base_url.rstrip("/") + "/models",
                    headers={"Authorization": "Bearer " + config.api_key},
                )
                entry["http_status"] = response.status_code
                entry["reachable"] = response.is_success
                if response.is_success:
                    entry["configured_model_listed"] = config.model in [
                        x.get("id") for x in response.json().get("data", [])
                    ]
        except Exception as exc:
            entry["error_type"] = type(exc).__name__
        rows.append(entry)
    target = Path("workspace/evaluation/preflight.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
