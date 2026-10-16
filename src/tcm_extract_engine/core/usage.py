"""Per-run client usage, including shared role aliases without double counting."""
from typing import Any


def snapshot_usage(clients: dict[str, Any]) -> dict[int, tuple[int, int, int]]:
    return {id(c): (c.usage.input_tokens, c.usage.output_tokens, c.usage.calls) for c in clients.values() if c is not None}


def usage_delta(clients: dict[str, Any], before: dict[int, tuple[int, int, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    seen: set[int] = set()
    for role, client in clients.items():
        values = (0, 0, 0)
        if client is not None and id(client) not in seen:
            old = before[id(client)]
            values = (client.usage.input_tokens-old[0], client.usage.output_tokens-old[1], client.usage.calls-old[2])
            seen.add(id(client))
        out.update({role+"_input": values[0], role+"_output": values[1], role+"_calls": values[2]})
    return out
