"""Logging and metrics.

Deliberately dependency-free: a counter registry and a Prometheus text
renderer are a hundred lines, and a gateway that drags in a metrics stack to
report six numbers is a gateway people fork to rip it out.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from typing import Any

from .config import Settings


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stderr)
    if settings.log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
    root = logging.getLogger("claudegate")
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level)
    root.propagate = False


class Metrics:
    """Counters and a couple of latency summaries."""

    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.labelled: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.started = time.time()

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def inc_label(self, name: str, label: str, value: float = 1.0) -> None:
        self.labelled[name][label] += value

    def observe_latency(self, name: str, seconds: float) -> None:
        self.counters[f"{name}_seconds_total"] += seconds
        self.counters[f"{name}_count"] += 1

    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        data: dict[str, Any] = dict(self.counters)
        for name, values in self.labelled.items():
            for label, value in values.items():
                data[f"{name}:{label}"] = value
        data["uptime_seconds"] = round(time.time() - self.started, 1)
        if extra:
            data.update(extra)
        return data

    def prometheus(self, extra: dict[str, Any] | None = None) -> str:
        lines: list[str] = []
        for key, value in sorted(self.counters.items()):
            lines.append(f"claudegate_{key} {value}")
        for name, values in sorted(self.labelled.items()):
            for label, value in sorted(values.items()):
                lines.append(f'claudegate_{name}{{kind="{label}"}} {value}')
        for key, value in sorted((extra or {}).items()):
            if isinstance(value, (int, float)):
                lines.append(f"claudegate_{key} {value}")
        lines.append(f"claudegate_uptime_seconds {round(time.time() - self.started, 1)}")
        return "\n".join(lines) + "\n"
