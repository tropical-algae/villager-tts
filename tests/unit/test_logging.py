"""Structured logging tests."""

import json
import logging

from villager_tts.common.logging import JsonFormatter, bind_request_id, reset_request_id


def test_json_formatter_adds_request_context_and_safe_stage_fields() -> None:
    record = logging.getLogger("test").makeRecord(
        "test",
        logging.INFO,
        __file__,
        10,
        "stage finished",
        (),
        None,
        extra={
            "stage": "tts",
            "duration_ms": 12.5,
            "output_samples": 480,
            "unapproved_field": "excluded",
        },
    )
    token = bind_request_id("request-123")
    try:
        payload = json.loads(JsonFormatter().format(record))
    finally:
        reset_request_id(token)

    assert payload["request_id"] == "request-123"
    assert payload["stage"] == "tts"
    assert payload["duration_ms"] == 12.5
    assert payload["output_samples"] == 480
    assert "unapproved_field" not in payload
