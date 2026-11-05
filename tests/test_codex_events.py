import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import strot
from strot._codex import collect_turn
from strot.exceptions import CompilationError
from test_compiler import REGISTRY, SAMPLE, VALID_PATHS


@pytest.fixture
def stream(monkeypatch):
    sdk = pytest.importorskip("openai_codex")
    from openai_codex import models

    types = dict(zip(
        ["item/agentMessage/delta", "item/completed", "thread/tokenUsage/updated", "turn/completed"],
        [models.AgentMessageDeltaNotification, models.ItemCompletedNotification,
         models.ThreadTokenUsageUpdatedNotification, models.TurnCompletedNotification]))
    records = json.loads((Path(__file__).parent / "fixtures/codex_events.json").read_text())
    records[2]["params"]["item"]["text"] = VALID_PATHS
    events = [models.Notification(record["method"], types[record["method"]].model_validate(record["params"]))
              for record in records]
    closed = []
    def generate():
        try:
            yield from events
        finally:
            closed.append(True)
    turn = SimpleNamespace(id="turn-1", stream=generate, interrupt=Mock())
    client = MagicMock()
    client.__enter__.return_value = client
    client.thread_start.return_value.turn.return_value = turn
    monkeypatch.setattr(sdk, "Codex", Mock(return_value=client))
    monkeypatch.setattr("strot._codex.find_cli", lambda cli=None: None)
    return sdk, turn, events, closed, client


def test_hooks_report_progress_metadata_and_preserve_cached_text(stream, tmp_path):
    _, turn, events, closed, client = stream
    seen, results, turns = [], [], []
    llm = strot.codex_sdk_llm_client(on_result=results.append, on_event=seen.append, on_turn=turns.append)
    for cached in (False, True):
        result = strot.run(SAMPLE, "triage", REGISTRY, llm=llm, state=tmp_path, execute=False)
        assert result.cache_hit is cached
    assert turns == [turn] and seen == events and closed == [True]
    assert results[0].final_response == VALID_PATHS
    assert results[0].usage.total.total_tokens == 130 and results[0].duration_ms == 125
    client.__exit__.assert_called_once()


@pytest.mark.parametrize("phase,expected", [
    ("final_answer", VALID_PATHS), (None, "Later unphased text"),
    ("partial_answer", "Later unphased text"),
])
def test_progress_collection_matches_sdk_final_message_rules(stream, phase, expected):
    sdk, turn, events, _, _ = stream
    from openai_codex.generated.v2_all import MessagePhase

    events[2].payload.item.root.phase = MessagePhase(phase) if phase else None
    actual = collect_turn(turn, lambda _: None)
    assert actual == sdk.TurnHandle.run(turn)
    assert actual.final_response == expected


@pytest.mark.parametrize("failure", [
    "observer", "transport", "failed", "failed-empty", "missing", "interrupted", "commentary-only",
])
def test_progress_errors_close_stream_and_runtime(stream, failure):
    sdk, turn, events, closed, client = stream
    hook = lambda _: None
    error = None
    if failure == "observer":
        error = OSError("observer failed")
        hook = Mock(side_effect=error)
    elif failure == "transport":
        error = sdk.TransportClosedError("connection lost")
        def broken():
            try:
                yield events[0]
                raise error
            finally:
                closed.append(True)
        turn.stream = broken
    elif failure == "missing":
        events.pop()
    elif failure == "commentary-only":
        del events[2:4]
    else:
        from openai_codex.generated.v2_all import TurnStatus, TurnError

        events[-1].payload.turn.status = TurnStatus("failed" if failure == "failed-empty" else failure)
        events[-1].payload.turn.error = TurnError(message="" if failure == "failed-empty" else "turn stopped")
    with pytest.raises(type(error) if error else CompilationError) as caught:
        strot.codex_sdk_llm_client(on_event=hook)("compile")
    if error:
        assert caught.value is error
    if failure == "failed-empty":
        assert "turn failed with status failed" in str(caught.value)
    assert closed == [True]
    client.__exit__.assert_called_once()


def test_progress_hook_can_interrupt_and_report_the_result(stream):
    _, turn, events, closed, _ = stream
    from openai_codex.generated.v2_all import TurnStatus

    handles, results = [], []
    turn.interrupt.side_effect = lambda: setattr(events[-1].payload.turn, "status", TurnStatus.interrupted)
    def cancel(event):
        if event.method == "item/agentMessage/delta":
            handles[0].interrupt()
    with pytest.raises(CompilationError, match="interrupted"):
        strot.codex_sdk_llm_client(on_turn=handles.append, on_event=cancel, on_result=results.append)("compile")
    turn.interrupt.assert_called_once()
    assert results[0].status is TurnStatus.interrupted and closed == [True]
