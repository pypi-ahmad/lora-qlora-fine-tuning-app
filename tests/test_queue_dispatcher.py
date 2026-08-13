from lora_finetune_studio import queue_dispatcher


def test_handoff_waits_for_parent_exit_before_dispatch(monkeypatch) -> None:
    events: list[str] = []

    class ParentProcess:
        def wait(self, timeout: int) -> None:
            assert timeout == 60
            events.append("parent exited")

    monkeypatch.setattr(
        queue_dispatcher.psutil, "Process", lambda _pid: ParentProcess()
    )
    monkeypatch.setattr(
        queue_dispatcher,
        "dispatch_next_run",
        lambda: events.append("next dispatched"),
    )

    assert queue_dispatcher.wait_for_parent_and_dispatch(1234) == 0
    assert events == ["parent exited", "next dispatched"]


def test_schedule_handoff_uses_base_project_interpreter(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("LORA_STUDIO_PYTHON", "C:/project/.venv/Scripts/python.exe")

    def fake_popen(command, **options):
        captured.update(command=command, options=options)

    monkeypatch.setattr(queue_dispatcher.subprocess, "Popen", fake_popen)

    queue_dispatcher.schedule_queue_handoff(1234)

    assert captured["command"] == [
        "C:/project/.venv/Scripts/python.exe",
        "-m",
        "lora_finetune_studio.queue_dispatcher",
        "1234",
    ]
