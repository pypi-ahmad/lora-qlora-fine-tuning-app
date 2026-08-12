from lora_finetune_studio import lifecycle


def test_schedule_application_exit_uses_daemon_timer(monkeypatch) -> None:
    scheduled: dict[str, object] = {}

    class FakeTimer:
        daemon = False

        def __init__(self, delay, function, args) -> None:
            scheduled.update(delay=delay, function=function, args=args, timer=self)

        def start(self) -> None:
            scheduled["started"] = True

    exit_function = lambda _code: None
    monkeypatch.setattr(lifecycle.threading, "Timer", FakeTimer)

    lifecycle.schedule_application_exit(1.5, exit_function)

    assert scheduled["delay"] == 1.5
    assert scheduled["function"] is exit_function
    assert scheduled["args"] == (0,)
    assert scheduled["timer"].daemon is True
    assert scheduled["started"] is True
