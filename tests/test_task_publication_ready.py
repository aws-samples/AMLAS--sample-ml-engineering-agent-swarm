"""The original capture boundary follows successful normal TASK publication."""

import asyncio
import threading
import time

from normal_sft_cold_fixture import close, original_precommit, setup

from amlas.control_plane.normal_run_checkpoint import NormalRunContext


def test_original_capture_waits_for_successful_task_publication(monkeypatch, tmp_path):
    async def scenario():
        saved = threading.Event()
        original_save = NormalRunContext.save_admitted_task

        def save(context, *args, **kwargs):
            time.sleep(0.25)
            result = original_save(context, *args, **kwargs)
            saved.set()
            return result

        monkeypatch.setattr(NormalRunContext, "save_admitted_task", save)
        state = setup(monkeypatch, tmp_path)
        try:
            await original_precommit(state, monkeypatch)
            assert saved.is_set(), "original fixture returned before successful TASK publication"
        finally:
            await close(state)

    asyncio.run(scenario())
