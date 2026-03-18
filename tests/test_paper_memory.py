"""Tests for roboclaw.paper.agent.memory (StructuredMemory)."""

import numpy as np
import pytest

from roboclaw.paper.agent.memory import (
    RoleIdentity,
    StructuredMemory,
    SubtaskRecord,
    TaskMemory,
    WorkingMemory,
    _encode_image,
)


class TestRoleIdentity:
    def test_to_messages_basic(self):
        role = RoleIdentity(system_prompt="You are a robot.")
        msgs = role.to_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert "You are a robot." in msgs[0]["content"]

    def test_to_messages_with_tools(self):
        role = RoleIdentity(
            system_prompt="System",
            tool_descriptions=[
                {"name": "pick", "description": "Pick up object"},
                {"name": "place", "description": "Place object"},
            ],
        )
        msgs = role.to_messages()
        content = msgs[0]["content"]
        assert "pick" in content
        assert "Place object" in content


class TestTaskMemory:
    def test_to_messages_basic(self):
        task = TaskMemory(task_instruction="pick up the red block")
        msgs = task.to_messages()
        assert len(msgs) == 1
        assert "pick up the red block" in msgs[0]["content"]

    def test_add_observation(self):
        task = TaskMemory(task_instruction="test")
        task.add_observation("saw red block at (0.1, 0.2)")
        task.add_observation("gripper is open")
        assert len(task.observation_log) == 2

    def test_observation_log_truncation_in_messages(self):
        task = TaskMemory(task_instruction="test")
        for i in range(10):
            task.add_observation(f"obs_{i}")
        msgs = task.to_messages()
        content = msgs[0]["content"]
        # Only last 5 observations should appear
        assert "obs_5" in content
        assert "obs_9" in content
        assert "obs_0" not in content

    def test_add_subtask_result(self):
        task = TaskMemory(task_instruction="test")
        task.add_subtask_result("pick up block", "success", ["block picked"])
        assert len(task.subtask_history) == 1
        assert task.subtask_history[0].subtask == "pick up block"
        assert task.subtask_history[0].status == "success"

    def test_subtask_history_in_messages(self):
        task = TaskMemory(task_instruction="test")
        task.add_subtask_result("step1", "success")
        task.add_subtask_result("step2", "failed")
        msgs = task.to_messages()
        content = msgs[0]["content"]
        assert "[success] step1" in content
        assert "[failed] step2" in content


class TestWorkingMemory:
    def test_empty_returns_no_messages(self):
        wm = WorkingMemory()
        assert wm.to_messages() == []

    def test_with_reasoning(self):
        wm = WorkingMemory(cot_reasoning="The block is near the gripper")
        msgs = wm.to_messages()
        assert len(msgs) == 1
        # Content is a list (multimodal format)
        text_parts = [p for p in msgs[0]["content"] if p["type"] == "text"]
        assert any("block is near" in p["text"] for p in text_parts)

    def test_with_image(self):
        wm = WorkingMemory(
            current_image=np.zeros((32, 32, 3), dtype=np.uint8),
            cot_reasoning="test",
        )
        msgs = wm.to_messages()
        content = msgs[0]["content"]
        img_parts = [p for p in content if p["type"] == "image_url"]
        assert len(img_parts) == 1
        assert img_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_with_tool_call(self):
        wm = WorkingMemory(
            tool_name="start_policy",
            tool_args={"instruction": "pick up"},
            tool_result="started",
        )
        msgs = wm.to_messages()
        text_parts = [p for p in msgs[0]["content"] if p["type"] == "text"]
        text = text_parts[0]["text"]
        assert "start_policy" in text
        assert "started" in text

    def test_reset(self):
        wm = WorkingMemory(
            current_image=np.zeros((32, 32, 3), dtype=np.uint8),
            cot_reasoning="think",
            tool_name="test",
        )
        wm.reset()
        assert wm.current_image is None
        assert wm.cot_reasoning == ""
        assert wm.tool_name == ""
        assert wm.to_messages() == []


class TestStructuredMemory:
    def test_to_messages_combines_all(self):
        mem = StructuredMemory()
        mem.role.system_prompt = "You are a robot."
        mem.task.task_instruction = "pick up block"
        mem.working.cot_reasoning = "I see the block"

        msgs = mem.to_messages()
        # system + task + working = at least 3 messages
        assert len(msgs) >= 3
        assert msgs[0]["role"] == "system"

    def test_reset_working(self):
        mem = StructuredMemory()
        mem.working.cot_reasoning = "think"
        mem.task.task_instruction = "test"
        mem.reset_working()
        assert mem.working.cot_reasoning == ""
        # Task memory preserved
        assert mem.task.task_instruction == "test"

    def test_reset_all(self):
        mem = StructuredMemory()
        mem.role.system_prompt = "test"
        mem.task.task_instruction = "test"
        mem.working.cot_reasoning = "test"
        mem.reset_all()
        assert mem.role.system_prompt == ""
        assert mem.task.task_instruction == ""
        assert mem.working.cot_reasoning == ""


class TestEncodeImage:
    def test_encode_decode(self):
        import base64

        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        b64 = _encode_image(img)
        # Should be valid base64
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0
