"""OpenPI client for VLA policy inference (paper Sec 3.2).

Supports two protocols:
1. WebSocket (remote inference): ws://host:port
2. HTTP POST (serve_policy.py default): http://host:port/act

OpenPI serve_policy.py 默认使用 HTTP POST:
  POST /act
  Request:  {"observation": {...}, "prompt": "instruction"}
  Response: {"actions": [[...], ...]}

OpenPI remote inference (websocket_client_policy) 使用 WebSocket.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from loguru import logger

from roboclaw.paper.config import PolicyServerConfig
from roboclaw.paper.policy.base import PolicyInterface
from roboclaw.paper.sim.base_env import ActionChunk, Observation


class OpenPIClient(PolicyInterface):
    """Client for OpenPI (π0/π0.5) policy server.

    Supports both HTTP and WebSocket protocols.
    Default (HTTP) matches OpenPI's serve_policy.py endpoint.
    """

    def __init__(
        self,
        config: PolicyServerConfig,
        policy_name: str = "openpi_forward",
        protocol: str = "http",
    ):
        self.config = config
        self._name = policy_name
        self._protocol = protocol

        # HTTP client
        self._http_client = None

        # WebSocket client
        self._ws = None
        self._connected = False

        # Action queue (OpenPI returns full chunk, we step through it)
        self._action_queue: list[ActionChunk] = []
        self._queue_idx = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_ready(self) -> bool:
        if self._protocol == "ws":
            return self._connected
        return True  # HTTP is stateless

    def reset(self) -> None:
        self._action_queue = []
        self._queue_idx = 0

    # ── HTTP protocol (default for serve_policy.py) ──

    @property
    def _http_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/act"

    async def _infer_http(self, obs: Observation, instruction: str) -> np.ndarray:
        """Send observation via HTTP POST to /act endpoint."""
        import httpx

        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        from roboclaw.paper.agent.memory import _encode_image

        # OpenPI serve_policy.py 的请求格式
        request = {
            "observation": {
                "image": _encode_image(obs.image),
                "joint_positions": obs.joint_positions.tolist(),
                "gripper_open": obs.gripper_open,
            },
            "prompt": instruction,
        }

        response = await self._http_client.post(self._http_url, json=request)
        response.raise_for_status()
        data = response.json()
        return np.array(data["actions"], dtype=np.float32)

    # ── WebSocket protocol (remote inference) ──

    async def connect(self) -> None:
        """Establish WebSocket connection to OpenPI server."""
        try:
            import websockets

            self._ws = await websockets.connect(self.config.ws_url)
            self._connected = True
            logger.info(f"Connected to OpenPI at {self.config.ws_url}")
        except Exception as e:
            self._connected = False
            raise ConnectionError(f"Failed to connect to OpenPI server: {e}") from e

    async def disconnect(self) -> None:
        """Close connections."""
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._connected = False

        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def _infer_ws(self, obs: Observation, instruction: str) -> np.ndarray:
        """Send observation via WebSocket."""
        if self._ws is None or not self._connected:
            await self.connect()

        from roboclaw.paper.agent.memory import _encode_image

        request = {
            "image": _encode_image(obs.image),
            "instruction": instruction,
            "joint_positions": obs.joint_positions.tolist(),
        }

        await self._ws.send(json.dumps(request))
        response_text = await self._ws.recv()
        data = json.loads(response_text)
        return np.array(data["actions"], dtype=np.float32)

    # ── PolicyInterface ──

    async def infer(self, obs: Observation, instruction: str) -> ActionChunk:
        """Send observation to OpenPI and receive action chunk.

        If the action queue from a previous inference is not exhausted,
        returns the next action from the queue.
        """
        # Check if we have queued actions
        if self._queue_idx < len(self._action_queue):
            return self._dequeue_chunk()

        # Fresh inference
        if self._protocol == "ws":
            raw_actions = await self._infer_ws(obs, instruction)
        else:
            raw_actions = await self._infer_http(obs, instruction)

        # Parse: raw_actions shape (chunk_size, num_joints + 1), last dim = gripper
        num_joints = obs.joint_positions.shape[0]

        if raw_actions.ndim == 1:
            raw_actions = raw_actions.reshape(1, -1)

        if raw_actions.shape[1] > num_joints:
            joint_targets = raw_actions[:, :num_joints]
            gripper_targets = raw_actions[:, num_joints]
        else:
            joint_targets = raw_actions
            gripper_targets = np.full(raw_actions.shape[0], 0.5, dtype=np.float32)

        # Store as per-step queue
        self._action_queue = [
            ActionChunk(
                joint_targets=joint_targets[i : i + 1],
                gripper_targets=gripper_targets[i : i + 1],
            )
            for i in range(len(joint_targets))
        ]
        self._queue_idx = 0

        return self._dequeue_chunk()

    def _dequeue_chunk(self) -> ActionChunk:
        """Return next action chunk from the queue."""
        chunk_size = min(
            self.config.action_chunk_size,
            len(self._action_queue) - self._queue_idx,
        )

        joints = np.concatenate(
            [self._action_queue[self._queue_idx + i].joint_targets for i in range(chunk_size)]
        )
        grippers = np.concatenate(
            [self._action_queue[self._queue_idx + i].gripper_targets for i in range(chunk_size)]
        )
        self._queue_idx += chunk_size

        return ActionChunk(joint_targets=joints, gripper_targets=grippers)
