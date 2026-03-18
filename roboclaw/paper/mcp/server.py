"""MCP Server exposing RoboClaw paper tools via Model Context Protocol.

This wraps the 6 paper tools + 2 skills as MCP-compliant tool endpoints,
making them callable by any MCP client (Claude Desktop, other agents, etc.).

Usage:
    # Start MCP server
    server = PaperMCPServer(env=env, policy_manager=pm, provider=provider)
    await server.serve(port=8001)

    # Or register with existing roboclaw MCP infrastructure
    server.register_with_tool_registry(registry)
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from roboclaw.paper.sim.base_env import BaseEnvironment
from roboclaw.paper.policy.manager import PolicyManager
from roboclaw.paper.tools.start_policy import StartPolicyTool
from roboclaw.paper.tools.terminate_policy import TerminatePolicyTool
from roboclaw.paper.tools.switch_policy import SwitchPolicyTool
from roboclaw.paper.tools.env_summary import EnvSummaryTool
from roboclaw.paper.tools.fetch_robot_stats import FetchRobotStatsTool
from roboclaw.paper.tools.call_human import CallHumanTool


class PaperMCPServer:
    """MCP Server that exposes paper tools.

    Provides two modes:
    1. Standalone MCP server (via `serve()`)
    2. Integration with existing ToolRegistry (via `register_with_tool_registry()`)
    """

    def __init__(
        self,
        env: BaseEnvironment,
        policy_manager: PolicyManager,
        provider: Any | None = None,
        vlm_model: str = "gpt-4o",
        human_callback: Any | None = None,
    ):
        self.env = env
        self.policy_manager = policy_manager
        self.provider = provider
        self.vlm_model = vlm_model

        # Instantiate all tools
        self.tools: dict[str, Any] = {
            "start_policy": StartPolicyTool(policy_manager),
            "terminate_policy": TerminatePolicyTool(policy_manager),
            "switch_policy": SwitchPolicyTool(policy_manager),
            "fetch_robot_stats": FetchRobotStatsTool(env),
            "call_human": CallHumanTool(callback=human_callback),
        }

        # EnvSummary requires a provider
        if provider is not None:
            self.tools["env_summary"] = EnvSummaryTool(env, provider, model=vlm_model)

    def get_tools(self) -> dict[str, Any]:
        """Get all tool instances."""
        return self.tools

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get MCP-compatible tool definitions."""
        definitions = []
        for tool in self.tools.values():
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            })
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool by name (MCP tool/call handler)."""
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            result = await tool.execute(**arguments)
            return result
        except Exception as e:
            logger.error(f"MCP tool call failed: {name} - {e}")
            return json.dumps({"error": str(e)})

    def register_with_tool_registry(self, registry: Any) -> None:
        """Register all tools with an existing roboclaw ToolRegistry.

        Args:
            registry: A roboclaw.agent.tools.registry.ToolRegistry instance.
        """
        for tool in self.tools.values():
            registry.register(tool)
        logger.info(f"Registered {len(self.tools)} paper tools with ToolRegistry")

    async def serve(self, host: str = "localhost", port: int = 8001) -> None:
        """Start a standalone MCP server using the MCP Python SDK.

        This creates an MCP-compliant server that any MCP client can connect to.
        """
        try:
            from mcp.server import Server
            from mcp.server.stdio import stdio_server
            from mcp import types
        except ImportError:
            # Fallback: simple HTTP server
            logger.warning("MCP SDK not available, falling back to HTTP server")
            await self._serve_http(host, port)
            return

        server = Server("roboclaw-paper")

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [
                types.Tool(
                    name=defn["name"],
                    description=defn["description"],
                    inputSchema=defn["inputSchema"],
                )
                for defn in self.get_tool_definitions()
            ]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            result = await self.call_tool(name, arguments or {})
            return [types.TextContent(type="text", text=result)]

        logger.info(f"Starting MCP server on stdio")
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    async def _serve_http(self, host: str, port: int) -> None:
        """Simple HTTP fallback when MCP SDK is not available."""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        import uvicorn

        async def handle_list_tools(request: Request) -> JSONResponse:
            return JSONResponse({"tools": self.get_tool_definitions()})

        async def handle_call_tool(request: Request) -> JSONResponse:
            body = await request.json()
            name = body.get("name", "")
            arguments = body.get("arguments", {})
            result = await self.call_tool(name, arguments)
            return JSONResponse({"result": result})

        app = Starlette(routes=[
            Route("/tools/list", handle_list_tools, methods=["GET"]),
            Route("/tools/call", handle_call_tool, methods=["POST"]),
        ])

        logger.info(f"Starting HTTP MCP server on {host}:{port}")
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
