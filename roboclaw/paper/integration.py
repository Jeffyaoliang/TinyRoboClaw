"""Integration with existing roboclaw infrastructure.

Bridges the paper module with the main roboclaw framework:
- Registers paper tools with ToolRegistry
- Connects VLM agent to existing LLMProvider
- Integrates skills with SkillsLoader
"""

from __future__ import annotations

from typing import Any

from loguru import logger


def register_paper_tools(
    registry: Any,
    env: Any,
    policy_manager: Any,
    provider: Any | None = None,
    vlm_model: str = "gpt-4o",
    human_callback: Any | None = None,
) -> dict[str, Any]:
    """Register all paper tools with an existing roboclaw ToolRegistry.

    Args:
        registry: roboclaw.agent.tools.registry.ToolRegistry
        env: BaseEnvironment instance
        policy_manager: PolicyManager instance
        provider: LLMProvider for VLM tools (EnvSummary)
        vlm_model: Model name for VLM calls
        human_callback: Callback for CallHuman tool

    Returns:
        Dict of tool name → tool instance
    """
    from roboclaw.paper.tools.start_policy import StartPolicyTool
    from roboclaw.paper.tools.terminate_policy import TerminatePolicyTool
    from roboclaw.paper.tools.switch_policy import SwitchPolicyTool
    from roboclaw.paper.tools.env_summary import EnvSummaryTool
    from roboclaw.paper.tools.fetch_robot_stats import FetchRobotStatsTool
    from roboclaw.paper.tools.call_human import CallHumanTool

    tools = {}

    tools["start_policy"] = StartPolicyTool(policy_manager)
    tools["terminate_policy"] = TerminatePolicyTool(policy_manager)
    tools["switch_policy"] = SwitchPolicyTool(policy_manager)
    tools["fetch_robot_stats"] = FetchRobotStatsTool(env)
    tools["call_human"] = CallHumanTool(callback=human_callback)

    if provider is not None:
        tools["env_summary"] = EnvSummaryTool(env, provider, model=vlm_model)

    for tool in tools.values():
        registry.register(tool)

    logger.info(f"Registered {len(tools)} paper tools with ToolRegistry")
    return tools


def create_paper_agent(
    config: Any,
    provider: Any,
    env: Any,
    policy_manager: Any,
    mode: str = "data_collection",
    human_callback: Any | None = None,
) -> Any:
    """Create a VLMAgentLoop with paper tools and skills.

    This is the high-level factory that wires everything together.

    Args:
        config: PaperConfig
        provider: LLMProvider
        env: BaseEnvironment
        policy_manager: PolicyManager
        mode: "data_collection" or "deployment"
        human_callback: Callback for CallHuman tool

    Returns:
        VLMAgentLoop instance with tools and skills ready.
    """
    from roboclaw.paper.agent.loop import VLMAgentLoop
    from roboclaw.paper.skills.data_collection import DataCollectionSkill
    from roboclaw.paper.skills.long_horizon import LongHorizonSkill
    from roboclaw.paper.mcp.server import PaperMCPServer

    # Create MCP server (which instantiates all tools)
    mcp = PaperMCPServer(
        env=env,
        policy_manager=policy_manager,
        provider=provider,
        vlm_model=config.vlm.model,
        human_callback=human_callback,
    )

    tools = mcp.get_tools()

    # Add skills as callable tools
    data_skill = DataCollectionSkill()
    horizon_skill = LongHorizonSkill()

    class SkillToolWrapper:
        """Wraps a RoboticSkill as a callable tool for the agent."""

        def __init__(self, skill, all_tools):
            self._skill = skill
            self._tools = all_tools
            self.name = f"skill_{skill.name.replace('-', '_')}"
            self.description = skill.description

        async def execute(self, **kwargs):
            instruction = kwargs.get("instruction", "")
            result = await self._skill.execute(instruction, self._tools, **kwargs)
            return f"[{result.status.value}] {result.message}"

    tools["skill_data_collection"] = SkillToolWrapper(data_skill, tools)
    tools["skill_long_horizon_execution"] = SkillToolWrapper(horizon_skill, tools)

    # Create agent loop
    agent = VLMAgentLoop(
        config=config,
        provider=provider,
        env=env,
        tools=tools,
        mode=mode,
    )

    logger.info(
        f"Created paper agent: mode={mode}, "
        f"tools={list(tools.keys())}"
    )
    return agent


def create_full_pipeline(
    config: Any,
    env: Any,
    provider: Any,
    forward_policy: Any,
    inverse_policy: Any,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """Create the complete RoboClaw pipeline: agent + EAP + flywheel + deployment.

    Returns a dict of all component instances for flexible use.
    """
    from pathlib import Path
    from roboclaw.paper.policy.manager import PolicyManager
    from roboclaw.paper.policy.pool import PolicyPool
    from roboclaw.paper.policy.trainer import PolicyTrainer
    from roboclaw.paper.eap.engine import EAPEngine
    from roboclaw.paper.eap.judge import SuccessJudge
    from roboclaw.paper.eap.trajectory import TrajectoryDataset
    from roboclaw.paper.flywheel.flywheel import DataFlywheel
    from roboclaw.paper.flywheel.recycler import DeploymentRecycler
    from roboclaw.paper.deployment.supervisor import DeploymentSupervisor
    from roboclaw.paper.deployment.executor import LongHorizonExecutor

    data_dir = data_dir or config.flywheel.data_dir

    # Core components
    policy_manager = PolicyManager(config, env, forward_policy, inverse_policy)
    policy_pool = PolicyPool()
    policy_pool.register(config.task_name, forward_policy, inverse_policy)

    trainer = PolicyTrainer(config.trainer)
    judge = SuccessJudge(provider, model=config.eap.judge_model or config.vlm.model)

    # Datasets
    training_dataset = TrajectoryDataset(Path(data_dir) / "training")
    deployment_dataset = TrajectoryDataset(Path(data_dir) / "deployment")

    # EAP engine
    eap_engine = EAPEngine(config, env, forward_policy, inverse_policy, judge, training_dataset)

    # Flywheel
    flywheel = DataFlywheel(config, eap_engine, training_dataset, trainer, policy_manager)

    # Deployment
    supervisor = DeploymentSupervisor(
        config.deployment, env, policy_manager, judge, deployment_dataset
    )
    executor = LongHorizonExecutor(config, provider, env, supervisor)

    # Recycler
    recycler = DeploymentRecycler(training_dataset, deployment_dataset)

    # Agent
    agent = create_paper_agent(config, provider, env, policy_manager, mode="data_collection")

    components = {
        "agent": agent,
        "policy_manager": policy_manager,
        "policy_pool": policy_pool,
        "trainer": trainer,
        "judge": judge,
        "training_dataset": training_dataset,
        "deployment_dataset": deployment_dataset,
        "eap_engine": eap_engine,
        "flywheel": flywheel,
        "supervisor": supervisor,
        "executor": executor,
        "recycler": recycler,
    }

    logger.info(f"Full pipeline created: {list(components.keys())}")
    return components
