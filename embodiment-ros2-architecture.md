# RoboClaw Embodied Framework

这份文档描述的是 RoboClaw 目前代码里已经落下来的具身框架，不是未来愿景草图。

当前阶段只围绕一个核心目标设计：

- 让普通用户通过对话完成 `连接 / 校准 / 移动 / debug / 复位`
- 尽快把更多开源、知名本体接进 RoboClaw
- 为后续跨本体技能、研究助手、本体接入范式预留边界，但不提前过度实现

## 当前优先级

四个版图的当前优先级是：

1. 通用具身入口版图
2. 本体接入范式版图
3. 跨本体技能底座版图
4. 研究助手版图

原因很直接：

- 版图 1 是现在必须真的跑起来的东西
- 版图 4 决定版图 1 能不能快速铺满更多本体
- 版图 2 需要建立在更稳的统一 contract 之上
- 版图 3 需要 telemetry / replay / trace 更成熟之后再做

## 设计原则

- `roboclaw/embodied/` 只放 framework 代码，不放某个用户现场的 setup
- 具体 setup 由 agent 生成到 `~/.roboclaw/workspace/embodied/`
- framework 提供通用 contract、通用组件和可复用 robot/sensor 定义
- workspace 提供某个现场的 assembly、deployment、adapter、simulator 资产
- 运行时通过 catalog 把 framework definitions 和 workspace assets 合并
- 最终执行统一经过 ROS2
- ROS2 下面不是一个万能桥接，而是按本体域拆分的 bridge contract

## 文字框架图

```text
用户自然语言
  ↓
RoboClaw Agent
  ├── 理解用户意图
  ├── 发现设备与收集 intake 信息
  ├── 引导用户补齐缺失配置
  ├── 在 ~/.roboclaw/workspace/embodied/ 生成或更新 setup 文件
  └── 选择并执行 connect / calibrate / move / debug / reset procedure
  ↓
Embodied Catalog
  ├── framework definitions
  │   ├── robots
  │   ├── sensors
  │   ├── schema
  │   ├── assemblies contracts
  │   ├── deployment contracts
  │   └── simulator contracts
  └── workspace-generated assets
      ├── intake notes
      ├── local robots / sensors
      ├── assemblies
      ├── deployments
      ├── adapters
      └── simulator worlds / scenarios
  ↓
Embodied Runtime
  ├── runtime session
  ├── procedure definitions
  ├── adapter bindings
  └── telemetry / diagnostics
  ↓
Execution Integration
  ├── transports
  ├── carriers
  ├── adapters
  └── domain bridges
  ↓
ROS2
  ├── topics
  ├── services
  └── actions
  ↓
真实本体 / 仿真本体
  ├── arm / hand
  ├── humanoid / whole-body
  ├── mobile base / fleet
  ├── drone
  └── simulator
```

## 代码结构

当前具身框架主路径是 [roboclaw/embodied](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/roboclaw/embodied)。

```text
roboclaw/embodied/
  ├── definition/
  │   ├── foundation/schema/
  │   ├── components/robots/
  │   ├── components/sensors/
  │   └── systems/
  │       ├── assemblies/
  │       ├── deployments/
  │       └── simulators/
  ├── execution/
  │   ├── integration/
  │   │   ├── carriers/
  │   │   ├── transports/
  │   │   ├── adapters/
  │   │   └── bridges/
  │   ├── orchestration/
  │   │   ├── runtime/
  │   │   └── procedures/
  │   └── observability/telemetry/
  ├── catalog.py
  └── workspace.py
```

职责边界：

- `definition/`: 描述系统是什么
- `execution/integration/`: 描述请求如何到达 ROS2 和 bridge
- `execution/orchestration/`: 描述当前会话如何被选择和驱动
- `execution/observability/`: 描述执行过程如何被观测
- `catalog.py`: 合并 framework definitions 与 workspace assets
- `workspace.py`: 校验并加载 `~/.roboclaw/workspace/embodied/`

## framework 和 workspace 的边界

### framework 里应该放什么

- 通用 schema
- 可复用 robot manifests
- 可复用 sensor manifests
- assembly / deployment / simulator contract
- adapter lifecycle / compatibility / result models
- procedure contract
- domain bridge contract

### framework 里不应该放什么

- 某个用户实验台的 assembly
- 某个 lab 的串口、IP、camera device id
- 某个 demo 的安全边界和 reset 细节
- 某个用户的本地 world/scenario

### workspace 里应该放什么

`~/.roboclaw/workspace/embodied/` 当前应该承载：

- `intake/`: 设备发现与用户提供事实
- `robots/`: 暂时还不够通用的本地 robot manifest
- `sensors/`: 暂时还不够通用的本地 sensor manifest
- `assemblies/`: 当前 setup 的组合拓扑
- `deployments/`: 当前现场的连接参数和安全覆盖
- `adapters/`: 当前 setup 的 adapter binding
- `simulators/`: 当前 setup 的 world / scenario

## 第 1 版图的最小运行链路

如果 RoboClaw 要在第 1 版图里真正 work，一次完整链路应该是：

1. 用户启动 RoboClaw
2. RoboClaw 读取 workspace bootstrap 文档和具身规则
3. RoboClaw 询问或发现：
   - 本体类型
   - 传感器
   - ROS2 namespace / topics / actions / services
   - real target / sim target
   - deployment 连接细节
4. RoboClaw 在 workspace 下写 intake note
5. RoboClaw 复用 framework 中已有 robot / sensor 定义，必要时补 local-only 定义
6. RoboClaw 生成 assembly / deployment / adapter / simulator 资产
7. `build_catalog(workspace)` 把这些资产读回运行时
8. RoboClaw 选择 procedure：
   - `connect`
   - `calibrate`
   - `move`
   - `debug`
   - `reset`
9. procedure 通过 typed action ref 驱动：
   - orchestrator actions
   - adapter actions
10. adapter 通过 typed result models 把状态、健康、兼容性、执行结果返回给 runtime
11. runtime 把结果组织成用户可读反馈

这条链路是当前所有文档和代码设计的中心。

## 当前必须稳定的 contract

当前阶段最重要的不是“再加更多功能”，而是让下面这些 contract 足够稳定：

### 1. Action / Observation Contract

当前已经有：

- action schema
- observation schema
- health schema
- command mode
- tolerance / completion

当前意义：

- 让不同本体至少能在统一动作/状态语义下被描述

### 2. Assembly Topology Contract

当前已经有：

- robot attachments
- sensor attachments
- frame transforms
- tool attachments
- control groups
- safety boundaries
- failure domains
- resource ownership

当前意义：

- 让一个 setup 不只是“挂了几个对象”，而是有基本可检查拓扑

### 3. Adapter Lifecycle Contract

当前已经有：

- connect / disconnect / ready / stop / reset / recover
- dependency checks
- timeout policy
- compatibility constraints
- degraded mode
- typed result models

当前意义：

- 让接入不会停留在“能 import 一个 driver 就算接入”

### 4. Procedure Contract

当前已经有：

- step graph
- preconditions
- retry / timeout
- cancel / compensation / rollback
- idempotency
- typed action ref

当前意义：

- 第 1 版图里的 `connect / calibrate / move / debug / reset` 终于不再只是散落 prompt

### 5. Workspace Asset Contract

当前已经有：

- export convention
- schema version
- migration policy
- lint / inspect
- duplicate detection
- provenance metadata

当前意义：

- agent 生成的 setup 文件可以被 catalog 稳定读回

## 当前仍然保留的风险

下面这些是现在仍然明确存在的风险，但不是这轮文档要掩盖的：

### 1. procedure 已经 typed，但 orchestrator executor 还没有完全落下去

也就是：

- `ProcedureActionRef` 已经不再是裸字符串
- 但 orchestrator action 还需要后续绑定真实执行入口

### 2. adapter 已经 typed，但还没有真实本体验证

也就是：

- `EmbodiedAdapter` 已经不再返回 `dict[str, Any]`
- 但这些 result models 还需要被真实本体 adapter 消费和验证

### 3. 现在的 built-in robot / sensor 还很少

所以当前框架更像“接入底座”，还不是“已经铺满大量本体的平台”。

## 文档关系

当前推荐这样理解文档层级：

- [README.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/README.md)
  对外介绍项目方向、当前目标、核心路径
- [embodiment-ros2-architecture.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/embodiment-ros2-architecture.md)
  介绍当前具身框架是什么、为什么这样分层、第一版图怎么跑
- [roboclaw-installation-guide.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/roboclaw-installation-guide.md)
  面向第一次实际跑 RoboClaw 的 checklist
- [roboclaw/embodied/README.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/roboclaw/embodied/README.md)
  介绍代码目录边界
- [roboclaw/templates/EMBODIED.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/roboclaw/templates/EMBODIED.md)
  告诉 agent workspace-first 的总规则
- [roboclaw/templates/embodied/README.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/roboclaw/templates/embodied/README.md)
  告诉 agent workspace 目录里各个子目录的用途
- [roboclaw/skills/embodied-setup/SKILL.md](/Users/elvin/Workspace/Project/embodied_ai/claw/RoboClaw/roboclaw/skills/embodied-setup/SKILL.md)
  告诉 agent 写具身资产时的行为约束

## 下一步最重要的事情

如果接下来不是继续写文档，而是继续推代码，最优先的是：

1. 用一个真实本体把当前 framework 跑通
2. 把 orchestrator action 真正绑定到 runtime executor
3. 让 agent 按 workspace-first 流程真正生成一套 setup
4. 用第一次实跑 checklist 检查哪里还会卡住
