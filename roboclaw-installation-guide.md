# RoboClaw Installation Guide

## 1. Prerequisites

假设你是从零开始：

```bash
git clone https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
```

## 2. Step 1: Install RoboClaw

执行：

```bash
pip install -e ".[dev]"
```

安装成功后，`roboclaw` 命令应该已经可用，可以直接检查：

```bash
roboclaw --help
```

预期结果：

- 能看到 `onboard`、`status`、`agent`、`provider` 等命令

## 3. Step 2: Initialize RoboClaw

执行：

```bash
roboclaw onboard
```

这一步应该创建 `~/.roboclaw/config.json`、`~/.roboclaw/workspace/` 和具身相关 scaffold，可以直接检查：

```bash
find ~/.roboclaw -maxdepth 4 -type f | sort
```

至少应该看到这些文件：

```text
~/.roboclaw/config.json
~/.roboclaw/workspace/AGENTS.md
~/.roboclaw/workspace/EMBODIED.md
~/.roboclaw/workspace/HEARTBEAT.md
~/.roboclaw/workspace/SOUL.md
~/.roboclaw/workspace/TOOLS.md
~/.roboclaw/workspace/USER.md
~/.roboclaw/workspace/memory/HISTORY.md
~/.roboclaw/workspace/memory/MEMORY.md
~/.roboclaw/workspace/embodied/README.md
~/.roboclaw/workspace/embodied/intake/README.md
~/.roboclaw/workspace/embodied/robots/README.md
~/.roboclaw/workspace/embodied/sensors/README.md
```

## 4. Step 3: Verify Status Output

执行：

```bash
roboclaw status
```

你要验证：

- `Config` 显示为 `✓`
- `Workspace` 显示为 `✓`
- 当前 `Model` 显示正常
- provider 状态和你机器上的真实情况一致

## 5. Step 4: Configure the Model Provider

在验证 `roboclaw agent` 之前，先把模型 provider 配好。

先执行：

```bash
roboclaw status
```

看当前机器上哪些 provider 已经可用。

常见情况有两类：

### 5.1 OAuth provider

如果你用的是 OAuth provider，可以直接登录。

目前代码里已经实现了这两种：

```bash
roboclaw provider login openai-codex
roboclaw provider login github-copilot
```

### 5.2 API key provider

如果你用的是 API key provider，就直接编辑：

```bash
~/.roboclaw/config.json
```

把对应 provider 的 key 和默认 model 配好。

常见的 API key provider 包括：

- `openai`
- `anthropic`
- `openrouter`
- `deepseek`
- `gemini`
- `zhipu`
- `dashscope`
- `moonshot`
- `minimax`
- `aihubmix`
- `siliconflow`
- `volcengine`
- `azureOpenai`
- `custom`
- `vllm`

配好以后，再执行一次：

```bash
roboclaw status
```

你要验证：

- 当前 `Model` 显示正确
- 你要用的 provider 已经不是 `not set`

## 6. Step 5: Verify the Model Path

现在执行一条最简单的消息，确认 RoboClaw 已经能正常对话：

```bash
roboclaw agent -m "hello"
```

你要检查：

- agent 能启动
- agent 能正常返回内容
- 如果失败，错误能明确指向模型配置、provider、网络或权限问题

## 7. Step 6: Let RoboClaw Start the Robot Setup Flow

确认基础对话正常后，就可以开始让 RoboClaw 带你做机器人接入。

然后直接用自然语言告诉 RoboClaw 你的目标。

如果你接的是实机，可以这样说：

```bash
roboclaw agent -m "我想接入一台真实机器人，请一步一步带我完成配置。"
```

如果你已经知道是机械臂，也可以这样说：

```bash
roboclaw agent -m "我想接入一台真实机械臂，请告诉我需要准备哪些信息，并一步一步带我完成配置。"
```

如果你接的是仿真环境，可以这样说：

```bash
roboclaw agent -m "我想接入一个机器人仿真环境，请一步一步带我完成配置。"
```

这一步你要检查：

- RoboClaw 能理解你是在做第一次机器人接入
- RoboClaw 会主动问你缺少的信息
- RoboClaw 的提问是面向普通用户的
- RoboClaw 不要求你先理解内部代码结构

如果它开始引导你填写设备信息、连接方式、传感器信息或运行环境，就说明这条流程开始工作了。

继续这轮对话后，可以再检查一次：

```bash
find ~/.roboclaw/workspace/embodied -maxdepth 3 -type f | sort
git status --short
```

你要检查：

- `~/.roboclaw/workspace/embodied/` 下面开始出现新的文件
- RoboClaw 没有把这些接入内容直接写回仓库源码

理想状态是：

- 用户只描述自己的目标
- RoboClaw 自己遵守 framework / workspace 的边界

## 8. Step 7: Verify That Embodied Assets Are Organized Correctly

你不需要一次性生成所有资产，但至少要验证路径语义是对的。

重点检查这些目录是否被正确使用：

```text
~/.roboclaw/workspace/embodied/intake/
~/.roboclaw/workspace/embodied/robots/
~/.roboclaw/workspace/embodied/sensors/
~/.roboclaw/workspace/embodied/assemblies/
~/.roboclaw/workspace/embodied/deployments/
~/.roboclaw/workspace/embodied/adapters/
~/.roboclaw/workspace/embodied/simulators/
```

你要验证：

- intake 信息先进入 `intake/`
- robot/sensor/setup 资产被写入语义正确的目录
- 目录结构没有混乱到看不出边界

这一步的目标不是验证“内容已经完美”，而是验证“这条路径是可维护、可继续扩展的”。

## 9. Step 8: If You Have a Real Robot or Simulator, Test the Embodied Flow

只有在你已经具备真实本体或仿真环境时，才进入这一段。

如果这一步里 RoboClaw 判断本机还没有 ROS2，不要让它临场自由发挥安装教程。  
应当让它读取并遵循仓库里的这份文档：

```text
roboclaw/templates/embodied/guides/ROS2_INSTALL.md
```

目标是：

- 优先走受支持的平台安装路径
- 优先走 Ubuntu binary install，而不是一上来就 source build
- 把安装结果和 ROS2 distro 记录进 intake / workspace
- 安装完成后再继续 deployment / adapter 生成

这里开始验证第一个版图的核心目标：

- 连接
- 校准
- 移动
- debug
- 复位

### 9.1 Connect

让 RoboClaw 帮你进入连接流程，例如：

```bash
roboclaw agent -m "Connect my robot and tell me what information is still missing."
```

你要验证：

- RoboClaw 能识别当前是 `real` 还是 `sim`
- RoboClaw 能识别本体类型
- 如果信息不完整，它会先补问，而不是假设
- 如果失败，失败原因可读

### 9.2 Calibrate

```bash
roboclaw agent -m "Calibrate this robot if calibration is supported. If not, explain why."
```

你要验证：

- RoboClaw 能区分“支持 calibration”和“不支持 calibration”
- 不支持时不会瞎编流程

### 9.3 Move

```bash
roboclaw agent -m "Do one minimal safe movement for verification."
```

你要验证：

- RoboClaw 会优先选择最小安全动作
- 它能清楚说明动作意图
- 失败时能说清是 setup、ROS2、adapter 还是安全限制问题

### 9.4 Debug

```bash
roboclaw agent -m "Debug the current setup and summarize the most likely blocking issue."
```

你要验证：

- RoboClaw 能输出可读的 debug 结果
- debug 不是泛泛而谈，而是能定位到具体层

### 9.5 Reset

```bash
roboclaw agent -m "Reset the robot to a known safe state."
```

你要验证：

- RoboClaw 会优先考虑安全状态
- reset 的结果或失败位置是清楚的

## 10. What to Record During Validation

每次验证都建议记录这几类信息：

- 当前命令
- 当前本体类型
- 当前是 `real` 还是 `sim`
- 当前 provider / model 状态
- 当前生成了哪些 workspace 文件
- 当前失败点是在安装、初始化、workspace、agent、ROS2、adapter 还是具体本体流程

## 11. Final Pass Criteria

当下面这些都成立时，才可以说这次 PR 的核心功能基本可用：

- [ ] `pip install -e ".[dev]"` 成功
- [ ] `roboclaw onboard` 成功
- [ ] `roboclaw status` 成功
- [ ] `roboclaw agent -m "hello"` 成功
- [ ] RoboClaw 能把具身 setup 写到 `~/.roboclaw/workspace/embodied/`
- [ ] RoboClaw 没有直接污染 framework 源码
- [ ] 如果有真实本体或仿真，RoboClaw 至少能进入 `connect` 流程并给出合理反馈

如果前四项都成立，但后面不成立，说明 RoboClaw 基础启动链路是通的，但具身入口链路还不够强。

如果前四项都不稳定，这个 PR 还不能作为对外展示的首跑流程。
