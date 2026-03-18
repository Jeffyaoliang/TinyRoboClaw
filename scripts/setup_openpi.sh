#!/usr/bin/env bash
# ============================================================
# OpenPI 部署脚本 — 从零搭建推理 + 训练环境
#
# 目标机器: 阿里云 4090 (推理) / GCP A100 (训练)
# 用法:
#   bash scripts/setup_openpi.sh install   # 安装 OpenPI
#   bash scripts/setup_openpi.sh serve     # 启动推理服务
#   bash scripts/setup_openpi.sh train     # 启动 LoRA 训练
#   bash scripts/setup_openpi.sh test      # 测试推理连通性
# ============================================================

set -euo pipefail

OPENPI_DIR="${OPENPI_DIR:-$HOME/openpi}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/.cache/openpi}"
SERVE_PORT="${SERVE_PORT:-8000}"

# 默认模型配置
MODEL_CONFIG="${MODEL_CONFIG:-pi0_fast_base}"
# 训练时使用的配置
TRAIN_CONFIG="${TRAIN_CONFIG:-pi0_fast_base}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[openpi]${NC} $*"; }
warn() { echo -e "${YELLOW}[openpi]${NC} $*"; }
err()  { echo -e "${RED}[openpi]${NC} $*" >&2; }

# ── 安装 ──────────────────────────────────────────────────

cmd_install() {
    log "=== 安装 OpenPI ==="

    # 1. 检查 NVIDIA GPU
    if ! command -v nvidia-smi &>/dev/null; then
        err "未检测到 nvidia-smi，请先安装 NVIDIA 驱动"
        exit 1
    fi
    log "GPU 信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

    # 2. 安装 uv (OpenPI 依赖的包管理器)
    if ! command -v uv &>/dev/null; then
        log "安装 uv 包管理器..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    log "uv 版本: $(uv --version)"

    # 3. 克隆 OpenPI
    if [ ! -d "$OPENPI_DIR" ]; then
        log "克隆 OpenPI 仓库..."
        git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
    else
        log "OpenPI 已存在: $OPENPI_DIR"
        cd "$OPENPI_DIR"
        git pull --recurse-submodules || true
    fi

    cd "$OPENPI_DIR"

    # 4. 安装依赖
    log "安装 Python 依赖 (uv sync)..."
    GIT_LFS_SKIP_SMUDGE=1 uv sync
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

    # 5. 验证安装
    log "验证安装..."
    uv run python -c "from openpi.training import config; print('OpenPI 安装成功')"

    log "=== 安装完成 ==="
    log "模型 checkpoint 将在首次使用时自动下载到 $CHECKPOINT_DIR"
}

# ── 启动推理服务 ──────────────────────────────────────────

cmd_serve() {
    local config="${1:-$MODEL_CONFIG}"
    local checkpoint="${2:-}"
    local port="${3:-$SERVE_PORT}"

    log "=== 启动 OpenPI 推理服务 ==="
    log "配置: $config"
    log "端口: $port"

    cd "$OPENPI_DIR"

    if [ -n "$checkpoint" ]; then
        # 使用自定义 checkpoint (训练后的 LoRA)
        log "Checkpoint: $checkpoint"
        uv run scripts/serve_policy.py \
            policy:checkpoint \
            --policy.config="$config" \
            --policy.dir="$checkpoint" \
            --port="$port"
    else
        # 使用 base model (自动从 GCS 下载)
        log "使用 base model (首次会自动下载)"
        uv run scripts/serve_policy.py \
            --policy.config="$config" \
            --port="$port"
    fi
}

# ── 后台启动推理服务 ──────────────────────────────────────

cmd_serve_bg() {
    local config="${1:-$MODEL_CONFIG}"
    local port="${2:-$SERVE_PORT}"
    local logfile="$OPENPI_DIR/serve.log"

    log "=== 后台启动 OpenPI 推理服务 ==="
    log "配置: $config, 端口: $port"
    log "日志: $logfile"

    cd "$OPENPI_DIR"

    nohup uv run scripts/serve_policy.py \
        --policy.config="$config" \
        --port="$port" \
        > "$logfile" 2>&1 &

    local pid=$!
    echo "$pid" > "$OPENPI_DIR/serve.pid"
    log "服务已启动, PID=$pid"
    log "查看日志: tail -f $logfile"
    log "停止服务: kill $pid"
}

# ── 停止推理服务 ──────────────────────────────────────────

cmd_stop() {
    local pidfile="$OPENPI_DIR/serve.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            log "已停止 OpenPI 服务 (PID=$pid)"
        else
            warn "进程 $pid 已不存在"
        fi
        rm -f "$pidfile"
    else
        warn "未找到 PID 文件，尝试查找进程..."
        pkill -f "serve_policy.py" && log "已停止" || warn "未找到运行中的服务"
    fi
}

# ── LoRA 训练 ─────────────────────────────────────────────

cmd_train() {
    local config="${1:-$TRAIN_CONFIG}"
    local data_dir="${2:?用法: $0 train <config> <data_dir> [exp_name]}"
    local exp_name="${3:-roboclaw_lora}"

    log "=== 启动 LoRA 训练 ==="
    log "配置: $config"
    log "数据: $data_dir"
    log "实验: $exp_name"

    cd "$OPENPI_DIR"

    # 1. 计算归一化统计量
    log "计算归一化统计量..."
    uv run scripts/compute_norm_stats.py --config-name="$config"

    # 2. 启动训练
    log "开始训练..."
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
        "$config" \
        --exp-name="$exp_name" \
        --overwrite

    log "=== 训练完成 ==="
    log "Checkpoints: $OPENPI_DIR/checkpoints/$config/$exp_name/"
}

# ── 测试推理连通性 ────────────────────────────────────────

cmd_test() {
    local host="${1:-localhost}"
    local port="${2:-$SERVE_PORT}"

    log "=== 测试 OpenPI 推理服务 ==="
    log "连接: ws://$host:$port"

    python3 -c "
import asyncio
import json
import base64
import numpy as np

async def test():
    try:
        import websockets
    except ImportError:
        print('安装 websockets: pip install websockets')
        return

    url = 'ws://$host:$port'
    print(f'连接 {url}...')

    try:
        async with websockets.connect(url, open_timeout=10) as ws:
            print('已连接!')

            # 发送测试 observation
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            img_b64 = base64.b64encode(dummy_image.tobytes()).decode()

            request = {
                'image': img_b64,
                'instruction': 'pick up the object',
                'joint_positions': [0.0] * 7,
            }

            print('发送测试请求...')
            await ws.send(json.dumps(request))

            print('等待响应...')
            response = await asyncio.wait_for(ws.recv(), timeout=30)
            data = json.loads(response)

            if 'actions' in data:
                actions = np.array(data['actions'])
                print(f'收到动作! shape={actions.shape}')
                print(f'动作范围: [{actions.min():.4f}, {actions.max():.4f}]')
                print('测试通过!')
            else:
                print(f'意外响应: {data}')

    except ConnectionRefusedError:
        print(f'连接被拒绝 — 确认服务是否在 {url} 运行')
    except asyncio.TimeoutError:
        print('响应超时 (30s)')
    except Exception as e:
        print(f'错误: {e}')

asyncio.run(test())
"
}

# ── 查看状态 ──────────────────────────────────────────────

cmd_status() {
    log "=== OpenPI 状态 ==="

    # 检查安装
    if [ -d "$OPENPI_DIR" ]; then
        log "安装目录: $OPENPI_DIR"
    else
        warn "未安装 (目录不存在: $OPENPI_DIR)"
    fi

    # 检查服务
    local pidfile="$OPENPI_DIR/serve.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            log "推理服务运行中 (PID=$pid)"
        else
            warn "推理服务已停止 (陈旧 PID=$pid)"
        fi
    else
        warn "推理服务未启动"
    fi

    # 检查 GPU
    if command -v nvidia-smi &>/dev/null; then
        log "GPU 使用:"
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    fi

    # 检查 checkpoints
    if [ -d "$CHECKPOINT_DIR" ]; then
        log "已下载的 checkpoints:"
        ls -d "$CHECKPOINT_DIR"/checkpoints/*/ 2>/dev/null || echo "  (无)"
    fi
}

# ── 主入口 ────────────────────────────────────────────────

case "${1:-help}" in
    install)    cmd_install ;;
    serve)      shift; cmd_serve "$@" ;;
    serve-bg)   shift; cmd_serve_bg "$@" ;;
    stop)       cmd_stop ;;
    train)      shift; cmd_train "$@" ;;
    test)       shift; cmd_test "$@" ;;
    status)     cmd_status ;;
    *)
        echo "用法: $0 <command> [args]"
        echo ""
        echo "命令:"
        echo "  install              安装 OpenPI (克隆 + 依赖)"
        echo "  serve [config] [ckpt] [port]  启动推理服务 (前台)"
        echo "  serve-bg [config] [port]      启动推理服务 (后台)"
        echo "  stop                 停止推理服务"
        echo "  train <config> <data_dir> [exp_name]  启动 LoRA 训练"
        echo "  test [host] [port]   测试推理连通性"
        echo "  status               查看状态"
        echo ""
        echo "环境变量:"
        echo "  OPENPI_DIR           OpenPI 安装目录 (默认: ~/openpi)"
        echo "  SERVE_PORT           推理服务端口 (默认: 8000)"
        echo "  MODEL_CONFIG         模型配置 (默认: pi0_fast_base)"
        echo ""
        echo "示例:"
        echo "  # 阿里云 4090 上部署推理"
        echo "  $0 install"
        echo "  $0 serve-bg pi0_fast_base"
        echo "  $0 test"
        echo ""
        echo "  # GCP A100 上训练"
        echo "  $0 train pi0_fast_base /path/to/data my_experiment"
        echo ""
        echo "  # 使用训练后的 checkpoint 推理"
        echo "  $0 serve pi0_fast_base checkpoints/pi0_fast_base/my_experiment/10000"
        ;;
esac
