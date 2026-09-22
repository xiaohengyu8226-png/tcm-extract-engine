#!/usr/bin/env bash
# tcm-extract-engine Web 服务停止脚本（按进程组回收 + 端口兜底 + 遗留 vLLM 引擎清扫）
# 用法: ./stop.sh [port]      默认端口 8692
# 与 restart.sh 的回收逻辑保持一致，只停不启。
set -euo pipefail

cd "$(dirname "$0")"

PORT="${1:-8692}"
LOG_DIR="logs"
PGID_FILE="$LOG_DIR/tcm-engine.pgid"
REPO_ROOT="$(pwd -P)"

# ── 回收服务 ─────────────────────────────────────────────────────────────────
# 逐步控制抽取的解析阶段会 in-process 调用 MinerU 的 do_parse；其 VLM/hybrid 后端
# 会派生 VLLM::EngineCore 子进程：它只听 ZMQ 随机端口、不是 ${PORT} 的监听者，且
# 命令行被改写看不出归属。只 kill 监听进程会让它被 init 收养，继续抱着十几 GiB
# 显存，下一次解析就 CUDA OOM。所以整组回收 + 兜底清扫遗留孤儿。

group_has_service() {
  # 组内必须真有 tcm-engine serve 才动手，防 pgid 被回收后复用误杀别人
  ps -eo pgid=,args= 2>/dev/null | awk -v g="$1" '$1 == g' | grep -q "[t]cm-engine serve"
}

kill_group() {
  kill -TERM -- "-$1" 2>/dev/null || true
  sleep 3
  kill -KILL -- "-$1" 2>/dev/null || true
}

STOPPED=0

echo "停止 tcm-engine 进程（端口 ${PORT}）..."

# 1) 按记录的进程组整组回收
if [[ -f "${PGID_FILE}" ]]; then
  OLD_PGID=$(tr -d '[:space:]' < "${PGID_FILE}")
  if [[ -n "${OLD_PGID}" ]] && group_has_service "${OLD_PGID}"; then
    echo "  回收进程组: pgid=${OLD_PGID}"
    kill_group "${OLD_PGID}"
    STOPPED=1
  fi
fi

# 2) 兜底：pgid 记录缺失或过期时，退回按端口定位监听进程
OLD_PID=$(ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
  echo "  按端口回收监听进程: pid=${OLD_PID}"
  PARENT_PID=$(ps -o ppid= -p "${OLD_PID}" 2>/dev/null | tr -d ' ' || true)
  kill "${OLD_PID}" 2>/dev/null || true
  [[ -n "${PARENT_PID:-}" ]] && kill "${PARENT_PID}" 2>/dev/null || true
  sleep 1
  kill -9 "${OLD_PID}" 2>/dev/null || true
  [[ -n "${PARENT_PID:-}" ]] && kill -9 "${PARENT_PID}" 2>/dev/null || true
  STOPPED=1
fi

# 3) 清扫遗留的 vLLM 引擎：ppid=1（父进程已死）且工作目录就是本仓库，才认定是自家
#    孤儿；别人容器里的引擎读不到 cwd 或指向别处，天然被排除
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
  [[ "$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')" == "1" ]] || continue
  [[ "$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" == "${REPO_ROOT}" ]] || continue
  echo "  清扫遗留 vLLM 引擎: pid=${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 2
  kill -KILL "${pid}" 2>/dev/null || true
  STOPPED=1
done

# 进程已回收，删除 pgid 记录避免下次误用
if [[ -f "${PGID_FILE}" ]]; then
  rm -f "${PGID_FILE}"
fi

if [[ "${STOPPED}" -eq 1 ]]; then
  echo "tcm-engine Web 服务已停止"
else
  echo "未发现运行中的 tcm-engine 服务（端口 ${PORT}）"
fi
