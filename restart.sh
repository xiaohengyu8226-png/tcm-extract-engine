

# tcm-extract-engine Web 服务重启脚本（后台运行，按进程组回收）
# 用法: ./restart.sh [port]      默认端口 8692
set -euo pipefail

cd "$(dirname "$0")"

PORT="${1:-8692}"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/web-serve.log"
PGID_FILE="$LOG_DIR/tcm-engine.pgid"
REPO_ROOT="$(pwd -P)"

# Web 服务可能惰性用到的可选依赖：web（FastAPI）、mineru（PDF 解析）、
# export（审核版 Excel 的 openpyxl）。按需裁剪：EXTRAS="web" ./restart.sh
EXTRAS="${EXTRAS:-web mineru export}"

mkdir -p "$LOG_DIR"

# ── 回收上一代服务 ───────────────────────────────────────────────────────────
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

echo "停止已有 tcm-engine 进程（端口 ${PORT}）..."

if [[ -f "${PGID_FILE}" ]]; then
  OLD_PGID=$(tr -d '[:space:]' < "${PGID_FILE}")
  if [[ -n "${OLD_PGID}" ]] && group_has_service "${OLD_PGID}"; then
    kill_group "${OLD_PGID}"
  fi
fi

# 兜底：pgid 记录缺失或过期时，退回按端口定位监听进程
OLD_PID=$(ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | head -1 || true)
if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
  PARENT_PID=$(ps -o ppid= -p "${OLD_PID}" 2>/dev/null | tr -d ' ' || true)
  kill "${OLD_PID}" 2>/dev/null || true
  [[ -n "${PARENT_PID:-}" ]] && kill "${PARENT_PID}" 2>/dev/null || true
  sleep 1
  kill -9 "${OLD_PID}" 2>/dev/null || true
  [[ -n "${PARENT_PID:-}" ]] && kill -9 "${PARENT_PID}" 2>/dev/null || true
fi

# 清扫遗留的 vLLM 引擎：ppid=1（父进程已死）且工作目录就是本仓库，才认定是自家
# 孤儿；别人容器里的引擎读不到 cwd 或指向别处，天然被排除
for pid in $(pgrep -f "VLLM::EngineCore" 2>/dev/null || true); do
  [[ "$(ps -o ppid= -p "${pid}" 2>/dev/null | tr -d ' ')" == "1" ]] || continue
  [[ "$(readlink -f "/proc/${pid}/cwd" 2>/dev/null)" == "${REPO_ROOT}" ]] || continue
  echo "  清扫遗留 vLLM 引擎: pid=${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  sleep 2
  kill -KILL "${pid}" 2>/dev/null || true
done

# ── 启动 ─────────────────────────────────────────────────────────────────────
# setsid 让服务自成会话与进程组，整棵进程树（uv → python → EngineCore 们）都能
# 被上面按组一次性收掉；不支持 setsid 的环境退回 nohup 直接后台。
# uv run 默认同步会剔除可选 extra，必须显式把所需 extra 全带上（否则 serve 缺
# uvicorn、PDF 解析缺 mineru 都会在运行期抛 ModuleNotFoundError）。
UV_EXTRA_ARGS=()
for _x in ${EXTRAS}; do UV_EXTRA_ARGS+=(--extra "${_x}"); done
if command -v setsid >/dev/null 2>&1; then
  setsid uv run "${UV_EXTRA_ARGS[@]}" tcm-engine serve --port "${PORT}" >> "${LOG_FILE}" 2>&1 &
else
  nohup uv run "${UV_EXTRA_ARGS[@]}" tcm-engine serve --port "${PORT}" >> "${LOG_FILE}" 2>&1 &
fi
NEW_PID=$!
sleep 1
NEW_PGID=$(ps -o pgid= -p "${NEW_PID}" 2>/dev/null | tr -d ' ' || true)
[[ -n "${NEW_PGID}" ]] && echo "${NEW_PGID}" > "${PGID_FILE}"

echo "tcm-engine Web 服务已后台启动"
echo "  地址: http://0.0.0.0:${PORT}"
echo "  extras: ${EXTRAS}"
echo "  日志: ${LOG_FILE}"
echo "  PID:  ${NEW_PID}"
echo "  PGID: ${NEW_PGID:-未知} → ${PGID_FILE}"
