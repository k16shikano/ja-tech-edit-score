#!/usr/bin/env bash
# SSH 切断後も学習を続けるため、setsid + nohup でジョブを起動する。
set -euo pipefail

ACTION=""
NAME=""
PID_FILE=""
LOG_FILE=""
WORKDIR=""
FORCE=0
ENV_VARS=()

usage() {
  cat <<'EOF'
usage:
  run_detached_job.sh start --name NAME --pid-file PATH --log-file PATH --workdir DIR [--force] [--env KEY=VAL]... -- COMMAND...
  run_detached_job.sh status --pid-file PATH [--log-file PATH]
  run_detached_job.sh stop --pid-file PATH
EOF
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo ""
    return
  fi
  tr -d '[:space:]' < "$PID_FILE"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    start|status|stop)
      ACTION="$1"
      shift
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --env)
      ENV_VARS+=("$2")
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$ACTION" ]]; then
  usage >&2
  exit 2
fi

case "$ACTION" in
  status)
    if [[ -z "$PID_FILE" ]]; then
      echo "status requires --pid-file" >&2
      exit 2
    fi
    pid="$(read_pid)"
    if pid_alive "$pid"; then
      echo "running name=${NAME:-?} pid=$pid log=${LOG_FILE:-?}"
      exit 0
    fi
    if [[ -n "$pid" ]]; then
      echo "stopped (stale pid $pid) log=${LOG_FILE:-?}"
    else
      echo "not running (no pid file) log=${LOG_FILE:-?}"
    fi
    exit 1
    ;;
  stop)
    if [[ -z "$PID_FILE" ]]; then
      echo "stop requires --pid-file" >&2
      exit 2
    fi
    pid="$(read_pid)"
    if ! pid_alive "$pid"; then
      echo "not running"
      rm -f "$PID_FILE"
      exit 0
    fi
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! pid_alive "$pid"; then
        rm -f "$PID_FILE"
        echo "stopped pid=$pid"
        exit 0
      fi
      sleep 1
    done
    echo "still running pid=$pid (send SIGKILL manually if needed)" >&2
    exit 1
    ;;
  start)
    if [[ -z "$NAME" || -z "$PID_FILE" || -z "$LOG_FILE" || -z "$WORKDIR" ]]; then
      echo "start requires --name, --pid-file, --log-file, --workdir" >&2
      exit 2
    fi
    if [[ $# -eq 0 ]]; then
      echo "start requires a command after --" >&2
      exit 2
    fi
    mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
    old_pid="$(read_pid)"
    if pid_alive "$old_pid" && [[ "$FORCE" -eq 0 ]]; then
      echo "already running name=$NAME pid=$old_pid log=$LOG_FILE" >&2
      exit 1
    fi
    runner="${PID_FILE%.pid}.sh"
    {
      echo "#!/usr/bin/env bash"
      echo "set -euo pipefail"
      printf 'cd %q\n' "$WORKDIR"
      for ev in "${ENV_VARS[@]}"; do
        printf 'export %q\n' "$ev"
      done
      printf 'exec'
      for arg in "$@"; do
        printf ' %q' "$arg"
      done
      echo
    } > "$runner"
    chmod +x "$runner"
    {
      echo "===== detached start $(date -Is) name=$NAME ====="
      echo "command: $*"
    } >> "$LOG_FILE"
    setsid nohup "$runner" >> "$LOG_FILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PID_FILE"
    echo "started name=$NAME pid=$pid log=$LOG_FILE"
    ;;
esac
