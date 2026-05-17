#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BACKEND_PORT="${BACKEND_PORT:-8008}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
VENV_DIR="$ROOT_DIR/backend/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

info() {
  printf '\033[1;34m[info]\033[0m %s\n' "$1"
}

warn() {
  printf '\033[1;33m[warn]\033[0m %s\n' "$1"
}

fail() {
  printf '\033[1;31m[error]\033[0m %s\n' "$1" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing command: $1"
}

port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1
  fi
}

ensure_node() {
  need_command node
  need_command npm
  local node_major
  node_major="$(node -p "Number(process.versions.node.split('.')[0])")"
  if [ "$node_major" -lt 20 ]; then
    fail "Node.js >= 20 is required. Current: $(node -v)"
  fi
  info "Node $(node -v), npm $(npm -v)"
}

ensure_python() {
  need_command "$PYTHON_BIN"
  "$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required")
print("Python", sys.version.split()[0])
PY
}

ensure_frontend_deps() {
  if [ ! -d "$ROOT_DIR/node_modules" ]; then
    info "Installing npm packages..."
    npm install
  else
    info "npm packages found"
  fi
}

ensure_backend_deps() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    info "Creating Python virtualenv..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  info "Checking Python packages..."
  if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import fastapi, httpx, pydantic, uvicorn
PY
  then
    info "Installing Python packages..."
    "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/backend/requirements.txt"
  else
    info "Python packages found"
  fi
}

check_ollama() {
  if ! command -v ollama >/dev/null 2>&1; then
    warn "Ollama command not found. App will still run, but AI review needs Ollama at http://127.0.0.1:11434."
    return
  fi

  if curl -sS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    info "Ollama is running"
    return
  fi

  warn "Ollama is installed but not responding. Start it separately with: ollama serve"
}

kill_port() {
  local port="$1"
  local pids
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -ti TCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  fi
  if [ -n "$pids" ]; then
    warn "Killing old process(es) on port ${port}: ${pids}"
    echo "$pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

check_ports() {
  if port_in_use "$BACKEND_PORT"; then
    kill_port "$BACKEND_PORT"
  fi
  if port_in_use "$FRONTEND_PORT"; then
    kill_port "$FRONTEND_PORT"
  fi
  # Also kill any leftover next-server processes for this directory
  local next_pids
  next_pids="$(pgrep -f "next dev.*$(basename "$ROOT_DIR")" 2>/dev/null || true)"
  if [ -n "$next_pids" ]; then
    warn "Killing leftover Next.js dev server(s): ${next_pids}"
    echo "$next_pids" | xargs kill -9 2>/dev/null || true
    sleep 1
  fi
}

run_checks() {
  info "Checking environment..."
  ensure_node
  ensure_python
  ensure_frontend_deps
  ensure_backend_deps
  check_ollama
  check_ports
}

start_app() {
  info "Starting AI-Powered Decision Making"
  info "Frontend: http://localhost:${FRONTEND_PORT}"
  info "Backend:  ${BACKEND_URL}"
  export DECISION_BACKEND_URL="$BACKEND_URL"
  export BACKEND_PORT
  npm run dev
}

run_checks
start_app
