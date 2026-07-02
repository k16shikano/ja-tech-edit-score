#!/usr/bin/env python3
"""Client for ja-tech-edit-score daemon."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def home_dir() -> Path:
  return Path(os.environ.get("JA_TECH_EDIT_SCORE_HOME", Path.home() / "dev/ja-tech-edit-score")).expanduser()


def run_dir() -> Path:
  return home_dir() / "run"


def socket_path() -> Path:
  override = os.environ.get("JA_TECH_EDIT_SCORE_SOCKET", "").strip()
  if override:
    return Path(override)
  return run_dir() / "daemon.sock"


def pid_path() -> Path:
  return run_dir() / "daemon.pid"


def daemon_script() -> Path:
  return home_dir() / "scripts" / "pref_daemon.py"


def daemon_python() -> str:
  override = os.environ.get("JA_TECH_EDIT_SCORE_PYTHON", "").strip()
  if override:
    return override
  venv_py = home_dir() / ".venv" / "bin" / "python3"
  if venv_py.is_file():
    return str(venv_py)
  return sys.executable


def is_daemon_running() -> bool:
  sock = socket_path()
  if not sock.exists():
    return False
  try:
    ping()
    return True
  except OSError:
    return False


def ping(timeout: float = 0.5) -> None:
  resp = request({"cmd": "ping"}, timeout=timeout)
  if not resp.get("ok"):
    raise OSError(resp.get("error", "ping failed"))


def request(payload: dict, *, timeout: float = 30.0) -> dict:
  sock = socket_path()
  if not sock.exists():
    raise FileNotFoundError(f"daemon socket missing: {sock}")
  conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  conn.settimeout(timeout)
  with conn:
    conn.connect(str(sock))
    conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
    chunks: list[bytes] = []
    while True:
      data = conn.recv(65536)
      if not data:
        break
      chunks.append(data)
      if b"\n" in data:
        break
    line = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def start_daemon() -> None:
  run_dir().mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  env.setdefault("JA_TECH_EDIT_SCORE_HOME", str(home_dir()))
  subprocess.Popen(
    [daemon_python(), str(daemon_script())],
    cwd=str(home_dir()),
    env=env,
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
  )


def ensure_daemon(*, wait_seconds: float = 15.0) -> None:
  if os.environ.get("JA_TECH_EDIT_SCORE_NO_DAEMON", "").strip():
    raise FileNotFoundError("daemon disabled")
  try:
    ping(timeout=0.2)
    return
  except OSError:
    pass
  start_daemon()
  deadline = time.time() + wait_seconds
  while time.time() < deadline:
    time.sleep(0.15)
    try:
      ping(timeout=0.3)
      return
    except OSError:
      continue
  raise TimeoutError("pref daemon failed to start")


def check_via_daemon(params: dict) -> dict:
  ensure_daemon()
  req = {"cmd": "check", **params}
  resp = request(req)
  if not resp.get("ok"):
    raise RuntimeError(resp.get("error", "daemon check failed"))
  return resp["payload"]
