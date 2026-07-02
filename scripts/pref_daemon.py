#!/usr/bin/env python3
"""Keep pref model loaded; serve check requests over Unix socket."""

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from check_revision_core import build_check_payload, default_model_dir  # noqa: E402
from pref_runtime import load_pref_model  # noqa: E402


def home_dir() -> Path:
  return Path(os.environ.get("JA_TECH_EDIT_SCORE_HOME", Path.home() / "dev/ja-tech-edit-score")).expanduser()


def run_dir() -> Path:
  path = home_dir() / "run"
  path.mkdir(parents=True, exist_ok=True)
  return path


def socket_path() -> Path:
  override = os.environ.get("JA_TECH_EDIT_SCORE_SOCKET", "").strip()
  if override:
    return Path(override)
  return run_dir() / "daemon.sock"


def pid_path() -> Path:
  return run_dir() / "daemon.pid"


def handle_client(conn: socket.socket, loaded) -> None:
  with conn:
    chunks: list[bytes] = []
    while True:
      data = conn.recv(65536)
      if not data:
        return
      chunks.append(data)
      if b"\n" in data:
        break
    line = b"".join(chunks).split(b"\n", 1)[0]
    try:
      req = json.loads(line.decode("utf-8"))
      if req.get("cmd") == "ping":
        resp = {"ok": True, "cmd": "ping"}
      elif req.get("cmd") == "check":
        payload = build_check_payload(req, loaded)
        resp = {"ok": True, "cmd": "check", "payload": payload}
      else:
        resp = {"ok": False, "error": f"unknown cmd: {req.get('cmd')!r}"}
    except Exception as exc:  # noqa: BLE001
      resp = {"ok": False, "error": str(exc)}
    conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))


def main() -> None:
  model_dir = Path(os.environ.get("JA_TECH_EDIT_SCORE_MODEL", "") or default_model_dir())
  loaded = load_pref_model(model_dir)

  sock_file = socket_path()
  if sock_file.exists():
    sock_file.unlink()

  server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  server.bind(str(sock_file))
  server.listen(8)
  pid_path().write_text(str(os.getpid()), encoding="utf-8")

  stop = threading.Event()

  def _shutdown(*_args) -> None:
    stop.set()
    server.close()

  signal.signal(signal.SIGTERM, _shutdown)
  signal.signal(signal.SIGINT, _shutdown)

  try:
    while not stop.is_set():
      try:
        conn, _addr = server.accept()
      except OSError:
        break
      thread = threading.Thread(target=handle_client, args=(conn, loaded), daemon=True)
      thread.start()
  finally:
    if sock_file.exists():
      sock_file.unlink()
    if pid_path().exists():
      pid_path().unlink()


if __name__ == "__main__":
  main()
