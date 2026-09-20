#!/usr/bin/env python3
"""japanese-tech-writing の SKILL.md から YAML 先頭を外して生成用規範ファイルを書く。"""
from __future__ import annotations

import argparse
from pathlib import Path


def strip_frontmatter(text: str) -> str:
  if not text.startswith("---"):
    return text.strip() + "\n"
  parts = text.split("---", 2)
  if len(parts) < 3:
    return text.strip() + "\n"
  return parts[2].strip() + "\n"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--src",
    default=str(
      Path.home() / ".cursor" / "skills" / "japanese-tech-writing" / "SKILL.md"
    ),
  )
  parser.add_argument("--out", default="data/a1_probe/japanese-tech-writing.md")
  args = parser.parse_args()
  src = Path(args.src)
  if not src.is_file():
    raise SystemExit(f"missing {src}")
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(strip_frontmatter(src.read_text(encoding="utf-8")), encoding="utf-8")
  print(f"wrote {out} from {src}")


if __name__ == "__main__":
  main()
