#!/usr/bin/env python3
"""推敲比較点が履歴構造から取れていることを検証する。

表面的なラベルや件数ではなく、次を全 manifest の edit/* について確認する。
1. resolve_pre_merge_pair が返す base は edit の祖先
2. マージ済みなら base == merge-base(p1, p2)（p1 ではない）
3. 未マージなら base == merge-base(mainline, edit)
4. assert_structural_edit_pair が通る
5. 既知の逆向き（p1..p2 で main だけの推敲が minus）が fork..edit に出ない

失敗したら exit 1。再採掘前に必ず通す。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from git_pre_merge import (  # noqa: E402
  assert_structural_edit_pair,
  detect_mainline,
  resolve_pre_merge_pair,
)


def list_edit_refs(repo: Path) -> list[str]:
  out = subprocess.check_output(["git", "-C", str(repo), "branch", "-a"], text=True)
  names: set[str] = set()
  for line in out.splitlines():
    name = line.strip().lstrip("* ").strip()
    if name.startswith("remotes/origin/"):
      name = name[len("remotes/origin/") :]
    if name.startswith("edit/"):
      names.add(name)
  return sorted(names)


def resolve_ref(repo: Path, edit_name: str) -> str | None:
  for cand in (edit_name, f"origin/{edit_name}"):
    proc = subprocess.run(
      ["git", "-C", str(repo), "rev-parse", "--verify", cand],
      capture_output=True,
      text=True,
    )
    if proc.returncode == 0:
      return cand
  return None


def verify_fb9_fixture(repo: Path) -> None:
  """compute-programming の既知逆向きが fork..edit に出ないこと。"""
  main = detect_mainline(repo)
  if not main:
    raise SystemExit("fb9 fixture: no mainline")
  got = resolve_pre_merge_pair(repo, main, "edit/sec2")
  if not got:
    print("WARN: edit/sec2 unresolved; skip fb9 fixture")
    return
  base, edit, meta = got
  if meta.get("method") != "merge_commit":
    print("WARN: edit/sec2 not merge_commit; skip fb9 fixture")
    return
  p1 = meta["merge_parent1"]
  path = "article/article.md"
  polished = "もう少し一般的に拡大，応用できます"
  draftier = "もう少し一般的に拡大・応用することができます"
  diff_p1 = subprocess.check_output(
    ["git", "-C", str(repo), "diff", "--unified=0", f"{p1}..{edit}", "--", path],
    text=True,
    errors="replace",
  )
  diff_fork = subprocess.check_output(
    ["git", "-C", str(repo), "diff", "--unified=0", f"{base}..{edit}", "--", path],
    text=True,
    errors="replace",
  )

  def sides(diff: str) -> tuple[str, str]:
    minus = "\n".join(
      ln[1:] for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---")
    )
    plus = "\n".join(
      ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")
    )
    return minus, plus

  m1, p1s = sides(diff_p1)
  mf, pf = sides(diff_fork)
  if not (polished in m1 and draftier in p1s):
    raise SystemExit("fb9 fixture broken: p1..p2 no longer shows classic inversion")
  if polished in mf and draftier in pf:
    raise SystemExit("FAIL: fork..edit still contains classic p1..p2 inversion")
  if meta["fork_sha"] == p1:
    raise SystemExit("FAIL: fork_sha equals p1 for sec2 (expected diverge)")
  print(
    f"OK fb9 fixture: p1..p2 inverts; fork..edit does not "
    f"(fork={base[:10]} p1={p1[:10]} edit={edit[:10]})"
  )


def verify_late_unmerged_rejected(repo: Path) -> None:
  """推敲済み main から生えた未マージ tip は掘らないこと（pfvm）。"""
  main = detect_mainline(repo)
  if not main:
    raise SystemExit("late fixture: no mainline")
  tip = "origin/edit/kernelfusion-memorysave"
  got = resolve_pre_merge_pair(repo, main, tip)
  if got is not None:
    base, edit, meta = got
    raise SystemExit(
      f"FAIL: late unmerged tip resolved "
      f"(method={meta.get('method')} base={base[:10]} edit={edit[:10]})"
    )
  print("OK late-unmerged rejected: edit/kernelfusion-memorysave")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--manifest",
    action="append",
    default=[],
    help="manifest JSON (repeatable). default: section + ww + extra",
  )
  parser.add_argument(
    "--raw",
    action="append",
    default=[],
    help="mined JSONL to check source_reference base/edit (repeatable)",
  )
  args = parser.parse_args()
  root = Path(__file__).resolve().parents[1]
  manifests = [Path(p) for p in args.manifest] or [
    root / "data" / "section_mining_manifest.json",
    root / "data" / "section_mining_manifest.wwtawwta.json",
    root / "data" / "section_mining_manifest.extra.json",
  ]

  n_ok = 0
  n_none = 0
  n_fail = 0
  seen_repos: set[str] = set()

  for mp in manifests:
    if not mp.is_file():
      print(f"skip missing manifest: {mp}")
      continue
    for entry in json.loads(mp.read_text(encoding="utf-8")):
      repo_s = entry.get("repo")
      pid = entry["project_id"]
      if not repo_s or not Path(repo_s).is_dir():
        print(f"FAIL missing repo: {pid} {repo_s}")
        n_fail += 1
        continue
      repo = Path(repo_s)
      if str(repo) in seen_repos:
        continue
      seen_repos.add(str(repo))
      main = detect_mainline(repo)
      if not main:
        print(f"FAIL no mainline: {pid}")
        n_fail += 1
        continue
      for edit_name in list_edit_refs(repo):
        ref = resolve_ref(repo, edit_name)
        if not ref:
          continue
        try:
          got = resolve_pre_merge_pair(repo, main, ref)
        except ValueError as exc:
          print(f"FAIL resolve raised {pid} {edit_name}: {exc}")
          n_fail += 1
          continue
        if not got:
          n_none += 1
          continue
        base, edit, meta = got
        try:
          assert_structural_edit_pair(repo, base, edit)
        except ValueError as exc:
          print(f"FAIL structural {pid} {edit_name}: {exc}")
          n_fail += 1
          continue
        if meta.get("method") == "merge_commit":
          if base != meta.get("fork_sha"):
            print(f"FAIL base!=fork_sha {pid} {edit_name}")
            n_fail += 1
            continue
          if base == meta.get("merge_parent1") and meta.get("merge_parent1") != meta.get(
            "fork_sha"
          ):
            print(f"FAIL base==p1!=fork {pid} {edit_name}")
            n_fail += 1
            continue
        n_ok += 1

  # known inversion fixture
  cp = None
  for mp in manifests:
    if not mp.is_file():
      continue
    for entry in json.loads(mp.read_text(encoding="utf-8")):
      if entry.get("project_id") == "compute-programming":
        cp = Path(entry["repo"])
        break
    if cp:
      break
  if cp and cp.is_dir():
    try:
      verify_fb9_fixture(cp)
    except SystemExit as exc:
      print(exc)
      n_fail += 1

  pfvm = None
  for mp in manifests:
    if not mp.is_file():
      continue
    for entry in json.loads(mp.read_text(encoding="utf-8")):
      if entry.get("project_id") == "pfvm":
        pfvm = Path(entry["repo"])
        break
    if pfvm:
      break
  if pfvm and pfvm.is_dir():
    try:
      verify_late_unmerged_rejected(pfvm)
    except SystemExit as exc:
      print(exc)
      n_fail += 1

  # mined raw: source_reference の base/edit が構造不変条件を満たすこと
  import re

  ref_re = re.compile(r":([0-9a-f]{7,40})@[0-9a-f]+->([0-9a-f]{7,40})@")
  repos_by_pid: dict[str, Path] = {}
  for mp in manifests:
    if not mp.is_file():
      continue
    for entry in json.loads(mp.read_text(encoding="utf-8")):
      if entry.get("repo"):
        repos_by_pid[entry["project_id"]] = Path(entry["repo"])

  for raw_s in args.raw:
    raw_path = Path(raw_s)
    if not raw_path.is_file():
      print(f"FAIL missing raw: {raw_path}")
      n_fail += 1
      continue
    seen_pairs: set[tuple[str, str, str]] = set()
    raw_ok = 0
    raw_fail = 0
    for line in raw_path.open(encoding="utf-8"):
      if not line.strip():
        continue
      obj = json.loads(line)
      m = ref_re.search(obj.get("source_reference") or "")
      if not m:
        print(f"FAIL raw bad ref: {raw_path.name} id={obj.get('id')}")
        raw_fail += 1
        n_fail += 1
        continue
      base, edit = m.group(1), m.group(2)
      pid = str(obj.get("project_id") or "")
      key = (pid, base, edit)
      if key in seen_pairs:
        continue
      seen_pairs.add(key)
      repo = repos_by_pid.get(pid)
      if not repo or not repo.is_dir():
        print(f"FAIL raw unknown repo: {pid}")
        raw_fail += 1
        n_fail += 1
        continue
      try:
        assert_structural_edit_pair(repo, base, edit)
        raw_ok += 1
      except ValueError as exc:
        print(f"FAIL raw structural {raw_path.name} {pid} {base[:10]}..{edit[:10]}: {exc}")
        raw_fail += 1
        n_fail += 1
    print(f"raw {raw_path.name}: unique_pairs_ok={raw_ok} fail={raw_fail}")

  print(f"resolved_ok={n_ok} unresolved={n_none} fail={n_fail} repos={len(seen_repos)}")
  if n_fail:
    raise SystemExit(1)
  print("VERIFY OK")


if __name__ == "__main__":
  main()
