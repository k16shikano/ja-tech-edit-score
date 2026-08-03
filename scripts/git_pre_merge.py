#!/usr/bin/env python3
"""edit ブランチと mainline の正しい比較点を求める。

運用（徹底）:
  推敲対では、対になる main 側コミットのほうが edit 側より「古い」。
  main 側のほうが新しい並び（進行した tip との逆差分）は推敲ではない。

現行 main は edit/... をマージしたあとも進んでいる。
`main..edit/foo` や、遅れたローカル main を正にした巨大 fork..edit は使わない。

正しい対は次のいずれか。
1. マージ済み: merge-base(第1親, 第2親)..第2親
   （第1親..第2親の two-dot は使わない。main 側だけの推敲が「逆差分」になる）
2. 未マージ: merge-base(mainline, edit)..edit
   ただし edit がすでに最新 mainline の祖先なら、マージ検出失敗として捨てる
     （遅れたローカル main で未マージ扱いしない）

mainline 参照は `detect_mainline` で origin/main を優先する。
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

# PR 件名の owner/branch を末尾まで取る。
# `[^ ]*?edit/` のような部分一致は reedit/arch を edit/arch と誤認する。
MERGE_SUBJECT_RE = re.compile(
  r"Merge (?:pull request #\d+ from (?:(?P<pr_owner>[^\s/]+)/)?(?P<pr_branch>[^\s']+)"
  r"|branch '(?P<local_branch>[^']+)')"
)


def git_output(repo: Path, *args: str) -> str:
  return subprocess.check_output(
    ["git", *args],
    cwd=repo,
    text=True,
    errors="replace",
  )


def git_ok(repo: Path, *args: str) -> bool:
  return (
    subprocess.run(
      ["git", *args],
      cwd=repo,
      capture_output=True,
      text=True,
    ).returncode
    == 0
  )


def normalize_branch_name(ref: str) -> str:
  name = ref.strip()
  if name.startswith("refs/heads/"):
    name = name[len("refs/heads/") :]
  if name.startswith("refs/remotes/origin/"):
    name = name[len("refs/remotes/origin/") :]
  if name.startswith("origin/"):
    name = name[len("origin/") :]
  return name


def branch_name_from_subject(subject: str) -> str | None:
  m = MERGE_SUBJECT_RE.search(subject)
  if not m:
    return None
  return m.group("pr_branch") or m.group("local_branch")


def is_primary_edit_branch(name: str | None) -> bool:
  """推敲前後として掘ってよいブランチ名か。

  reedit/*・*-reedit・*-summary は、すでに推敲が入った稿の再編集であり
  「下書き→推敲」の対にならない。
  """
  if not name:
    return False
  n = normalize_branch_name(name)
  if n.startswith("reedit/") or "/reedit/" in n:
    return False
  if "-reedit" in n or "_reedit" in n:
    return False
  if n.endswith("-summary") or n.endswith("_summary"):
    return False
  return n.startswith("edit/") or n.startswith("fix_")


def is_editorial_merge_subject(subject: str) -> bool:
  """main に載った推敲系マージか（件名から）。"""
  branch = branch_name_from_subject(subject)
  if branch and (
    normalize_branch_name(branch).startswith(("edit/", "reedit/", "fix_"))
  ):
    return True
  s = subject.replace("\\", "/")
  return any(
    p in s
    for p in (
      "/edit/",
      " edit/",
      "/reedit/",
      " reedit/",
      "/fix_",
      " fix_",
    )
  )


def fork_has_prior_editorial_merge(repo: Path, mainline: str, fork_sha: str) -> bool:
  """fork 時点の main に、別の推敲マージがすでに取り込まれているか。

  取り込まれたあとに生えた未マージ tip は「推敲済みからの再編集」になりうる。
  未マージ対としては採用しない（マージされたときだけ merge_commit 経路で掘る）。
  """
  fork_sha = git_output(repo, "rev-parse", fork_sha).strip()
  for merge, _p1, _p2, subject in list_merges(repo, mainline):
    if not is_editorial_merge_subject(subject):
      continue
    if is_ancestor(repo, merge, fork_sha):
      return True
  return False


def commit_committer_ts(repo: Path, sha: str) -> int:
  return int(git_output(repo, "log", "-1", "--format=%ct", sha).strip())


def is_ancestor(repo: Path, maybe_ancestor: str, maybe_descendant: str) -> bool:
  return git_ok(repo, "merge-base", "--is-ancestor", maybe_ancestor, maybe_descendant)


def has_diff(repo: Path, base: str, edit: str) -> bool:
  proc = subprocess.run(
    ["git", "diff", "--quiet", f"{base}..{edit}"],
    cwd=repo,
    capture_output=True,
  )
  # diff --quiet: 0=同一, 1=差分あり
  return proc.returncode == 1


def main_side_is_older(repo: Path, base: str, edit: str) -> bool:
  """対の main 側 (base) が edit 側より古いか。

  - edit が base の祖先 → base のほうが新しい並び（逆差分）→ False
  - base が edit の祖先 → fork 型。main 側が履歴上古い → True
  - どちらでもない → コミッター時刻で base < edit
  """
  if is_ancestor(repo, edit, base):
    return False
  if is_ancestor(repo, base, edit):
    return True
  return commit_committer_ts(repo, base) < commit_committer_ts(repo, edit)


def detect_mainline(repo: Path | str) -> str | None:
  """最新の mainline 参照。origin/* をローカルより優先する。"""
  repo = Path(repo)
  for ref in ("origin/main", "origin/master", "main", "master"):
    if git_ok(repo, "rev-parse", "--verify", ref):
      return ref
  return None


def list_merges(repo: Path, mainline: str) -> list[tuple[str, str, str, str]]:
  """(merge_hash, parent1, parent2, subject)"""
  out = git_output(
    repo, "log", "--merges", "--first-parent", "--format=%H%x00%P%x00%s", mainline, "--"
  )
  merges: list[tuple[str, str, str, str]] = []
  for line in out.splitlines():
    parts = line.split("\x00")
    if len(parts) != 3:
      continue
    commit, parents, subject = parts
    parent_list = parents.split()
    if len(parent_list) != 2:
      continue
    merges.append((commit, parent_list[0], parent_list[1], subject))
  return merges


def find_merges_for_branch(
  repo: Path, mainline: str, edit_branch: str
) -> list[tuple[str, str, str, str]]:
  """当該ブランチ名に対応するマージコミット一覧（新しい順）。

  件名から取り出したブランチ名との完全一致のみ。
  部分一致（edit/arch ⊂ reedit/arch）はしない。
  """
  want = normalize_branch_name(edit_branch)
  found: list[tuple[str, str, str, str]] = []
  for commit, p1, p2, subject in list_merges(repo, mainline):
    branch = branch_name_from_subject(subject)
    if branch and normalize_branch_name(branch) == want:
      found.append((commit, p1, p2, subject))
  return found


def assert_structural_edit_pair(repo: Path, base_sha: str, edit_sha: str) -> None:
  """履歴構造上、base..edit が推敲対の比較点として許されるか。

  表面的な「それっぽさ」ではなく次だけを見る。
  - base は edit の祖先（でなければ p1..p2 級の逆差分になりうる）
  - edit があるマージの第2親なら、base はそのマージの merge-base(p1,p2)
    （第1親 p1 を base にしてはならない）
  - 未マージなら base は merge-base(mainline, edit)
  """
  repo = Path(repo)
  base_sha = git_output(repo, "rev-parse", base_sha).strip()
  edit_sha = git_output(repo, "rev-parse", edit_sha).strip()
  if base_sha == edit_sha:
    raise ValueError(f"base and edit are the same commit: {base_sha[:12]}")
  if not is_ancestor(repo, base_sha, edit_sha):
    raise ValueError(
      "base must be an ancestor of edit "
      f"(base={base_sha[:12]} edit={edit_sha[:12]}). "
      "Rejects merge-parent1..edit when parent1 is not the fork."
    )

  # 汚染（向きの逆転）を止める核は「base が edit の祖先」だけ。
  # p1..edit かつ p1≠fork のとき p1 は edit の祖先でない → 上で拒否される。
  #
  # さらに、edit が mainline 上マージの第2親なら、正は fork=merge-base(p1,p2)。
  # p1 を base にする明示を拒否する（祖先でない場合は既に上で落ちるが、
  # 文言を残す）。
  mainline = detect_mainline(repo)
  if not mainline:
    return

  for _merge, p1, p2, _subject in list_merges(repo, mainline):
    if p2 != edit_sha:
      continue
    fork = git_output(repo, "merge-base", p1, p2).strip()
    if base_sha == p1 and p1 != fork:
      raise ValueError(
        "base is merge first-parent (main at merge), not fork "
        f"(p1={p1[:12]} fork={fork[:12]} edit={edit_sha[:12]}). "
        "Use merge-base(p1,p2)..p2 only."
      )
    if base_sha != fork:
      raise ValueError(
        "for a merged edit tip, base must be merge-base(p1,p2) "
        f"(fork={fork[:12]} got base={base_sha[:12]} edit={edit_sha[:12]})"
      )
    return

  # 未マージ（または mainline 以外の draft 参照: generated..edited 等）。
  # base が edit の祖先であること以上は、detect_mainline との MB 一致を強制しない。
  # （draft ブランチ tip が origin/main の祖先でないプロジェクトがある）
  return


def resolve_pre_merge_pair(
  repo: Path,
  mainline: str,
  edit_ref: str,
) -> tuple[str, str, dict] | None:
  """推敲の before/after コミットを返す。

  Returns:
    (base_sha, edit_sha, meta) または解決不能・非推敲なら None

  base / edit はブランチ名・「それっぽさ」ではなく履歴構造から決める。
  - マージ済み: base = merge-base(第1親, 第2親), edit = 第2親
  - 未マージ: base = merge-base(mainline, edit tip), edit = edit tip
    ただし fork 時点ですでに別の edit/reedit/fix マージが取り込まれていれば捨てる
    （推敲済み main から生えた再編集 tip を「下書き→推敲」にしない）
  reedit/* や *-summary は掘らない。
  「main 側のほうが古い」を満たさない対は返さない。
  返す対は必ず assert_structural_edit_pair を通る。
  """
  repo = Path(repo)
  edit_branch = normalize_branch_name(edit_ref)
  if not is_primary_edit_branch(edit_branch):
    return None

  mainline_sha = git_output(repo, "rev-parse", mainline).strip()
  try:
    edit_sha = git_output(repo, "rev-parse", edit_ref).strip()
  except subprocess.CalledProcessError:
    return None

  # 遅れたローカル main を渡されても、origin/main が先に進んでいればそちらを正とする
  fresh = detect_mainline(repo)
  if fresh:
    fresh_sha = git_output(repo, "rev-parse", fresh).strip()
    if fresh_sha != mainline_sha and is_ancestor(repo, mainline_sha, fresh_sha):
      mainline = fresh
      mainline_sha = fresh_sha

  common_meta = {
    "mainline": mainline,
    "edit_ref": edit_ref,
    "edit_branch": edit_branch,
    "mainline_tip": mainline_sha,
    "edit_ref_tip": edit_sha,
  }

  merges = find_merges_for_branch(repo, mainline, edit_branch)
  if merges:
    # base は merge-base(p1, p2)。第1親 p1 との two-dot は使わない。
    # p1 にだけある推敲が「削除」、未リベースの edit tip の旧稿が「追加」に見えるため。
    for merge_commit, p1, p2, subject in merges:
      try:
        fork = git_output(repo, "merge-base", p1, p2).strip()
      except subprocess.CalledProcessError:
        continue
      if not fork or fork == p2:
        continue
      if not main_side_is_older(repo, fork, p2):
        continue
      if not has_diff(repo, fork, p2):
        continue
      if is_ancestor(repo, p2, p1):
        continue
      assert_structural_edit_pair(repo, fork, p2)
      return (
        fork,
        p2,
        {
          **common_meta,
          "method": "merge_commit",
          "merge_commit": merge_commit,
          "merge_subject": subject,
          "merge_parent1": p1,
          "merge_parent2": p2,
          "fork_sha": fork,
          "main_at_merge": p1,
          "main_side_older": True,
          "age_witness": fork,
          "diff_range": "fork..edit",
        },
      )

  # 未マージ: edit がすでに最新 mainline の祖先なら、遅れた main 誤認を捨てる
  if is_ancestor(repo, edit_sha, mainline_sha):
    return None

  try:
    fork = git_output(repo, "merge-base", mainline_sha, edit_sha).strip()
  except subprocess.CalledProcessError:
    return None
  if not fork or fork == edit_sha:
    return None
  # 推敲済み main から後出しの未マージ tip は掘らない
  if fork_has_prior_editorial_merge(repo, mainline, fork):
    return None
  if not main_side_is_older(repo, fork, edit_sha):
    return None
  if not has_diff(repo, fork, edit_sha):
    return None
  assert_structural_edit_pair(repo, fork, edit_sha)
  return (
    fork,
    edit_sha,
    {
      **common_meta,
      "method": "merge_base_unmerged",
      "fork_sha": fork,
      "main_side_older": True,
      "age_witness": fork,
      "diff_range": "fork..edit",
    },
  )
