# Cursor Agent Skills（配布物）

ja-tech-edit-score 向けの Cursor Skill を置く。
リポジトリを clone したあと、次のいずれかで有効化する。

## インストール（推奨）

```bash
cd ~/dev/ja-tech-edit-score   # clone 先
make install-skills      # ~/.cursor/skills/ に symlink
```

あわせて CLI も PATH に入れる場合:

```bash
make venv && make install-bin && make install-skills
```

## 手動インストール

```bash
mkdir -p ~/.cursor/skills
ln -sfn /path/to/ja-tech-edit-score/skills/ja-tech-edit-score-check ~/.cursor/skills/ja-tech-edit-score-check
```

またはコピー:

```bash
cp -r skills/ja-tech-edit-score-check ~/.cursor/skills/
```

## 同梱 Skill

| Skill | 用途 |
|-------|------|
| [ja-tech-edit-score-check](ja-tech-edit-score-check/SKILL.md) | 推敲中の Markdown を hunk 単位で選好採点 |

推敲文の生成規範には **japanese-tech-writing**（別リポジトリまたは各自の Skill）を使う。
ja-tech-edit-score の Skill は選好採点専用である。
