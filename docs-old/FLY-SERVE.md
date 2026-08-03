# Fly.io への採点 Web 公開

CPU 上で三軸採点を公開する手順。

- 主軸: `outputs/pref-sentseq-keep`（節 keep の文列）
- ゲート: `outputs/pref-bt-keep`（hunk keep の BT）
- 方向: `outputs/pref-sentseq-anchor-2stage-v2`

旧 `pref-sentseq-3e4` / `pref-bt` は使わない。

## 前提

- [flyctl](https://fly.io/docs/flyctl/install/) が入っていること
- `fly auth login` 済みであること
- 次の成果物がローカルにあること
  - `outputs/pref-sentseq-keep/`
  - `outputs/pref-bt-keep/`
  - `outputs/pref-sentseq-anchor-2stage-v2/`
  - `outputs/acceptance_margin_calibration.json`

## 初回

```bash
fly auth login
fly apps create ja-tech-edit-score   # 名前が埋まっていれば別名にして fly.toml の app も合わせる
fly deploy
```

`fly.toml` の `primary_region` は既定で `nrt`（東京）。別リージョンにするなら変更する。

## 更新

```bash
fly deploy
```

## 環境変数

| 変数 | 既定 | 意味 |
|------|------|------|
| `PRIMARY_MODEL` | `/app/outputs/pref-sentseq-keep` | 主軸 |
| `GATE_MODEL` | `/app/outputs/pref-bt-keep` | ゲート |
| `DIRECTION_MODEL` | `/app/outputs/pref-sentseq-anchor-2stage-v2` | 方向検出（`none` で無効） |
| `RATE_LIMIT_PER_IP` | 10 | IP あたりの採点回数上限 |
| `RATE_LIMIT_WINDOW_SECONDS` | 86400 | 上限の窓（秒） |
| `MAX_TEXT_CHARS` | 8000 | 下書き・推敲それぞれの最大文字数 |
| `MIN_MARGIN` / `GATE_MIN_MARGIN` | 5.5 / 0.0 | 合格ライン（主軸は節 keep valid の self p50） |

変更例:

```bash
fly secrets set RATE_LIMIT_PER_IP=5
# または fly.toml の [env] を編集して fly deploy
```

## マシン規模

既定は `shared-cpu-2x` + メモリ 2GB。起動時に埋め込みモデルと選好モデルを載せるので、1GB だと足りないことがある。

モデル読込はポート bind のあとに裏で行う（`/healthz` は読込前でも 200）。
読込完了までは `/api/score` が 503 を返す。コールドスタートの初回待ちを減らすため
`min_machines_running = 1` にしている（常時課金が増える）。止めてよければ 0 にする。

ヘルスチェックの猶予は 180 秒（`/healthz`）。

## ローカルでのイメージ確認

```bash
docker build -f Dockerfile.serve -t ja-tech-edit-score:serve .
docker run --rm -p 8080:8080 ja-tech-edit-score:serve
curl -s http://127.0.0.1:8080/api/meta | head
```
