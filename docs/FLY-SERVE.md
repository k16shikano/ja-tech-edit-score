# Fly.io への採点 Web 公開

CPU 上で三軸採点（`pref-sentseq-3e4` / `pref-bt` / 方向検出）を公開する手順。

## 前提

- [flyctl](https://fly.io/docs/flyctl/install/) が入っていること
- `fly auth login` 済みであること
- 次の成果物がローカルにあること
  - `outputs/pref-sentseq-3e4/`
  - `outputs/pref-bt/`
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
| `RATE_LIMIT_PER_IP` | 10 | IP あたりの採点回数上限 |
| `RATE_LIMIT_WINDOW_SECONDS` | 86400 | 上限の窓（秒） |
| `MAX_TEXT_CHARS` | 8000 | 下書き・推敲それぞれの最大文字数 |
| `MIN_MARGIN` / `GATE_MIN_MARGIN` | 3.7 / 0.0 | 合格ライン |

変更例:

```bash
fly secrets set RATE_LIMIT_PER_IP=5
# または fly.toml の [env] を編集して fly deploy
```

## マシン規模

既定は `shared-cpu-2x` + メモリ 2GB。起動時に埋め込みモデルと選好モデルを載せるので、1GB だと足りないことがある。コールドスタート後のヘルスチェック猶予は 120 秒。

## ローカルでのイメージ確認

```bash
docker build -f Dockerfile.serve -t ja-tech-edit-score:serve .
docker run --rm -p 8080:8080 ja-tech-edit-score:serve
curl -s http://127.0.0.1:8080/api/meta | head
```
