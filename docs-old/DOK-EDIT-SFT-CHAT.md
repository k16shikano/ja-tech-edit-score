# DOK で編集モデルを対話試用する

手元に GPU が無いとき、高火力 DOK の **SSH** で素の Qwen / SFT 済み LoRA を対話試用する手順。

SSH は中でコマンドを打つためのもの。`scp` / `rsync` ではファイルを出せない（[公式](https://manual.sakura.ad.jp/cloud/koukaryoku-container/use-ssh.html)）。  
原稿は端末に表示されるだけなので、ログに残さない・外部に送らない。

## 対話の既定（検証用）

バッチ評価（`generate_edit_sft`）とは設定を分ける。

| 項目 | 対話（この手順） | バッチ評価 |
|------|------------------|------------|
| デコード | サンプリング（温度 0.7 / top_p 0.9） | 貪欲（`do_sample=False`） |
| 指示 | 推敲＋「本文のみ」（メタ禁止） | 学習時の短指示のみ |

同じ下書きを再度貼ると別案が出る。base と adapter に同じ指示・同じデコードを使う。

学習時短指示や貪欲デコードに揃えるときは `--train-prompt` / `--greedy`。

## 事前: 対話用イメージ

評価用イメージに対話脚本と LoRA を同梱する。
既定の adapter は採用ラン `outputs/Qwen__Qwen3-8B-revised-2000/adapter`（全 keep・EPOCHS=2）。

```bash
export REGISTRY=ja-tech-edit.sakuracr.jp
# 省略可（既定が Qwen__Qwen3-8B-revised-2000）
# export ADAPTER_SRC=outputs/Qwen__Qwen3-8B-revised-2000/adapter
./scripts/build_push_edit_sft_eval_image.sh
```

`edit-sft-eval:latest` には `/app/adapter`・`chat_edit_sft.py`・依存が入る。  
ベース重みは実行時に Hugging Face から取得する（初回は数分〜）。

## DOK タスク

| 項目 | 値 |
|------|-----|
| イメージ | `ja-tech-edit.sakuracr.jp/edit-sft-eval:latest` |
| GPU | V100 で可。厳しければ H100 |
| **SSH** | **ON** |
| コマンド / エントリーポイント | バッチを走らせない。例: `sleep infinity` |

バッチ用 `ENTRYPOINT`（`dok_edit_sft_eval.sh`）が付いたイメージなので、**対話のときはエントリーポイントを上書き**して `sleep infinity` などにする。

## SSH 後

```bash
cd /app
# 素の Qwen
PYTHONPATH=scripts python scripts/chat_edit_sft.py \
  --mode base --base-model Qwen/Qwen3-8B --load-in-4bit

# 別セッション（または終了後）で SFT 済み
PYTHONPATH=scripts python scripts/chat_edit_sft.py \
  --mode adapter --base-model Qwen/Qwen3-8B \
  --adapter /app/adapter --load-in-4bit
```

操作:

- 下書きを貼る → **空行だけの行**で生成開始
- 同じ下書きをもう一度貼ると別案が出る（サンプリング）
- `/quit` で終了
- `/clear` で入力バッファ消去

温度を変える例: `--temperature 0.9`。再現したいときだけ `--seed 0`。

## 注意

- 課金はコンテナ稼働中続く。試し終わったらタスクを止める
- 成果物ディレクトリに下書きを書き出さない
- いまの adapter は Qwen3-8B 用。別ベースを試すときは学習からやり直し
- 既存の DOK コンテナは古い脚本のまま。脚本変更後はイメージを再 push し、タスクを作り直す
