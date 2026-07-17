# 訓練データ活用ロードマップ

採掘済みの推敲ペアデータで何ができるかを整理する構想メモ。
[SFT-DPO.md](SFT-DPO.md) / [EDIT-MODEL.md](EDIT-MODEL.md) が生成 fine-tune（系統1）の検討、[ACTIVATION-STEERING.md](ACTIVATION-STEERING.md) が活性化ステアリング（系統3）の検討である。
本メモは評価モデルの強化、絶対スコア化、ループエンジニアリングを含む全体像を扱う。
評価・BT・ループの実装済み部分以外に、生成への埋め込み実験は計画段階である。

## 現状の資産

| 資産 | 内容 |
|------|------|
| `data/examples.raw.jsonl` | Git diff から採掘した推敲ペア（ローカル、非公開） |
| `data/dpo_curated.jsonl` | キュレーション済みペア 約 5,900 件 |
| `data/pref_dataset.jsonl` | swap 拡張済みの選好学習データ 約 11,800 行 |
| pref-static | 静的埋め込み（hotchpotch）+ ロジスティック回帰の選好評価モデル |
| `check` / `compare` | hunk 単位採点と 2 候補比較の CLI |

`check` は内部で「編集後 vs 修正前」のペア推論を行っており、固定の下書きを基準にした候補スコアリングは現行モデルでも可能である。

## 評価プロトコルの整備（すべての前提）

モデルの差し替えや大規模化の効果を判定するには、判断基準となる固定評価セットが要る。
これは以後のすべての実験の前提となる。

**held-out は測定のための分割であり、学習データを減らす話ではない。**
実際に配布・使用するモデルは、従来どおり全書籍のデータで学習する。
分割は「モデル構成 A と B のどちらが良いか」を測るときにだけ使う。

評価軸は 2 つ用意する。

| 評価軸 | 分割方法 | 測れるもの |
|--------|----------|-----------|
| in-domain | 現行の `base_id` ハッシュ分割 | 既知の書籍の、見ていない段落への性能 |
| cross-project | leave-one-project-out 交差検証 | 未知の書籍への汎化性能 |

現行の `base_id` 分割では、同じ章の隣り合う段落が train と eval に分かれて落ちる。
同じ著者・同じ章の文体という手がかりを eval にも与えるため、未知の書籍への性能を過大評価する。
cross-project 評価はこれを補う。
`project_id` ごとに 1 書籍を除外して学習し、除外した書籍で評価する。
これを全書籍分繰り返して平均すれば、特定の書籍の癖に結果が左右されない。

なお、1 書籍だけで学習したモデルの汎化が悪いという既知の観察は、書籍間の文体差が大きいことを示している。
これは cross-project 評価が必要である根拠でもある。

**実装済み**: `scripts/eval_pref_xproject.py`（`make eval-xproject`）。
`project_id` ごとに 1 書籍を held-out して残りで学習・評価し、fold 別と平均（micro / macro accuracy）を出力する。
レポートは `outputs/eval_xproject.json` に書く。

現行構成（hotchpotch 静的埋め込み + ロジスティック回帰）の初回計測（2026-07-14、13 書籍、11,774 行）:

| 指標 | in-domain（base_id 分割） | cross-project |
|------|--------------------------|---------------|
| accuracy | 0.972 | micro 0.959 / macro 0.953 |
| log_loss | 0.071 | 0.103 |

未知の書籍でも 1.3〜1.9 ポイント程度の低下にとどまり、現行構成の汎化は当初の懸念より良い。
書籍別では `raft-practical`（0.914）、`compute-whatis`（0.904）、`what-is-monad`（0.923）が低く、埋め込み差し替えの効果はこれらの fold で見るのがよい。

## 評価モデルの強化

段階的に進める。
各段階の採否は上記の評価プロトコルで判定する。

### 段階 1: 凍結埋め込みの差し替え（CPU で完結）

現在のアーキテクチャ（埋め込みは凍結、分類器だけ学習）のまま、埋め込みモデルだけを差し替えて比較する。

**計測済み**（leave-one-project-out、`max_seq_length=512`、適切なプレフィックス、`outputs/eval_xproject/summary.json`）:

| 埋め込み | micro acc | macro acc | log_loss |
|----------|-----------|-----------|----------|
| `intfloat/multilingual-e5-base`（`passage: `） | **0.981** | **0.975** | 0.057 |
| `cl-nagoya/ruri-v3-30m`（`文章: `） | 0.980 | 0.975 | **0.049** |
| `cl-nagoya/ruri-v3-70m`（`文章: `） | 0.965 | 0.959 | 0.089 |
| `hotchpotch/static-embedding-japanese`（現行） | 0.959 | 0.953 | 0.103 |
| `cl-nagoya/ruri-v3-130m`（`文章: `） | 0.954 | 0.945 | 0.126 |

所見:

- 文脈つき埋め込みへの差し替えだけで、cross-project 精度が約 2 ポイント上がる（e5-base / ruri-30m）。
- Ruri は大きいほど悪化した。凍結埋め込み + 線形分類器では次元や表現の細かさが必ずしも効かず、小さなモデルのほうが安定した。
- `pkshatech/GLuCoSE-base-ja-v2` は現行の transformers / sentence-transformers と tokenizer 非互換のため未計測。
- 再現は `scripts/run_embed_compare.sh` または `make eval-xproject EMBED_MODEL=... TRUNCATE_DIM=0 TEXT_PREFIX=...`。

段階 1 の結論: **`cl-nagoya/ruri-v3-30m` を採用**（軽量・log_loss 最良・e5-base とほぼ同等の精度）。
`make train` の既定埋め込みを ruri に切り替え済み（`TEXT_PREFIX=文章: `、`max_seq_length=512`）。
in-domain valid accuracy は 0.983（旧 hotchpotch 時 0.972）。

### 段階 2a: 凍結 ruri + Bradley-Terry 報酬（実装済み・CPU）

候補 1 件にスカラー \( s(\text{source}, \text{candidate}) \) を出す報酬ヘッドを、凍結 ruri 埋め込みの上に学習する。
損失は \( P(A \succ B) = \sigma(s(A) - s(B)) \)。

| コマンド | 内容 |
|----------|------|
| `make train-bt` | `outputs/pref-bt` に学習 |
| `make eval-bt-xproject` | 書籍単位 held-out |
| `make score-bt SOURCE=... CANDIDATE=...` | 絶対スコア推論 |

初回計測（ruri-v3-30m、2026-07-15）:

| 指標 | in-domain（valid） | cross-project |
|------|--------------------|---------------|
| pair accuracy | 0.985 | micro 0.975 / macro 0.965 |
| bt_loss | 0.064 | 0.073 |

これで閾値判定・Best-of-N・収束判定のループに直接使える絶対スコアが手に入った。
弱い fold は引き続き `compute-whatis`（0.859）。

### 段階 2b: cross-encoder の fine-tune（実装済み・計測待ち・GPU）

凍結埋め込みで頭打ちが見えたら、エンコーダ自体を学習する。

- `(source, candidate)` を連結して 1 つのスカラーを出す cross-encoder 構成
- ベース候補: `sbintuitions/modernbert-ja-130m` / `310m`（ruri と同系統）
- GPU 1 枚で数十分から数時間の規模。さくらの高火力 DOK（コンテナ実行、秒課金）が合う

**実装済み**: `scripts/train_pref_ce.py` / `scripts/eval_pref_ce_xproject.py` / `scripts/pref_ce_runtime.py`（`make train-ce` / `make eval-ce-xproject`）。
DOK 手順は [DOK-PREF-CE.md](DOK-PREF-CE.md)。
採否の **配線確認・相対比較** は LOPO micro pair accuracy（pref-bt は 0.975）。
全 fold で CE（modernbert-ja-130m）は micro **0.9995** まで到達した（甘い試験での勝ち）。

**難試験の結果（2026-07-17、bases_v1、20項目 × 6候補、人手全順位つき）**:

| 指標 | pref-bt | pref-ce |
|------|--------:|--------:|
| Top-1 一致 | 0.45 (9/20) | **0.50 (10/20)** |
| ペア一致率 | 0.830 (249/300) | **0.837 (251/300)** |
| human の中央順位 | 1 位 | **0.5 位** |
| スコア↔長さ Spearman | 0.50 | 0.57 |

- 両モデルとも `base`（無改変）は全 20 項目で最下位に置けた。
- Top-1 で判定が割れた 5 項目は CE 3 勝・BT 2 勝。似た LLM 推敲の中から人間編集を拾う場面で CE が上回る。
- 差は小さい（Top-1 +1 件、ペア +2 件）が、全集計指標で CE が上回り、LOPO の傾向とも一貫する。
- 留意: 今回のラベルは全項目で human が best（Top-1 は実質「human を当てる」試験）。また長さ相関は CE の方が高く、長文バイアスの監視は継続する。

**判定: pref-ce を選好評価モデルの本線として採用する。**
Top-1 0.50 は「似た候補の選抜」にはまだ不足なので、難試験の拡充と長さバイアス対策を並行する。

## ループエンジニアリング

「一定水準の推敲済み日本語を安定して得る」ためのループは、3 通り設計できる。
**実装済み**: `ja-tech-edit-score-rank`（Best-of-N）と `ja-tech-edit-score-converge`（収束判定）。
生成そのものは行わず、外で出した候補を採点・停止判定する。

### 閾値ループ（BT マージン）

`make rank ... MIN_MARGIN=τ` で、最良案でも source 自己スコアからの改善が τ 未満なら reject できる。
BT の生スコアはスケールが学習ごとに動くため、絶対閾値より **source との差分** を使う。

### Best-of-N（実装済み）

```bash
ja-tech-edit-score-rank \
  --source-file /tmp/source.txt \
  --candidate-file /tmp/a.txt \
  --candidate-file /tmp/b.txt \
  --candidate-file /tmp/c.txt \
  --format markdown

make rank SOURCE_FILE=/tmp/source.txt CANDIDATE_FILES="/tmp/a.txt /tmp/b.txt /tmp/c.txt" FORMAT=markdown
```

- BT モデル（`outputs/pref-bt`）で各候補の \( s(\text{source}, \text{cand}) \) を出し、最大を採用する
- 既定で下書き自身も候補に含め、改善なしを選べる
- `--min-margin` / `MIN_MARGIN` で改善幅の下限を課せる

### 収束判定（実装済み）

```bash
# pair: P(revised ≻ current) ≈ 0.5 なら収束（pref-static）
ja-tech-edit-score-converge \
  --mode pair \
  --current-file /tmp/cur.txt \
  --revised-file /tmp/rev.txt

# bt: s(source, revised) - s(source, current) が閾値未満なら収束
ja-tech-edit-score-converge \
  --mode bt \
  --source-file /tmp/source.txt \
  --current-file /tmp/cur.txt \
  --revised-file /tmp/rev.txt
```

「これ以上直しても良くならない = 推敲済み」の操作的定義。
推奨は `mode=pair`（確率差 ε、既定 0.08）。

## 要推敲検出器（採掘の拡張）

現在の hunk 採掘（`mine_branch_pair.py`）には構成情報が落ちる問題がある（下記「節単位採掘」参照）。
推敲ブランチで変更されなかった段落は「推敲不要と判定された文章」の正例であり、無償の学習データになる。
これを使えば「この段落は推敲が必要か」の二値分類器が作れる。

[WORKFLOW.md](WORKFLOW.md) では問題箇所の機械的検出を「行わないこと」としているが、この検出器はループの入口（どの段落を直すか）を自動化する。

## 節単位採掘（構成レベルの教師データ）

**問題（2026-07-17 判明）**: 既存の `examples.raw.jsonl`（6,055 キュレーション後ペア）は、
`mine_branch_pair.py` が `--unified=0` の hunk だけを取り、`clean_lines` で空行を除去するため、
**段落境界を含むペアが 0 件**。節・章レベルで推敲していても、学習データは文レベルの表層差分に潰されている。
パラグラフライティング（段落分割・統合・接続）の選好はモデルが一度も見ていない。

**対処（実装済み）**:

| 脚本 | 役割 |
|------|------|
| `scripts/markdown_sections.py` | Markdown を見出しパンくず単位に分割 |
| `scripts/mine_section_pairs.py` | base..edit の同見出し節をペア化（空行保持） |
| `scripts/batch_mine_sections.sh` | `data/section_mining_manifest.json` に基づく一括再採掘 |
| `scripts/analyze_section_pairs.py` | 段落境界・段落数変化の統計 |

再採掘（14 プロジェクト、`/home/k16/work/Nmonthly` / `Nspecial` 等）の初回結果:

| 指標 | hunk (`examples.raw.jsonl`) | 節 (`examples.section.raw.jsonl`) |
|------|----------------------------|-----------------------------------|
| ペア数 | 19,532 raw / 6,055 curated | **484**（現行ブランチ差分が残る分のみ） |
| 空行（段落境界）あり | **0%** | **100%** |
| 段落数が変化 | 0% | **50.8%** |

次: CE を再学習して LOPO / 難試験で構成選好が載るか測る。
長い節は `max_length=512` を超えうるため、トークン上限と切り詰め方針も要検討。

節ペアの pref 化（`make section-pref-data`）:

| 段階 | 件数 |
|------|------|
| section raw | 484 |
| DPO（accepted） | 249 |
| curated（max_chars=4000） | 219 |
| pref（swap 増強） | 438 |
| **マージ後 pref_dataset** | **12,548**（hunk 12,110 + section 438） |
| train / valid / test | 9,984 / 1,302 / 1,262 |

CE 再学習用イメージ: `make build-pref-ce-image` → `pref-ce:local`（6.3GB）。DOK push は `REGISTRY=... ./scripts/build_push_pref_ce_image.sh`。

**再学習と採用（2026-07-17）**: DOK `MODE=train` で節ペア込み CE を学習（in-domain valid pair accuracy 0.992）。
Hard Eval v1 では旧 CE と実質同等（Top-1 0.45 vs 0.50、ペア一致 0.840 vs 0.837）。
v1 は文レベルの推敲しか測っておらず、段落構成の識別力は試験に含まれていない。
文レベルで同等なら、構成レベルの信号を学習に含む新モデルが期待値で優位（旧 CE は段落境界を一度も見ていない）なので、**節ペア込み CE を本線に採用**した。

- 配置: `outputs/pref-ce-beyond-para/`（実体）、`outputs/pref-ce` はそこへの symlink
- 旧 CE は `outputs/pref-ce-hunk-only/` に退避（比較用）
- 構成識別力の実測は難試験 v2（節単位、3e）で行う

## 生成モデルへの埋め込み（系統1 / 系統3）

規範スキル付きフロンティアモデルへの依存を減らすため、同じ推敲ペアを生成側へ埋め込む経路を二系統で検討する。
kNN 実例注入（系統4）は過去に失敗しており、採用しない。

| 系統 | 文書 | 要旨 | 状態 |
|------|------|------|------|
| 1 編集モデル | [EDIT-MODEL.md](EDIT-MODEL.md) / [DOK-EDIT-SFT.md](DOK-EDIT-SFT.md) | SFT + on-policy DPO/RAFT。生産置き換えではなく選好測定→必要ならベンダー FT | フェーズ0済み・フェーズ1脚本あり |
| 3 activation steering | [ACTIVATION-STEERING.md](ACTIVATION-STEERING.md) | 対照対から方向ベクトルを読み、生成時に加算 | フェーズ A 計測済み（弱め）・縮小 |

推奨順: 系統1フェーズ1（QLoRA SFT）を優先。系統3のフェーズ B/C は読み取りが弱いため見送り。

生成品質だけではフロンティア＋規範スキル＋Best-of-N が当面の本線である。
系統1の問いは「言語化から漏れた選好が学習で載るか」であり、載る場合のみベンダー FT を検討する。
詳細は [EDIT-MODEL.md](EDIT-MODEL.md) の「本系統の位置づけ」。

### 生成モデルの fine-tune（編集モデル）

詳細は [EDIT-MODEL.md](EDIT-MODEL.md)（旧メモ [SFT-DPO.md](SFT-DPO.md) も参照）。

- 「日本語をゼロから生成するモデル」ではなく「下書きを推敲済みへ変換する編集モデル」として作る。
  手持ちデータの形が `(source_text, edited_text)` のペアであり、変換タスクの教師としてはそのまま使える。
  ゼロから生成する能力の教師としては約 5,900 件では足りない。
- 変換タスクなら内容はソースが与えるので、モデルは直し方だけを学べばよく、LoRA 程度の軽い fine-tune で成立しやすい。
- DPO の rejected を「修正前と同一」だけにすると信号が弱い。
  SFT 済みモデル自身に温度を上げて複数候補を出させ、報酬モデルで負けた候補を rejected に回すと、iterative DPO のループが回る。
- 規模感: 7B〜13B クラス（`llm-jp-3`、`sarashina2.2`、Qwen3 など）の QLoRA なら GPU 1 枚で数時間。

編集モデルができると、副産物として \( \log P(\text{candidate} \mid \text{source}) \) が「推敲済みらしさ」の連続スコアとして使える。
報酬モデルとは独立の信号なので、アンサンブルすると頑健になる。

## 外部 GPU とデータの扱い

学習データは非公開原稿由来である。
外部 GPU サービスに上げる前に、持ち出しの範囲と消去の手順を決めておく。
高火力 DOK はジョブ終了でコンテナが破棄されるため、比較的扱いやすい。

## 優先順位

| 順 | 項目 | 依存 | 計算資源 | 状態 |
|----|------|------|----------|------|
| 1 | 評価プロトコルの整備（cross-project 分割） | なし | CPU | 済み |
| 2 | 凍結埋め込みの差し替え比較 → ruri-v3-30m 採用 | 1 | CPU | 済み |
| 3a | BT 報酬モデル（凍結 ruri + 線形ヘッド） | 2 | CPU | 済み |
| 3b | BT 報酬の cross-encoder 化（ModernBERT-ja） | 3a | GPU | **採用**（難試験で BT を上回る。Top-1 0.50 / ペア 0.837） |
| 3c | 選抜難試験（LLM ベース＋人手） | 3a | CPU＋人手 | v1 実施済み（20項目）。拡充は継続（[HARD-EVAL.md](HARD-EVAL.md)） |
| 3d | CE の運用組み込み（rank / converge） | 3b | CPU | 済み（meta.json で BT/CE 自動判別。rank の既定は pref-ce） |
| 3e | 難試験 v2: 節（複数段落）単位の項目 | 3c | CPU＋人手 | 定義ラベル済み（human>deg-join>deg-split>base>deg-reverse）。採点比較へ |
| 3f | 節単位ペアの再採掘と CE 再学習 | 3b | CPU＋GPU | **済み・採用**。節ペア込み CE（`pref-ce-beyond-para`）を本線に。構成識別力の実測は 3e（難試験 v2）へ。次: `MAX_LENGTH=2048` 再学習 |
| 4 | Best-of-N と収束判定のループ実装 | 3a | CPU | 済み |
| 5 | 要推敲検出器の採掘拡張 | 1 | CPU | 未着手 |
| 6a | activation steering 読み取り（フェーズ A） | データ | GPU（短） | 計測済み（弱め）・B/C 見送り |
| 6b | 編集モデルの SFT（フェーズ0〜1） | 3a | GPU | 実施済み・BT 評価で規範スキルに敗北（勝率 0.269）・縮小 |

「推敲済み日本語の安定生成」という大目標に対しては、生成モデルより先に良い報酬モデル（絶対スコア）を持つことがボトルネック解消になる。
報酬モデルがあれば、生成側はフロンティア LLM + 規範スキルでも水準に届き、自前の fine-tune / steering はその後の最適化になる。
