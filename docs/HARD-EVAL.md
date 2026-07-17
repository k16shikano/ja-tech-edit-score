# 選抜の難試験（Hard Eval）

人間推敲ペアの LOPO（「下書き vs 推敲後」）は構成比較には使えるが、実運用の Best-of-N より易しい。
似た LLM 候補を並べる力は、別試験で測る。

## なぜ held-out 下書きをそのまま使わないか

held-out 書籍の下書きを \(x\) にすると、表層は未知でも「人間の原稿差分」に近い分布のままになる。
採点器は学習時と同じ種類の対照を当てる試験になりやすい。

そこで **意味の種だけ** を既存段落から取り、表層は LLM で書き直した文をベース \(B\) にする。
候補も **\(B\) を直したもの** に揃える。採点は常に \(s(B, c_i)\)。

## プロトコル

1. **種（seed）**  
   held-out 書籍など、学習に使っていない段落から意味内容だけ取る。  
   採点入力には使わない（漏洩監視・再現用のメタとして残してよい）。

2. **ベース \(B\)**  
   LLM に「意味を保った別表層の日本語」を生成させる。  
   これが `base_text`（従来の rank における source）。

3. **候補 \(c_1,\ldots,c_N\)**  
   いずれも \(B\) の推敲案。
   - **複数のモデルによる推敲**（同一指示・モデルだけ替える）。  
     フロンティア（Composer / Fable / GPT など）、安価帯（Kimi / GMK など）、癖の強いモデル（Grok など）を混ぜる。  
     項目数が少ないので、1 base あたり 4〜6 案を目安に厚くする。
   - **人間による編集**（\(B\) を自分で推敲したもの）
   - \(B\) 自身（改善なし）

   規範スキルを付けた生成を「良い候補」と仮定して混ぜる設計は取らない。
   「規範プロンプト＝良い推敲」という循環を試験に持ち込まないため。

4. **人手ラベル**  
   最良 1 件（`best_id`）を必須。余裕があれば全順位（`rank`）。  
   人間編集の候補を自動的に最良と決めつけず、他候補と同じ土俵で並べて付ける。
   `bases_v1` では `bases_v1_candidates_preview.md` の各項目内にある候補サブセクションを良い順に並べ替える。
   候補見出しと本文は一緒に動かし、本文は編集しない。
   並べ替え後に `make hard-eval-label` を実行すると、先頭候補を `best_id`、全候補の順序を `rank` とした `bases_v1_labeled.jsonl` が作られる。

5. **モデル採点**  
   pref-bt / pref-ce などで \(s(B, c_i)\) を出し、人間との一致を測る。

## 指標

| 指標 | 定義 |
|------|------|
| Top-1 一致 | モデル最良案の id が `human.best_id` と一致 |
| ペア一致率 | 人間の順位から作った全ペア \(A \succ B\) のうち、モデルも \(s(A)>s(B)\) の割合（`rank` があるとき） |
| 長さ相関 | スコアと文字数の Spearman（短文化バイアス監視。参考） |

### 試験の解釈

この試験は Best-of-N の代理に限定しない。**編集者本人の選好との近さ**を測る。
編集者自身の編集（`human`）が best であることは定義上動かないので、v1 のように全項目で `human.best_id = "human"` になるのは偏りではなく前提である。
指標はそれぞれ次を測る。

- **Top-1 一致**: 採点器が選好の頂点（人間編集）を識別できるか
- **ペア一致率**: モデル候補どうしの序列（人間の選好への近さの順）を再現できるか。順位分解能はこちらで見る

構成比較（BT vs CE）は **同じラベル付き JSONL** で両モデルを走らせ、Top-1 / ペア一致を並べる。

LOPO micro accuracy が高くても、この試験で差が無ければ選抜本線は差し替えない。

## データ形式

スキーマ: [`data/hard_eval.schema.json`](../data/hard_eval.schema.json)  
テンプレート: [`data/hard_eval.template.jsonl`](../data/hard_eval.template.jsonl)

ラベル済み実データは原稿由来になりうるので `data/hard_eval/` に置き、gitignore する。

```json
{
  "id": "he-001",
  "seed_text": "（任意）意味の種。採点には使わない",
  "seed_meta": { "project_id": "what-is-monad", "note": "held-out seed" },
  "base_text": "LLM が書いた同意味のベース文",
  "base_generator": "composer",
  "candidates": [
    { "id": "human", "text": "…", "generator": "human", "prompt_tag": "human-edit" },
    { "id": "model-a", "text": "…", "generator": "composer", "prompt_tag": "revise" },
    { "id": "model-b", "text": "…", "generator": "gpt", "prompt_tag": "revise" },
    { "id": "base", "text": "（base_text と同じ）", "generator": "copy", "prompt_tag": "identity" }
  ],
  "human": {
    "best_id": "human",
    "rank": ["human", "model-a", "base", "model-b"],
    "notes": "人間編集が常に best とは限らない"
  },
  "status": "labeled"
}
```

制約:

- `candidates[].text` は **base の推敲**であること（seed / 元下書きの推敲を流用しない）
- `human.best_id` は `candidates[].id` のいずれか
- `status=labeled` の行だけ採点対象

## 生成プロンプト（目安）

### ベース \(B\)（seed → 同意味・別表層）

```
次の日本語技術文と、意味・主張・技術的内容を保ったまま、
表層の言い回しと文の区切りを変えて書き直せ。
前置き・解説・「以下は〜」は禁止。本文のみを出力せよ。

【原文】
{seed_text}
```

### モデル候補（\(B\) → 推敲）

**全モデル同一の指示**で、モデルだけ替える。規範スキルは付けない。

```
次の文章を、意味を変えずに日本語の技術文書として読みやすく推敲してください。
出力は本文のみ。前置き・解説は不要です。

【文章】
{base_text}
```

### 人間候補

\(B\) を自分で推敲する。他候補は見ずに行うのが望ましい（アンカリング回避）。

## 規模の目安

初回は **20〜30 項目**（各 4〜6 候補）で足りる。  
Top-1 の差がはっきりしなければ項目を増やす。

## 採点コマンド

```bash
# pref-bt
make hard-eval-score \
  INPUT=data/hard_eval/labeled.jsonl \
  SCORER=bt \
  MODEL=outputs/pref-bt

# pref-ce（outputs/pref-ce があるとき）
make hard-eval-score \
  INPUT=data/hard_eval/labeled.jsonl \
  SCORER=ce \
  MODEL=outputs/pref-ce
```

レポートは `outputs/hard_eval_report.json`（および `.md`）。

## 作業順

1. seed を 20〜30 段落選ぶ（held-out 書籍推奨）
2. ベース \(B\) と候補を生成し、`status: pending` の JSONL を作る
3. Markdown プレビューの候補サブセクションを良い順に並べ、`make hard-eval-label` で `labeled` JSONL に変換する
4. BT / CE を同じ INPUT で採点し比較する

## 関連

| パス | 役割 |
|------|------|
| `scripts/import_hard_eval_preview_rank.py` | Markdown の候補順を `best_id` / `rank` に変換 |
| `scripts/score_hard_eval.py` | ラベル付き JSONL を BT/CE で採点 |
| `data/hard_eval.schema.json` | 行スキーマ |
| `data/hard_eval.template.jsonl` | 1 行サンプル |
| [DOK-PREF-CE.md](DOK-PREF-CE.md) | CE 学習。難試験の前段（甘い LOPO） |
| [ROADMAP.md](ROADMAP.md) | 全体位置づけ |
