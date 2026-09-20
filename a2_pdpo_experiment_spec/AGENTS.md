# AGENTS.md

## 実験の目的

A2を使って、一人の編集者に固有の推敲差分をP-DPOで学習する。

A2は379件の固定データセットで、各項目は次の2要素からなる。

- `draft`: 下書き
- `human_revision`: 同じ下書きを一人の編集者が実際に推敲した文章

基礎LLMがすでに持つ一般的な推敲能力を `pi_0` とし、A2の編集者に固有の差分を学習した方策を `pi_H` とする。
概念的には次を狙う。

    pi_H(y|x) ∝ pi_0(y|x) * exp(Delta_H(x, y))

この実験では `Delta_H` を editor LoRA として近似する。

## 学習データ

各A2項目 i の `draft` から、固定したローカルLLMで一般的な推敲を3件生成する。

    g_i1, g_i2, g_i3 ~ pi_0(. | draft_i)

A2の人間推敲を `h_i` とする。

P-DPOへ与える関係は常に、

    h_i > g_ij

だけである。

### 絶対に作らない関係

- `g_i1 > g_i2` のようなLLM推敲同士の順位
- `g_i1 = g_i2` のような同値関係
- `g_i1 || g_i2` のような比較不能ラベル
- `human_revision > draft` のようなdraftをrejectedとするpair
- A2全体をa<b<c<dのような一次元序列へ変換したラベル

LLM推敲同士は、この実験ではそもそも比較対象ではない。

## 実施する学習

**P-DPOだけを実施する。**

- SFTは行わない。
- vanilla DPOは行わない。
- SFT -> P-DPOの二段階学習も行わない。
- SFTやvanilla DPOをbaselineとして追加しない。

adapter OFFのbase modelは `pi_0` のreference policyとして必要だが、これは別の学習条件ではない。

## 文書単位についての前提

- 段落単位で推敲が進むとは仮定しない。
- before/afterの段落alignmentを作らない。
- 段落の統合・分割・移動を許容する。
- 局所的な文意が変わっていてもよい。
- 原稿全体の主張・技術内容・事実関係が保たれていることを重視する。
- 編集判断を「構成」「簡潔さ」「論理性」などの独立軸へ分解しない。

## A2の分割

A2 379件をsource item単位で固定分割する。

- train: 303
- dev: 38
- test: 38
- seed: `20260914`

`data/A2/split_manifest.jsonl` を一度生成したら固定する。
同一原稿の別versionがある場合は同じsplitへ置く。

## 一般的なLLM推敲の生成

各 `draft` から3件生成する。

    K = 3

既定generator:

    Qwen3.8-27B Q4_K_M
    llama.cpp

model path / endpoint / sampling設定は `config/experiment.yaml` から変更可能にする。

全A2で同じmodel、同じprompt、同じsampling設定を使い、sampleごとにseedだけを変える。

### 生成prompt

正本は `prompts/generic_revision_system.txt` と `prompts/generic_revision_user.txt`。
コードへpromptを直書きしてはいけない。

ローカルLLMには次だけを要求する。

- 一般的な日本語編集者として推敲する。
- 原稿全体の主張、技術内容、事実関係を維持する。
- 段落の統合・分割・移動、説明順序変更、局所的な言い換えを許す。
- 原稿にない新事実を追加しない。
- A2の編集者の文体を模倣しない。
- 変更理由、講評、要約を出さない。
- 推敲後本文だけを出力する。

## モデル構造

base model本体は凍結する。

    base model
      ├─ adapter OFF = reference / generic policy pi_0
      └─ editor LoRA ON = editor policy pi_H

一人の編集者しか扱わないため、最初の実験では別個のuser embeddingを導入しない。
editor-specific residualはLoRAに持たせる。

## P-DPO loss

各pair `(x_i, h_i, g_ij)` について、

    s_theta(x,y) = log pi_H(y|x) - log pi_0(y|x)

とし、

    L_ij = -log sigmoid(beta * [s_theta(x_i,h_i) - s_theta(x_i,g_ij)])

を使う。

同じA2 itemから3 negativesを作るので、item単位で重みを平均する。

    L_i = (L_i1 + L_i2 + L_i3) / 3

negativeを3件持つA2 itemの重みを3倍にしてはいけない。

## reference log-prob

GPUメモリ節約のため、`pi_0` のreference log-probは事前計算して保存してよい。
学習時にreference modelを同時ロードする必要はない。

保存する値:

- sequence total log-prob
- completion token count
- mean token log-prob（診断用）

P-DPOのlossは標準のsequence log-probを使う。

## 長さリークの確認

A2の人間推敲とgeneric revisionの長さだけで識別できていないかは必ず確認する。
ただし、これは別の学習baselineではなく診断である。

最低限、次を出力する。

- `len(human_revision) / len(draft)` の分布
- `len(generic_revision) / len(draft)` の分布
- P-DPO marginと長さ比の相関

必要なら、文字数等だけを使う単純classifierを診断用に実装してよい。

## 評価

主評価はtest 38件で行う。

各test draftから、学習用とは異なるseedでfresh generic revisionを3件生成する。
評価対象は常に、

    human_revision vs fresh_generic_revision

だけである。

fresh generic revisions同士を比較しない。

各pairについて、

    margin = s_theta(x, human) - s_theta(x, generic)

を計算し、A2 item単位で平均する。

さらに各test draftについて、

- adapter OFFによるgeneric revision
- adapter ONによるeditor-adapted revision

を1件ずつ生成し、blind A/B用ファイルを作る。
勝敗は自動で付けない。

## 実装順序

`docs/IMPLEMENTATION_ORDER.md` の順に進める。

## 禁止事項

- A2を379件以外に再定義しない。
- human_revision以外をchosenにしない。
- draftをrejectedにしない。
- generic revision同士の比較pairを作らない。
- SFTを追加しない。
- vanilla DPOを追加しない。
- 段落alignmentを前提にしない。
- testをmodel selectionへ使わない。
- P-DPOが成功しただけで「編集者の原稿の見え方を学習した」と結論しない。

## 完了条件

`docs/ACCEPTANCE_CRITERIA.md` をすべて満たすこと。
