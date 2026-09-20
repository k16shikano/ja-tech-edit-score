# ead R（A2 のみ）作業指示

**やること**

1. A2 で**編集者尤度** $R$ を学習する。
2. 学習した $R$ に編集者尤度が載っているか、**評価データ E** で確かめる。

**前提**：`evaluator-design.md` の $R$ 定義を使う。G1/G2、大局性 A、C のブラインド判定、選好評価器との対決は本 plan に含めない。

---

## 1. 編集者尤度 $R$

下書き $x$、候補 $y$ に対し

$$s_{\mathrm{den}}(x,y) = \log p_\theta(y \mid x) - \log p_0(y \mid x)$$

を返す QLoRA 付き Qwen3-8B である。$\theta$ が $R$、$p_0$ がベースモデル。

$R$ は選好評価器でも関門でもない。E では $s_{\mathrm{den}}$ を付け、分布として報告する。

---

## 2. 学習（A2 のみ）

| 項目 | 内容 |
| --- | --- |
| 教師 | 人間の推敲のみ（下書き → 人間 gold）。ペア負例なし |
| データ | A2 379 節：`data/edit_sft_section/train.jsonl`（329）+ `heldout.jsonl`（50）。元 corpus は `data/revision_corpus/keep_section.jsonl` |
| ベース | Qwen3-8B |
| 方式 | QLoRA SFT（`scripts/ead/den_train.py` と同型） |
| `max_seq_length` | **5632**（4096 では 5/379 が truncate） |
| 成果物 | `outputs/ead/adapters/ead-den-a2/` |

379 節すべて学習に使ってよい。A2 内の held-out 50 も学習に含める（$R$ 用の別 valid は設けない）。

**学習に使わないもの**：A1（`edit_sft_all`）、選好三つ組、D、C の人手判定。

**起動**：`make ead-den-a2-train`（`den_train.py --corpus a2 --max-seq-length 5632` → `outputs/ead/adapters/ead-den-a2/`）。

---

## 3. 評価データ E

**E** = 人間の推敲ではない LLM 生成 $(x,y)$。整備済み（`outputs/ead/data/validation.json` ok）。

$R$ 学習後、各行に $s_{\mathrm{den}}$ を付ける。C の判定 jsonl や選好評価器とは突合しない。

### 3.1 三系統

| 系統 | 件数 | $y$ の生成 | 下書き $x$ |
| --- | --- | --- | --- |
| **E1** | 378 | Composer | A2 379 節のうち `judgments.jsonl` で `degraded` 1 件を除いた節 |
| **E2** | 60 | Qwen3-8B + **編集 SFT** LoRA、貪欲 1 本 | 評価用データ C と同じ 60 下書き（`data/blind_eval/items.jsonl`） |
| **E3** | 60 | 素 Qwen3-8B、貪欲 1 本 | E2 と同一 id |

**読み方**

- **E1**：$s_{\mathrm{den}}$ の分布を単独で報告する。E2/E3 と比較しない。
- **E2 と E3**：同一 id 60 件で $s_{\mathrm{den}}$ を対照する（編集 SFT あり / なし）。比較はここだけ。

E1 と E2/E3 は id も件数も独立。E2/E3 の 60 件（manifest: `c_blind60`）は節 50 + A1 断片 10。うち 50 節の下書きは A2 held-out 由来で、$R$ 学習データの下書きと重なる。学習に入るのは人間 gold であり、E2/E3 の生成 $y$ ではない。

### 3.2 E2/E3 と C の関係

C（BRIEF の評価用データ C）は 60 下書き + アダプタ 8 本サンプリング + 人手判定である。E から使うのは **下書き id と E2 用の貪欲 1 本だけ**。

| 使う | 使わない |
| --- | --- |
| `data/blind_eval/items.jsonl`（下書き） | C のペア・判定 jsonl |
| `outputs/edit-sft-eval-v3/adapter_greedy.jsonl` → E2 | `adapter_samples`（8 本）、評価器選抜 1 本 |
| E3 は同 id で `mode=base` 新規生成 | `base_greedy.jsonl`（C にも E にも使わない） |

E2 の LoRA は**編集 SFT**（例：`outputs/Qwen__Qwen3-8B-pairsplit-v2/adapter`）。$R$（`ead-den-a2`）とは別物。

### 3.3 共通プロンプト

E1/E2/E3 は同じ user 本文。`scripts/ead/e_prompt.py` の `build_e_user_content(draft)`（`export_edit_sft.INSTRUCTION + "\n\n" + draft`）。`prompt_tag` = `edit-sft-instruction`。

E1 は Composer API に 1 文字列。E2/E3 は chat template の user 1 件（E2: `mode=adapter`、E3: `mode=base`、`enable_thinking=false`）。E2/E3 の差は LoRA の有無のみ。

### 3.4 成果物

`outputs/ead/data/`：`e1.jsonl`（378）、`e2.jsonl`（60）、`e3.jsonl`（60）、統合 `e.jsonl`（498）、`manifest.json`、`validation.json`。

1 行 = $(x,y)$。フィールド：`item_id`、`draft`、`generated`、`e_kind`、`prompt_tag`、`generation`、`generator`。E2 は `adapter` と `source: c_adapter_greedy`。E3 は `adapter: null`。

### 3.5 整備コマンド（済）

`make ead-e-pipeline`（= `ead-build-e1` → `ead-import-e2-from-c` → `ead-generate-e3` → `ead-build-e` → `ead-validate-e`）。

---

## 4. E で何を見るか

E の評価は、**人手の勝敗や他評価器との一致率を当てにいく作業ではない**。

BRIEF の目的にさかのぼると、問いは次の一点である。

> 人間の推敲履歴から学習した $R$ が、**人間の推敲ではない** LLM 生成 $(x,y)$ に対しても、編集者尤度 $s_{\mathrm{den}}$ として意味のある値を返すか。

E には人手の正解ラベルがない（C の判定は E の成功条件に使わない）。したがって E で見るのは次の二つに限る。

1. **$s_{\mathrm{den}}$ の分布**（E1 378 件）。異常な偏り、外れ値、系統ごとのずれがないか。
2. **同一下書きでの対照**（E2/E3 60 id）。編集 SFT の有無だけが違うとき、$s_{\mathrm{den}}$ がどう変わるか。

「$s_{\mathrm{den}}(E2) > s_{\mathrm{den}}(E3)$ が何件か」だけを勝ち負けに数えて終えるのは、上の問いに答えていない。件数の大小は報告してよいが、**採否の根拠にしてはならない**。

旧系列（`ead-den-a-excl` など）で B valid 一致率 0.936 や C 一致率 0.589 が出ても、postmortem が示したとおり、それは表層規則（編集距離・expand・長さ）と同天井に張り付いていた。エージェントは「一致率が床を上回った」だけで合格と報告しがちである。**一致率・勝率は E 評価の主指標に置かない。**

---

## 5. E 評価の検証（交絡の否証）

$s_{\mathrm{den}}$ を付けたあと、**$R$ が見ているのが編集者尤度か、それ以外か**を検証する。これは E 評価の必須部分である。分布報告だけで終えてはならない。

検証は二段構成とする。どちらも **E 上で BT/GPM/P-DPO や選好評価器を走らせて勝率を比べる作業ではない**。過去の各種検証で判明した交絡パターンを手がかりに、$s_{\mathrm{den}}$ 単体が同種の誤りをしていないかを調べる。

### 5.1 第一段：計測

| 対象 | 内容 |
| --- | --- |
| E1（378） | $s_{\mathrm{den}}$ の分布（要約統計、ヒストグラム、外れ値） |
| E2/E3（60 id） | 同一 id での $s_{\mathrm{den}}$ 対照、$\Delta s = s(E2) - s(E3)$ の分布 |
| 共通 | truncate 件数、`max_seq_length`、計測不能行 |

### 5.2 第二段 A：ML 系の安い信号

BT、GPM、P-DPO、旧 $s_{\mathrm{den}}$ の評価で、**表層量だけで説明できた**ことが記録されている。E では $s_{\mathrm{den}}$ と同型の表層特徴の相関・層別を取り、同型の偏りが出ないかを見る。

| 既知の傾向（参照記録） | E での検査 |
| --- | --- |
| P-DPO：文字数だけで human/generic 分類 accuracy 0.775；margin と文字数比 Spearman 0.183（A2 test 38） | $s_{\mathrm{den}}$ と `\|y\|`、draft 比、負例側長さ比の Spearman |
| D 床：`edit_distance_norm` AUC 0.642（`delta_chars` は 0.489） | $s_{\mathrm{den}}$ と `edit_distance_norm`、`abs_delta_chars` 等 |
| 旧 $s_{\mathrm{den}}$ on C：edit_distance 大 41/56、expand 34/56；表層三規則も 32/56 で同天井 | E 498 行で表層ルールと $s_{\mathrm{den}}$ の順位一致率（参考）。**採否には使わない** |
| `ead-den-a-excl-ablation`：$\log p_\theta$・$\log p_0$ 単独は低一致、差だけ高一致 | E でも三成分を分解し、差が表層量相関を相殺しただけか |

E2/E3 対照では、$\Delta s_{\mathrm{den}}$ が LoRA の質ではなく、**E2 の方が編集距離大・長い・expand 寄り**というプロファイル差と一致し続けないかを重点的に見る。

### 5.3 第二段 B：非 ML の下書き→推敲変化

機械学習評価器を介さず、**$(x,y)$ の差分そのもの**で説明できる交絡にも注意する。曖昧性プローブ（`ead-ambiguity-probe`）が示したのはこの類型である。解消・保持・消失、Δ文字数符号ごとの解消率の差、追記型と圧縮型の区別など、**編集現象のラベル**で層を切るとスコアと人間方針がずれる、という話である。

**A1 の T1–T10 表や GiNZA アラインメント結果を E にそのまま当てはめる指示ではない。** E の 498 行に対し、同じ考え方で特徴を付け直す。

| 非 ML 軸（E 向け） | 検査 |
| --- | --- |
| 変化のトポロジ | 段落数・空行・文数の増減；全面書き換え vs 局所修正（アラインメント比率） |
| 変化の方向 | expand / shrink / neutral；追記（注・括弧・ hedging）vs 削除・圧縮 |
| 生成源プロファイル | E1 / E2 / E3 で上記分布がずれ、$s_{\mathrm{den}}$ が「生成源の癖」に引かれていないか |
| コーパス典型とのずれ | A2 学習コーパスは shrink 寄り（中央値 Δ=-2）なのに $s_{\mathrm{den}}$ が expand 側を系統的に上げないか |
| 読みやすさ系（非 ML） | GiNZA 等の読みやすさ $z$ スコア（`ead-readability-probe` と同種）と $s_{\mathrm{den}}$ の層別 |

表層量三規則（編集距離・Δ文字数・長さ）は C で一致率 32/56 と同値だった。非 ML 軸は、**なぜ同じ天井に張り付くか**を分解する側である。

### 5.4 レポート

成果物：`outputs/ead/reports/ead-r-a2-e.{md,json}`

- 第一段（分布・E2/E3 対照）
- 第二段 A（ML 系安い信号）
- 第二段 B（非 ML 変化属性）
- 各節で「$s_{\mathrm{den}}$ がこの要因だけで説明できるか」の結論を書く。説明できるなら、編集者尤度が載っているとは言えない。

### 5.5 やらないこと（検証）

- C の判定 jsonl との一致率を E の採否に使う
- B valid 一致率、選好評価器（BT/GPM/P-DPO）との勝率比較
- E 上で他評価器を走らせて「$R$ が勝った」で合格とする
- 一致率・勝率だけを報告して交絡検証を省略する
- E1 と E2/E3 の id 不一致を無視した横並び比較
- `adapter_samples` 選抜 1 本や C の `base_greedy` を E2/E3 に混ぜる

---

## 6. 手順と状態

| 段 | 内容 | 状態 |
| --- | --- | --- |
| 1 | E 整備 | **済** |
| 2 | $R$ 学習（`make ead-den-a2-train`） | **実行中** |
| 3 | E へ $s_{\mathrm{den}}$ 付与（`den_score.py`。生の `s_sum`） | 未 |
| 4 | レポート `ead-r-a2-e`（§5：分布 + 交絡否証） | 未 |

---

## 7. 成功条件

- `ead-den-a2` が学習完了している
- E1 378 件に $s_{\mathrm{den}}$ が付き、分布が報告されている
- E2/E3 同一 id 60 件に $s_{\mathrm{den}}$ が付き、対照が報告されている
- truncate 件数と `max_seq_length` が記録されている
- §5.2・§5.3 の交絡検証が `ead-r-a2-e` に含まれ、$s_{\mathrm{den}}$ が表層量や非 ML 変化属性だけで説明できないかが論じられている

**成功条件に含めないもの**：C/B 人手一致率、選好評価器との勝率、$s_{\mathrm{den}}$ の符号が何件「正解」だったか。

---

## 8. 残タスク

| タスク | 状態 |
| --- | --- |
| `make ead-den-a2-train` | 実行中 |
| E 計測 + 交絡検証レポート（`ead-r-a2-score-e`） | 未 |

---

## 9. 参考

A2 379 件、`max_seq_length=4096`：truncate 5/379（5632 で 0/379）。

交絡検証の参照記録（E 計測時に読む）：`outputs/ead/reports/ead-floor.md`、`ead-ambiguity-probe.md`、`ead-readability-probe.md`、`editor-as-distribution/postmortem.md`、BRIEF §P-DPO、`docs/plan-d-interval.md` §H1。
