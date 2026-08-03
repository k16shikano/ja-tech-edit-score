# 編集 SFT 学習コーパスの再整備

## 作業記録（2026-08-02）

この日までに確定した事実と、実施した手順を時系列で残す。
方針の要約は後続節。手順の再現は本節と「実装の置き場所」を見る。

### 問題とゲート

1. **比較点**  
   対は常に `fork..edit`。`第1親..edit` は使わない（p1≠fork のとき逆向き教師になる）。  
   `resolve_pre_merge_pair` / `assert_structural_edit_pair` で強制。  
   検証: `scripts/verify_edit_revision_pairs.py`（fb9 逆向き＋ pfvm late tip）。

2. **「下書き→推敲」でない対**  
   git 上は祖先条件を満たしても、推敲済み main から生えた未マージ tip は教師にならない。  
   例: pfvm の `edit/kernelfusion-memorysave`（画面上ほぼ同一に見えた対）。  
   対策（`scripts/git_pre_merge.py`）:
   - `reedit/*`・`*-reedit`・`*-summary` は掘らない
   - **未マージ**で、fork 時点にすでに `edit/` / `reedit/` / `fix_` マージが取り込まれていれば捨てる  
     （マージ済みは従来どおり `merge_commit` 経路）

3. **keep 汚染の定義**  
   汚染が成立するのは base が fork でなく、かつ p1≠fork（または tip main≠fork）のときだけ。  
   証明できないものは汚染扱いしない。

### 再取得（late-gate）

退避: `data/archive-before-20260802-late-gate/`（旧 raw・8311/8312 キュー）。

| raw | 件数（再取得後） |
|-----|-----------------:|
| `examples.section.raw.jsonl` | 336 |
| `examples.section.extra.raw.jsonl` | 264 |
| `examples.raw.jsonl` | 3570 |
| `examples.wwtawwta.premerge.raw.jsonl` | 1058 |

レビューサーバは起動時に JSONL/state を載荷する。データ差し替え後は再起動が必要（8311/8312 で実施）。
8310 キューは raw 再取得の対象外だった（すでにレビュー済み keep を保持していたため）。学習上は他キューの keep と同列である。

### 人手レビュー完了

| キュー | PORT | pending | keep | exclude |
|--------|-----:|--------:|-----:|--------:|
| 節 | 8310 | 0 | 200 | 0 |
| hunk | 8311 | 0 | 1659 | 51 |
| 節追加 | 8312 | 0 | 121 | 1 |

三キューの keep はいずれも学習採用。8310 だけを別カテゴリにはしない。

### 学習用への書き出し

規則: 推敲前か後に空行（`\\n\\n`）があれば節系統。どちらにも無ければ空行なし hunk。  
8310 と 8312 は同質として節へ。8311 の空行ありも節へ合流。

| 系統 | 件数 | パス |
|------|-----:|------|
| 節（空行あり） | 379 | `data/edit_sft_section/` |
| 空行なし hunk | 1597 | `data/edit_sft_hunk_nopara/` |
| **合流（SFT 本線）** | 1976（train 1717 / heldout 259） | `data/edit_sft_all/` |

内訳（空行あり 379）: 8310 から 200、8312 から 121、8311 から 58。出典キューの違いであって、学習上の等級差はない。

- 書き出し: `make edit-sft-export-keeps`（`scripts/export_reviewed_keeps.py` → `build_revision_corpus.py`）
- heldout は各レビューキューでもともと heldout 側だった keep を維持したもの。学習箱は train のみ読む
- 正本: `data/revision_corpus/keep_section.jsonl` / `keep_hunk_nopara.jsonl` / `canonical.jsonl`

### Qwen SFT（DOK）

- イメージ: `ja-tech-edit.sakuracr.jp/edit-sft:latest`（全 keep = `data/edit_sft_all/train.jsonl`）
- ベース: `Qwen/Qwen3-8B`、QLoRA r=16、系列長 8192、lr 2e-4、有効バッチ 8
- **採用ラン**: `EPOCHS=2`（件数 1717 に合わせた周回。15 は中断）
- 成果物: `outputs/Qwen__Qwen3-8B-revised-2000/`（`adapter/` + `train_meta.json`。n_train=1717, epochs=2）
- 手順本文: [DOK-EDIT-SFT.md](DOK-EDIT-SFT.md)
- 系統別だけの学習は `make edit-sft-section-only` / `edit-sft-hunk`（任意）

### まだやっていないこと

- DOK SSH での対話試用（[DOK-EDIT-SFT-CHAT.md](DOK-EDIT-SFT-CHAT.md)）
- 空行なし hunk だけの別イメージ学習（本線では合流に含め済み）

held-out バッチ生成（`outputs/edit-sft-eval/`、259 件）と手元採点は済み。採点器勝率は副次。

## 再確認した2点

1. **段落**  
   コーパス全体に段落付きが無いのは問題である（構成を含む推敲を学べない）。  
   ただし段落の無い短い対も、前後と対応が正しければ重要な正しい推敲データである。  
   「段落が無い＝壊れた教師」ではない。

2. **正しい推敲データだけを使う**  
   次を除去した対だけを学習に入れる。  
   - 推敲の順序を取り違えた対比（進行した main と edit など）  
   - 推敲でない対比  
   - コード片や英文などしかないノイズ  

## 学習データの管理（レビュー keep）

人手 keep はすべて学習に使う。空行（段落境界）の有無だけであらためて二系統に分ける。

| 系統 | 判定 | 学習用 chat | 正本 |
|------|------|-------------|------|
| 節 | 推敲前か後に空行あり | `data/edit_sft_section/` | `data/revision_corpus/keep_section.jsonl` |
| 空行なし hunk | どちらにも空行なし | `data/edit_sft_hunk_nopara/` | `data/revision_corpus/keep_hunk_nopara.jsonl` |
| **合流（SFT 本線）** | 上記の和 | `data/edit_sft_all/` | `canonical.jsonl` |

- 8310 と 8312 は同質。keep は節系統へ
- 8311 のうち空行ありも節系統へ合流
- 8311 のうち空行なしは空行なし hunk 系統
- Qwen SFT は **合流（`edit_sft_all`）** で回す。系統別ディレクトリは分析・再学習用に残す
- 書き出し: `make edit-sft-export-keeps`
- 学習: `make edit-sft`（合流）/ 系統別は `edit-sft-section-only` / `edit-sft-hunk`

## 素材の棚卸し（2026-08-02）

| 素材 | 件数 | 判定 |
|------|-----:|------|
| 8310 keep（`data/edit_sft/`） | 200 | 節系統へ |
| 8312 keep | 121 | 節系統へ（8310 と同質） |
| 8311 keep | 1659 | 空行あり→節、空行なし→hunk_nopara |
| 合流 `edit_sft_all` | 1976 | SFT 本線 |
| `data/examples.raw.jsonl` | 3570 | `resolve` の fork..edit。late unmerged / reedit 除外後 |
| `data/examples.wwtawwta.premerge.raw.jsonl` | 1058 | 同上 |
| `data/examples.section.raw.jsonl` | 336 | 同上 |
| `data/examples.section.extra.raw.jsonl` | 264 | 同上 |

### 節追加から除外するリポジトリ

| project_id | 理由 |
|------------|------|
| `peering` | 利用不可 |
| `treasure-data` | 利用不可 |
| `voyage-techbook` | 利用不可 |
| `wwtawwta-systems` | 節追加キューでは掘らない。hunk は pre-merge で本リポ側に取り直す |
| `progo` | ブランチが複雑で、推敲済みの判定ができない |

## 比較点の運用（徹底）

推敲対では **対になる main 側が edit 側より古い**。main 側のほうが新しい並び（進行した tip との逆差分）は推敲ではない。

| 規則 | 内容 |
|------|------|
| mainline | `origin/main`（なければ `origin/master`）を正とする。遅れたローカル `main` は使わない |
| 差分範囲 | 常に `fork..edit`。`第1親..edit` の two-dot は禁止（p1≠fork のとき逆向き教師になる） |
| マージ済み | fork = `merge-base(p1,p2)`、edit = 第2親。**base は必ず edit の祖先** |
| 未マージ | fork = `merge-base(mainline, edit)`。edit がすでに最新 mainline の祖先なら捨てる |
| 再編集 tip | `reedit/*`・`*-reedit`・`*-summary` は掘らない。未マージで、fork 時点にすでに `edit/`/`reedit/`/`fix_` マージが取り込まれていれば捨てる（推敲済み main からの後出し） |
| 実装 | `scripts/git_pre_merge.py` の `resolve_pre_merge_pair` / `assert_structural_edit_pair` |
| 検証 | `scripts/verify_edit_revision_pairs.py`（再採掘前に通す。fb9 逆向き＋ pfvm late tip） |

`edit/*` に記号修正など推敲以外が混ざっていてもよい。上記を満たさない対だけ掘らない。
採掘時点の `review_result` は `pending`。学習昇格は人手 `keep` のみ（pending を keep 扱いしない）。

`editor-ai-starter` の `mine_branch_examples.py` は `git log <editブランチ> -p` の全履歴を掘るため、翻訳ラウンド・main 共有履歴・WIP が混ざる。  
本リポでは `git_pre_merge` + `mine_branch_pair` で **マージ直前の fork↔edit** だけを掘る。`batch_import_repos.sh`（旧 tip main 直比較）は拒否する。

## 人間レビューの分離（必須）

| キュー | ディレクトリ | make | 既定 PORT | 想定 |
|--------|--------------|------|----------:|------|
| 節 | `data/edit_sft/` | `edit-sft-review` | 8310 | keep は節系統へ。8312 と同質 |
| hunk | `data/edit_sft_hunk_review/` | `edit-sft-review-hunk` | 8311 | 空行ありは節へ、なしは hunk_nopara へ |
| 節追加 | `data/edit_sft_section_extra_review/` | `edit-sft-review-section-extra` | 8312 | 8310 と同質。節系統へ合流 |

### keep 汚染の切り方（雰囲気判定禁止）

汚染が成立するのは **比較 base が fork ではなく、かつ p1≠fork（または tip main≠fork）のとき**だけ。  
`p1==fork` や `base==fork` なら、この経路では汚染しない。証明できないものは汚染扱いしない。

| 集合 | 件数 | 構造判定 |
|------|-----:|----------|
| 8310 keep | 200 | 照合できた範囲に `CONTAMINATED_P1` なし |
| hunk keep（purge 退避から復元した期） | 574 | **全件 `base==fork`（OK）**。一括破棄は誤りだった |
| before_threedot 期の hunk keep | 558 | OK 555 / `CONTAMINATED_P1` **3** のみ |

件数（レビュー完了・書き出し後）:

- 8310 / 8311 / 8312: 上表のレビュー完了どおり
- 学習合流: train 1717 / heldout 259（`data/edit_sft_all/`）
- late-gate 再取得の退避: `data/archive-before-20260802-late-gate/`

## 作り直しの方針

### レビュー候補の作り方（実施済み）

`scripts/prepare_hunk_training_pairs.py` が hunk raw をマスク／フィルタし、`hunk_trainready.jsonl`（`quality=candidate`）を出す。  
節追加は `batch_mine_sections_extra.sh` → `export_edit_sft.py` → 8312。  
**学習正本には candidate を入れない。** 人手 keep だけを `export_reviewed_keeps` で昇格する。

### 正本フォーマット

`examples.schema.json` 互換（`source_text` / `edited_text` / `project_id` / `source_reference` / …）に統一する。  
SFT 用 chat messages への変換は下流で行う。

追加メタ（corpus 用）:

- `unit`: `section` | `hunk` | `span`
- `compare_kind`: `pre_merge` | `branch_tip` | `human_reviewed` | `trusted_import`
- `quality`: `keep` | `candidate` | `reject`（自動＋人手）
- `has_paragraph_break`: bool
- `corpus_source`: 由来パス
- `meta.bucket`: `section` | `hunk_nopara`

### パイプライン順

1. **採掘**: `resolve_pre_merge_pair`（late-gate 込み）で raw を取る
2. **レビュー**: 8310 / 8311 / 8312 で keep を確定
3. **書き出し**: `make edit-sft-export-keeps` → 節 / 空行なし / 合流
4. **正本**: `canonical.jsonl` = keep_section + keep_hunk_nopara（いずれも `quality=keep`）
5. **SFT**: `make edit-sft` または DOK 箱（合流 `edit_sft_all`）

### 段落バランス

節系統と空行なし hunk はディレクトリを分けて管理する。  
SFT 本線では合流して学習する。

## 実装の置き場所

| パス | 役割 |
|------|------|
| `data/edit_sft_all/` | **SFT 本線**（全 keep の合流 chat） |
| `data/edit_sft_section/` | 空行あり keep（chat） |
| `data/edit_sft_hunk_nopara/` | 空行なし keep（chat） |
| `data/revision_corpus/keep_section.jsonl` | 節系統の正本 |
| `data/revision_corpus/keep_hunk_nopara.jsonl` | 空行なし hunk の正本 |
| `data/revision_corpus/canonical.jsonl` | 上記の統合 |
| `data/edit_sft/` | 8310 レビューキュー |
| `data/edit_sft_hunk_review/` | 8311 レビューキュー |
| `data/edit_sft_section_extra_review/` | 8312 レビューキュー |
| `data/section_extra_exclude.json` | 節追加から外す project_id と理由 |
| `data/archive-before-20260802-late-gate/` | late-gate 再取得前の退避 |
| `scripts/git_pre_merge.py` | fork..edit / late-gate |
| `scripts/verify_edit_revision_pairs.py` | 比較点検証 |
| `scripts/export_reviewed_keeps.py` | keep → 二系統＋合流 |
| `scripts/build_revision_corpus.py` | 正本組み立て |
| `scripts/prepare_hunk_training_pairs.py` | hunk 機械整形（レビュー候補） |
| `scripts/export_corpus_for_review.py` | corpus → レビュー用 chat JSONL |
| `scripts/batch_mine_sections_extra.sh` | 件外リポの節採掘 |
| `scripts/batch_mine_hunks_premerge.sh` | hunk 再採掘（pre-merge） |
| `scripts/preserve_review_keeps.py` | 再出力後の keep/exclude 引き継ぎ |
