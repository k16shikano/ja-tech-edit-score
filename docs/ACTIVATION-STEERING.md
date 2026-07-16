# 系統3：activation steering（推敲方向の読み取りと注入）

推敲ペアのコントラストから、ローカル LLM の残差ストリーム上の**方向ベクトル**を取り出し、生成時に加算して文体・編集傾向を寄せる構想。
重みは変更しない。学習はほぼ不要（前向き計算と線形プローブ）。
現時点では **フェーズ A（読み取り）の脚本まで用意**、生成への書き込み（フェーズ C）は未着手。

関連：[EDIT-MODEL.md](EDIT-MODEL.md)（系統1）、[ROADMAP.md](ROADMAP.md)、[WORKFLOW.md](WORKFLOW.md)。

## なぜこの形か

規範スキルは、同じ差分コーパスを LLM に読ませて言語化した規則である。
生成品質がベースモデルに強く依存する、という制約は残る。

kNN 実例注入は、個別実例を文脈に入れるため表面を模倣しやすく、過去の試行では場当たりな悪文になった。
steering は約 5,900 対の**平均差分方向**という統計量だけを活性化に足す。
個別実例の癖は原理的に消えやすく、「抽象化して与える」をベクトルでやる経路に近い。

本リポジトリの pref-static は、埋め込み差分上の線形分類で cross-project 精度が高い。
「推敲済みらしさ」が小さな埋め込み空間でも線形に分離できることは既測である。
より表現力の高い LLM の残差でも同様の方向が取れるかは、フェーズ A で検証する。

## 理論の要約

線形表現仮説のもと、層 \(l\) の隠れ状態について対照対の平均差

\[
v_l = \mathbb{E}\bigl[h_l(\text{推敲後})\bigr] - \mathbb{E}\bigl[h_l(\text{下書き})\bigr]
\]

を取る。
対ごとの内容差は平均で打ち消され、全対に共通する「推敲でどう変わるか」の方向が残りやすい。
生成時に \(h \leftarrow h + \alpha v_l\) とすると、出力がその方向へ偏る。

representation engineering の作法では、先に**読み取り**（その方向で下書き／推敲後を分類できるか）を検証し、通ってから**書き込み**（生成への注入）に進む。
読み取りが弱い層・モデルなら、生成介入の期待値も低い。

限界：制御は粗い。脚注作法や見出し規則など規範の個別条項までは表現しにくい。
効いても「推敲済みらしい圧力」であって、規範スキルの代替ではなく併用が前提である。

## 実験計画（外部 GPU 前提）

データは非公開原稿由来。
外部ジョブではペアテキストの消去を徹底し、恒久成果物は統計量（ベクトル、精度表）に限る。

### フェーズ A：読み取り（本リポジトリで用意済み）

下書き／推敲後をローカル LLM に通し、層ごとの平均プーリング活性を取る。
leave-one-project-out の線形プローブで「下書きか推敲後か」を予測し、層プロファイルを出す。

**中止判定の目安**：最良層の cross-project 精度が pref-static（おおむね 0.95 前後）に遠く及ばない場合、以降のフェーズは縮小する。

手順は後述。

### フェーズ B：ベクトル構成（未着手）

最良層帯で mean-difference と PCA 版の \(v_l\) を構成する。
held-out 下書き活性に \(v_l\) を足したとき、プローブスコアが推敲後側へ動くかを生成前に確認する。

### フェーズ C：生成実験（未着手）

推敲プロンプト生成に \(\alpha\) を振って注入する。
評価は系統1と揃える：BT マージン、pref-static／BT 勝率、長さ比、記法破壊率。
条件は無介入 / steering のみ / 規範スキルのみ / スキル＋steering。
知りたいのは「スキルに載らない暗黙選好がベクトルで再現されるか」である。

## フェーズ A の実行

### 依存

通常の `requirements.txt` に加え、GPU ジョブ用:

```bash
pip install -r requirements-steering.txt
```

`torch` / `transformers` が必要。CUDA 付き環境を想定する。

### 1. 対照ペアの書き出し（ローカル可）

```bash
make steering-pairs
# または
python scripts/export_revision_pairs.py \
  --input data/dpo_curated.jsonl \
  --out data/revision_pairs.jsonl
```

各行は `id`, `project_id`, `draft`, `revised`。`rejected`（下書き）と `chosen`（推敲後）を使い、`[参照]` ブロックは落とす。

### 2. 活性の抽出（GPU）

```bash
make steering-extract \
  MODEL=Qwen/Qwen3-8B \
  DEVICE=cuda \
  LIMIT=0

# スモーク（件数制限。書籍横断のラウンドロビンで選ばれる）
make steering-extract MODEL=... DEVICE=cuda LIMIT=64

# 読み取り方の指定
make steering-extract MODEL=... STEERING_PROMPT_MODE=norms
```

`--prompt-mode` で読み取り方を選ぶ。

| モード | 表象 | 出力先 |
|--------|------|--------|
| `none` | 生テキストの全トークン平均 | `outputs/steering/<slug>/` |
| `reading` | 「推敲済みか考えよ」プロンプトの最終トークン | `outputs/steering/<slug>--reading/` |
| `norms` | `data/tech-writing-norms.md`（規範スキルのスナップショット）を前置した判定プロンプトの最終トークン | `outputs/steering/<slug>--norms/` |

`reading` / `norms` では全サンプル共通の前置部分の KV キャッシュを一度だけ計算して使い回す
（規範前置は約 4,000 トークンあるが、増分は本文ぶんだけになる）。
draft / revised で前置は同一なので、対照差分では前置の寄与は打ち消される。

原稿本文は成果物に含めない（id / project_id / 形状のみ）。

### 3. 層プローブ（CPU でも可）

```bash
make steering-probe MODEL=Qwen/Qwen3-8B [VARIANT=reading|norms]
# または
python scripts/probe_revision_activations.py \
  --activations outputs/steering/<slug>/activations.npz \
  --report outputs/steering/<slug>/probe_report.json
```

各層で 2 つの読み取りを leave-one-project-out で評価する。

- **single**：単独ベクトルが下書き側か推敲後側か（mean-difference 射影＋中点しきい値）
- **pair**：差分ベクトル \(h(\text{推敲後})-h(\text{下書き})\) の符号当て（内容成分が打ち消える。pref-static の対照構造に近い）

Markdown 要約も同ディレクトリに書く。
フル次元のロジスティック回帰は 6,000 ペア規模で数時間かかるため使わない
（PCA128＋LR でも mean-diff 射影を超えないことを確認済み）。

### Makefile 変数

| 変数 | 既定 | 意味 |
|------|------|------|
| `MODEL` | （必須） | Hugging Face モデル ID（`STEERING_MODEL` でも可） |
| `STEERING_DEVICE` | `cuda` | `cuda` / `cpu` / `mps` |
| `STEERING_LIMIT` / `LIMIT` 相当 | `0` | ペア数上限（0＝全件）。`make steering-pairs STEERING_LIMIT=64` も可 |
| `STEERING_BATCH_SIZE` | `1` | 活性抽出のバッチ（VRAM に合わせる） |
| `STEERING_MAX_LENGTH` | `2048` | トークン長上限 |
| `TRUST_REMOTE_CODE` | 空 | 非空なら `--trust-remote-code` |

外部 GPU（さくらの高火力 DOK）での手順は、前提を置かずに書いた次を読む。

- **[DOK-STEERING-PHASE-A.md](DOK-STEERING-PHASE-A.md)**（オブジェクトストレージなし・イメージ同梱・タスク実行）

`scp` / `rsync` は公式に利用不可である。

### 高火力 DOK でのフェーズ A（オブジェクトストレージなし）

マニュアル上、タスクへのデータ投入は次のどちらかである（[FAQ](https://manual.sakura.ad.jp/cloud/koukaryoku-container/faqs.html)）。

1. **コンテナイメージに同梱する**（本節。ストレージ追加不要）
2. 外部ストレージから実行中に取る

`scp` / `rsync` は不可（[SSH](https://manual.sakura.ad.jp/cloud/koukaryoku-container/use-ssh.html)）。
対話ノートへのファイル転送で詰まらないこと。**イメージを push してタスク実行**する。

#### 手順概要

1. 手元: `make steering-pairs`（済みならそのままでよい）
2. さくらのクラウドで **コンテナレジストリ**（非公開）と push 用ユーザーを作る
   （[コンテナレジストリ](https://manual.sakura.ad.jp/cloud/appliance/container-registry/index.html)）
3. 手元でイメージを build & push（`Dockerfile.steering`）
4. DOK でタスク作成: そのイメージ、レジストリ認証、GPU（まず V100）、必要なら `LIMIT=64`
5. 終了後、アーティファクト（`SAKURA_ARTIFACT_DIR`）をダウンロード
6. 手元で `make steering-probe`

#### 手元の build / push

```bash
# 一度だけ: レジストリにログイン
docker login （レジストリ名）.sakuracr.jp

export REGISTRY=（レジストリ名）.sakuracr.jp
chmod +x scripts/build_push_steering_image.sh
./scripts/build_push_steering_image.sh
```

イメージには `revision_pairs.jsonl` と脚本が入る。LLM 重みは入らない（実行時に Hugging Face から取得）。
**非公開レジストリのみ**に push すること（原稿由来のため）。

#### DOK タスク

| 項目 | 例 |
|------|-----|
| イメージ | `（レジストリ）.sakuracr.jp/steering-phase-a:latest` |
| レジストリ認証 | 作成したユーザー |
| GPU | V100 32GB（足りなければ H100） |
| 環境変数 | スモーク `LIMIT=64`。全件なら `LIMIT=0` または未設定。読み取り方は `PROMPT_MODES`（既定 `reading norms`） |
| コマンド | 空でよい（ENTRYPOINT が抽出する） |

成果は自動で `SAKURA_ARTIFACT_DIR` にコピーされる。タスク終了後に詳細から取得する。

## 実装ファイル

| パス | 内容 |
|------|------|
| `scripts/export_revision_pairs.py` | dpo_curated → 清潔な draft/revised |
| `scripts/steering_utils.py` | 共通（ロード、slug、参照除去） |
| `scripts/extract_revision_activations.py` | 層活性の抽出 |
| `scripts/probe_revision_activations.py` | LOPO 線形プローブ |
| `requirements-steering.txt` | torch / transformers 等 |

## フェーズ A 第1回の結果（2026-07-15、Qwen3-8B・mean-pool）

全 6,055 ペア・14 書籍で DOK により抽出し、LOPO でプローブした。

| 読み取り | 最良層 | micro |
|----------|--------|-------|
| 単独ベクトル分類（mean-diff 射影） | 33 | 0.54 |
| 同・PCA128＋ロジスティック回帰 | — | 0.60 前後 |
| ペア差分の符号当て | 30 | 0.64 |

解釈：

- 単独分類はチャンス付近。素の因果 LM の mean-pool 隠れ状態には、
  「推敲済みらしさ」の軸がほぼ線形に露出していない。
- ペア差分にすると 0.64。書籍横断で汎化する推敲方向は**存在するが弱い**。
- 強い分類器でも上がらないので、測定器不足ではなく表象の問題である。
- pref-static の経験（素の BERT 系で 0.5、対照学習済み文埋め込みで 0.9 超）と同型。
  情報が無いのではなく、素の表象では軸が取り出せない。

対策として、読み取りプロンプト（`reading`）と規範前置（`norms`）＋最終トークン抽出を実装した（上記モード）。
規範スキルと推敲ペアは同じ差分コーパス由来なので、規範条件付きの判定軸は
ペアの実差分方向と一致しやすい、という見込みである。第2回はこの 2 条件で計測する。

## 実装状況

| 項目 | 状態 |
|------|------|
| 理論・計画（本ドキュメント） | あり |
| フェーズ A 第1回（mean-pool）：計測済み | 最良 micro 0.64（ペア差分） |
| フェーズ A 第2回（reading / norms・最終トークン） | 脚本あり・未計測 |
| フェーズ B / C | 未着手（第2回の結果で判断） |
