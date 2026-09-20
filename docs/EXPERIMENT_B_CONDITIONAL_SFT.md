# データBによる条件付き生成と、データDによる選好学習の実験仕様

この文書は、2026年9月7日時点のBRIEF.mdと、今回提示されたデータD・実験結果に基づく実装指示である。
実データ、既存コード、学習済み重みは未確認であり、以下の中間形式・コマンド名・初期設定は新規実装の仕様である。
学習・推論を実施した結果ではない。

実行環境はローカルのPro 4000、単一GPUに固定する。
既存のtrain_edit_sft.pyは使用せず、6節のprompt/labelsをそのまま扱う学習処理を新規実装する。

今回固定するのは、評価式、分割と除外の規則、モデルrevision、入出力形式である。
D epoch7のコード由来の設定値、Bの長さ分布、Cの1,024トークン超過件数、Pro 4000での最大系列のメモリ使用量は、手元のデータ・コードで取得する実測項目として残る。
未取得の件数を0として扱わず、確認レポートではnullとstatus="not_measured"を使う。

## 1. 研究目的と、今回の実験から分かる範囲

目的は、一人の編集者の推敲履歴と理由を伴わない判断から、言語化できない推敲の質をモデルに学習させることにある。
文法性・冗長性などの評価項目への分解や、編集意図の説明は教師信号に加えない。

Bによる生成実験は、この目的に向けた一つの学習経路を検証する。
生成性能の改善だけで、汎用の評価器や、任意の推敲の質を表すスカラーを獲得したとは結論しない。

以下では、下書きを x、Composerによる推敲を y、人間による推敲本文を h とする。
ユーザーが示した p(a | x,y) の a は、この h に相当する。
Dの評価ラベル a との混同を避け、本文の変数名には h を使う。

Bには (x,y,h) があるので、pθ(h | x,y) の教師あり学習はそのまま成立する。
編集操作への変換も、人間による追加の理由説明も必要ない。
出力は h の全文とする。

ただし、Bの h は、y を見て人間が訂正した結果として収集されたものではない。
したがって、Bは「yに対する人間の実際の訂正行動」の記録ではなく、同じ下書きに由来する候補と参照本文の組である。
yを無視してxからhを予測する解も成立する。
また、hを生成の教師に使うことと、すべての原稿でhがyより優れていると仮定することは別である。
後者の順位はこの生成損失に加えない。

## 2. データDに関する次の検証

### 2.1 現在判明している事実

- DはA1由来の200原稿、原稿ごとに生成候補3本と人間の推敲1本、計800行。
- ラベルは a < b < c < d。bは下書き相当、dは人間の推敲相当。
- trainは640行、validは160行。原稿単位で分割されているかは未確認。
- Dは4段階位置ラベル付きの選好データとして作成された。
- 実施済みのpref-sentseq-a2b-modernbert-dでは、Dの区間損失で学習したModernBERTを凍結し、文埋め込みとして利用した。
- 後段のTransformerと採点層はA2、続いてBのBT損失で学習した。
- Cの人手で同等だった4件を除いた56件で、人手選好との一致は25件、44.6%。

この44.6%は、Dで学習した元の区間判定器の評価値ではない。
Dで学んだ表現を文埋め込みとして転用し、別の教師信号で採点を学び直した系の結果である。
凍結されたModernBERTについて「A2/B学習で忘却した」とは説明しない。
Dの判定に必要な情報が後段の採点に利用されているかは、この結果だけでは分からない。

### 2.2 D epoch7判定器の復元と直接評価

#### 2.2.1 元の実装を固定する

Dの7 epoch目の判定器を、当時の入力単位・前処理・判定ヘッドを含めて復元する。
Dの区間損失を計算していたモデル全体を対象にする。
文埋め込みの抽出や、別の分類ヘッドへの差し替えを加えない。

既存コード、実行時設定、checkpointから次を取得し、d_epoch7.lock.jsonに保存する。

| 項目 | 固定する内容 |
| --- | --- |
| 実装 | git commit、対象ファイルとハッシュ、未commitの差分があればその保存先 |
| 重み | checkpointのパスとSHA-256、epoch表記と実際の更新step、モデル全体を復元できるか |
| 入力 | 下書き・候補のどちらをどの順番で渡すか、区切り文字、特殊トークン、前処理関数 |
| tokenizer | モデル名・revisionまたはローカルファイルのハッシュ、padding/truncationの当時の設定 |
| 長さ | max_length=1024。当時の切り詰め側・方式も記録 |
| 判定 | poolingの方式とmask、ヘッド構造・活性化、連続得点の定義と大小の向き |
| 損失 | 実際の区間損失式、a/b/c/d各区間の上下端・端点の包含、reduction |
| 学習履歴 | checkpoint選択指標、train/validの原稿ID、既存の選択ログ |
| 推論 | dtype、evalモード、使用ライブラリのversion |

区間の端点や入力の順番を、この文書から推定して埋めない。
元のヘッドの重みが保存されていなければ、epoch7判定器の復元はできないと報告する。
その場合、新しいヘッドや再学習モデルの成績をepoch7の成績として出さない。
NaN、無限大、欠損得点は同点扱いにせず実行エラーとして記録する。

原稿IDによりDのtrain/validの重複を確認する。
640/160行であることから160/40原稿の分割とは断定しない。
重複があれば、元のcheckpointの評価には、そのtrainに一候補も入っていない原稿だけを使う。
分割を作り直しただけでは旧checkpointへの漏洩は消えない。
独立した留保原稿が不足する場合は、原稿単位の分割で再学習する実験として区別する。
epoch7の選択に使ったvalid上の数値は、モデル選択後の診断値と明記する。

#### 2.2.2 「変更候補同士のラベル順位」の定義

原稿iの下書きをx_i、Dの4候補をy_ij、ラベルをℓ_ij∈{0,1,2,3}とする。
a→0、b→1、c→2、d→3は順位を符号化するだけであり、段階間の距離を仮定しない。
s_ijは、元の区間判定器が返す連続得点を「大きいほど良い」に揃えた値とする。
符号は元の損失と区間定義から固定し、評価結果を見て反転しない。
予測ラベルへ丸めてから順位を測らない。

まず、評価対象として予定する候補ペアを定める。

$$
P_i=\{(j,k):j<k,\ y_{ij}\ne x_i,\ y_{ik}\ne x_i,\ \ell_{ij}\ne\ell_{ik}\}.
$$

「変更」は保存本文の文字列不一致とし、strip、Unicode正規化、空白除去を挟まない。
下書きそのものを候補として追加しない。
同ラベルの組は、この順位指標から除く。
生成候補3本と人間候補1本の全組が対象だが、生成候補同士だけの値も必ず別に報告する。
後者により、生成元の違いを当てるだけの系でも得られる成績を区別できる。

このうち、両候補を元の入力形式で全文処理でき、有限な得点を得たペアをQ_i⊆P_iとする。
初回の全文評価では1,024トークンを超える入力を切り詰めない。
判定器の入力が下書きと候補の連結なら、特殊トークンを含む連結後の長さを数える。
各候補の得点は一度だけ計算し、候補ペア間で使い回す。

$$
m_{ijk}=(\ell_{ij}-\ell_{ik})(s_{ij}-s_{ik}),\qquad
u(m)=
\begin{cases}
1 & m>0\\
\tfrac12 & m=0\\
0 & m<0 .
\end{cases}
$$

主指標は、比較できるペアがある原稿の平均とする。

$$
I=\{i:|Q_i|>0\},\qquad
A_{\rm macro}=
\frac1{|I|}\sum_{i\in I}\frac1{|Q_i|}
\sum_{(j,k)\in Q_i}u(m_{ijk}).
$$

補助指標は全ペアをまとめた一致率とする。

$$
A_{\rm micro}=
\frac{\sum_i\sum_{(j,k)\in Q_i}u(m_{ijk})}
{\sum_i|Q_i|}.
$$

得点の同点は、保存した未丸めの得点の厳密一致とする。
後から許容幅を最適化しない。
計算時の精度をlockし、低精度の値をfloat32へ変換しただけでは元の精度が上がらないことにも注意する。
ペアが0件の値はnullとし、0%や50%で埋めない。

両指標に加え、正順・逆順・得点同点の数、予定ペア数Σ|P_i|、評価ペア数Σ|Q_i|、対象原稿数、原稿ごとの比較数を保存する。
無変更・同ラベル・長さ超過・実行エラー・学習原稿との重複による対象外件数も別々に出す。
生成候補同士の集計でも、同じ式と除外規則を用いる。
同じ本文に異なるラベルが付いた組は消さず、注記して残す。
同じ入力から同じ得点を返す系では、その組は得点同点になる。
信頼区間を付ける場合は、原稿または既知の重複範囲group単位のbootstrapを使う。

#### 2.2.3 Cへの適用と1,024トークン上限

Cの60原稿について、既存の人手判定に対応する2候補を、Dの入力関数とtokenizerでtruncation=Falseとして長さ計測する。
人が選んだ既存候補以外のサンプルへ差し替えない。
同等4件を含む全60件の長さを先に集計する。

片方でも1,024を超えれば、その原稿の比較はstatus="unevaluable_length"とする。
片側だけの得点から優劣を推定せず、文分割・先頭切り出し・平均得点への変更も行わない。
そのような長文処理は、別の評価器を定義する追加実験になる。

c_length_report.jsonには、左側超過・右側超過・両側超過・いずれか超過、両側が上限内の件数を保存する。
全60件と、人が優劣を付けた56件の双方について数える。
ここでの左右は既存Cの2候補を示す固定名であり、新しいブラインド比較の乱択左右とは別である。

56件のうち全文評価できた原稿だけで、正順1、得点同点0.5、逆順0の一致率を計算する。
その分母nとcoverage=n/56を必ず併記する。
人手の同等4件は、得点差の分布を別に残し、順位一致率に加えない。
実行エラーは長さ超過と別に数える。
旧評価器の25/56と直接比較する場合、旧評価器も今回評価可能だった同じ原稿集合で再集計する。
異なる分母の数値だけで改善を主張しない。
Cを使ってDのcheckpoint、得点の向き、長文処理方式を選び直さない。

### 2.3 負の信号を最終段まで使う比較

次の学習比較では、Dから作った選好を、最終的に使う採点ヘッドの教師信号にする。
A2/Bだけの後段学習で置き換えない。

Dの各原稿内で、4候補からラベルが異なる全組を作る。
生成元ではなく、実際のラベルを使って勝者・敗者を決める。
同ラベルの候補は比較未観測として、この最初のペア損失には入れない。
同ラベルを「真に同等」とみなして得点差を0に押し込まない。
別原稿の候補間に順位を作らない。
一原稿当たりの比較数は最大6であり、比較数の多い原稿だけが強く重み付けされないよう、原稿内の損失平均を取り、その後原稿間で平均する。

D由来のペア台帳を教師にする選好学習（スカラー BT と GPM 選好ベクトル）の手順・次元・評価は [plan-d-gpm.md](plan-d-gpm.md) に書く。
ベース、入力、更新範囲、分割、候補組、最適化条件を BT と GPM で揃える。

優先順は、元のD判定器の直接評価、Dを最終段の教師にする比較（plan-d-gpm）とする。

## 3. Bによる生成実験の仮説と比較条件

新規に学習するのは次の3条件である。
各条件は同じ初期モデルから独立に開始する。

| 条件ID | 入力 | 教師本文 | 検証する点 |
| --- | --- | --- | --- |
| b-gen-x | 下書き x | h | 同じBによる通常の直接SFT |
| b-gen-y | Composer候補 y | h | 候補からの再生成だけで足りるか |
| b-gen-xy | 下書き x と候補 y | h | 両方を与える利益があるか |

既存の「Aで学習したQwen」との比較だけでは、学習データや初期状態も変わる。
b-gen-xを同じB、同じ初期状態、同じ更新回数で学習し、主たる対照とする。

追加学習なしの初期モデルを同じxy入力で動かす b-gen-xy-base も用意する。
これは追加の学習runを必要としない。
出力をそのまま使うComposer候補 y も比較対象として保存する。

## 4. Bの取り出し方と分割

### 4.1 既存ファイル

- data/revision_corpus/keep_section.jsonl：A2の下書きと人間推敲。
- data/section_middle/revisions.jsonl：Composerの推敲。
- data/section_middle/pref_train.jsonl、pref_valid.jsonl：Bの原稿集合と既存の分割の確認。

既存JSONLのフィールド名は未確認である。
実装者はdocs/DATA.mdと既存の読み込みコードを確認し、原稿IDで結合する。
chosen/rejectedという位置だけから人間・Composer・下書きを推測しない。
対応IDがない場合、元の下書き本文の厳密一致を補助に使い、不一致を類似度だけで自動結合しない。

3種類の選好ペアを、そのまま3倍の学習例にしない。
Bの一原稿から、x、y、hを持つ一件を作る。
不合格としてBから除外済みのComposer候補を復活させない。

### 4.2 新規の中間形式とmanifest語彙

本文はrecords.jsonlに一原稿一行で保存する。
分割と適格性はmanifest.jsonを唯一の正とする。
以下は新規の形式であり、既存ファイルのスキーマではない。

records.jsonlの一行：

~~~json
{
  "source_id": "<既存の原稿ID>",
  "draft": "<下書き全文>",
  "composer": "<Composerの推敲全文>",
  "human": "<人間の推敲全文>"
}
~~~

manifest.jsonのトップレベル：

~~~json
{
  "schema_version": "b-generation-manifest-v1",
  "expected_source_count": 378,
  "records_file": "records.jsonl",
  "split_seed": 20260907,
  "dev_group_fraction": 0.1,
  "model_lock_file": "model.lock.json",
  "length_report_file": "b_length_report.json",
  "sources": []
}
~~~

sourcesの各要素は次のフィールドを持つ。

| フィールド | 型・許容値 | 意味 |
| --- | --- | --- |
| source_id | 必須string、一意 | records.jsonlとの結合キー |
| document_id | stringまたはnull、キーは必須 | 取得できた場合の親文書ID。分割の必須情報ではない |
| group_id | 必須string | 同一splitに置く原稿群 |
| grouping_basis | source_id / known_overlap | 単独原稿か、既知の重複・包含関係によるgroupか |
| original_split | train / valid | 既存Bのpref_train / pref_validのどちらに属したか |
| split | train / dev / holdout | 今回の用途。validやtestという別名を許さない |
| split_reason | original_valid / group_contains_valid / hash_dev / remaining_train | 分割した理由 |
| eligible | boolean | 3条件共通で今回の実験に使用するか |
| exclusion_reasons | string配列 | 適格なら空。長さ超過時の語彙は7.1節 |
| lengths | object | x/y/xy各条件のprompt_tokens・train_tokens・inference_total_tokensとtarget_tokens |
| text_sha256 | object | draft/composer/humanそれぞれ、保存本文のUTF-8バイト列のSHA-256 |

original_splitにholdoutやdevを入れない。
元の同一source_idがpref_trainとpref_validの両方に現れた場合、元データの矛盾として報告して結合処理を止める。
document_idを推測して作らない。
書籍・章の区別は、既存のIDの意味をschema_mapping.jsonに記録する。
document_idが同じという理由だけで、一冊全体を自動的に一つのgroupにしない。

本文の改行、段落、空白、表記を保存する。
前後の差分による文分割や、編集タイプの付与は加えない。
全378原稿を復元できることを先に確認する。
欠落・重複・不一致があれば、長さ超過による予定された除外と区別し、IDをdata_checks.jsonへ記録する。
378原稿を復元できない状態で、黙って学習集合を確定しない。
本文・元ファイルのフィールド対応と結合根拠もschema_mapping.jsonへ記録する。

### 4.3 実験用の分割

分割は本文の長さによる除外前に一度だけ行う。
同じ原稿のx、y、h、および派生した全条件を同じsplitに置く。
重複・包含する本文の範囲が分かる場合、その関係の連結成分を一つのgroupにする。
group_idは、そのgroupのsource_id配列をソートし、UTF-8の正規化したJSON表現をSHA-256にかけた値にgroup:を付ける。
ここでのJSONはensure_ascii=False、separators=(",", ":")とし、IDのソートはUnicodeコードポイント順とする。
既知の関係がなければgroup_id="source:"+source_id、grouping_basis="source_id"とする。
この場合、検証済みなのは原稿ID単位の分離までであり、未知の本文重複や書籍をまたぐ汎化まで保証しない。

既存Bのvalid原稿を含むgroupをholdoutにする。
そのgroup内にoriginal_split=trainの別原稿があれば、その原稿もholdoutへ移し、split_reason=group_contains_validとして記録する。
元のvalid原稿はsplit_reason=original_validとする。
BRIEFでは50原稿だが、実ファイルのID集合と、このgroup規則適用後の件数を報告する。
50に合わせるために原稿を移動しない。

残ったG個のtrain由来groupから、ceil(0.1×G)個をdevにする。
順序はSHA-256(UTF-8("20260907" + "\n" + group_id))の16進文字列の昇順、同値ならgroup_id順とする。
先頭をdev、残りをtrainにする。
G<2でtrain/devの両方を確保できなければ、その問題を報告して学習集合の確定を止める。
split_reasonはそれぞれhash_dev、remaining_trainとする。

長さ超過原稿もsources配列に残し、splitを保ったままeligible=falseとする。
除外後にdevの数合わせや分割のやり直しを行わない。
いずれかのsplitが除外後に空になった場合は、その旨を報告して初回比較を開始しない。
3条件、全seedで同じmanifestのハッシュを使う。

既存の50原稿は過去のモデル選択にも使われているので、プロジェクト全体に対する未使用の最終テストとは呼ばない。
この探索で差が確認された後の確証には、別途固定した未使用原稿が必要となる。
今回の初回実験を、その追加収集の完了まで止める必要はない。

Dとこの生成実験の学習データは初回では混ぜない。
後でDへの表現転用を評価する場合は、Dの留保原稿・包含範囲がB学習に入っていないことも確認する。

## 5. 入力と教師系列

systemメッセージは全条件で同じにする。

~~~text
入力された文章を推敲し、推敲後の本文だけを返してください。
~~~

userメッセージは以下のとおり。
「人間」「Composer」という作成元の名前や、評価ラベル、過去の人手判定はモデル入力に含めない。

b-gen-x：

~~~text
【下書き】
{x}
~~~

b-gen-y：

~~~text
【推敲候補】
{y}
~~~

b-gen-xy：

~~~text
【下書き】
{x}

【推敲候補】
{y}
~~~

assistantの教師本文は、どの条件でも同じhの全文。
編集操作、解説、理由、スコア、思考文を出力の教師に加えない。
入力にはhを含めない。

初回の学習に、(x,h)を入力してhをコピーする恒等例や、(x,x)を入力する追加例は混ぜない。
それらを混ぜる効果は別の仮説になるので、最初の3条件の比較には含めない。

## 6. トークン化と損失マスク

モデルとtokenizerは、両方とも次のrevisionを指定する。

~~~yaml
model_id: Qwen/Qwen3-8B
model_revision: b968826d9c46dd6066d109eabc6255188de91218
tokenizer_revision: b968826d9c46dd6066d109eabc6255188de91218
eos_token: <|im_end|>
eos_token_id: 151645
pad_token: <|endoftext|>
pad_token_id: 151643
~~~

上記commitは2026年9月7日に確認した公開リポジトリのrevisionである。
mainを実行のたびに解決し直さない。
ローカルの重みが別のrevisionなら、そのまま同じものと扱わず、指定snapshotを取得するか、全条件の実行前にlockを一括改訂する。
[Qwen公開リポジトリのrevision](https://huggingface.co/api/models/Qwen/Qwen3-8B)、[公式tokenizer設定](https://huggingface.co/Qwen/Qwen3-8B/raw/main/tokenizer_config.json)

Qwen3ではenable_thinking=Falseでchat templateを適用する。
準備・学習・推論で同一のbuild_prompt_ids(condition, x, y, tokenizer)関数を共有する。
学習だけ別の全文chat templateを適用する経路を作らない。
生成prefixにはテンプレート由来の空のthinkブロックが入る。
この制御部分は入力であり、hの教師本文には加えない。

実装例は次の処理を満たすこと。
これは学習スクリプト全体ではなく、教師系列とマスクの定義である。

~~~python
messages = [
    {"role": "system", "content": system_text},
    {"role": "user", "content": user_text},
]
prompt_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
)
end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
assert end_id == tokenizer.eos_token_id == 151645
assert tokenizer.encode("<|im_end|>", add_special_tokens=False) == [end_id]
assert tokenizer.pad_token_id == 151643
target_ids = tokenizer(human_text, add_special_tokens=False)["input_ids"]
target_ids = target_ids + [end_id]

input_ids = prompt_ids + target_ids
labels = [-100] * len(prompt_ids) + target_ids
attention_mask = [1] * len(input_ids)
~~~

Qwenのテンプレートが挿入する制御用prefixはprompt側に含め、損失をかけない。
モデル側のcausal shiftを使い、labelsを実装側で重ねてshiftしない。
padding位置のlabelsも-100にする。
教師系列の最後のim_endにも損失をかける。
paddingは追加した位置で判別し、トークンIDの一致による一括maskを使わない。
末尾へ二つ目のEOSや改行を自動追加しない。

model.lock.jsonにcommit、tokenizer関連ファイルのSHA-256、chat_template文字列のSHA-256、特殊トークンIDを保存する。
各runで照合し、不一致なら開始しない。
全378原稿・3条件について、準備時のprompt_idsと学習・推論用関数のprompt_idsが一致することを確認する。
この検査は本文を外部へ送らず、ローカルで行える。

損失はh本文とassistant終了トークンに対する通常の次トークン交差エントロピー。
更新単位の有効な教師トークン全体で平均する。
xやyの再生成は損失に含めない。
DPO、BT、編集距離による重み、既存評価器による重み付けはこの実験に加えない。

新規の学習処理はPyTorch、Transformers、PEFTで実装し、上記のトークン化済みinput_ids/labels/attention_maskをそのまま受け取る。
既存train_edit_sft.pyへのオプション追加では実装しない。
初回はTRLの自動データ整形も通さず、labelsを保存する専用collatorを使う。
packing=False、truncation=Falseとし、長さ適格性は7.1節で先に決める。

gradient accumulation中の長さが違う例を、単なる例ごとの損失平均にしない。
一更新分の最大16原稿をCPU側で確定し、各原稿の有効な教師トークン数をN_i、合計をT=ΣN_iとする。
モデルが返す原稿iの平均CEをloss_iとして、各microbatchでloss_i×N_i/Tをbackwardする。
N_iはcausal shift後に教師になるlabels[1:]のうち-100でない個数であり、この入力形式ではtarget_idsの長さと一致する。
一更新分を処理した後に一度だけclip、optimizer.step、scheduler.step、zero_gradを行う。
さらに16で割る処理を重ねない。
最後の端数更新にも、その実際のTを使う。
dev NLLも全有効教師トークンの合計で正規化する。

## 7. 共通の学習設定

初回比較の開始点として、以下に固定する。
これらは本データに対する最適値を主張するものではない。

~~~yaml
model_id: Qwen/Qwen3-8B
model_revision: b968826d9c46dd6066d109eabc6255188de91218
tokenizer_revision: b968826d9c46dd6066d109eabc6255188de91218
initial_adapter: none
enable_thinking: false
device: cuda:0
num_gpus: 1
bf16: true
attn_implementation: sdpa
use_cache_during_training: false

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: bfloat16

lora:
  task_type: CAUSAL_LM
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: none
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

optimizer: adamw_torch
learning_rate: 0.0001
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 0.00000001
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
num_train_epochs: 3

per_device_train_batch_size: 1
effective_batch_size: 16
gradient_accumulation_steps: 16
gradient_checkpointing: true
packing: false
max_total_tokens: 32768
save_strategy: epoch
seeds: [0, 1, 2]
~~~

初回はseed 0で3条件を通す。
実装と入力・損失に問題がない状態で、同じ設定のseed 1・2を実行する。
最良seedだけを報告しない。
seed 0の結果を見て他のseedだけ学習率やepochを変更しない。

ローカルのPro 4000一枚で、3条件を順に実行する。
gradient_accumulation_steps=16とする。
最後の端数batchも、有効な教師トークン数に応じた損失正規化を保つ。
全条件で同じ原稿順序、同じepoch数、同じ更新回数を使う。

4bitモデルはPEFTのk-bit学習準備を済ませてからLoRAを付ける。
ベースの量子化重みは固定し、各Transformer層のLoRAを学習する。
学習時に推論用のdevice_map="auto"を流用しない。
上記module名が実モデルに存在し、想定した層にLoRAが付いたかを確認する。
[量子化と追加パラメーター学習](https://huggingface.co/docs/transformers/quantization/bitsandbytes)、[LoRA設定](https://huggingface.co/docs/peft/developer_guides/lora)

AやBで既に学習したadapterを初期値には使わず、共通の公開checkpointから比較を開始する。
既存adapterを使う比較は、その学習済み原稿との重複を確認した別runとする。

torch、transformers、peft、bitsandbytes、CUDA、GPUドライバの実行時version、実際に使用したその他の依存パッケージ、モデルcommit、tokenizer/chat templateのhashを保存する。
新しいライブラリの未確認のversion番号を仕様に固定せず、実装環境で動作確認した組をlockする。

### 7.1 全378原稿の長さ計測と共通除外

学習系列の長さはsystem、x、y、assistant prefix、h、終了トークンを合わせて数える。
Qwen3-8Bの標準文脈長の範囲として、本実験では32,768トークンを上限にする。
[Qwen3-8Bの仕様とthinking切り替え](https://huggingface.co/Qwen/Qwen3-8B)

原稿i、条件c∈{x,y,xy}について、6節の関数で次を数える。

$$
P_{ic}=|\mathrm{prompt\_ids}_{ic}|,\quad
H_i=|\mathrm{target\_ids}_i|,\quad
L^{\rm train}_{ic}=P_{ic}+H_i,\quad
L^{\rm infer}_{ic}=P_{ic}+8192.
$$

H_iには教師本文末尾のim_endを含む。
文字数からの換算、x/y/hを別に数えて足すだけの近似、tokenizerによる暗黙の切り詰めを使わない。
条件ごとの長さは正式なprompt_idsから取得する。

今回の共通適格条件を次のように固定する。

$$
E_i=
\mathbf1\left[
\max_{c\in\{x,y,xy\}}
\max(L^{\rm train}_{ic},L^{\rm infer}_{ic})\le32768
\right].
$$

E_i=0の原稿は、train/dev/holdoutのどのsplitでも3条件すべてから除外する。
一条件だけ除外したり、本文を途中で切ったりしない。
群内の一原稿が超過しても、他の原稿を機械的に除外する必要はない。
groupの全原稿を同一splitに保つ規則は継続する。
除外理由はtrain_total_over_32768、inference_budget_over_32768の該当するものを両方残す。
どの条件で超えたかはlengthsから追跡できるようにする。
生成時の8,192トークン予算を条件別に減らして採用する処理は入れない。

b_length_report.jsonには、全378原稿それぞれの長さと判定、および以下の集計を保存する。

- 条件別のprompt/target/train/inference合計の最大値と分布。
- 条件別と3条件の和集合での、学習長超過数・推論予算超過数。
- split別の元原稿数、除外数、採用数、採用group数。
- 人間本文の教師系列が8,192を超える原稿数。これは参考値であり、追加の除外基準にはしない。

「378原稿すべてが収まるか」は、このレポートの実測結果が出るまで未確認である。
除外が生じた場合、得られる結論の対象はこの長さ条件を満たす原稿に限定される。

### 7.2 Pro 4000での事前実行

32,768は系列長の上限であり、GPUメモリへ収まる保証ではない。
まずCPUで7.1節までを実行し、採用された実データの長さを確定する。
次に各条件で、学習合計長が最大の例、教師本文が最長の例、推論promptが最長の例を使ってローカルGPUで確認する。
同じ例に該当するものは重複実行しなくてよい。
実際のGPU名・VRAM容量は実行環境から記録する。

学習の確認では、予定した4bit、LoRA、BF16、SDPA、gradient checkpointingの構成でforward/backwardを行い、optimizer状態が確保される最初のstepと、その後のforward/backwardまで通す。
推論の確認では、最大のpromptと8,192トークン分のKV cacheを使う上限付近のメモリも確認する。
このメモリ検査だけはEOSによる早期終了を抑えてもよいが、検査出力は品質比較へ使わない。
peak allocated/reserved memory、処理時間、終了状態をlocal_preflight.jsonに残す。
検査用に更新したadapterは本学習の初期値に使わない。

OOMになった原稿を、その都度飛ばして学習を続ける処理は入れない。
まず同じ損失・全文入力を保つ省メモリ化を検討し、変更した実装と環境を再固定する。
特に、全位置の語彙logitsとCE計算の一時領域もメモリを使うため、量子化した重みの容量だけで実行可能とは判断しない。
低い共通長上限で実験範囲を変える場合は、別manifestと別実験IDにする。
この仕様は、未測定のPro 4000で32,768トークン学習が完了すると主張しない。

### 7.3 checkpoint

初回の主比較には、全条件で3 epoch目のcheckpointを使う。
これにより、更新回数を揃えた比較になる。
各epochのdev損失も保存し、過学習の診断に使う。

別にdevのhの平均トークンNLLが最小だったcheckpointを記録するが、主比較と混ぜない。
devで選んだcheckpointの結果を追加報告する場合は、その選択規則を明示する。
NLLは参照本文hを予測する学習の診断値であり、推敲品質の評価器の点数とは扱わない。
holdoutの人手選好や既存評価器の点数を見てcheckpointを選ばない。

## 8. 推論条件と出力保存

各条件は学習時と同じ入力形式でholdoutを処理する。
hは推論入力に含めない。
学習時と同じ4bitロード方式とモデルrevisionを使う。

初回の主比較は、条件差を見やすくする固定の推論条件とする。

~~~yaml
enable_thinking: false
do_sample: false
num_beams: 1
num_return_sequences: 1
repetition_penalty: 1.0
max_new_tokens: 8192
eos_token_id: 151645
pad_token_id: 151643
use_cache: true
~~~

greedyは本比較の制御条件であり、Qwenの品質上の最適設定を意味しない。
Qwen公式はnon-thinkingでsampling設定を推奨している。
必要な生成品質比較としてsamplingを追加する場合は、全条件共通でtemperature=0.7、top_p=0.8、top_k=20、固定した生成seedを使い、greedyの結果と混ぜない。

推論入力は6節と同じprompt_idsをそのまま渡す。
追加のBOSや別のchat templateを適用しない。
公式generation_configにはEOS候補が151645と151643の二つあるが、この実験では教師末尾と揃え、停止IDを151645に明示設定する。
公開設定のsamplingパラメーターも暗黙に継承せず、実際に使ったGenerationConfigを保存する。
[公式generation_config](https://huggingface.co/Qwen/Qwen3-8B/blob/main/generation_config.json)

全条件で入力長+max_new_tokensが32,768以下となる原稿だけが7.1節で採用される。
この規則を推論時にもassertする。
生成が上限で止まった例を通常の完成出力として処理しない。
truncated=trueを残し、件数を必ず報告する。
EOSで正常終了したか、EOSなしで上限に達したかは、生成トークンIDを見て判定する。
末尾のEOSだけを本文から除き、decodeではclean_up_tokenization_spaces=Falseとする。
生成本文をstripしたり、思考文・見出しを後処理で削除したりして品質を整えない。
意図せず出力された制御トークンもraw token IDsとともに記録する。
予算変更による再推論は別runとし、初回結果を書き換えない。

全出力についてsource_id、condition、train_seed、checkpoint、generation設定、本文、token数、終了理由を保存する。
Best-of-Nや既存評価器による候補選択は主比較に挟まない。

## 9. 評価

### 9.1 主比較

下書きxを共通の文脈として提示し、作成元と条件名を伏せて本人が判定する。
評価項目への分解や判定理由は求めない。
以下を原稿ごとに比較する。

1. b-gen-xy対b-gen-x：候補yを条件に加える利益。
2. b-gen-xy対b-gen-y：元原稿xも参照する利益。
3. b-gen-xy対Composer候補y：追加の生成に実際の利益があるか。
4. b-gen-xy対b-gen-xy-base：Bによる学習の利益。

判定は左が良い、右が良い、同等、判定不能を別々に記録する。
左右は原稿・比較ごとに乱択し、その対応表は表示データと分離して保存する。
同等と判定不能を一つにまとめない。
片方が空出力・未完了なら、その失敗を除外して勝率を上げない。

以下の形式と集計規則を、判定を始める前に固定する。

#### 表示用JSONL

pairs.jsonlは一比較一行とする。以下の整形例を、実ファイルでは一行に直列化する。

~~~json
{
  "schema_version": "b-blind-pair-v1",
  "pair_id": "<条件名を含まないランダムID>",
  "context": {"draft": "<下書きx全文>"},
  "left": {"text": "<左の候補全文>"},
  "right": {"text": "<右の候補全文>"}
}
~~~

下書きは候補と別の欄に表示する。
表示用ファイルにはh、条件、seed、生成元、checkpoint、内部source_idを含めない。
本文の改行・空白を保持し、差分強調や自動整形を行わない。
作成元を本文自身から推測できる可能性は残るため、完全な盲検を保証するとは記述しない。

#### 非表示の対応表

keys.jsonlはpair_idで表示用ファイルと一対一に対応する。

~~~json
{
  "schema_version": "b-blind-key-v1",
  "pair_id": "<表示用と同じID>",
  "source_id": "<原稿ID>",
  "group_id": "<manifestのgroup_id>",
  "split": "holdout",
  "train_seed": 0,
  "comparison": "xy:x",
  "left": {"condition": "xy", "generation_id": "<出力ID>"},
  "right": {"condition": "x", "generation_id": "<出力ID>"},
  "manifest_sha256": "<manifestのハッシュ>"
}
~~~

comparisonはxy:x、xy:y、xy:composer、xy:xy-baseだけを許可する。
conditionはx、y、xy、composer、xy-baseとし、yは学習済みb-gen-y、composerは保存済み候補本文を意味する。
対応する出力台帳に、本文のSHA-256、checkpoint、終了理由、エラー内容を保持する。
Composerとgreedyのxy-baseは原稿ごとに一度用意し、全train_seedで同じgeneration_idを参照する。
比較されるxyのtrain_seedを対応表に記録し、baseに架空の学習seedを付けない。

blind_seed=20260907で左右割当と表示順を生成し、使用した乱数実装・versionもblind_config.jsonへ保存する。
対応表を確定した後は、判定の途中で並べ直した別ファイルへ差し替えない。
全seedを一度に表示しても、seedを識別できるまとまりや表示名を付けない。

#### 判定JSONLと生成失敗

~~~json
{
  "schema_version": "b-blind-judgment-v1",
  "pair_id": "<表示用と同じID>",
  "judgment": "tie"
}
~~~

judgmentはleft、right、tie、unjudgeableの4値とする。
理由は必須にせず、任意のnoteだけを許可する。
行がない比較は未判定とし、unjudgeableを自動で補わない。
未知のpair_id、重複した判定行、許可していない値は集計エラーにする。
訂正する場合は履歴を別に保存し、集計対象の確定ファイルには一比較一判定だけを残す。

長さ上限で止まった非空出力も、生成された本文のまま比較に含める。
空出力や実行エラーでも予定した比較行を消さず、text=""として表示する。
空欄のUIには候補本文の外側に「出力なし」と表示する。
失敗の技術的理由は対応表側で保持し、判定者に条件を知らせない。
これらも人手のleft/right/tie/unjudgeableで判定し、実装側で自動的な勝敗を付けない。
終了理由別の件数を集計に併記し、判定不能が増えた系を勝率だけで有利に見せない。

#### 集計式

比較種別とtrain_seedごとに、xy側から見た勝W、負L、同等T、判定不能U、未判定Rを数える。
予定数Nは、適格holdout原稿数と一致させ、N=W+L+T+U+Rをassertする。
左右の対応はkeys.jsonlから復元する。

$$
S=\frac{W+0.5T}{W+L+T},\qquad
\mathrm{coverage}=\frac{W+L+T}{N}.
$$

分母が0ならSはnullとする。
補助値W/(W+L)は別名decisive_win_rateで保存し、同様に分母0はnullとする。
W/L/T/U/R/N、coverage、生成失敗数をすべて出す。
R>0の結果は未完了の中間集計と明記する。
各seedの値に加え、3seedのSの単純平均を報告する。
いずれかのSがnullなら3seed平均もnullとし、有効seedだけの平均に置き換えない。
判定不能の選択性を確認するため、未完了でない集計では区間[(W+0.5T)/N, (W+0.5T+U)/N]も補助的に示す。
これは判定不能を全敗／全勝とした感度範囲であり、信頼区間ではない。

信頼区間を出す場合はmanifestのgroup_idを再標本化単位とする。
同じgroupの全原稿・全seed・全比較をまとめて抽出し、集計式を再計算する。
候補やseedを独立の原稿として数えない。
反復数10,000、bootstrap_seed=20260907、percentile 2.5%/97.5%とし、分母0となる反復数も報告する。
無効反復を黙って除いた区間は出さず、その場合は区間をnullにする。

hとのBLEU、ROUGE、編集距離、生成NLL、既存報酬モデルの点数を主たる品質判定に使わない。
それらを記録する場合は、模倣やコピー挙動を調べる補助情報と位置付ける。

### 9.2 yを使っているか

まず主比較のxy対xを見る。
必要な場合に、同じxyモデルへy=xを入力する追加診断を実行する。
学習時と異なる入力分布なので、この診断だけで品質理解の有無を結論しない。
無関係な別原稿をyに入れたときの変化も、入力依存性の診断にはなるが、編集の質を理解した証拠にはならない。

xyがxに勝ったとしても、任意のyに対する改善能力が得られたとは限らない。
その一般化を次に確かめる場合は、同じ留保原稿から、hを一切与えず別の固定生成器で候補y'を作る。
xyモデルの(x,y')からの出力を本人が評価する。
Bとは異なる生成元への転用として結果を別に集計する。

Cの既存候補を使う場合、候補を生成したSFTモデルが該当するhを学習していないことを既存manifestで確認する。
未確認なら、原稿とhへの未学習が分かる初期モデルから候補を作る。

### 9.3 結果から言えること

| 結果 | 解釈 |
| --- | --- |
| xyがx・y・未調整モデルより良く、Composer候補自体にも勝つ | 2本文を条件とするB学習が、当該分布で推敲を改善する証拠 |
| xyとxが同程度 | yを加える利益を確認できていない。yを無視する解とも整合する |
| xyとyが同程度でxより良い | 改善済みの候補から生成する利益はあるが、xを追加する利益は未確認 |
| NLLだけが下がり、人手選好が改善しない | hの予測・模倣の改善を、推敲品質の改善とは解釈しない |
| Composerには効くが別生成器には効かない | 特定の生成元に依存する可能性。汎用の質の学習とは未確定 |
| どの条件でも改善しない | このモデル・データ・設定での失敗。「言語化できない質は学習不能」とは結論しない |

この生成実験からlog p(h|x,y)をそのまま候補yの品質点として導出しない。
hを予測しやすい候補であることと、候補自体の質が高いことは同値ではない。

## 10. 実装するコマンドと成果物

以下はこれから実装するインターフェース名であり、既存の実行可能ファイルではない。
既存コードを利用できる部分は利用し、同じ機能を重複実装しない。
データ整形・集計は既存プロジェクトの言語を使う。
学習部分はPyTorch、Transformers、PEFTの利用を想定する。

~~~text
inspect-d-epoch7
  --code-root <既存プロジェクト>
  --checkpoint <Dのepoch7 checkpoint>
  --out outputs/d_epoch7/d_epoch7.lock.json

evaluate-d-epoch7
  --lock outputs/d_epoch7/d_epoch7.lock.json
  --d-train data/d/train.jsonl
  --d-valid data/d/valid.jsonl
  --c <既存Cの候補と人手判定>
  --out outputs/d_epoch7/evaluation

prepare-b-generation
  --a2 data/revision_corpus/keep_section.jsonl
  --composer data/section_middle/revisions.jsonl
  --pref-train data/section_middle/pref_train.jsonl
  --pref-valid data/section_middle/pref_valid.jsonl
  --dev-fraction 0.1
  --split-seed 20260907
  --model Qwen/Qwen3-8B
  --model-revision b968826d9c46dd6066d109eabc6255188de91218
  --out data/b_generation

check-b-generation-local
  --manifest data/b_generation/manifest.json
  --config configs/b_generation/common.yaml
  --out outputs/b_generation/local_preflight.json

train-b-generation
  --data data/b_generation
  --condition x|y|xy
  --seed 0|1|2
  --config configs/b_generation/common.yaml
  --out outputs/b_generation/<condition>/seed-<seed>

infer-b-generation
  --manifest data/b_generation/manifest.json
  --split holdout
  --condition x|y|xy|xy-base
  --checkpoint <checkpoint-path-or-fixed-base-model>
  --out outputs/b_generation/predictions/<condition>-<seed>.jsonl

make-b-generation-blind
  --manifest data/b_generation/manifest.json
  --predictions outputs/b_generation/predictions
  --comparisons xy:x,xy:y,xy:composer,xy:xy-base
  --blind-seed 20260907
  --out data/b_generation/blind

summarize-b-generation
  --manifest data/b_generation/manifest.json
  --pairs data/b_generation/blind/pairs.jsonl
  --keys data/b_generation/blind/keys.jsonl
  --judgments data/b_generation/blind/judgments.jsonl
  --out outputs/b_generation/results
~~~

inspect-d-epoch7は手元のコードとcheckpointを調べる作業を含む。
未知のアーキテクチャを自動推定する汎用コマンドとして実装する必要はない。
当該プロジェクト専用の復元処理と、取得根拠を持つlockを作る。
evaluate-d-epoch7は長さレポート、原稿漏洩検査、候補得点、原稿内ペア、2.2節の集計を別々に保存する。

prepare-b-generationはCPU上で結合・分割・正式templateによる全件tokenize・共通除外までを完了する。
check-b-generation-localの成功後に、train-b-generationを3条件×3seedで順次実行する。
train-b-generationは新規のprompt/labels SFTとし、既存train_edit_sft.pyへ委譲しない。
infer-b-generationは共通のprompt構築関数を使い、xy-baseも同じ量子化・推論設定で処理する。
make-b-generation-blindは必要な全出力IDを照合し、欠けた出力を隠して比較数を減らさない。
処理未実行と記録済みの生成エラーを区別し、前者があれば比較ファイルの確定を止める。

2.3節（[plan-d-gpm.md](plan-d-gpm.md)）は、2.2節の出力を確認した後に着手する。
まずペア台帳を固定し、スカラー BT 対照のあと GPM（head_dim≥32）を同じ台帳で学習する。

成果物は、D復元lockと直接評価、B/Cの長さレポート、split manifest、データ対応の検査結果、学習設定、環境lock、GPU事前実行結果、全checkpointのdev損失、選択記録、生成出力、判定用データ、対応表、人手判定、集計結果。
元のA・B・C・Dのファイルへ生成実験用の列や分割を書き戻さず、新規の中間データへ保存する。

実装の確認は、原稿の取り違え・学習漏洩・損失マスク・切り詰めという結果を無効にする要因を対象にする。
少数例のtoken decodeで、損失対象がhと終了トークンだけであることを確認する。
学習用の少数例に対して損失が下がることとLoRAに勾配があることを確認する。
この動作確認を人手の品質評価の代わりにしない。

最低限の検証例を以下に固定する。

- Dの順位式：正順・逆順・厳密同点を含む小例でmacroとmicroを手計算に照合し、無変更・同ラベル・超過・非有限値を区別する。
- 分割：同一原稿の全候補と既知の重複groupがsplitをまたがず、元validを含むgroupがholdoutになること。
- 長さ：32,768を採用、32,769を全条件共通で除外し、分割の再抽選が起きないこと。
- SFT：prepare/train/inferのprompt token IDsが完全一致し、教師本文とim_endだけがlabelsに残ること。
- 勾配累積：長さの異なる小例について、全教師tokenの合計で正規化した損失と累積更新の勾配が、同じ目的関数の参照計算と一致すること。dropout等は検証時だけ無効化する。
- ブラインド集計：左右反転してもxy側の勝敗が変わらず、同等・判定不能・未判定・空出力を含めて予定件数が保存されること。

## 11. 次の意思決定

最初に、Dの元の判定器と、後段でA2/Bを学んだ現在の評価器を区別した結果を得る。
同時期の生成実験として、Bの3条件を同一条件で比較する。
両実験の点数を相互の正解にはしない。

生成実験で利益が確認された後、B学習済みモデルの潜在表現がDの選好学習を助けるかを検証する余地がある。
その場合、未調整モデルとB学習済みモデルに同じ選好ヘッドを付け、Dの同じ原稿分割と同じ更新条件で比較する。
Bのhを生成できたことだけで「質の表現を獲得した」とせず、別の判断への転用を確かめる。
この転用実験は、BとDの原稿・本文範囲の重複を解決した後に実施する。
