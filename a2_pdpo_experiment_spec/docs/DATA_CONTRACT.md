# DATA_CONTRACT.md

## A2

A2は379件の固定データセットとして扱う。

最低限、各項目に次を持たせる。

```json
{"id":"A2-0001","draft":"...","human_revision":"..."}
```

元データに別のIDがある場合はそれを保持する。IDがない場合だけ `A2-0001` ... `A2-0379` を割り当てる。

## split

splitはA2 item単位で一度だけ作る。

- train 303
- dev 38
- test 38

seed: 20260914

`data/A2/split_manifest.jsonl` を作り、以後固定する。

同一原稿の別versionがあることが判明した場合は、同じsplitへまとめる。その調整をした場合はmanifestへ理由を記録する。

## generic revision

各A2 itemから3件生成する。

保存先:

```text
data/generated/{split}/{a2_id}/sample-0.json
data/generated/{split}/{a2_id}/sample-1.json
data/generated/{split}/{a2_id}/sample-2.json
```

生成テキストだけでなく、model identifier、seed、sampling parameters、prompt hash、生成日時、llama.cpp commit/versionを保存する。

## preference records

train/devの各A2 itemについて3行作る。

```json
{
  "a2_id": "A2-0001",
  "split": "train",
  "draft": "...",
  "chosen": "...human_revision...",
  "rejected": "...generic_revision...",
  "rejected_sample_index": 0,
  "source_weight": 0.3333333333
}
```

`source_weight` の合計は各A2 itemについて1になること。

generic revisions同士の関係を表すrecordを作ってはいけない。draftをchosen/rejectedへ入れてはいけない。
