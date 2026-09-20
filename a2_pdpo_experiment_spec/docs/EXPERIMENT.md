# EXPERIMENT.md

## 仮説

一般的なLLMはすでに推敲能力を持つ。A2のhuman revisionと、同じdraftから一般的なローカルLLMが生成したrevisionとの差を学習すれば、一人の編集者に固有の推敲 residual を得られる。

基礎方策:

```text
pi_0(y|x)
```

編集者方策:

```text
pi_H(y|x) ∝ pi_0(y|x) exp(Delta_H(x,y))
```

この実験では `Delta_H` をeditor LoRAとして近似する。

## 学習関係

A2 item i のhuman revisionを `h_i`、ローカルLLMのsample jを `g_ij` とする。

教師信号は

```text
h_i > g_ij
```

のみ。

`g_i1`, `g_i2`, `g_i3` の相互関係は定義しない。

## モデル

base modelは凍結する。同じbase modelにeditor LoRAを付ける。

- adapter OFF: reference / generic policy
- adapter ON: editor policy

一人の編集者しか扱わないので、最初の実験では別個のuser embeddingを置かない。「personalization」はeditor LoRAそのものが担う。

## DPO objective

各pairについて

```text
s_theta(x,y) = log pi_theta^H(y|x) - log pi_0(y|x)
```

とし、

```text
L_ij = -log sigmoid(beta * [s_theta(x_i,h_i) - s_theta(x_i,g_ij)])
```

を最小化する。

3 negativesを持つsourceについては、source単位で平均する。

```text
L_i = (L_i1 + L_i2 + L_i3) / 3
```

batch内でもA2 itemごとの総重みが等しくなるようにする。

## 学習条件

学習はP-DPOのみ。base modelはreference policyとして固定し、editor LoRAだけを更新する。SFT、vanilla DPO、SFTからの二段階学習は行わない。

## length leakage

humanとgenericを、文字数やtoken数だけで分類できないかを必ず検査する。

最低限以下の特徴だけを使ったclassifierを作る。

- source chars
- revision chars
- revision/source length ratio
- sentence count
- average sentence length
- punctuation counts

これだけで高いtest accuracyが出る場合、P-DPO結果をeditorial residualと解釈しない。

P-DPO評価ではmarginとlength ratioの相関も報告する。
