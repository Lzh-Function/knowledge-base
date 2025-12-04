## これは何
transformersを利用したモデルの構築、trainerの利用、推論・評価の基本形を理解してく

Gemini 3.0 Pro先生

## 基本の流れを教えてくれ
Hugging Face `transformers` ライブラリを使用したモデル開発フローにおいて、ユーザーが意識すべき「クラス継承」と「主要メソッド」に焦点を当てて解説します。

一般的に、Hugging Faceを利用する際、以下の3つのコンポーネントが重要になります。

1.  **データセット (Dataset):** データの読み込みと前処理。
2.  **モデル (Model):** ニューラルネットワークの定義。
3.  **トレーナー (Trainer):** 訓練ループと評価の抽象化。

これらを実装する際に、継承すべきクラスとその主要メソッドを整理し、その後に実際のコード例を提示します。

-----

### 1\. 主要なクラスとメソッドの構造

Hugging Faceのエコシステムで開発を行う際、特に継承（サブクラス化）を意識するのは `Dataset` クラスです。高度なカスタマイズを行う場合は `Trainer` や `PreTrainedModel` も継承します。

#### A. データセット (継承元: `torch.utils.data.Dataset`)

独自のデータをモデルに入力できる形式に変換するために継承します。

| メソッド名 | 役割 | 実装必須 | 詳細 |
| :--- | :--- | :---: | :--- |
| `__init__` | 初期化 | Yes | データ（テキストやラベル）とトークナイザーを受け取り、保存します。 |
| `__len__` | 長さの返却 | Yes | データセットの総サンプル数を返します。 |
| `__getitem__` | データの取得 | Yes | 指定されたインデックス `idx` のデータを取得し、トークナイズしてテンソル形式で返します。 |

#### B. モデル (継承元: `transformers.PreTrainedModel`)

*※通常は `AutoModelForSequenceClassification.from_pretrained()` 等を使用するため、自作する必要はありませんが、構造を理解するために重要です。*

| メソッド名 | 役割 | 実装必須 (自作時) | 詳細 |
| :--- | :--- | :---: | :--- |
| `__init__` | レイヤー定義 | Yes | BERT等のバックボーンと、分類用のヘッド（Linear層など）を定義します。 |
| `forward` | 順伝播 | Yes | 入力（`input_ids`, `attention_mask`）を受け取り、モデルに通して `logits` (や `loss`) を返します。 |

#### C. トレーナー (継承元: `transformers.Trainer`)

*※標準の学習ループで十分な場合は継承不要です。独自の損失関数(Loss Function)を使いたい場合に継承します。*

| メソッド名 | 役割 | 実装必須 (自作時) | 詳細 |
| :--- | :--- | :---: | :--- |
| `compute_loss` | 損失計算 | No (Override用) | 特別な損失関数や重み付けを行いたい場合にオーバーライドします。 |

-----

### 2\. 実装フローの例 (テキスト分類)

以下は、PyTorchの `Dataset` を継承してデータセットを作成し、標準的な `AutoModel` と `Trainer` を使用して訓練・評価を行うフローです。

#### Step 1: 必要なライブラリのセットアップ

```python
# Install necessary libraries
# pip install transformers datasets torch scikit-learn

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
```

#### Step 2: データセットクラスの作成 (継承のポイント)

ここでは `torch.utils.data.Dataset` を継承します。

```python
class CustomTextDataset(Dataset):
    """
    Custom Dataset class inheriting from torch.utils.data.Dataset.
    This class handles tokenization and input formatting.
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Store the data and tokenizer
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Return the total number of samples
        return len(self.texts)

    def __getitem__(self, idx):
        # Retrieve the text and label at the specified index
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text using the HuggingFace tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return PyTorch tensors
        )

        # Return a dictionary compatible with the model's forward method
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
```

#### Step 3: データ準備とモデルのロード

モデル自体は既存の事前学習済みモデル（BERTなど）を利用するのが一般的です。

```python
# Sample data (dummy data for demonstration)
train_texts = ["I love this movie!", "This is terrible.", "Absolutely fantastic.", "Not my cup of tea."]
train_labels = [1, 0, 1, 0] # 1: Positive, 0: Negative

eval_texts = ["Good logic.", "Bad implementation."]
eval_labels = [1, 0]

# Load Pretrained Tokenizer (e.g., BERT base uncased)
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create Dataset instances
train_dataset = CustomTextDataset(train_texts, train_labels, tokenizer)
eval_dataset = CustomTextDataset(eval_texts, eval_labels, tokenizer)

# Load Pretrained Model for Sequence Classification
# num_labels=2 for Binary Classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
```

#### Step 4: 評価指標の定義

`Trainer` に渡すための関数です。予測結果と正解ラベルを比較します。

```python
def compute_metrics(pred):
    """
    Compute metrics for evaluation.
    Args:
        pred: An object containing predictions and label_ids.
    Returns:
        A dictionary of metrics.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

#### Step 5: 訓練設定と実行 (Trainerの利用)

`Trainer` クラスを使用することで、学習ループ（エポック回し、バッチ処理、Backwardパスなど）を記述せずに済みます。

```python
# Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=3,              # Total number of training epochs
    per_device_train_batch_size=2,   # Batch size per device during training
    per_device_eval_batch_size=2,    # Batch size for evaluation
    warmup_steps=10,                 # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # Evaluate at the end of each epoch
    save_strategy="epoch",           # Save checkpoint at the end of each epoch
    load_best_model_at_end=True,     # Load the best model at the end of training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                     # The instantiated 🤗 Transformers model to be trained
    args=training_args,              # Training arguments, defined above
    train_dataset=train_dataset,     # Training dataset
    eval_dataset=eval_dataset,       # Evaluation dataset
    compute_metrics=compute_metrics  # The callback that computes metrics of interest
)

# Start Training
print("Starting training...")
trainer.train()

# Evaluate the model
print("Starting evaluation...")
eval_result = trainer.evaluate()
print(f"Evaluation results: {eval_result}")

# Save the final model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")
```

-----

### まとめ

  * **継承必須:** `torch.utils.data.Dataset`
      * `__len__` と `__getitem__` を実装し、モデルが理解できる形式（Input IDs, Attention Maskなど）でデータを返すようにします。
  * **継承推奨 (カスタマイズ時のみ):** `transformers.Trainer`
      * 独自の損失関数 (`compute_loss`) を定義したい場合にサブクラス化します。
  * **通常は継承不要:** `PreTrainedModel`
      * 既存のアーキテクチャ (`BertForSequenceClassification` 等) を使うのが基本です。

もし、損失関数をカスタマイズしたい場合（例：不均衡データに対する重み付けなど）の `Trainer` の継承方法についても詳しく知りたい場合は、お知らせください。

## カスタイマイズしたいんだが？
`from_pretrained`（既存の重み）を使わず、アーキテクチャ定義から自分で行う場合（いわゆるスクラッチ学習や独自モデル開発）、**「設定 (Configuration)」** と **「モデル定義 (Model Definition)」** のクラス継承が追加で必要になります。

Hugging Faceのエコシステムに乗っ取りつつ、完全なカスタムモデルを作るための構成要素とフローを解説します。

-----

### 1\. 必要なクラスと役割の全体像

スクラッチでモデルを作る場合、以下の3つのクラスを連携させます。

| コンポーネント | 継承する親クラス | 役割 | 主な実装メソッド |
| :--- | :--- | :--- | :--- |
| **設定 (Config)** | `transformers.PretrainedConfig` | ハイパーパラメータ（層の数、隠れ層のサイズなど）の管理と保存。 | `__init__` |
| **モデル (Model)** | `transformers.PreTrainedModel` | ニューラルネットワーク本体の定義。`Trainer` と連携するためのインターフェース。 | `__init__`, `forward`, `_init_weights` |
| **データセット** | `torch.utils.data.Dataset` | （前回と同じ）データの供給。 | `__len__`, `__getitem__` |

-----

### 2\. 各クラスの詳細と実装ポイント

#### A. 設定クラス (`PretrainedConfig` の継承)

モデルの構造を決定するパラメータを定義します。これを継承することで、モデルと一緒に設定ファイル（`config.json`）として保存・読み込みが可能になります。

  * **`model_type`**: モデルの種類を識別する文字列（任意）。
  * **`__init__`**: デフォルト値を設定し、親クラスに渡します。

#### B. モデルクラス (`PreTrainedModel` の継承)

ここが核心部分です。PyTorchの `nn.Module` の代わりに `PreTrainedModel` を継承することで、Hugging Faceの `Trainer` や `save_pretrained` メソッドが使えるようになります。

  * **`config_class`**: 上で作ったConfigクラスを紐付けます。
  * **`__init__(self, config)`**: `config` を受け取り、レイヤー（`nn.Linear` や `nn.Transformer` 等）を定義します。最後に `self.post_init()` を呼んで重みを初期化します。
  * **`_init_weights(self, module)`**: 重みの初期化ロジック（XavierやHeの初期化など）を記述します。`post_init()` から自動で呼ばれます。
  * **`forward(self, input_ids, labels=...)`**:
      * 計算を行い、`labels` が渡された場合は内部で **Loss（損失）** を計算します。
      * 戻り値は `transformers.modeling_outputs.SequenceClassifierOutput` などの専用クラス、または辞書形式で返すと `Trainer` がスムーズに動きます。

-----

### 3\. 実装コード例

ここでは、シンプルな「Embedding + 簡単なEncoder + 分類器」という構造のカスタムモデルを定義し、ランダムな重みから学習させる例を示します。

#### Step 1: ConfigとModelの定義

```python
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# 1. Define Custom Configuration
class SimpleCustomConfig(PretrainedConfig):
    model_type = "simple_custom"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=256,
        num_labels=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels

# 2. Define Custom Model
class SimpleCustomModel(PreTrainedModel):
    config_class = SimpleCustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Define layers based on config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = nn.Linear(config.hidden_size, config.hidden_size) # Simplified encoder
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        # Initialize weights (Standard HF method)
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize the weights. This method is called by post_init().
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids=None, labels=None, **kwargs):
        """
        Forward pass. 
        Note: The arguments must match what the Dataset/Collator produces.
        """
        # 1. Model Logic
        x = self.embeddings(input_ids) # [batch, seq_len, hidden]
        x = torch.mean(x, dim=1)       # Simple pooling: [batch, hidden]
        x = self.encoder(x)
        x = self.activation(x)
        logits = self.classifier(x)    # [batch, num_labels]

        # 2. Loss Calculation
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.view(-1))

        # 3. Return Output Object (Required for Trainer)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )
```

#### Step 2: データセットの準備 (前回同様)

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
```

#### Step 3: 実行フロー (Configから初期化 -\> Train)

`from_pretrained` は使わず、`Config` オブジェクトを作成し、それをモデルのコンストラクタに渡して初期化します。

```python
from transformers import Trainer, TrainingArguments, AutoTokenizer

# --- Setup Data (Dummy) ---
train_data = [("This is good", 1), ("This is bad", 0), ("Great work", 1)]
# We still use a pretrained tokenizer for text processing, 
# but the model itself is from scratch.
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = CustomDataset(train_data, tokenizer)

# --- Initialize Model from Scratch ---
# 1. Create Config
config = SimpleCustomConfig(vocab_size=tokenizer.vocab_size, hidden_size=128, num_labels=2)

# 2. Initialize Model with Config (Random Weights)
model = SimpleCustomModel(config)

print(f"Model initialized with hidden size: {model.config.hidden_size}")

# --- Training ---
training_args = TrainingArguments(
    output_dir="./scratch_model",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_steps=1,
    use_cpu=True # Use GPU if available
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# --- Save ---
# This saves both config.json and pytorch_model.bin
model.save_pretrained("./my_scratch_model")
```

### まとめ：カスタムモデル作成の要点

1.  **`PretrainedConfig` を継承**: 自分のモデル専用の設定クラスを作る。
2.  **`PreTrainedModel` を継承**:
      * `__init__` でレイヤー定義と `post_init()` (重み初期化) を呼ぶ。
      * `forward` で `loss` を計算し、`SequenceClassifierOutput` (または `loss` を含む辞書/タプル) を返すようにする。
3.  **初期化方法**: `from_pretrained` ではなく、`MyModel(config)` としてインスタンス化する。

これで、Hugging Faceの便利な機能（Trainer、保存/ロード、Hubへのアップロードなど）を維持したまま、中身が完全にオリジナルのモデルを作成・学習できます。

もし、モデルの中身（Attention機構など）の実装詳細や、既存のBERTなどのコードをベースに改造する方法に興味があれば、お知らせください。

## なんかPreTrainedModelの初期化特殊っぽくね？

（参考：[PreTrainedModel](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L1044)）

おっしゃる通り、`PreTrainedModel` の重み初期化（Weight Initialization）は、通常の PyTorch (`nn.Module`) の書き方とは少し異なる **「お作法」** があります。

具体的には、**`post_init()`** と **`_init_weights()`** という2つのメソッドが鍵になります。

通常のPyTorchでは `model.apply(init_fn)` を手動で呼ぶことが多いですが、Hugging Faceではこれをクラス構造の中に組み込んでいます。

### 1\. 仕組みの概要

`PreTrainedModel` を継承したクラスでは、以下のフローで初期化が行われます。

1.  **`__init__`**: レイヤーを定義する（ここは通常通り）。
2.  **`post_init()`**: `__init__` の**最後**に必ず呼び出すメソッド。
3.  **`_init_weights()`**: `post_init()` の内部から自動的に呼び出されるメソッド。**ここに具体的な初期化ロジック（正規分布の標準偏差など）を書きます。**

### 2\. なぜ特殊なのか？（PyTorch標準との違い）

PyTorchの標準的なレイヤー（`nn.Linear`など）は、作成された時点でデフォルトの初期化（Kaiming初期化など）が行われています。しかし、BERTやGPTなどのTransformerモデルは、学習を安定させるために **「平均0、標準偏差0.02の切断正規分布」** など、非常に特定の初期化分布を必要とします。

そのため、Hugging Faceでは「とりあえずモデルを作ってから、最後に全部まとめてHugging Face流に初期化し直す」という処理を `post_init()` で強制的に行っています。

### 3\. 実装パターン

カスタムモデルを作る際、以下のコードブロックのように実装します。

```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

class MyCustomModel(PreTrainedModel):
    config_class = PretrainedConfig # Link to your config

    def __init__(self, config):
        super().__init__(config)
        
        # 1. Define layers (Standard PyTorch)
        # At this point, these layers have PyTorch's default weights
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        # 2. Call post_init() at the VERY END of __init__
        # This triggers the weight initialization logic defined in _init_weights
        self.post_init()

    def _init_weights(self, module):
        """
        This method defines HOW to initialize each layer type.
        It is automatically called by post_init() for every submodule.
        """
        # Initialize Linear layers
        if isinstance(module, nn.Linear):
            # Transformers usually utilize a normal distribution with mean 0 and std 0.02
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
        # Initialize Embedding layers
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # LayerNorm is typically initialized with weight=1, bias=0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
```

### 4\. 重要なポイント

1.  **`_init_weights(self, module)` をオーバーライドする:**
    親クラス（`PreTrainedModel`）にもデフォルトの実装はありますが、自分で作ったレイヤー構成に合わせてカスタマイズしたい場合は、このメソッドを再定義します。

2.  **`self.post_init()` を忘れない:**
    `__init__` メソッドの**最後**にこれを書かないと、せっかく定義した `_init_weights` が実行されず、PyTorchのデフォルト初期値のまま学習が始まってしまいます（学習が収束しない原因になります）。

3.  **再帰的に適用される:**
    `post_init()` は内部で `self.apply(self._init_weights)` を実行しています。これにより、モデル内のすべてのサブモジュール（定義したレイヤーすべて）に対して `_init_weights` が適用されます。

### まとめ

  * **`_init_weights`**: 「**どうやって**初期化するか」のルールブックを書く場所。
  * **`post_init`**: 「**いつ**初期化するか」のスイッチ。`__init__`の最後に押す。

この2つをセットで使うのが、Hugging Face流のモデル作成ルールです。

## Validationはどう挟まれるん？evaluationと一緒？
結論から言うと、**Hugging Faceの `Trainer` を使っている場合、学習中のValidationは内部的に `evaluate()` メソッドを呼び出しているため、実質的に「同じもの」と考えて大丈夫です。**

ただし、「いつ、どのように呼び出すか」という制御の方法が異なります。

### 1\. Trainerを利用する場合（自動化）

`Trainer` を使う場合、学習中のValidationは **`TrainingArguments` の設定だけ** で制御します。自分でループを書く必要はありません。

`Trainer` は設定されたタイミング（エポックごとか、ステップごとか）が来ると、学習を一時中断し、内部で `self.evaluate()` を実行して、その結果をログに記録したり、ベストモデルの保存判定を行ったりします。

**設定方法 (`TrainingArguments`):**

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    
    # --- ここでValidationのタイミングを制御 ---
    eval_strategy="steps",      # "steps": ステップごと, "epoch": エポックごと, "no": しない
    eval_steps=500,             # 500ステップごとにValidation実行 (eval_strategy="steps"の場合)
    
    # --- その他の関連設定 ---
    per_device_eval_batch_size=16, # 評価時のバッチサイズ
    load_best_model_at_end=True,   # 学習終了時に、Validationで一番良かったモデルを読み込む
    metric_for_best_model="accuracy", # ベストモデル判定に使う指標
    save_total_limit=2,            # 保存するチェックポイントの最大数
)

# Trainer初期化時に eval_dataset を渡しておくことが必須
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,     # これがないとValidationできない
    compute_metrics=compute_metrics
)
```

**`evaluate()` との関係:**

  * 学習中の自動Validation: `Trainer` が `args.eval_steps` に到達するたびに `evaluate()` を呼び出します。
  * 手動実行: 学習完了後に `trainer.evaluate()` を呼ぶと、最終的なモデルで評価が行われます。

-----

### 2\. カスタムループ（スクラッチ）の場合（手動実装）

`Trainer` を使わず、自分で `for epoch in range(epochs):` を書く場合は、**明示的に「訓練モード」と「評価モード」を切り替える**必要があります。

ここで `evaluate()` に相当する処理（Validationループ）を自分で挟み込みます。

**実装のパターン:**

```python
import torch

# オプティマイザなどの定義
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    # ==========================
    # 1. 訓練フェーズ (Training)
    # ==========================
    model.train()  # 重要: 訓練モードへ (Dropout有効化, BatchNorm更新など)
    
    for batch in train_dataloader:
        # ... (順伝播、Loss計算、逆伝播) ...
        optimizer.step()
        optimizer.zero_grad()

    # ==========================
    # 2. 検証フェーズ (Validation) -> これが evaluate() の中身に相当
    # ==========================
    model.eval()   # 重要: 評価モードへ (Dropout無効化など)
    
    val_loss = 0
    all_preds = []
    all_labels = []

    # 勾配計算をしない（メモリ節約・高速化）
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, labels=labels)
            
            # Lossの集計
            val_loss += outputs.loss.item()
            
            # 予測結果の収集（後でAccuracyなどを計算するため）
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 指標の計算
    epoch_accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: Val Loss = {val_loss/len(val_dataloader)}, Accuracy = {epoch_accuracy}")

    # 次のエポックのために再度訓練モードに戻す（忘れがち！）
    model.train() 
```

### まとめ

  * **同一視していいか？**: はい。機能としては「評価データに対して推論し、指標を計算する」という点で同じです。
  * **Trainerの場合**: `eval_strategy` を設定するだけで、Trainerが勝手に `evaluate()` を呼び出してくれます。
  * **カスタムループの場合**: `model.eval()` と `torch.no_grad()` で囲んだブロックを作り、それをエポックの終わりなどに挟み込みます。

検証の結果を見て「学習を早めに打ち切る（Early Stopping）」などの処理を入れたい場合は、このValidationの結果を使って判断することになります。Early Stoppingの実装方法についても知りたい場合はおっしゃってください。

## どうEarlyStoppingを入れる？
Early Stopping（早期終了）を実装する方法は、**`Trainer` を使う場合**と、**カスタムループ（スクラッチ）の場合**でアプローチが異なります。

Hugging Faceの `Trainer` を使う場合は非常に簡単（コールバックを追加するだけ）ですが、カスタムループの場合はロジックを自分で記述する必要があります。

-----

### 1\. Trainerを使用する場合 (推奨)

`transformers` ライブラリには、すでに `EarlyStoppingCallback` というクラスが用意されています。これを `Trainer` に渡すだけで実装可能です。

**重要なポイント:**
ただコールバックを渡すだけでなく、**「何を基準にベストとするか」** を `TrainingArguments` で指定する必要があります。

#### 実装コード

```python
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# 1. TrainingArguments の設定（必須項目あり）
training_args = TrainingArguments(
    output_dir="./results",
    
    # --- 検証の設定 ---
    eval_strategy="steps",       # "epoch" でもOK。検証を実行するタイミング。
    eval_steps=500,              # 何ステップごとに検証するか。
    
    # --- ベストモデルの保存と読み込み（必須） ---
    load_best_model_at_end=True, # 学習終了時、最も良かったモデルを自動でロードする設定
    metric_for_best_model="eval_loss", # 監視する指標（"eval_accuracy"なども可）
    greater_is_better=False,     # lossならFalse（低い方が良い）、accuracyならTrue（高い方が良い）
    save_total_limit=1,          # 容量節約のため、保存するチェックポイントを最新/ベストのみに制限
)

# 2. Trainer に EarlyStoppingCallback を渡す
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics, # accuracy等を使う場合は定義が必要
    
    # ここに追加！
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0)]
)

trainer.train()
```

#### パラメータの説明

  * **`early_stopping_patience=3`**:
      * 「改善が見られなくなってから、あと何回（evalの回数）我慢するか」です。
      * 例：`3` に設定し、検証ロスが `0.5 -> 0.4 -> 0.42 -> 0.45 -> 0.46` と推移した場合、3回連続で改善しなかった時点でストップします。
  * **`load_best_model_at_end=True`**:
      * **超重要です。** これを `True` にしないと、学習は途中で止まりますが、手元に残るモデルは「最後に学習した（過学習し始めているかもしれない）モデル」になってしまいます。`True` にすると、過去のベスト時点の重みを復元してくれます。

-----

### 2\. カスタムループ（スクラッチ）の場合

自分で `for` ループを書いている場合は、Early Stoppingを管理するクラスを自作して、検証ループの後に呼び出します。

#### 管理用クラスの作成例

```python
import numpy as np
import torch

class EarlyStopper:
    def __init__(self, patience=3, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # ここでベストモデルの保存フラグを立てても良い
            return False # ストップしない（改善した）
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True # ストップする（我慢の限界）
            return False
```

#### ループへの組み込み方

```python
# インスタンス化
early_stopper = EarlyStopper(patience=3, min_delta=0.01)

for epoch in range(num_epochs):
    # ... (Train Loop) ...
    
    # ... (Validation Loop) ...
    # val_loss = ... (計算結果)

    print(f"Epoch {epoch}: Val Loss {val_loss}")

    # Early Stopping の判定
    if early_stopper(val_loss):
        print("Early stopping triggered!")
        break # ループを抜ける
    
    # ベストモデルの保存ロジックは別途必要
    # （EarlyStopper内で管理するか、ここで early_stopper.counter == 0 の時に保存する）
    if early_stopper.counter == 0:
         torch.save(model.state_dict(), 'best_model.pth')
```

### まとめ

  * **Trainerを使うなら:** `EarlyStoppingCallback` をリストに入れて `Trainer` に渡すだけ。`TrainingArguments` の `load_best_model_at_end=True` を忘れずに。
  * **自作ループなら:** `patience`（我慢する回数）と `best_score` を記録する簡単なクラスを作り、Validationの直後にチェックして `break` させます。

## Callbackはどのようなものがある？
（参考：[Callback](https://huggingface.co/docs/transformers/ja/main_classes/callback)）

Hugging Faceの `Trainer` クラスで使用できるCallback（コールバック）は、`transformers.TrainerCallback` クラスを継承して作成します。

これらのメソッドをオーバーライドすることで、学習ループの「特定の瞬間」にカスタム処理を差し込むことができます。

以下に、実行される順序や階層構造を意識したタイミング一覧と、それぞれの用途をまとめました。

-----

### 1\. タイミング一覧表

実行頻度の高い順（ステップ単位）から低い順（学習全体）へ分類しています。

| メソッド名 | 発火タイミング | 主な用途・実装例 |
| :--- | :--- | :--- |
| **ステップ（バッチ）単位** | | |
| `on_step_begin` | 1ステップ（バッチ処理）の**開始直前** | GPUメモリの監視、特定のステップでの入力データの操作など。 |
| `on_substep_end` | 勾配蓄積（Gradient Accumulation）の**各ミニバッチ終了時** | 勾配クリッピング前の詳細な監視など（あまり使いません）。 |
| `on_step_end` | 1ステップの処理（順伝播・逆伝播・重み更新）が**完了した後** | **最もよく使われます。** 損失(Loss)の監視、NaNのチェック、学習率の記録など。 |
| **エポック単位** | | |
| `on_epoch_begin` | 新しいエポックの**開始時** | エポックごとのデータシャッフルの確認、特定エポックでのパラメータ凍結解除など。 |
| `on_epoch_end` | エポックの**終了時** | エポック単位の統計情報のログ出力、カスタム評価の実行など。 |
| **学習全体・イベント** | | |
| `on_train_begin` | `trainer.train()` が呼ばれて、学習ループに入る**直前** | 外部ロガー（WandBなど）の初期化、独自変数のリセット。 |
| `on_train_end` | 全ての学習工程が**完了した後** | モデルの最終保存、終了通知の送信、リソースの解放。 |
| `on_evaluate` | 検証（Validation）ループが**完了した後** | 検証結果(`metrics`)に基づいた判断、推論結果の可視化。 |
| `on_save` | モデルのチェックポイントが**保存された直後** | 保存されたモデルをS3へアップロード、古いモデルの削除ロジックなど。 |
| `on_log` | `logging_steps` ごとにログが記録された時 | 標準ログ以外の場所への送信、特定条件下でのアラート。 |
| `on_prediction_step` | `predict` や `evaluate` 中の1ステップごと | (※これは `Trainer` を大幅改造しない限り呼ばれにくい特殊なフックです) |

-----

### 2\. メソッドが受け取る引数（情報の取り出し方）

すべてのCallbackメソッドは、共通して以下の引数を受け取ります。これらを使って現在の状態を確認したり、学習を操作したりします。

1.  **`args`** (`TrainingArguments`):
      * 設定値（バッチサイズ、学習率、保存パスなど）へのアクセス。
2.  **`state`** (`TrainerState`):
      * **現在の状況**を知るためのオブジェクト。
      * `state.global_step`: 現在の総ステップ数
      * `state.epoch`: 現在のエポック数（小数点で表現される）
      * `state.log_history`: これまでのログ履歴
3.  **`control`** (`TrainerControl`):
      * **学習フローを操作**するためのオブジェクト。ここで「次どうするか」を指令します。
      * `control.should_training_stop = True`: 学習を強制終了させる。
      * `control.should_save = True`: 今すぐ保存させる。
      * `control.should_evaluate = True`: 今すぐ評価を実行させる。

-----

### 3\. 実装例：特定のLoss以下になったら保存して終了する

「学習中にLossが0.1を下回ったら、即座にモデルを保存して学習を打ち切る」というカスタムCallbackの例です。

```python
from transformers import TrainerCallback

class StopOnLowLossCallback(TrainerCallback):
    """
    Lossが特定の閾値を下回ったら学習を停止するコールバック
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def on_step_end(self, args, state, control, **kwargs):
        # state.log_history は辞書のリストです。最新のログを確認します。
        # 注意: logging_steps のタイミングでないとログが空の場合があります。
        if len(state.log_history) > 0:
            last_log = state.log_history[-1]
            
            # 'loss' キーが存在するか確認（evalのログにはlossがない場合もあるため）
            if 'loss' in last_log:
                current_loss = last_log['loss']
                
                if current_loss < self.threshold:
                    print(f"\nStep {state.global_step}: Loss ({current_loss}) hit threshold. Stopping!")
                    
                    # Trainerに指令を出す
                    control.should_training_stop = True # 学習停止
                    control.should_save = True          # モデル保存
        
        return control
```

### 4\. 組み込み方

作成したCallbackはリスト形式で `Trainer` に渡します。

```python
my_callback = StopOnLowLossCallback(threshold=0.05)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    # 既存のCallback（EarlyStoppingなど）と併用可能
    callbacks=[my_callback] 
)
```

このように、`TrainerCallback` を継承し、介入したいタイミングのメソッド（`on_step_end` など）だけを記述すれば、簡単に学習フローを制御できます。

## DDPの時はどうなる？
`torchrun` を使用して **DDP (Distributed Data Parallel)** を実行すると、Pythonスクリプトが **「1つのGPUにつき1つの独立したプロセス」** として起動されます。

これらはメモリを共有せず、互いに通信（NCCLバックエンドを使用）しながら同期して学習を進めます。

以下に、シングルノード・マルチノードの違い、およびコード内での具体的な変更点（DataLoader, Loss, Validation）を詳細に解説します。

-----

### 1\. 実行コマンドと構成の違い

`torchrun` は、自動的に環境変数を設定し、プロセスを立ち上げるランチャーです。

#### A. 1ノード 複数GPU (例: 1台のマシンにGPU 4枚)

通信はマシン内の高速なバス（PCIeやNVLink）で行われます。

  * **コマンド:**
    ```bash
    torchrun --nproc_per_node=4 --nnodes=1 train.py
    ```
  * **挙動:**
      * 4つのプロセスが起動します。
      * `RANK` (全体でのID) は 0, 1, 2, 3。
      * `LOCAL_RANK` (マシン内でのID) も 0, 1, 2, 3。

#### B. 複数ノード (例: 2台のマシン、各1GPU)

通信はネットワーク（EthernetやInfiniBand）経由で行われるため、**Master（代表ノード）のアドレス指定**が必須になります。両方のマシンでコマンドを叩く必要があります。

  * **ノード1 (Master) での実行:**
    ```bash
    torchrun --nproc_per_node=1 \
             --nnodes=2 \
             --node_rank=0 \
             --master_addr="192.168.1.100" \
             --master_port=1234 \
             train.py
    ```
  * **ノード2 (Worker) での実行:**
    ```bash
    torchrun --nproc_per_node=1 \
             --nnodes=2 \
             --node_rank=1 \  # ここだけ変える
             --master_addr="192.168.1.100" \
             --master_port=1234 \
             train.py
    ```
  * **挙動:**
      * 合計2プロセスが、別々のマシンで起動します。
      * ノード1のプロセス: `RANK`=0, `LOCAL_RANK`=0
      * ノード2のプロセス: `RANK`=1, `LOCAL_RANK`=0 (自分のマシン内では0番目だから)

-----

### 2\. コード内でのセットアップ (共通)

スクリプト側では、`torchrun` がセットした環境変数を読み取り、プロセスグループを初期化する必要があります。

```python
import os
import torch
import torch.distributed as dist

# 環境変数からランク情報を取得
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# GPUの指定
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# プロセスグループの初期化 (通信の確立)
dist.init_process_group(backend="nccl")

print(f"I am rank {global_rank} (local {local_rank}) of {world_size}")
```

-----

### 3\. DataLoaderとSamplerの扱い (最重要)

DDPにおいて最も重要なのが **`DistributedSampler`** です。これを使わないと、**全てのGPUが全く同じデータを学習してしまい、DDPの意味がなくなります。**

#### データの分割

`DistributedSampler` は、データセット全体を `world_size` 分割し、現在の `rank` に割り当てられた部分だけをDataLoaderに渡します。

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Dataset作成 (前回と同じ)
dataset = CustomDataset(...)

# Samplerの定義
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size, # 全プロセス数
    rank=global_rank,        # 自分のランク
    shuffle=True             # 訓練時はシャッフル
)

# DataLoaderの定義
dataloader = DataLoader(
    dataset,
    batch_size=16,           # 「各GPUごとの」バッチサイズ (全体では 16 * world_size になる)
    sampler=sampler,         # ここにセット
    shuffle=False,           # Samplerを使う場合は必ず False にする！
    num_workers=2,
    pin_memory=True
)
```

#### 学習ループでの注意点: `set_epoch`

エポックごとにシャッフルのシード値を変えるため、ループの先頭で以下を呼ぶ必要があります（これを忘れると毎回同じ順序でデータが来ます）。

```python
for epoch in range(epochs):
    # エポックごとにシャッフルパターンを変える
    sampler.set_epoch(epoch) 
    
    for batch in dataloader:
        # ... 学習処理 ...
```
（追加：全てをTrainerに一任していて、標準的なDataLoaderとSamplerを使っている場合なら[ここ](https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/trainer.py#L2390)にあるように特に気にしなくてOK）

-----

### 4\. Lossの計算とValidationの方法

#### A. モデルの定義 (DDPラッパー)

モデルをGPUに転送した後、`DistributedDataParallel` でラップします。これにより、`.backward()` 時に自動的に全GPU間の勾配が平均化（All-Reduce）されます。

```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = MyModel().to(device)
model = DDP(model, device_ids=[local_rank])
```

#### B. Lossの表示と記録

各GPUでは「自分の担当分のデータのLoss」しか計算されていません。ログを表示する際は、全GPUのLossを集計（平均）する必要があります。**そうしないと、Rank 0 の結果だけが表示され、全体の学習状況を正確に把握できません。**

```python
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # 全GPUの値を合計
    rt /= world_size # プロセス数で割って平均化
    return rt

# 学習ループ内
loss = outputs.loss
loss.backward()
optimizer.step()

# ログ表示用 (計算グラフを切って集計)
with torch.no_grad():
    avg_loss = reduce_tensor(loss)
    
if global_rank == 0:
    print(f"Average Loss: {avg_loss.item()}")
```

#### C. Validation (評価)

評価時もデータは分散されています。正確なAccuracyなどを出すには、各GPUの予測結果 (`logits`) または正解数 (`correct`) を**全GPUから集める (`all_gather`)** 必要があります。

```python
def evaluate(model, dataloader):
    model.eval()
    total_correct = torch.tensor(0.0).to(device)
    total_samples = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            preds = outputs.logits.argmax(dim=-1)
            
            # ローカルでの正解数
            total_correct += (preds == labels).sum()
            total_samples += len(labels)

    # 全GPUの結果を集計
    dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

    # Rank 0 だけが表示・保存を行う
    if global_rank == 0:
        acc = total_correct / total_samples
        print(f"Validation Accuracy: {acc}")
```

-----

### まとめ：Hugging Face Trainerを使う場合

上記のような複雑な処理（Samplerの設定、`set_epoch`、`all_reduce`、`all_gather`）は、**Hugging Faceの `Trainer` を使えば全て自動で行われます。**

ユーザーが意識するのは以下の点だけです：

1.  **スクリプト内:** `Trainer` を通常通り使う（コード変更ほぼなし）。
2.  **実行時:** `python train.py` ではなく `torchrun ... train.py` で起動する。

これだけで、内部で `DistributedSampler` が割り当てられ、勾配同期やLoss集計も適切に行われます。自分でループを書く場合は、上記の `dist` 関連の処理を全て実装する必要があります。