# ms-swift-demo

このプロジェクトは、[MS-Swift](https://github.com/modelscope/swift) を使用して、Qwen3-8Bモデルに対する継続事前学習（Continued Pretraining）を実施するデモです。  
モデル形式の変換（Hugging Face形式⇔mcore形式）、学習、および推論まで一連の流れをサポートします。

---

### 実行環境

| 項目             | 内容                             |
|------------------|----------------------------------|
| GPU              | NVIDIA H200 × 8枚                |
| NVIDIA Driver    | 570 以降                         |

※ bfloat16はH200で正式サポートされています。

---

### リポジトリをクローン

```bash
git clone https://github.com/shanigoro48256/ms-swift-demo.git
cd ms-swift-demo
````

---

### Dockerイメージをビルド

```bash
docker build -t ms-swift-env .
```

---

### 環境変数の設定（任意）

```bash
vim ~/.bashrc
```

```bash
export HOST_WORKSPACE=/data/workspace
export CONTAINER_NAME=ms_swift_container
export HOST_MODEL_ROOT=/data/huggingface_cache
export HUGGINGFACE_TOKEN="HuggingFaceのAPIキーが入ります。"
export WANDB_API_KEY="WANDBのAPIキーが入ります。"
```

```bash
source ~/.bashrc
```

作業スペースディレクトリを作成
```bash
mkdir -p "$HOST_WORKSPACE"
sudo chmod -R 777 $HOST_WORKSPACE

sudo mkdir -p /data/huggingface_cache
sudo chmod -R 777 $HOST_MODEL_ROOT
```

---
### コンテナの起動

```bash
docker run --name $CONTAINER_NAME \
    -v $HOST_MODEL_ROOT:$HOME/.cache/huggingface \
    -v /data/workspace:/home/developer/workspace \
    --env HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    -itd --gpus=all --ipc=host \
    --privileged \
    ms-swift-env
```

---

### コンテナにログイン

```bash
docker exec -u root -it $CONTAINER_NAME /bin/bash
cd workspace/ms-swift-demo/ms-swift/
```

---

### データセットの前処理
以下の医療系データセットを使用します。
[kunishou/ApolloCorpus-ja](https://huggingface.co/datasets/kunishou/ApolloCorpus-ja)
```bash
vim create_dataset_apollo.py
```
```bash
def create_cpt_jsonl(
    dataset_name: str = "kunishou/ApolloCorpus-ja",
    output_dir: str = "data",
    output_filename: str = "apollo_cpt.jsonl",
    text_column: str = "response_ja",
):

    try:
        # 出力ディレクトリを確保
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        print(f"データセット '{dataset_name}' をロード中...")
        ds = load_dataset(dataset_name, split="train")
        print(f"ロード完了。総エントリ数: {len(ds)}")

        written = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            for i, row in enumerate(ds):
                text = row.get(text_column, "")
                if not text or not text.strip():
                    continue

                formatted = {
                    "messages": [
                        {"role": "assistant", "content": text.strip()}
                    ]
                }
                f_out.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                written += 1

            print(f"{written} 件を書き出しました → {output_path}")

    except Exception as e:
        print(f"エラー: {e}")


if __name__ == "__main__":
    create_cpt_jsonl()
```

実行
```bash
python create_dataset_apollo.py
```

---

### モデルのダウンロードおよび形式変換：Hugging Face → mcore

```bash
vim from_hf_to_mcore
```

```bash
#!/usr/bin/env bash

MODEL_ID="Qwen/Qwen3-8B-Base"  # 任意のHFモデルに変更可
BASE_NAME="$(basename "${MODEL_ID}")"
OUT_DIR="${BASE_NAME}-mcore"

CUDA_VISIBLE_DEVICES=0 \
swift export \
  --model "${MODEL_ID}" \
  --to_mcore true \
  --torch_dtype bfloat16 \
  --output_dir "${OUT_DIR}"
```

実行：

```bash
bash from_hf_to_mcore
```

---

### 継続事前学習の実行

```bash
vim run_qwen3_8b_dense.sh
```

```bash
#!/bin/bash
# run_qwen3_8b_dense.sh

RUN_ID="$(date +%Y%m%d-%H%M)"
MODEL_NAME="Qwen3-8B-Base-mcore"  # from_hf_to_mcore で出力したディレクトリ
DATA_PATH="data/apollo_cpt.jsonl"
TP_SIZE=2
GLOBAL_BS=512
MB_SIZE=2
LR=1e-5
WARMUP_ITERS=100
TRAIN_ITERS=1000
LR_MIN=1e-6
SAVE_INT=50
DS_NUMPROC=64
WANDB_PROJ="ms-swift"
LOG_INT=1

SAFE_MODEL_TAG="$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]' | tr -s ' _' '-')"
CKPT_DIR="checkpoints/${SAFE_MODEL_TAG}-${RUN_ID}"
LOG_DIR="logs/${SAFE_MODEL_TAG}-${RUN_ID}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron pt \
  --load "${MODEL_NAME}" \
  --dataset "${DATA_PATH}" \
  --tensor_model_parallel_size "${TP_SIZE}" \
  --context_parallel_size 1 \
  --sequence_parallel true \
  --micro_batch_size "${MB_SIZE}" \
  --global_batch_size "${GLOBAL_BS}" \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --train_iters "${TRAIN_ITERS}" \
  --finetune true \
  --cross_entropy_loss_fusion true \
  --lr "${LR}" \
  --lr_warmup_iters "${WARMUP_ITERS}" \
  --min_lr "${LR_MIN}" \
  --save "${CKPT_DIR}" \
  --save_interval "${SAVE_INT}" \
  --max_length 32768 \
  --truncation_strategy right \
  --num_workers 1 \
  --no_save_optim true \
  --no_save_rng true \
  --dataset_num_proc "${DS_NUMPROC}" \
  --packing true \
  --streaming true \
  --use_flash_attn true \
  --wandb_project "${WANDB_PROJ}" \
  --wandb_exp_name "${SAFE_MODEL_TAG}-${RUN_ID}" \
  --log_interval "${LOG_INT}"
```

実行：

```bash
bash run_qwen3_8b_dense.sh 2>&1 | tee run_qwen3_8b_dense.log
```

---

### モデル変換：mcore → Hugging Face形式

```bash
vim convert_mcore_to_hf
```

内容例：

```bash
checkpoint_path=./checkpoints/qwen3-8b-base-mcore-20250614-2254/v0-20250614-225451
converted_model_path=./converted_model/qwen3-8b-base-mcore-20250614-2254

CUDA_VISIBLE_DEVICES=0 \
swift export \
  --mcore_model "$checkpoint_path" \
  --output_dir "$converted_model_path" \
  --to_hf true \
  --torch_dtype bfloat16
```

実行：

```bash
bash convert_mcore_to_hf
```

---

### 推論の実行

```bash
python hf_inference_after.py
```

---

### ライセンス

商用利用には元モデルおよびMS-Swiftのライセンスに従ってください。
