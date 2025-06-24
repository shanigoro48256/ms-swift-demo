from datasets import load_dataset
import json
import os   # パス結合用

def create_cpt_jsonl(
    dataset_name: str = "kunishou/ApolloCorpus-ja",
    output_dir: str = "data",
    output_filename: str = "apollo_cpt.jsonl",
    text_column: str = "response_ja",
):
    """
    指定した Hugging Face データセットから `text_column` を取り出し、
    MS-Swift CPT 用 JSONL として保存する。

    各行のフォーマット:
    {
      "messages": [
        { "role": "assistant", "content": "<テキスト>" }
      ]
    }
    """
    try:
        # 出力ディレクトリを確保
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)

        print(f"データセット '{dataset_name}' をロード中...")
        ds = load_dataset(dataset_name, split="train")  # 'train' split を使用
        print(f"ロード完了。総エントリ数: {len(ds)}")

        written = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            for i, row in enumerate(ds):
                text = row.get(text_column, "")
                if not text or not text.strip():
                    # 空行をスキップ
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
    # そのまま実行すると ApolloCorpus-ja 全体を処理し、
    # data/apollo_cpt.jsonl に保存します。
    create_cpt_jsonl()

