import json
import struct
import numpy as np
from gensim.models import Word2Vec
from CorpusGenerator import food_tags



# 1. 导出为轻量级JSON格式（适合安卓端解析）
def export_model_for_android(model, food_tags, output_dir="android_export", output_prefix="food_model"):
    """导出模型和配置供安卓使用"""

    # 保存词向量（紧凑格式）
    vectors_dict = {}
    for word in model.wv.key_to_index:
        vectors_dict[word] = model.wv[word].tolist()

    with open(f"{output_dir}/{output_prefix}_vectors.json", "w", encoding="utf-8") as f:
        json.dump(vectors_dict, f, ensure_ascii=False, indent=2)

    # 保存食物配置
    food_config = {}
    for food, info in food_tags.items():
        food_config[food] = {
            "tags": info["tags"],
            "negative": info.get("negative", [])
        }

    with open(f"{output_dir}/{output_prefix}_config.json", "w", encoding="utf-8") as f:
        json.dump(food_config, f, ensure_ascii=False, indent=2)

    print(f"已导出安卓配置文件:")
    print(f"  - {output_dir}/{output_prefix}_vectors.json")
    print(f"  - {output_dir}/{output_prefix}_config.json")


# 加载训练好的模型
model = Word2Vec.load("model/food_word2vec.model")
print(f"加载模型成功，词表大小: {len(model.wv)}")

export_model_for_android(model, food_tags)


# 2. 导出为二进制格式（更小，加载更快）
def export_binary_vectors(model, output_file="android_export/food_vectors.bin"):
    """导出二进制词向量文件"""
    with open(output_file, "wb") as f:
        # 写入头部信息
        vocab_size = len(model.wv.key_to_index)
        vector_size = model.wv.vector_size
        f.write(struct.pack("ii", vocab_size, vector_size))

        # 写入词向量
        for word in model.wv.key_to_index:
            # 写入词（UTF-8编码）
            word_bytes = word.encode("utf-8")
            f.write(struct.pack("i", len(word_bytes)))
            f.write(word_bytes)

            # 写入向量（float32）
            vector = model.wv[word].astype(np.float32)
            f.write(vector.tobytes())

    print(f"已导出二进制词向量: {output_file}")


export_binary_vectors(model)