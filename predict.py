import jieba
import numpy as np
from gensim.models import Word2Vec

from CorpusGenerator import food_tags

# 加载模型
model = Word2Vec.load("model/food_word2vec.model")


class FoodRecommender:
    def __init__(self, model, food_tags):
        self.model = model
        self.food_tags = food_tags

        # 计算每种食物的代表向量（标签向量的平均值）
        self.food_vectors = {}
        for food, info in food_tags.items():
            vectors = []
            for tag in info["tags"]:
                # 标签可能包含多个词
                words = jieba.lcut(tag)
                for word in words:
                    if word in model.wv:
                        vectors.append(model.wv[word])
            if vectors:
                self.food_vectors[food] = np.mean(vectors, axis=0)

    def text_to_vector(self, text):
        """将文本转换为词向量"""
        words = jieba.lcut(text)
        vectors = []
        for word in words:
            if word in self.model.wv:
                vectors.append(self.model.wv[word])

        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def recommend(self, user_input, top_n=1):
        """推荐食物"""
        input_vec = self.text_to_vector(user_input)
        if input_vec is None:
            return [("无法识别", 0.0)]

        # 计算相似度
        results = []
        for food, food_vec in self.food_vectors.items():
            # 余弦相似度
            similarity = np.dot(input_vec, food_vec) / (
                    np.linalg.norm(input_vec) * np.linalg.norm(food_vec)
            )
            results.append((food, similarity))

        # 排序返回
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def test_cases(self):
        """测试用例"""
        test_inputs = [
            "今天天气真好",
            "好热啊想降温",
            "和朋友聚餐",
            "最近在减肥",
            "熬夜加班好困",
            "想吃宵夜了",
            "冬天好冷",
            "夏天到了",
            "一个人随便吃点",
            "想要健康饮食"
        ]

        print("\n" + "=" * 50)
        print("模型测试结果")
        print("=" * 50)

        for text in test_inputs:
            recommendations = self.recommend(text, top_n=1)
            food, score = recommendations[0]
            print(f"输入: {text:15} -> 推荐: {food:8} (相似度: {score:.3f})")


# 运行测试
recommender = FoodRecommender(model, food_tags)
recommender.test_cases()

# 交互式测试
print("\n" + "=" * 50)
print("交互测试模式（输入 'q' 退出）")
print("=" * 50)

while True:
    user_input = input("\n请输入内容: ").strip()
    if user_input.lower() == 'q':
        break

    results = recommender.recommend(user_input, top_n=3)
    print("推荐结果:")
    for food, score in results:
        print(f"  {food:8} (置信度: {score:.3f})")