from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

# 1. 训练模型
print("开始训练模型...")
model = Word2Vec(
    sentences=LineSentence("data/food_corpus.txt"),
    vector_size=64,           # 词向量维度（小数据用64足够）
    window=5,                 # 上下文窗口
    min_count=2,              # 最小词频
    workers=multiprocessing.cpu_count(),
    sg=1,                     # Skip-gram（适合小数据）
    hs=1,                     # 层次Softmax（小数据效果好）
    negative=5,               # 负采样数量
    epochs=100,               # 迭代次数（小数据多迭代）
    compute_loss=True
)

print(f"训练完成！词表大小: {len(model.wv)}")

# 2. 保存完整模型（可用于继续训练）
model.save("model/food_word2vec.model")

# 3. 只保存词向量（用于安卓部署）
model.wv.save_word2vec_format("vectors/food_vectors.txt", binary=False)

print("模型已保存:")
print("  - food_word2vec.model (完整模型)")
print("  - food_vectors.txt (词向量文件)")