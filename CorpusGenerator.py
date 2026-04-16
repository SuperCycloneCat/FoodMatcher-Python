import jieba
import random
from itertools import product

# 1. 定义食物和标签
food_tags = {
    "面条": {
        "tags": ["晴天", "天气好", "晚上", "宵夜", "面食", "热乎", "暖和"],
        "negative": ["热", "夏天", "炎热", "减肥", "冰"]
    },
    "冰淇淋": {
        "tags": ["热", "夏天", "凉爽", "甜点", "冷饮", "解暑", "高温"],
        "negative": ["冬天", "冷", "感冒", "暖胃"]
    },
    "火锅": {
        "tags": ["聚餐", "辣", "冬天", "热闹", "麻辣", "寒冷", "朋友"],
        "negative": ["夏天", "一个人", "清淡"]
    },
    "沙拉": {
        "tags": ["减肥", "健康", "清淡", "蔬菜", "低卡", "健身", "营养"],
        "negative": ["重口味", "油腻", "宵夜"]
    },
    "咖啡": {
        "tags": ["提神", "加班", "熬夜", "苦", "早上", "工作", "困"],
        "negative": ["晚上", "睡觉", "失眠"]
    },
    "烧烤": {
        "tags": ["夜宵", "啤酒", "朋友", "户外", "烟火气", "聚会", "夏天"],
        "negative": ["减肥", "健康", "清淡", "睡觉"]
    },
    "牛奶": {
        "tags": ["夜宵", "晚上", "早上", "睡觉", "晚安"],
        "negative": ["夜宵", "重口味", "油腻", "熬夜"]
    }


}

# 2. 丰富句子模板（包含正面、负面、中性）
templates = {
    "positive": [
        "今天{0}，好想吃{1}啊",
        "{0}的时候最适合吃{1}了",
        "天气{0}，来个{1}吧",
        "{0}的日子里，{1}是绝配",
        "感觉{0}，不如吃{1}",
        "{0}，我第一个想到的就是{1}",
        "{0}的季节，{1}不能少",
        "这么{0}，必须吃{1}",
        "一整天都{0}，晚上吃{1}犒劳自己",
        "{0}的心情，配上{1}刚刚好"
    ],
    "negative": [
        "虽然{0}，但还是想吃{1}",
        "{0}也不影响我对{1}的爱",
        "管它{0}不{0}，我就要吃{1}",
        "{0}了，但是{1}的诱惑挡不住",
        "就算{0}，也要吃{1}"
    ],
    "neutral": [
        "我推荐{1}，特别是{0}的时候",
        "{1}很好吃，尤其适合{0}",
        "说到{0}，我推荐{1}",
        "{1}不错，{0}的时候可以试试"
    ]
}

# 3. 生成语料（带数据增强）
corpus = []
sentence_count = 600  # 生成300条

for i in range(sentence_count):
    food = random.choice(list(food_tags.keys()))
    tags = food_tags[food]["tags"]

    # 80%概率使用正面标签，20%使用负面标签（增加模型鲁棒性）
    if random.random() < 0.8:
        tag = random.choice(tags)
        template_type = "positive"
    else:
        tag = random.choice(food_tags[food].get("negative", ["普通"]))
        template_type = "negative"

    template = random.choice(templates[template_type])

    # 随机选择第二个标签（用于更丰富的表达）
    if random.random() < 0.3:  # 30%概率使用两个标签
        tag2 = random.choice(tags)
        tag = f"{tag}又{tag2}"

    sentence = template.format(tag, food)

    # 分词
    words = " ".join(jieba.lcut(sentence))
    corpus.append(words)

# 4. 添加一些日常对话语料（让模型接触更多词汇）
common_phrases = [
    "今天 天气 真 不错",
    "我 喜欢 吃 美食",
    "中午 吃 什么 好 呢",
    "晚上 有 什么 推荐",
    "这 家店 的 口味 不错",
    "感觉 有点 饿 了",
    "想 吃 点 东西",
    "现在 有 什么 好吃 的"
]

for phrase in common_phrases:
    for _ in range(50):  # 每个短语重复50次
        corpus.append(phrase)

# 5. 保存语料
random.shuffle(corpus)
with open("data/food_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus))

print(f"生成 {len(corpus)} 条训练语料")
print("示例语料：")
for i in range(5):
    print(f"  {corpus[i]}")