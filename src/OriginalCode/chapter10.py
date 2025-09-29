--------------------------------------------------------------------------------------------------------------


import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# 模拟用户行为数据集
data = {
    'user_id': [1, 2, 3, 4, 5],
    'item_id': [101, 102, 103, 104, 105],
    'clicks': [5, 2, 3, 0, 1],  # 用户点击次数
    'purchases': [1, 0, 1, 0, 0],  # 是否购买 (1: 购买, 0: 未购买)
    'rating': [4.0, 3.5, 4.5, None, 2.0],  # 用户评分 (有些值缺失)
    'category': ['electronics', 'books', 'electronics', 'clothing', 'books']
}
# 将数据集转换为 DataFrame
df = pd.DataFrame(data)
# Step 1: 缺失值处理 - 使用平均值填充缺失评分
df['rating'].fillna(df['rating'].mean(), inplace=True)
# Step 2: One-Hot 编码 - 将类别特征转换为数值特征
encoder = OneHotEncoder()
category_encoded = encoder.fit_transform(df[['category']]).toarray()
# 将编码后的类别特征加入原始 DataFrame
encoded_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['category']))
df = pd.concat([df, encoded_df], axis=1).drop('category', axis=1)
# Step 3: 特征标准化 - 对数值特征进行标准化处理
scaler = StandardScaler()
df[['clicks', 'rating']] = scaler.fit_transform(df[['clicks', 'rating']])
print("处理后的特征数据：\n", df)
# Step 4: 标签生成 - 基于用户的购买行为构建标签
df['label'] = df['purchases']  # 将购买行为作为标签
df.drop('purchases', axis=1, inplace=True)  # 删除原始购买列
# Step 5: 构建训练集和测试集
X = df.drop(['user_id', 'item_id', 'label'], axis=1)  # 输入特征
y = df['label']  # 输出标签
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n训练集输入特征：\n", X_train)
print("\n训练集标签：\n", y_train)
print("\n测试集输入特征：\n", X_test)
print("\n测试集标签：\n", y_test)


--------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from surprise import Dataset, Reader, SVD, KNNBasic,accuracy
from surprise.model_selection import train_test_split
# 模拟用户-物品评分数据
data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'item_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 105],
    'rating': [5, 3, 4, 2, 4, 5, 3, 4, 4, 2]
}
df = pd.DataFrame(data)
# ------------------- 1. 数据预处理与矩阵构建 -------------------
# 构建用户-物品交互矩阵
user_item_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
print("用户-物品交互矩阵：\n", user_item_matrix)
# ------------------- 2. 基于用户的协同过滤实现 -------------------
# 计算用户之间的余弦相似度
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print("\n用户相似度矩阵：\n", user_similarity_df)
# 为指定用户生成推荐（基于用户的协同过滤）
def recommend_items_user_based(user_id, user_item_matrix, user_similarity_df, top_n=2):
    # 找到最相似的用户
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    # 聚合相似用户的评分
    recommendations = pd.Series(dtype=float)
    for similar_user in similar_users:
        user_ratings = user_item_matrix.loc[similar_user]
        recommendations = recommendations.add(user_ratings, fill_value=0)
    # 排除用户已评分的物品
    already_rated = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index
    recommendations = recommendations.drop(already_rated, errors='ignore')
    # 返回评分最高的N个物品
    return recommendations.sort_values(ascending=False).head(top_n)
print("\n基于用户的推荐：\n", recommend_items_user_based(1, user_item_matrix, user_similarity_df))
# ------------------- 3. 基于物品的协同过滤实现 -------------------
# 计算物品之间的余弦相似度
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
print("\n物品相似度矩阵：\n", item_similarity_df)
# 为指定用户生成推荐（基于物品的协同过滤）
def recommend_items_item_based(user_id, user_item_matrix, item_similarity_df, top_n=2):
    user_ratings = user_item_matrix.loc[user_id]
    # 根据已评分物品，聚合相似物品的评分
    recommendations = pd.Series(dtype=float)
    for item, rating in user_ratings.items():
        if rating > 0:
            similar_items = item_similarity_df[item] * rating
            recommendations = recommendations.add(similar_items, fill_value=0)
    # 排除用户已评分的物品
    recommendations = recommendations.drop(user_ratings[user_ratings > 0].index, errors='ignore')
    # 返回评分最高的N个物品
    return recommendations.sort_values(ascending=False).head(top_n)
print("\n基于物品的推荐：\n", recommend_items_item_based(1, user_item_matrix, item_similarity_df))
# ------------------- 4. 使用矩阵分解优化协同过滤 -------------------
# 使用SVD进行矩阵分解
svd = TruncatedSVD(n_components=2)
matrix_svd = svd.fit_transform(user_item_matrix)
print("\nSVD分解后的矩阵：\n", matrix_svd)
# ------------------- 5. 使用Surprise库处理推荐 -------------------
# 使用Surprise库构建推荐模型
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)
# 使用SVD模型训练
svd_model = SVD()
svd_model.fit(trainset)
# 在测试集上预测并评估模型
predictions = svd_model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"\nSVD模型的RMSE：{rmse}")
# 为用户生成推荐列表
def surprise_recommend(user_id, trainset, svd_model, top_n=2):
    items = trainset.all_items()
    anti_testset = [(user_id, item, 0) for item in items if trainset.ur[user_id]]
    predictions = svd_model.test(anti_testset)
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:top_n]
    return [(pred.iid, pred.est) for pred in recommendations]
print("\n基于Surprise库的推荐：\n", surprise_recommend(1, trainset, svd_model))


--------------------------------------------------------------------------------------------------------------


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# 模拟数据集：包含商品ID及其描述
data = {
    'item_id': [101, 102, 103, 104],
    'description': [
        "A high-quality smartphone with great camera",
        "A budget-friendly phone with basic features",
        "A powerful laptop for professionals",
        "A lightweight laptop for students"
    ]
}
# 构建 DataFrame
df = pd.DataFrame(data)
# 使用TF-IDF对物品描述进行向量化
vectorizer = TfidfVectorizer(stop_words='english')
item_vectors = vectorizer.fit_transform(df['description'])
# 查看TF-IDF矩阵
print("TF-IDF矩阵：\n", item_vectors.toarray())


# 计算物品之间的余弦相似度
similarity_matrix = cosine_similarity(item_vectors)
# 构建相似度矩阵的 DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=df['item_id'], columns=df['item_id'])
print("\n物品相似度矩阵：\n", similarity_df)


# 为用户推荐物品：假设用户浏览了物品101
def recommend_similar_items(item_id, similarity_df, top_n=2):
# 排除自身物品，并选择相似度最高的N个物品
    similar_items = similarity_df[item_id].sort_values(ascending=False).drop(item_id).head(top_n)
    return similar_items
# 示例：为用户推荐与物品101相似的物品
recommendations = recommend_similar_items(101, similarity_df)
print("\n与物品101相似的推荐：\n", recommendations)


--------------------------------------------------------------------------------------------------------------


OPENAI_API_KEY=your_openai_api_key
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
import pandas as pd
# 模拟用户-物品评分数据
ratings_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'item_id': [101, 102, 101, 103, 102, 104, 103, 105],
    'rating': [5, 3, 4, 2, 4, 5, 3, 4]
}
# 模拟物品特征数据
items_data = {
    'item_id': [101, 102, 103, 104, 105],
    'description': [
        "High-end smartphone with advanced camera",
        "Affordable smartphone with basic features",
        "Powerful laptop for professionals",
        "Lightweight laptop for students",
        "Noise-canceling headphones"
    ]
}
ratings_df = pd.DataFrame(ratings_data)
items_df = pd.DataFrame(items_data)


def gpt_recommendation(prompt):
    """使用OpenAI API进行自然语言推荐"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].text.strip()
# 示例：通过用户的聊天记录生成推荐
user_query = "Looking for a powerful laptop for coding and gaming."
recommendation = gpt_recommendation(f"Recommend a product based on the query: {user_query}")
print(f"GPT推荐：{recommendation}")


from sklearn.metrics.pairwise import cosine_similarity
# 构建用户-物品交互矩阵
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
# 计算物品之间的余弦相似度
item_vectors = items_df['description'].apply(lambda x: x.split())
tfidf = TfidfVectorizer()
item_features = tfidf.fit_transform(items_df['description'])
similarity_matrix = cosine_similarity(item_features)
# 基于内容的推荐
def content_based_recommend(item_id, similarity_matrix, top_n=2):
    similar_items = similarity_matrix[item_id].argsort()[::-1][1:top_n + 1]
    return items_df.iloc[similar_items]
# 示例推荐
print(content_based_recommend(101, similarity_matrix))


def hybrid_recommend(user_id, item_id, user_item_matrix, similarity_matrix, gpt_weight=0.3, cf_weight=0.4, content_weight=0.3):
    # Step 1: 基于用户的协同过滤推荐
    user_sim = cosine_similarity(user_item_matrix)
    similar_users = user_sim[user_id - 1].argsort()[::-1][1:3]  # 找到相似用户
    user_recommendations = user_item_matrix.iloc[similar_users].mean().sort_values(ascending=False)
    # Step 2: 基于内容的推荐
    content_recommendations = content_based_recommend(item_id, similarity_matrix, top_n=3)
    # Step 3: GPT自然语言推荐
    prompt = f"User is interested in item {item_id}. Recommend similar products."
    gpt_recommendation_result = gpt_recommendation(prompt)
    # Step 4: 合并推荐结果（根据权重）
    merged_recommendations = pd.Series(dtype=float)
    # 合并协同过滤推荐结果
    for item, score in user_recommendations.items():
        merged_recommendations[item] = cf_weight * score
    # 合并内容推荐结果
    for item in content_recommendations['item_id']:
        merged_recommendations[item] = merged_recommendations.get(item, 0) + content_weight
    # GPT推荐的物品分配权重
    print(f"GPT推荐的物品：{gpt_recommendation_result}")
    for item in gpt_recommendation_result.split(","):
        item = item.strip()
        merged_recommendations[int(item)] = merged_recommendations.get(int(item), 0) + gpt_weight
    # 返回最终推荐结果，按分数排序
    final_recommendations = merged_recommendations.sort_values(ascending=False).head(5)
    return final_recommendations
# 示例调用：为用户1推荐基于混合模型的物品
final_recommendations = hybrid_recommend(user_id=1, item_id=101, user_item_matrix=user_item_matrix, similarity_matrix=similarity_matrix)
print("\n混合推荐结果：\n", final_recommendations)


--------------------------------------------------------------------------------------------------------------


import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import wraps
from typing import List, Dict, Callable
import random
import time
# OpenAI API 配置
openai.api_key = "your_openai_api_key"
def gpt_completion(prompt: str) -> str:
    """调用 OpenAI 的 GPT 模型生成文本响应"""
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"GPT调用失败，正在重试...，{e}")
        return "推荐系统目前不可用"
class CollaborativeFiltering:
    """协同过滤模块，实现基于用户的推荐"""
    def __init__(self, user_item_data: pd.DataFrame):
        self.user_item_matrix = self._build_interaction_matrix(user_item_data)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
    def _build_interaction_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """构建用户-物品交互矩阵"""
        return data.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
    def recommend_user_based(self, user_id: int, top_n: int = 2) -> pd.Series:
        """基于用户的协同过滤推荐"""
        similar_users = self.user_similarity[user_id - 1].argsort()[::-1][1:]
        recommendations = pd.Series(dtype=float)
        for user in similar_users:
            recommendations = recommendations.add(self.user_item_matrix.iloc[user], fill_value=0)
        return recommendations.nlargest(top_n)
class ContentBasedRecommendation:
    """基于内容的推荐模块"""
    def __init__(self, items_data: pd.DataFrame):
        self.items_df = items_data
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.item_vectors = self.vectorizer.fit_transform(items_data['description'])
        self.similarity_matrix = cosine_similarity(self.item_vectors)
    def recommend_similar_items(self, item_id: int, top_n: int = 2) -> pd.Series:
        """基于内容的推荐"""
        similar_items = self.similarity_matrix[item_id - 101].argsort()[::-1][1:top_n + 1]
        return self.items_df.iloc[similar_items]['item_id']
class HybridRecommendation:
    """混合推荐模块，结合协同过滤、基于内容的推荐和 GPT 输出"""

    def __init__(self, cf: CollaborativeFiltering, content: ContentBasedRecommendation):
        self.cf = cf
        self.content = content
    def gpt_recommendation(self, query: str) -> List[int]:
        """调用 GPT 模型生成推荐"""
        prompt = f"根据以下用户输入推荐物品：{query}"
        gpt_response = gpt_completion(prompt)
        try:
            return [int(item.strip()) for item in gpt_response.split(",")]
        except ValueError:
            return []
    def hybrid_recommend(self, user_id: int, item_id: int, top_n: int = 3) -> pd.Series:
        """混合推荐逻辑，结合协同过滤、内容推荐和 GPT 输出"""
        user_based = self.cf.recommend_user_based(user_id, top_n)
        content_based = self.content.recommend_similar_items(item_id, top_n)
        gpt_based = self.gpt_recommendation(f"用户 {user_id} 的兴趣")
        all_recommendations = pd.Series(dtype=float)
        for item in user_based.index:
            all_recommendations[item] = 0.4 * user_based[item]
        for item in content_based:
            all_recommendations[item] = all_recommendations.get(item, 0) + 0.3
        for item in gpt_based:
            all_recommendations[item] = all_recommendations.get(item, 0) + 0.3
        return all_recommendations.nlargest(top_n)
# 性能监控装饰器
def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper
@performance_monitor
def main():
    """主程序入口，演示混合推荐系统的功能"""
    # 模拟数据
    user_item_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3, 4, 4],
        'item_id': [101, 102, 101, 103, 102, 104, 103, 105],
        'rating': [5, 3, 4, 2, 4, 5, 3, 4]
    })
    items_data = pd.DataFrame({
        'item_id': [101, 102, 103, 104, 105],
        'description': [
            "High-end smartphone with advanced camera",
            "Affordable smartphone with basic features",
            "Powerful laptop for professionals",
            "Lightweight laptop for students",
            "Noise-canceling headphones"
        ]
    })
    # 初始化各模块
    cf = CollaborativeFiltering(user_item_data)
    content = ContentBasedRecommendation(items_data)
    hybrid = HybridRecommendation(cf, content)
    # 调用混合推荐
    print("用户1的推荐结果：\n", hybrid.hybrid_recommend(user_id=1, item_id=101))
if __name__ == "__main__":
    main()


--------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_split
from sklearn.feature_extraction.text import TfidfVectorizer
import random
# -------------------- 数据准备 --------------------
# 用户-物品评分数据
ratings_data = {
    'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'item_id': [101, 102, 101, 103, 102, 104, 103, 105, 104, 105],
    'rating': [5, 3, 4, 2, 4, 5, 3, 4, 4, 2]
}
# 物品描述数据
items_data = {
    'item_id': [101, 102, 103, 104, 105],
    'description': [
        "High-end smartphone with excellent camera.",
        "Affordable phone with basic features.",
        "Laptop for gaming and professional use.",
        "Lightweight laptop for students.",
        "Noise-canceling wireless headphones."
    ]
}
# 创建 DataFrame
ratings_df = pd.DataFrame(ratings_data)
items_df = pd.DataFrame(items_data)
# -------------------- 数据预处理与矩阵构建 --------------------
# 创建用户-物品交互矩阵
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
# 使用TF-IDF向量化物品描述
tfidf = TfidfVectorizer(stop_words='english')
item_features = tfidf.fit_transform(items_df['description'])
# 计算物品相似度矩阵
item_similarity = cosine_similarity(item_features)
# -------------------- 训练：协同过滤与SVD --------------------
# 使用Surprise库的SVD进行协同过滤模型训练
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['user_id', 'item_id', 'rating']], reader)
trainset, testset = surprise_split(data, test_size=0.25)
# 训练SVD模型
svd_model = SVD()
svd_model.fit(trainset)
# 在测试集上进行预测并评估模型性能
predictions = svd_model.test(testset)
rmse = accuracy.rmse(predictions)
print(f"SVD模型的RMSE: {rmse}")
# -------------------- GPT推荐系统集成 --------------------
def gpt_recommendation(prompt):
    """GPT推荐逻辑，返回推荐结果。"""
    recommendations = {
        "Looking for a professional laptop.": "103, 104",
        "Looking for a smartphone.": "101, 102",
        "Need some good headphones.": "105"
    }
    return recommendations.get(prompt, "101, 102")
# 示例：使用GPT推荐生成结果
user_query = "Looking for a professional laptop."
gpt_result = gpt_recommendation(user_query)
print(f"GPT推荐：{gpt_result}")
# -------------------- 混合推荐系统实现 --------------------
def hybrid_recommend(user_id, item_id, user_item_matrix, item_similarity, svd_model, gpt_weight=0.3, cf_weight=0.4, content_weight=0.3):
    # Step 1: 协同过滤推荐
    svd_predictions = [svd_model.predict(user_id, iid).est for iid in user_item_matrix.columns]
    svd_recommendations = pd.Series(svd_predictions, index=user_item_matrix.columns)
    # Step 2: 基于内容的推荐
    content_recommendations = pd.Series(item_similarity[item_id - 101], index=items_df['item_id'])
    # Step 3: GPT推荐结果
    gpt_recommendations = [int(i) for i in gpt_result.split(",") if i.isdigit()]
    gpt_scores = pd.Series([1.0] * len(gpt_recommendations), index=gpt_recommendations)
    # 合并所有推荐结果并加权
    final_recommendations = (
        svd_recommendations * cf_weight +
        content_recommendations * content_weight +
        gpt_scores.reindex_like(svd_recommendations).fillna(0) * gpt_weight
    ).sort_values(ascending=False).head(5)
    return final_recommendations
# 示例调用：为用户1生成混合推荐
recommendations = hybrid_recommend(1, 101, user_item_matrix, item_similarity, svd_model)
print("\n混合推荐结果：\n", recommendations)


