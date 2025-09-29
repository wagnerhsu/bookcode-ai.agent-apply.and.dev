--------------------------------------------------------------------------------------------------------------


# celery_config.py —— Celery配置文件
from celery import Celery
app = Celery('recruitment_tasks', broker='redis://localhost:6379/0')
@app.task
def parse_resume(resume_data):
    print(f"Parsing resume: {resume_data}")
    return f"Parsed: {resume_data}"
# 启动Redis
redis-server
# 启动Celery worker
celery -A celery_config worker --loglevel=info
# add_task.py —— 添加任务到Celery队列
from celery_config import parse_resume
resume_data = {"name": "Alice", "skills": ["Python", "Machine Learning"]}
parse_resume.delay(resume_data)

# /etc/nginx/nginx.conf —— Nginx配置文件
http {
    upstream backend {
        server localhost:8001;
        server localhost:8002;
    }
    server {
        listen 80;
        location / {
            proxy_pass http://backend;
        }
    }
}

@app.task(bind=True, max_retries=3)
def parse_resume(self, resume_data):
    try:
        print(f"Parsing resume: {resume_data}")
        return f"Parsed: {resume_data}"
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)  # 5秒后重试

import redis
# 初始化Redis客户端
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
def cache_interview_schedule(interview_id, schedule):
    redis_client.set(interview_id, schedule)
def get_interview_schedule(interview_id):
    return redis_client.get(interview_id)


--------------------------------------------------------------------------------------------------------------


# models.py —— 用户模型定义
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import generate_password_hash, check_password_hash
db = SQLAlchemy()
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # 用户角色：如招聘官、面试官
    def set_password(self, password):
        self.password_hash = generate_password_hash(password).decode('utf8')
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# app.py —— 初始化数据库
from flask import Flask
from models import db, User
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db.init_app(app)
with app.app_context():
    db.create_all()  # 创建数据库表
    # 创建管理员用户
    admin = User(username='admin', role='admin')
    admin.set_password('admin123')
    db.session.add(admin)
    db.session.commit()


# auth.py —— 用户登录与JWT生成
from flask import request, jsonify, Flask
from flask_jwt_extended import JWTManager, create_access_token
app = Flask(__name__)
app.config['JWT_SECRET_KEY'] = 'super-secret'  # 秘钥配置
jwt = JWTManager(app)
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username', None)
    password = request.json.get('password', None)
    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity={'username': user.username, 'role': user.role})
        return jsonify(access_token=access_token), 200
    return jsonify({"msg": "用户名或密码错误"}), 401


# rbac.py —— 基于角色的访问控制
from flask_jwt_extended import jwt_required, get_jwt_identity
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    if current_user['role'] != 'admin':
        return jsonify({"msg": "权限不足"}), 403
    return jsonify({"msg": "访问成功，管理员权限"}), 200


# audit.py —— 记录用户操作日志
import logging
logging.basicConfig(filename='audit.log', level=logging.INFO)
@app.route('/log_action', methods=['POST'])
@jwt_required()
def log_action():
    current_user = get_jwt_identity()
    action = request.json.get('action')
    logging.info(f"用户 {current_user['username']} 执行了操作：{action}")
    return jsonify({"msg": "操作记录成功"}), 200


export FLASK_APP=app.py
flask run


POST /login
Content-Type: application/json
{
    "username": "admin",
    "password": "admin123"
}


--------------------------------------------------------------------------------------------------------------


import spacy
# 加载 spaCy 预训练的英文模型
nlp = spacy.load('en_core_web_sm')

import re
import json
import spacy
# 样例简历文本
resume_text = """
John Doe
Email: john.doe@example.com
Phone: +1-123-456-7890
Skills: Python, Machine Learning, Data Analysis, SQL
"""
# 加载 spaCy 语言模型
nlp = spacy.load('en_core_web_sm')
def parse_resume(resume_text):
    # 使用 spaCy 解析文本
    doc = nlp(resume_text)
    # 提取姓名（假设第一个人名即为姓名）
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break
    # 使用正则表达式提取邮箱和电话
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text)
    phone = re.search(r'\+?\d[\d -]{8,12}\d', resume_text)
    # 提取技能（假设技能在 "Skills" 后列出）
    skills = None
    skills_match = re.search(r'Skills:\s*(.*)', resume_text)
    if skills_match:
        skills = [skill.strip() for skill in skills_match.group(1).split(',')]
    # 将解析结果结构化为 JSON 格式
    parsed_data = {
        "name": name,
        "email": email.group(0) if email else None,
        "phone": phone.group(0) if phone else None,
        "skills": skills
    }
    return json.dumps(parsed_data, indent=4)
# 调用解析函数并输出结果
parsed_resume = parse_resume(resume_text)
print(parsed_resume)


--------------------------------------------------------------------------------------------------------------


import sqlite3
import json
import os
import asyncio
import logging
from typing import Dict, Optional, Callable  # 确保正确导入 Callable
from functools import wraps
import smtplib
from email.mime.text import MIMEText
import time
# 从环境变量获取数据库名称和管理员邮箱
DB_NAME = os.getenv("DB_NAME", "resumes.db")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class DatabaseManager:
    """数据库管理器：负责数据库连接和操作的封装"""
    def __init__(self):
        self.conn = None
        self.cursor = None
    async def connect(self):
        """异步建立数据库连接并初始化表结构"""
        try:
            self.conn = sqlite3.connect(DB_NAME)
            self.cursor = self.conn.cursor()
            logging.info(f"Connected to database: {DB_NAME}")
            await self._create_table()
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}")
            await self._alert_admin(f"Database connection failed: {e}")
            raise
    async def _create_table(self):
        """创建存储简历的表"""
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Resumes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT,
                phone TEXT,
                skills TEXT
            )
        ''')
        self.conn.commit()
        logging.info("Resumes table initialized.")
    async def save_resume(self, parsed_data: Dict[str, Optional[str]]):
        """保存解析后的简历数据到数据库"""
        skills = ', '.join(parsed_data.get('skills', []))
        try:
            self.cursor.execute('''
                INSERT INTO Resumes (name, email, phone, skills) 
                VALUES (?, ?, ?, ?)
            ''', (parsed_data.get('name'),
                  parsed_data.get('email'),
                  parsed_data.get('phone'),
                  skills))
            self.conn.commit()
            logging.info(f"Resume saved for {parsed_data.get('name')}.")
        except sqlite3.Error as e:
            logging.error(f"Failed to save resume: {e}")
            await self._alert_admin(f"Failed to save resume: {e}")
            raise
    async def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")
    async def _alert_admin(self, message: str):
        """向管理员发送异常警报"""
        try:
            msg = MIMEText(message)
            msg["Subject"] = "Database Error Alert"
            msg["From"] = "noreply@example.com"
            msg["To"] = ADMIN_EMAIL
            with smtplib.SMTP("localhost") as server:
                server.sendmail("noreply@example.com", ADMIN_EMAIL, msg.as_string())
            logging.info(f"Alert sent to {ADMIN_EMAIL}.")
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")
async def load_and_save_resume(json_data: str):
    """加载 JSON 数据并保存到数据库"""
    try:
        parsed_data = json.loads(json_data)
        db_manager = DatabaseManager()
        await db_manager.connect()
        await db_manager.save_resume(parsed_data)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {e}")
    finally:
        await db_manager.close()
# JSON 数据
parsed_resume = '''
{
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1-123-456-7890",
    "skills": ["Python", "Machine Learning", "Data Analysis", "SQL"]
}
'''
# 性能监控装饰器
def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper
@performance_monitor
async def main():
    """主程序入口，简历数据存储"""
    logging.info("Starting resume processing...")
    await load_and_save_resume(parsed_resume)

if __name__ == "__main__":
    asyncio.run(main())


--------------------------------------------------------------------------------------------------------------


import pandas as pd
import numpy as np
import random
import logging
from typing import List, Dict, Any, Callable
from functools import wraps
import time
# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class NLPProcessor:
    """NLP 模块：负责职位描述与简历语义分析"""
    def __init__(self):
        logging.info("Initializing NLP Processor...")
    def encode_text(self, text: str) -> np.ndarray:
        """模拟将文本编码为向量"""
        vector = np.random.rand(1, 768)  # 模拟 768 维的文本向量
        logging.info(f"Encoded text '{text[:10]}...' to vector.")
        return vector
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """语义相似度计算"""
        score = random.uniform(0.5, 1.0) 
        logging.info(f"Calculated semantic similarity: {score:.2f}")
        return score
class ResumeMatcher:
    """简历匹配模块，结合NLP和评分逻辑"""
    def __init__(self, job_description: str, resumes: pd.DataFrame):
        self.job_description = job_description
        self.resumes = resumes
        self.nlp = NLPProcessor()
    def calculate_match_score(self, resume: pd.Series) -> float:
        """职位与简历的匹配分数"""
        job_vector = self.nlp.encode_text(self.job_description)
        resume_vector = self.nlp.encode_text(resume["summary"])
        skill_match = self.nlp.semantic_similarity(resume["skills"], self.job_description)
        score = 0.7 * np.dot(job_vector, resume_vector.T)[0][0] + 0.3 * skill_match
        logging.info(f"Calculated match score for {resume['name']}: {score:.2f}")
        return score
    def generate_match_report(self, resume: pd.Series) -> str:
        """匹配分析报告"""
        report = f"候选人 {resume['name']} 匹配报告：\n - 匹配分数：{random.uniform(0.6, 0.9):.2f}\n"
        report += f" - 技能匹配度：{resume['skills']}\n"
        logging.info(f"Generated match report for {resume['name']}.")
        return report
    def match_resumes(self) -> pd.DataFrame:
        """对所有简历进行匹配并生成报告"""
        results = []
        for _, resume in self.resumes.iterrows():
            score = self.calculate_match_score(resume)
            report = self.generate_match_report(resume)
            results.append({"name": resume["name"], "score": score, "report": report})
        return pd.DataFrame(results).sort_values(by="score", ascending=False)
# 人员数据
job_description = "软件开发工程师岗位，要求熟悉Python、机器学习和数据分析。"
resumes = pd.DataFrame([
    {"name": "Alice", "summary": "具有五年Python开发经验和数据分析背景", "skills": "Python, Data Analysis"},
    {"name": "Bob", "summary": "机器学习专家，精通深度学习和大数据处理", "skills": "Machine Learning, Big Data"},
    {"name": "Charlie", "summary": "熟悉SQL和数据可视化工具，具备项目管理经验", "skills": "SQL, Project Management"}
])
# 性能监控装饰器
def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper
@performance_monitor
def main():
    """主程序入口，进行简历匹配并生成报告"""
    matcher = ResumeMatcher(job_description, resumes)
    results = matcher.match_resumes()
    print("匹配结果：\n", results)
if __name__ == "__main__":
    main()


--------------------------------------------------------------------------------------------------------------


import random
import logging
from typing import List, Dict, Callable, Any
from functools import wraps
import time
# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class SentimentAnalyzer:
    """情感分析模块：负责分析面试过程中候选人的情绪状态"""
    def __init__(self):
        logging.info("Initializing Sentiment Analyzer...")
    def analyze_sentiment(self, transcript: str) -> Dict[str, float]:
        """模拟情感分析，返回积极、消极和中立的伪造概率"""
        sentiment_scores = {
            "positive": random.uniform(0.2, 0.7),
            "negative": random.uniform(0.1, 0.5),
            "neutral": random.uniform(0.3, 0.6)
        }
        logging.info(f"Analyzed sentiment: {sentiment_scores}")
        return sentiment_scores
class BehaviorAnalyzer:
    """行为模式分析模块：分析候选人的回答模式和行为倾向"""
    def __init__(self):
        logging.info("Initializing Behavior Analyzer...")
    def analyze_behavior(self, responses: List[str]) -> Dict[str, float]:
        """伪造行为分析结果，如自信、犹豫和主动性"""
        behavior_scores = {
            "confidence": random.uniform(0.5, 0.9),
            "hesitation": random.uniform(0.1, 0.4),
            "proactiveness": random.uniform(0.4, 0.8)
        }
        logging.info(f"Analyzed behavior: {behavior_scores}")
        return behavior_scores
class InterviewAnalysis:
    """面试分析模块：结合情感与行为分析，生成面试报告"""
    def __init__(self, sentiment_analyzer: SentimentAnalyzer, behavior_analyzer: BehaviorAnalyzer):
        self.sentiment_analyzer = sentiment_analyzer
        self.behavior_analyzer = behavior_analyzer
    def generate_interview_report(self, transcript: str, responses: List[str]) -> Dict[str, Any]:
        """综合分析情感和行为数据，生成面试报告"""
        sentiment = self.sentiment_analyzer.analyze_sentiment(transcript)
        behavior = self.behavior_analyzer.analyze_behavior(responses)
        overall_score = self._calculate_overall_score(sentiment, behavior)
        report = {
            "sentiment": sentiment,
            "behavior": behavior,
            "overall_score": overall_score
        }
        logging.info(f"Generated interview report: {report}")
        return report
def _calculate_overall_score(self, sentiment: Dict[str, float], behavior: Dict[str, float]) -> float:
    """计算综合评分，权重分配给情感和行为"""
    score = (
        0.5 * sentiment["positive"] +
        0.3 * behavior["confidence"] -
        0.2 * behavior["hesitation"]
    )
    return round(score, 2)


# 人员面试数据
transcript = "我认为我在团队协作和项目管理方面有很强的能力，但我也希望能不断提升技术技能。"
responses = [
    "我很乐意接受挑战，并喜欢探索新技术。",
    "在过去的项目中，我会根据情况调整策略。",
    "我认为团队合作是取得成功的关键。"
]
# 性能监控装饰器
def performance_monitor(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper
@performance_monitor
def main():
    """主程序入口，进行面试情感与行为分析并生成报告"""
    sentiment_analyzer = SentimentAnalyzer()
    behavior_analyzer = BehaviorAnalyzer()
    analysis = InterviewAnalysis(sentiment_analyzer, behavior_analyzer)
    report = analysis.generate_interview_report(transcript, responses)
    print("面试报告：\n", report)
if __name__ == "__main__":
    main()


--------------------------------------------------------------------------------------------------------------


import json
# 自动化评价模型的权重配置
WEIGHTS = {
    "language": 0.3,
    "emotion": 0.2,
    "expression": 0.2,
    "job_fit": 0.3
}
# 候选人评价数据（模拟的面试数据）
candidate_data = {
    "name": "John Doe",
    "language": 85,  # 语言表达得分（0-100）
    "emotion": 75,   # 情感稳定性得分（0-100）
    "expression": 80, # 面部表情与肢体语言得分（0-100）
    "job_fit": 90    # 岗位匹配度得分（0-100）
}
# 自动化评价模型：计算综合评分
def calculate_score(data, weights):
    total_score = 0
    for key, weight in weights.items():
        total_score += data[key] * weight
    return round(total_score, 2)
# 生成候选人评价报告
def generate_report(data):
    score = calculate_score(data, WEIGHTS)
    report = {
        "candidate": data["name"],
        "scores": {
            "language": data["language"],
            "emotion": data["emotion"],
            "expression": data["expression"],
            "job_fit": data["job_fit"]
        },
        "total_score": score,
        "recommendation": "Recommended" if score >= 80 else "Not Recommended"
    }
    return report
# 将评价报告输出为JSON格式
evaluation_report = generate_report(candidate_data)
print(json.dumps(evaluation_report, indent=4))


--------------------------------------------------------------------------------------------------------------


import time
import random
import threading
import asyncio
from functools import wraps
class APIError(Exception):
    """API错误类，用于API请求失败的情况。"""
    pass
def retry(retries=3, delay=2):
    """
    重试逻辑的装饰器：如果发生异常，则重试指定次数。
    参数:
    - retries: 重试次数
    - delay: 每次重试之间的延迟
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
            raise APIError("All retries failed.")
        return wrapper
    return decorator
class InterviewScheduler:
    """
    负责面试计划的类，支持多线程并发安排。
    """
    def __init__(self):
        self.schedule = {}
    def add_schedule(self, candidate, time_slot):
        """将面试安排加入计划表中。"""
        self.schedule[candidate] = time_slot
        print(f"Scheduled {candidate} at {time_slot}")
    def get_schedule(self, candidate):
        """查询候选人的面试安排。"""
        return self.schedule.get(candidate, "No interview scheduled")
    def async_schedule(self, candidates):
        """多线程安排候选人的面试。"""
        threads = []
        for candidate, slot in candidates.items():
            t = threading.Thread(target=self.add_schedule, args=(candidate, slot))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
class AIResponseGenerator:
    """
    生成面试问题的类，包含模拟API调用。
    """
    def __init__(self):
        self.questions = [
            "Describe a challenge you faced at work.",
            "What motivates you?",
            "Where do you see yourself in 5 years?",
            "How do you handle failure?"
        ]
    @retry(retries=2)
    def generate_question(self):
        """模拟API调用以生成问题。"""
        if random.random() < 0.4:
            raise APIError("Simulated API failure")
        return random.choice(self.questions)
class CandidateEvaluator:
    """
    候选人评估类，支持异步执行。
    """
    async def evaluate(self, candidate_data):
        """异步评估候选人数据，并生成报告。"""
        await asyncio.sleep(1)  # 模拟耗时操作
        score = sum(candidate_data.values()) / len(candidate_data)
        recommendation = "Recommended" if score >= 75 else "Not Recommended"
        return {"score": score, "recommendation": recommendation}


import asyncio
from assistant_modules import InterviewScheduler, AIResponseGenerator, CandidateEvaluator
import time
def log_execution(func):
    """
    记录函数执行时间的装饰器，用于性能监控。
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.2f} seconds.")
        return result
    return wrapper
@log_execution
def main():
    """系统主入口：安排面试并评估候选人。"""
    # 1. 安排面试
    scheduler = InterviewScheduler()
    candidates = {
        "Alice": "2024-11-15 10:00",
        "Bob": "2024-11-15 11:00"
    }
    scheduler.async_schedule(candidates)
    print(scheduler.get_schedule("Alice"))
    # 2. 生成面试问题
    generator = AIResponseGenerator()
    try:
        question = generator.generate_question()
        print(f"Generated Question: {question}")
    except Exception as e:
        print(f"Failed to generate question: {e}")
    # 3. 评估候选人
    evaluator = CandidateEvaluator()
    candidate_data = {"communication": 85, "problem_solving": 90, "leadership": 78}
    result = asyncio.run(evaluator.evaluate(candidate_data))
    print(f"Evaluation Result: {result}")
if __name__ == "__main__":
    main()


