from flask_sqlalchemy import SQLAlchemy
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# .env 파일 로드 (MongoDB URI와 DB 이름을 가져오기 위함)
load_dotenv()
db = SQLAlchemy()

# MongoDB 클라이언트 클래스
class MongoDBClient:
    def __init__(self, uri=None, db_name=None):
        mongo_uri = uri or os.getenv('MONGO_URI')
        mongo_db_name = db_name or os.getenv('MONGO_DB_NAME')

        if not mongo_uri or not mongo_db_name:
            raise ValueError("MongoDB URI or DB name not set in environment variables or config.")

        self.client = MongoClient(mongo_uri)
        self.db = self.client[mongo_db_name]
        self.inference_results_collection = self.db.inference_results

    def insert_result(self, result_data):
        try:
            self.inference_results_collection.insert_one(result_data)
            print(f"MongoDB 'inference_results'에 문서 삽입 성공.")
        except Exception as e:
            print(f"MongoDB 'inference_results' 문서 삽입 실패: {e}")
            raise

    def insert_into_collection(self, collection_name, document):
        try:
            collection = self.db[collection_name]
            result = collection.insert_one(document)
            print(f"MongoDB '{collection_name}' 컬렉션에 문서 삽입 성공: {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            print(f"MongoDB '{collection_name}' 문서 삽입 실패: {e}")
            raise

    def get_collection(self, collection_name):
        return self.db[collection_name]

    def close(self):
        self.client.close()

# ✅ 환자용 모델 (User)
class User(db.Model):
    __tablename__ = 'user'

    # ✅ BigInteger에서 Integer로 되돌리고 auto_increment는 기본값으로 사용
    user_id = db.Column(db.Integer, primary_key=True)
    register_id = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    birth = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    role = db.Column(db.String(1)) # 'P'

    def __repr__(self):
        return f'<User {self.register_id}>'

# ✅ 의사용 모델 (Doctor)
class Doctor(db.Model):
    __tablename__ = 'doctor'

    # ✅ BigInteger에서 Integer로 되돌리고 auto_increment는 기본값으로 사용
    doctor_id = db.Column(db.Integer, primary_key=True)
    register_id = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100))
    gender = db.Column(db.String(10))
    birth = db.Column(db.String(20))
    phone = db.Column(db.String(20))
    role = db.Column(db.String(1)) # 'D'

    def __repr__(self):
        return f'<Doctor {self.register_id}>'
