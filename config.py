# config.py
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

class DevelopmentConfig:
    DEBUG = True

    # MySQL
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DB = os.getenv('MYSQL_DB')
    SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@localhost/{MYSQL_DB}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # MongoDB
    MONGO_URI = os.getenv('MONGO_URI')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME')

    # 보안 키
    SECRET_KEY = os.getenv('SECRET_KEY') or 'default_fallback_key'

    # ✅ 이미지 저장 경로 (images 하위로 정리)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ✅ 현재 파일 기준으로 고정
    IMAGE_BASE_DIR = os.path.join(BASE_DIR, 'images')

    UPLOAD_FOLDER_ORIGINAL = os.path.join(IMAGE_BASE_DIR, 'original')
    PROCESSED_FOLDER_MODEL1 = os.path.join(IMAGE_BASE_DIR, 'model1')
    PROCESSED_FOLDER_MODEL2 = os.path.join(IMAGE_BASE_DIR, 'model2')
    PROCESSED_FOLDER_MODEL3 = os.path.join(IMAGE_BASE_DIR, 'model3')

    # 허용 확장자
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}