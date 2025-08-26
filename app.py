import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from config import DevelopmentConfig
from models.model import db, MongoDBClient

# ✅ Vertex AI 및 dotenv
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ JWT
from flask_jwt_extended import JWTManager

# ✅ Flask 앱 생성 및 설정
app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
app.config["SERVER_BASE_URL"] = None  # ✅ 최초에는 None으로 초기화
CORS(app)  # ← 기존 유지 (동작하던 설정)

# ✅ .env 설정 로드
load_dotenv()

# ✅ JWT 설정 (기존 방식 유지)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'super-secret-key')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600
jwt = JWTManager(app)

# ✅ Gemini API 키 설정 및 모델 로드
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API 키가 .env에 없습니다.")
genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
except Exception as e:
    raise ValueError(f"Gemini 모델 로드 실패: {e}")

# ✅ 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER_ORIGINAL'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_MODEL1'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_MODEL2'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_MODEL3'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_XMODEL1'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_XMODEL2'], exist_ok=True)

# ✅ DB 초기화 및 Mongo 연결
db.init_app(app)
mongo_client = MongoDBClient(uri=app.config['MONGO_URI'], db_name=app.config['MONGO_DB_NAME'])

try:
    mongo_client.client.admin.command('ping')
except Exception as e:
    print(f"❌ MongoDB 연결 실패: {e}")
    sys.exit(1)

# ✅ 앱 확장 객체 등록 (기존 그대로)
app.extensions = getattr(app, 'extensions', {})
app.extensions['mongo_client'] = mongo_client
app.extensions['gemini_model'] = gemini_model

# ✅ survey_routes.py에서 사용할 수 있도록 DB 핸들 안전 주입
#    (MongoDBClient에 db 속성이 없을 수도 있으니 안전하게 처리)
try:
    app.mongo_db = mongo_client.client[app.config['MONGO_DB_NAME']]
except Exception:
    app.mongo_db = getattr(mongo_client, 'db', None)

with app.app_context():
    db.create_all()

# ✅ 최초 요청 시 host_url을 캐싱
@app.before_request
def cache_host_url():
    if not app.config.get("SERVER_BASE_URL"):
        app.config["SERVER_BASE_URL"] = request.host_url.rstrip("/")

# ✅ 라우트 등록 (기존 + survey_bp 추가)
from routes.auth_routes import auth_bp
from routes.image_routes import image_bp
from routes.upload_routes import upload_bp
from routes.inference_routes import inference_bp
from routes.static_routes import static_bp
from routes.application_routes import application_bp
from routes.consult_routes import consult_bp
from routes.chatbot_routes import chatbot_bp
from routes.chatbot_routes_medgemma import chatbot_med_bp
from routes.multimodal_gemini_route import multimodal_gemini_bp
from routes.multimodal_gemini_xray_route import multimodal_gemini_xray_bp
from routes.xray_implant_classify_route import xray_implant_bp
from routes.survey_routes import survey_bp  # ✅ 추가

app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(image_bp)
app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(inference_bp, url_prefix='/api')
app.register_blueprint(static_bp)
app.register_blueprint(application_bp, url_prefix='/api')
app.register_blueprint(consult_bp, url_prefix='/api/consult')
app.register_blueprint(chatbot_bp, url_prefix='/api')
app.register_blueprint(chatbot_med_bp, url_prefix='/api')
app.register_blueprint(multimodal_gemini_bp, url_prefix='/api')
app.register_blueprint(multimodal_gemini_xray_bp, url_prefix='/api')
app.register_blueprint(xray_implant_bp, url_prefix='/api')
app.register_blueprint(survey_bp, url_prefix='/api')  # ✅ /api/survey/latest

# ✅ 기본 라우트
@app.route('/')
def index():
    return "Hello from MediTooth Backend!"

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"서버 내부 오류 발생: {error}")
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500

# ✅ 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
