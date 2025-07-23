import sys
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from config import DevelopmentConfig
from models.model import db, MongoDBClient

# ✅ Vertex AI 및 dotenv
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image as VertexAIImage
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ Blueprint 라우트 임포트
from routes.auth_routes import auth_bp
from routes.image_routes import image_bp
from routes.upload_routes import upload_bp
from routes.inference_routes import inference_bp
from routes.static_routes import static_bp
from routes.application_routes import application_bp
from routes.consult_routes import consult_bp
from routes.chatbot_routes import chatbot_bp
from routes.chatbot_routes_medgemma import chatbot_med_bp
from routes.multimodal_gemini_route import multimodal_gemini_bp  # ✅ 추가

# ✅ .env 설정 로드
load_dotenv()

# ✅ GCP 서비스 계정 키 설정
CREDENTIALS_FILE_NAME = "gcp_credentials.json"
CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), CREDENTIALS_FILE_NAME)

if not os.path.exists(CREDENTIALS_PATH):
    raise FileNotFoundError(f"GCP 서비스 계정 키 파일을 찾을 수 없습니다: {CREDENTIALS_PATH}")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
print(f"✅ GOOGLE_APPLICATION_CREDENTIALS 환경 변수 설정됨: {CREDENTIALS_PATH}")

# ✅ Vertex AI 초기화
PROJECT_ID = "meditooth"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

# ✅ MedGemma 모델 로드
MEDGEMMA_MODEL_NAME = "medgemma"
try:
    medgemma_model = GenerativeModel(MEDGEMMA_MODEL_NAME)
    print(f"✅ MedGemma 모델 '{MEDGEMMA_MODEL_NAME}' 로드 성공.")
except Exception as e:
    raise ValueError(f"MedGemma 모델 '{MEDGEMMA_MODEL_NAME}' 로드 실패: {e}")

# ✅ Gemini API 키 설정 및 모델 로드
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Gemini API 키가 .env에 없습니다.")
genai.configure(api_key=GEMINI_API_KEY)

try:
    gemini_model = genai.GenerativeModel("models/gemini-2.5-flash")
    print(f"✅ Gemini 모델 'gemini-2.5-flash' 로드 성공.")
except Exception as e:
    raise ValueError(f"Gemini 모델 로드 실패: {e}")

# ✅ Flask 앱 생성 및 설정
app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
CORS(app)

print(f"✅ 연결된 DB URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

# ✅ 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER_ORIGINAL'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_MODEL1'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_MODEL2'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER_MODEL3'], exist_ok=True)

# ✅ DB 초기화 및 Mongo 연결
db.init_app(app)
mongo_client = MongoDBClient(uri=app.config['MONGO_URI'], db_name=app.config['MONGO_DB_NAME'])

try:
    mongo_client.client.admin.command('ping')
    print(f"✅ MongoDB에 성공적으로 연결되었습니다: {app.config['MONGO_URI']}")
except Exception as e:
    print(f"❌ MongoDB 연결 실패: {e}")
    sys.exit(1)

# ✅ 앱 확장 객체 등록
app.extensions = getattr(app, 'extensions', {})
app.extensions['mongo_client'] = mongo_client
app.extensions['medgemma_model'] = medgemma_model
app.extensions['gemini_model'] = gemini_model

with app.app_context():
    db.create_all()

# ✅ 라우트 등록
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(image_bp)
app.register_blueprint(upload_bp, url_prefix='/api')
app.register_blueprint(inference_bp, url_prefix='/api')
app.register_blueprint(static_bp)
app.register_blueprint(application_bp, url_prefix='/api')
app.register_blueprint(consult_bp, url_prefix='/api/consult')
app.register_blueprint(chatbot_bp, url_prefix='/api')
app.register_blueprint(chatbot_med_bp, url_prefix='/api')
app.register_blueprint(multimodal_gemini_bp, url_prefix='/api')  # ✅ 추가

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
    app.run(debug=True, host='0.0.0.0', port=5000)
