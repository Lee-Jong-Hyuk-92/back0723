# 안쓰는 파일

from flask import Blueprint, request, jsonify, current_app
from models.location_model import db, Location
from models.application_model import ApplicationModel
from pymongo import MongoClient

application_bp = Blueprint('application', __name__)

@application_bp.route('/apply', methods=['POST', 'OPTIONS'])
def apply_for_consultation():
    # ✅ CORS preflight 요청 처리
    if request.method == 'OPTIONS':
        return '', 204

    try:
        data = request.json
        user_id = data.get("user_id")
        location = data.get("location")
        inference_result_id = data.get("inference_result_id")

        if not all([user_id, location, inference_result_id]):
            return jsonify({"error": "필수 항목 누락"}), 400

        # 1️⃣ MySQL: location 저장 또는 업데이트
        existing = Location.query.filter_by(user_id=user_id).first()
        if existing:
            existing.location = location
        else:
            db.session.add(Location(user_id=user_id, location=location))
        db.session.commit()

        # 2️⃣ MongoDB: 신청 정보 저장
        mongo_client = current_app.extensions.get("mongo_client")
        if not mongo_client:
            return jsonify({"error": "MongoDB 클라이언트 오류"}), 500

        collection = mongo_client.get_collection("application")
        app_model = ApplicationModel(collection)
        app_model.create_application(inference_result_id)

        return jsonify({"message": "신청 완료"}), 200

    except Exception as e:
        print(f"❌ 신청 처리 중 예외: {e}")
        return jsonify({"error": "서버 내부 오류"}), 500
