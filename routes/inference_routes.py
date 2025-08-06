from flask import Blueprint, jsonify, current_app, request
from bson import ObjectId
from models.consult_model import ConsultRequest
from models.model import db
from config import DevelopmentConfig

inference_bp = Blueprint('inference', __name__)

@inference_bp.route('/inference_results', methods=['GET'])
def get_inference_results():
    role = request.args.get('role')
    user_id = request.args.get('user_id')
    image_path = request.args.get('image_path')  # D용

    try:
        mongo_client = current_app.extensions.get('mongo_client')
        if not mongo_client:
            return jsonify({"error": "MongoDB 연결 실패"}), 500

        collection = mongo_client.get_collection("inference_results")
        server_base_url = DevelopmentConfig.INTERNAL_BASE_URL

        # ✅ 환자용: 전체 리스트
        if role == 'P':
            if not user_id:
                return jsonify({"error": "user_id가 필요합니다."}), 400

            documents = list(collection.find({"user_id": user_id}))

            for doc in documents:
                doc["_id"] = str(doc["_id"])
                image_path = doc.get("original_image_path", "")
                filename = image_path.split("/")[-1]
                full_image_path = server_base_url + image_path

                # 🔍 진단 신청 여부 확인
                consult = (
                    db.session.query(ConsultRequest)
                    .filter_by(image_path=full_image_path)
                    .order_by(ConsultRequest.request_datetime.desc())
                    .first()
                )
                doc["is_requested"] = consult.is_requested if consult else "N"
                doc["is_replied"] = consult.is_replied if consult else "N"

                # ✅ 마스크 이미지 경로 지정
                doc["model1_image_path"] = f"/images/model1/{filename}"
                doc["model2_image_path"] = f"/images/model2/{filename}"

                if doc.get("image_type") == "normal":
                    doc["model3_2_image_path"] = f"/images/model3_2/{filename}"
                else:
                    doc["model3_2_image_path"] = None

            return jsonify(documents), 200

        # ✅ 의사용: 단일 결과 조회
        elif role == 'D':
            if not user_id or not image_path:
                return jsonify({"error": "user_id와 image_path가 필요합니다."}), 400

            doc = collection.find_one({
                "user_id": user_id,
                "original_image_path": image_path
            })

            if doc:
                doc["_id"] = str(doc["_id"])
                filename = image_path.split("/")[-1]
                full_image_path = server_base_url + image_path

                # 🔍 진단 신청 여부 확인
                consult = (
                    db.session.query(ConsultRequest)
                    .filter_by(image_path=full_image_path)
                    .order_by(ConsultRequest.request_datetime.desc())
                    .first()
                )
                doc["is_requested"] = consult.is_requested if consult else "N"
                doc["is_replied"] = consult.is_replied if consult else "N"

                # ✅ 마스크 이미지 경로 지정
                doc["model1_image_path"] = f"/images/model1/{filename}"
                doc["model2_image_path"] = f"/images/model2/{filename}"

                if doc.get("image_type") == "normal":
                    doc["model3_2_image_path"] = f"/images/model3_2/{filename}"
                else:
                    doc["model3_2_image_path"] = None

                return jsonify(doc), 200
            else:
                return jsonify({"error": "해당 결과를 찾을 수 없습니다."}), 404

        return jsonify({"error": "Invalid role"}), 400

    except Exception as e:
        print(f"❌ MongoDB 오류: {e}")
        return jsonify({"error": "MongoDB 조회 실패"}), 500
