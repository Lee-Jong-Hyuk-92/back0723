from flask import Blueprint, jsonify, current_app, request
from models.consult_model import ConsultRequest
from models.model import db

inference_bp = Blueprint('inference', __name__)

@inference_bp.route('/inference-results', methods=['GET'])
def get_inference_results():
    role = request.args.get('role')
    user_id = request.args.get('user_id')

    if role == 'P':
        try:
            mongo_client = current_app.extensions.get('mongo_client')
            if not mongo_client:
                return jsonify({"error": "MongoDB 연결 실패"}), 500

            collection = mongo_client.get_collection("inference_results")
            documents = list(collection.find())

            server_base_url = "http://192.168.0.19:5000"  # ✅ MySQL에 저장된 경로 형식과 맞추기

            for doc in documents:
                doc["_id"] = str(doc["_id"])
                image_path = doc.get("original_image_path")

                full_image_path = server_base_url + image_path  # ✅ 일치시키기

                consult = (
                    db.session.query(ConsultRequest)
                    .filter_by(image_path=full_image_path)
                    .order_by(ConsultRequest.request_datetime.desc())
                    .first()
                )

                if consult:
                    doc["is_requested"] = consult.is_requested
                    doc["is_replied"] = consult.is_replied
                else:
                    doc["is_requested"] = "N"
                    doc["is_replied"] = "N"

            if user_id:
                documents = [doc for doc in documents if doc.get("user_id") == user_id]

            return jsonify(documents), 200

        except Exception as e:
            print(f"❌ MongoDB 오류: {e}")
            return jsonify({"error": "MongoDB 조회 실패"}), 500

    return jsonify({"error": "Invalid role"}), 400