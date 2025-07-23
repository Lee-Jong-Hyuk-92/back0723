from flask import Blueprint, jsonify, current_app, request

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

            for doc in documents:
                doc["_id"] = str(doc["_id"])

            if user_id:
                documents = [doc for doc in documents if doc.get("user_id") == user_id]

            return jsonify(documents), 200

        except Exception as e:
            print(f"❌ MongoDB 오류: {e}")
            return jsonify({"error": "MongoDB 조회 실패"}), 500

    return jsonify({"error": "Invalid role"}), 400
