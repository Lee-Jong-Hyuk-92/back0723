# routes/xray_implant_classify_route.py
import os
from flask import Blueprint, request, jsonify, current_app
from ai_model.predict_implant_manufacturer import classify_implants_from_xray

xray_implant_bp = Blueprint("xray_implant", __name__)

@xray_implant_bp.route("/xray_implant_classify", methods=["POST"])
def xray_implant_classify():
    data = request.get_json()
    image_path = data.get("image_path")

    if not image_path:
        return jsonify({"error": "image_path가 필요합니다."}), 400

    # 절대 경로 처리 (서버에서 직접 파일 읽을 수 있게)
    image_path_abs = os.path.join(current_app.root_path, image_path.strip("/"))

    if not os.path.exists(image_path_abs):
        return jsonify({"error": f"이미지 경로가 존재하지 않습니다: {image_path_abs}"}), 404

    try:
        results = classify_implants_from_xray(image_path_abs)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
