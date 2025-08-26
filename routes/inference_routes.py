# routes/inference_routes.py
from flask import Blueprint, jsonify, current_app, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson import ObjectId
import os

from models.consult_model import ConsultRequest
from models.model import db
from config import DevelopmentConfig

inference_bp = Blueprint('inference', __name__)

# -------------------------------
# 공통 유틸
# -------------------------------
def _is_xray(doc) -> bool:
    t = (doc.get("image_type") or "").strip().lower()
    return t in {"xray", "panorama", "panoramic", "cbct"}


def _remove_files(doc):
    """문서의 경로 정보를 바탕으로 서버 파일 삭제 (존재 시)."""
    try:
        filename = (doc.get("original_image_path") or "").split("/")[-1]
        if not filename:
            return

        # 원본
        original_abs = os.path.join(DevelopmentConfig.UPLOAD_FOLDER_ORIGINAL, filename)
        if os.path.exists(original_abs):
            os.remove(original_abs)

        # 마스크 파일들
        if _is_xray(doc):
            paths = [
                os.path.join(DevelopmentConfig.PROCESSED_FOLDER_XMODEL1, filename),
                os.path.join(DevelopmentConfig.PROCESSED_FOLDER_XMODEL2, filename),
            ]
        else:
            paths = [
                os.path.join(DevelopmentConfig.PROCESSED_FOLDER_MODEL1, filename),
                os.path.join(DevelopmentConfig.PROCESSED_FOLDER_MODEL2, filename),
                os.path.join(DevelopmentConfig.PROCESSED_FOLDER_MODEL3, filename),
            ]
        for p in paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def _delete_inference_core(mongo_client, inference_id: str, owner_user_id: str):
    """
    소유자 검증 후 Mongo 문서 + 파일 + ConsultRequest 삭제.
    - owner_user_id: JWT 토큰의 사용자 ID (registerId)
    """
    try:
        oid = ObjectId(inference_id)
    except Exception:
        return None, 400, "invalid id"

    coll = mongo_client.get_collection("inference_results")
    doc = coll.find_one({"_id": oid, "user_id": owner_user_id})
    if not doc:
        # 소유자가 아니거나 존재하지 않음
        return None, 404, "not found or no permission"

    # Mongo 삭제
    coll.delete_one({"_id": oid})

    # 서버 파일 삭제
    _remove_files(doc)

    # ConsultRequest 정리 (절대 URL 기준 + 파일명 fallback)
    try:
        base = (
            current_app.config.get("SERVER_BASE_URL")
            or current_app.config.get("INTERNAL_BASE_URL", "")
        ).rstrip("/")
        rel = (doc.get("original_image_path") or "").strip()
        full_url = f"{base}{rel}" if rel else None

        deleted = 0
        if full_url:
            deleted = db.session.query(ConsultRequest).filter_by(image_path=full_url).delete()

        if deleted == 0:
            filename = rel.split("/")[-1] if rel else ""
            if filename:
                db.session.query(ConsultRequest).filter(
                    ConsultRequest.image_path.like(f"%/{filename}")
                ).delete()

        db.session.commit()
    except Exception:
        db.session.rollback()

    return True, 200, "ok"


# -------------------------------
# (목록 조회) 기존 호환: GET /api/inference_results
#   - role=P & user_id=... : 환자 자신의 리스트
#   - role=D & user_id=... & image_path=... : (의사용) 단건
#   ※ 기존 프론트 호환 때문에 유지
# -------------------------------
@inference_bp.route('/inference_results', methods=['GET'])
def get_inference_results():
    role = request.args.get('role')
    user_id = request.args.get('user_id')
    image_path = request.args.get('image_path')  # 상대 경로

    try:
        mongo_client = current_app.extensions.get('mongo_client')
        if not mongo_client:
            return jsonify({"error": "MongoDB 연결 실패"}), 500

        collection = mongo_client.get_collection("inference_results")

        # 요청 호스트 우선, 없으면 INTERNAL_BASE_URL
        server_base_url = (
            current_app.config.get("SERVER_BASE_URL")
            or current_app.config.get("INTERNAL_BASE_URL", "")
        ).rstrip("/")

        def _to_str_id(doc):
            try:
                if isinstance(doc.get("_id"), ObjectId):
                    doc["_id"] = str(doc["_id"])
            except Exception:
                pass

        def _build_mask_paths(doc, filename: str):
            if _is_xray(doc):
                m1 = f"/images/xmodel1/{filename}"
                m2 = f"/images/xmodel2/{filename}"
                m3 = None
            else:
                m1 = f"/images/model1/{filename}"
                m2 = f"/images/model2/{filename}"
                m3 = f"/images/model3/{filename}"
            return m1, m2, m3

        def _attach_consult_flags(doc, full_image_url: str):
            # 1차: 절대 URL 완전 일치
            consult = (
                db.session.query(ConsultRequest)
                .filter_by(image_path=full_image_url)
                .order_by(ConsultRequest.request_datetime.desc())
                .first()
            )

            # 2차: 호스트가 달라 저장된 경우를 위한 파일명 기반 fallback
            if not consult:
                filename = (doc.get("original_image_path") or "").split("/")[-1]
                if filename:
                    consult = (
                        db.session.query(ConsultRequest)
                        .filter(ConsultRequest.image_path.like(f"%/{filename}"))
                        .order_by(ConsultRequest.request_datetime.desc())
                        .first()
                    )

            doc["is_requested"] = consult.is_requested if consult else "N"
            doc["is_replied"] = consult.is_replied if consult else "N"

        if role == 'P':
            if not user_id:
                return jsonify({"error": "user_id가 필요합니다."}), 400

            documents = list(collection.find({"user_id": user_id}))
            for doc in documents:
                _to_str_id(doc)
                img_path_rel = doc.get("original_image_path", "") or ""
                filename = img_path_rel.split("/")[-1] if img_path_rel else ""
                full_image_url = f"{server_base_url}{img_path_rel}" if img_path_rel else ""
                _attach_consult_flags(doc, full_image_url)
                m1, m2, m3 = _build_mask_paths(doc, filename)
                doc["model1_image_path"] = m1
                doc["model2_image_path"] = m2
                doc["model3_image_path"] = m3
            return jsonify(documents), 200

        elif role == 'D':
            if not user_id or not image_path:
                return jsonify({"error": "user_id와 image_path가 필요합니다."}), 400

            doc = collection.find_one({
                "user_id": user_id,
                "original_image_path": image_path
            })
            if not doc:
                return jsonify({"error": "해당 결과를 찾을 수 없습니다."}), 404

            _to_str_id(doc)
            filename = image_path.split("/")[-1]
            full_image_url = f"{server_base_url}{image_path}"
            _attach_consult_flags(doc, full_image_url)
            m1, m2, m3 = _build_mask_paths(doc, filename)
            doc["model1_image_path"] = m1
            doc["model2_image_path"] = m2
            doc["model3_image_path"] = m3
            return jsonify(doc), 200

        return jsonify({"error": "Invalid role"}), 400

    except Exception as e:
        print(f"❌ MongoDB 오류: {e}")
        return jsonify({"error": "MongoDB 조회 실패"}), 500


# -------------------------------
# (삭제) 레거시/단일 엔드포인트: POST /api/inference_delete
# body: { "inference_id": "<id>" }
# JWT 필수, 소유자만 삭제 가능
# -------------------------------
@inference_bp.route("/inference_delete", methods=["POST"])
@jwt_required()
def inference_delete_legacy():
    mongo_client = current_app.extensions.get('mongo_client')
    if not mongo_client:
        return jsonify({"ok": False, "error": "mongo not ready"}), 500

    data = request.get_json(silent=True) or {}
    inference_id = data.get("inference_id")
    if not inference_id:
        return jsonify({"ok": False, "error": "inference_id required"}), 400

    user_id = str(get_jwt_identity() or "")
    ok, code, msg = _delete_inference_core(mongo_client, inference_id, user_id)
    if code != 200:
        return jsonify({"ok": False, "error": msg}), code
    return jsonify({"ok": True}), 200
