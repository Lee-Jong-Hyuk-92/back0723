# routes/survey_routes.py
from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity

survey_bp = Blueprint("survey", __name__)

def _mongo_db():
    db = getattr(current_app, 'mongo_db', None)
    if db is None:
        mc = current_app.extensions.get('mongo_client')
        if mc is not None:
            db = mc.client[current_app.config['MONGO_DB_NAME']]
    if db is None:
        raise RuntimeError("Mongo DB handle not found on app.")
    return db

def _survey_coll(db):
    return db[current_app.config.get('SURVEY_COLLECTION', 'surveys')]

def _inference_coll(db):
    # 기존 업로드에서 쓰던 컬렉션 (기본: inference_results)
    return db[current_app.config.get('MONGO_COLLECTION', 'inference_results')]

@survey_bp.get("/survey/latest")
@jwt_required()
def get_latest_survey():
    """
    최신 문진 1건을 반환합니다.
    1) SURVEY_COLLECTION(기본 'surveys') 먼저 조회
    2) 없으면 MONGO_COLLECTION(기본 'inference_results')의 survey 필드에서 폴백
    정렬 키: created_at ↓, timestamp ↓, _id ↓
    """
    user_id = request.args.get("user_id") or str(get_jwt_identity() or "")
    if not user_id:
        return jsonify({"ok": False, "error": "user_id required"}), 400

    db = _mongo_db()

    # ---- 1) 전용 surveys 컬렉션 조회
    cur = (
        _survey_coll(db)
        .find({"user_id": user_id})
        .sort([("created_at", -1), ("timestamp", -1), ("_id", -1)])
        .limit(1)
    )
    latest = next(cur, None)

    # ---- 2) 없으면 inference_results에서 폴백 (survey 필드가 있는 문서만)
    if not latest:
        cur2 = (
            _inference_coll(db)
            .find({"user_id": user_id, "survey": {"$exists": True}})
            .sort([("created_at", -1), ("timestamp", -1), ("_id", -1)])
            .limit(1)
        )
        latest = next(cur2, None)
        if latest:
            answers = latest.get("answers") or latest.get("survey") or {}
        else:
            answers = {}
    else:
        answers = latest.get("answers") or latest.get("survey") or {}

    if not latest or not answers:
        return jsonify({"ok": True, "data": None})

    created = latest.get("created_at") or latest.get("timestamp")
    try:
        created_iso = created.isoformat() if hasattr(created, "isoformat") else str(created) if created else None
    except Exception:
        created_iso = None

    return jsonify({
        "ok": True,
        "data": {
            "id": str(latest.get("_id")),
            "user_id": user_id,
            "created_at": created_iso,
            "answers": answers
        }
    })

# (선택) 문진 저장 엔드포인트 — 전용 컬렉션에 명시 저장하고 싶을 때 사용
@survey_bp.post("/survey")
@jwt_required()
def save_survey():
    """
    body: { "user_id": "...", "answers": { "<질문>": <값>, ... } }
    """
    payload = request.get_json(silent=True) or {}
    user_id = payload.get("user_id") or str(get_jwt_identity() or "")
    answers = payload.get("answers") or {}

    if not user_id or not isinstance(answers, dict):
        return jsonify({"ok": False, "error": "invalid payload"}), 400

    from datetime import datetime, timezone
    doc = {
        "user_id": user_id,
        "answers": answers,
        "created_at": datetime.now(timezone.utc),
        "source": "manual",  # 구분용 태그
    }
    db = _mongo_db()
    res = _survey_coll(db).insert_one(doc)
    return jsonify({"ok": True, "id": str(res.inserted_id)})
