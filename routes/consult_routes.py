from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import func
from models.consult_model import ConsultRequest
from models.model import db, User, Doctor
from datetime import datetime, timedelta
import json
from flask_jwt_extended import jwt_required, get_jwt_identity

# ▶ 추가: Mongo 사용을 위해
from pymongo import MongoClient
import os

consult_bp = Blueprint('consult', __name__)

# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸: birth 문자열(YYYY-MM-DD 또는 YYYYMMDD) → 나이 계산
# ─────────────────────────────────────────────────────────────────────────────
def _birth_to_age(birth_str):
    if not birth_str:
        return None
    try:
        # '1999-02-02' 또는 '19990202' 모두 처리 (앞 4자리 연도만 사용)
        by = int(str(birth_str)[:4])
        this_year = datetime.now().year
        age = this_year - by
        if 0 <= age <= 120:
            return age
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸: 'YYYYMMDD' 또는 'YYYY-MM-DD' → datetime.date
# ─────────────────────────────────────────────────────────────────────────────
def _parse_ymd(date_str: str):
    if not date_str:
        return None
    try:
        if '-' in date_str:
            return datetime.strptime(date_str, '%Y-%m-%d').date()
        return datetime.strptime(date_str, '%Y%m%d').date()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ▶ 추가: MongoDB 컬렉션 핸들러
# ─────────────────────────────────────────────────────────────────────────────
def _get_mongo_collection():
    uri = current_app.config.get('MONGO_URI')
    dbname = current_app.config.get('MONGO_DB_NAME')
    collname = current_app.config.get('MONGO_COLLECTION', 'uploads')
    if not uri or not dbname:
        raise RuntimeError('MongoDB is not configured (MONGO_URI / MONGO_DB_NAME).')
    client = MongoClient(uri)
    return client[dbname][collname]

# ─────────────────────────────────────────────────────────────────────────────
# ▶ 추가: 경로 정규화 유틸 (호스트 접두사 제거, 쿼리스트링 제거 등)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_path(p: str) -> str:
    if not p:
        return ''
    # 쿼리/해시 제거
    p = p.split('?', 1)[0].split('#', 1)[0]
    # 내부 BASE URL 접두사 제거
    base = (current_app.config.get('INTERNAL_BASE_URL') or '').rstrip('/')
    if base and p.startswith(base):
        p = p[len(base):]
    # 슬래시 정리
    if not p.startswith('/'):
        p = '/' + p
    return p

# ✅ 1. 신청 등록
@consult_bp.route('', methods=['POST'])
@jwt_required()
def create_consult():
    data = request.json
    try:
        user_id = data.get('user_id')
        image_path = data.get('original_image_url')
        request_datetime_str = data.get('request_datetime')
        if not request_datetime_str or len(request_datetime_str) < 14:
            raise ValueError("Invalid request_datetime")
        request_datetime = datetime.strptime(request_datetime_str[:14], '%Y%m%d%H%M%S')

        # 사용자 존재 확인
        if User.query.filter_by(register_id=user_id).first() is None:
            return jsonify({'error': 'Invalid user_id'}), 400

        # 중복 신청 여부 확인
        existing = ConsultRequest.query.filter_by(user_id=user_id, is_requested='Y', is_replied='N').first()
        if existing:
            return jsonify({'error': '이미 신청 중인 진료가 있습니다.'}), 400

        # 새로운 신청 등록
        consult = ConsultRequest(
            user_id=user_id,
            image_path=image_path,
            request_datetime=request_datetime,
            is_requested='Y',
            is_replied='N'
        )

        db.session.add(consult)
        db.session.commit()

        return jsonify({'message': 'Consultation request created'}), 201

    except Exception as e:
        db.session.rollback()
        print(f"❌ 신청 실패: {e}")
        return jsonify({'error': f'Database error: {e}'}), 500

# ✅ 2. 신청 취소
@consult_bp.route('/cancel', methods=['POST'])
@jwt_required()
def cancel_consult():
    data = request.json
    user_id = data.get('user_id')
    image_path = data.get('original_image_url')  # 👈 프론트에서 보내주는 상대 경로 그대로 사용

    if not user_id or not image_path:
        return jsonify({'error': 'Missing parameters'}), 400

    consult = ConsultRequest.query.filter_by(
        user_id=user_id,
        image_path=image_path,
        is_requested='Y',
        is_replied='N'
    ).order_by(ConsultRequest.id.desc()).first()

    if consult:
        db.session.delete(consult)
        db.session.commit()
        return jsonify({'message': 'Request cancelled'}), 200

    return jsonify({'error': 'Cannot cancel this request'}), 400

# ✅ 3. 특정 이미지에 대한 신청 상태 조회
@consult_bp.route('/status', methods=['GET'])
def get_consult_status():
    user_id = request.args.get('user_id')
    image_path = request.args.get('image_path')

    if not user_id or not image_path:
        return jsonify({'error': 'Missing parameters'}), 400

    # ✅ 중복 대비: 최신 데이터 기준
    consult = ConsultRequest.query.filter_by(user_id=user_id, image_path=image_path) \
        .order_by(ConsultRequest.id.desc()).first()

    if consult:
        print(f"[CONSULT STATUS] user_id={user_id}, image_path={image_path} -> is_requested={consult.is_requested}, is_replied={consult.is_replied}")
        return jsonify({
            'is_requested': consult.is_requested,
            'is_replied': consult.is_replied
        }), 200

    return jsonify({
        'is_requested': 'N',
        'is_replied': 'N'
    }), 200

# ✅ 4. 의사 응답
@consult_bp.route('/reply', methods=['POST'])
def doctor_reply():
    data = request.json
    request_id = data.get('request_id')
    doctor_id = data.get('doctor_id')
    comment = data.get('comment')
    reply_datetime = data.get('reply_datetime')

    doctor = Doctor.query.filter_by(register_id=doctor_id).first()
    if not doctor:
        return jsonify({'error': 'Invalid doctor_id'}), 400

    consult = ConsultRequest.query.get(request_id)
    if consult and consult.is_requested == 'Y':
        consult.doctor_id = doctor_id
        consult.doctor_comment = comment
        consult.reply_datetime = reply_datetime
        consult.is_replied = 'Y'
        db.session.commit()
        return jsonify({'message': 'Reply submitted'}), 200

    return jsonify({'error': 'Request not found or already completed'}), 400

# ✅ 5. 통계 조회
@consult_bp.route('/stats', methods=['GET'])
def consult_stats():
    date_str = request.args.get('date')
    try:
        date_obj = datetime.strptime(date_str, '%Y%m%d')
        start = datetime.combine(date_obj, datetime.min.time())
        end = start + timedelta(days=1)

        all_requests = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime < end
        ).count()

        completed = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime < end,
            ConsultRequest.is_replied == 'Y'
        ).count()

        pending = all_requests - completed

        return jsonify({
            'date': date_str,
            'total': all_requests,
            'completed': completed,
            'pending': pending
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ✅ 6. 진료 신청 리스트 조회
@consult_bp.route('/list', methods=['GET'])
def list_consult_requests():
    try:
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())

        consults = ConsultRequest.query.filter(
            ConsultRequest.is_requested == 'Y',
            ConsultRequest.request_datetime >= start,  # 오늘만이 아닌 기존 날짜도 포함하려면 주석처리
            ConsultRequest.request_datetime <= end
        ).order_by(ConsultRequest.request_datetime.desc()).all()

        result = []
        for consult in consults:
            user = User.query.filter_by(register_id=consult.user_id).first()
            result.append({
                'request_id': consult.id,
                'user_id': consult.user_id,
                'user_name': user.name if user else '',
                'image_path': consult.image_path,
                'request_datetime': consult.request_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    if isinstance(consult.request_datetime, datetime) else consult.request_datetime,
                'is_replied': consult.is_replied
            })

        return jsonify({'consults': result}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to fetch consult list'}), 500

# ✅ 7. 사용자 진행 중 진료 조회
@consult_bp.route('/active', methods=['GET'])
@jwt_required()
def get_active_consult_request():
    user_id = get_jwt_identity()
    active = ConsultRequest.query.filter_by(user_id=user_id, is_requested='Y', is_replied='N').order_by(ConsultRequest.id.desc()).first()
    if active:
        return jsonify({
            'image_path': active.image_path,
            'request_id': active.id
        }), 200
    return jsonify({'image_path': None, 'request_id': None}), 200

# ✅ 8. 오늘 날짜 기준 요청 수 반환
@consult_bp.route('/today-count', methods=['GET'])
def today_request_count():
    try:
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())

        count = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime <= end
        ).count()

        return jsonify({'count': count}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ 9. 오늘 날짜 기준 상태별 요청 수 반환
@consult_bp.route('/today-status-counts', methods=['GET'])
def today_status_counts():
    try:
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())

        total = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime <= end
        ).count()

        pending = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime <= end,
            ConsultRequest.is_requested == 'Y',
            ConsultRequest.is_replied == 'N'
        ).count()

        completed = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime <= end,
            ConsultRequest.is_requested == 'Y',
            ConsultRequest.is_replied == 'Y'
        ).count()

        canceled = ConsultRequest.query.filter(
            ConsultRequest.request_datetime >= start,
            ConsultRequest.request_datetime <= end,
            ConsultRequest.is_requested == 'N',
            ConsultRequest.is_replied == 'N'
        ).count()

        return jsonify({
            'total': total,
            'pending': pending,
            'completed': completed,
            'canceled': canceled
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ 10. 최근 7일 신청 건수 API
@consult_bp.route('/recent-7-days', methods=['GET'])
def recent_7_days():
    try:
        today = datetime.now().date()
        start_date = today - timedelta(days=6)  # 오늘 포함 7일 전

        results = []
        for i in range(7):
            day = start_date + timedelta(days=i)
            start = datetime.combine(day, datetime.min.time())
            end = datetime.combine(day, datetime.max.time())

            count = ConsultRequest.query.filter(
                ConsultRequest.request_datetime >= start,
                ConsultRequest.request_datetime <= end
            ).count()

            results.append({
                'date': day.strftime('%Y-%m-%d'),
                'count': count
            })

        return jsonify({'data': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ 11. 성별·연령대 통계 (신청 이력이 있는 사용자만 대상)
@consult_bp.route('/demographics', methods=['GET'])
def consult_demographics():
    """
    응답:
    {
      "ok": true,
      "data": {
        "gender": {"male": 5, "female": 3},
        "age": {"20대": 2, "30대": 4, "40대": 1, "50대": 1}
      }
    }
    """
    try:
        # 신청 테이블에 등장한 user_id(=user.register_id)만 추출
        subq = (
            db.session.query(ConsultRequest.user_id)
            .filter(ConsultRequest.is_requested == 'Y')  # 필요에 따라 조건 조정
            .distinct()
            .subquery()
        )

        # user 테이블 조인으로 gender, birth 확보
        rows = (
            db.session.query(User.register_id, User.gender, User.birth)
            .join(subq, subq.c.user_id == User.register_id)
            .all()
        )

        # ----- 성별 집계 -----
        gender_counts = {"male": 0, "female": 0}
        for _, g, _ in rows:
            if g in ('M', '남', 'm', 'Male', 'male'):
                gender_counts["male"] += 1
            elif g in ('F', '여', 'f', 'Female', 'female'):
                gender_counts["female"] += 1

        # ----- 연령대 집계 -----
        age_buckets = {"20대": 0, "30대": 0, "40대": 0, "50대": 0}
        for _, _, birth in rows:
            age = _birth_to_age(birth)
            if age is None:
                continue
            if 20 <= age < 30:
                age_buckets["20대"] += 1
            elif 30 <= age < 40:
                age_buckets["30대"] += 1
            elif 40 <= age < 50:
                age_buckets["40대"] += 1
            elif 50 <= age < 60:
                age_buckets["50대"] += 1
            # 10대/60대+ 등은 현재 UI 요구사항에 없으면 생략

        return jsonify({"ok": True, "data": {"gender": gender_counts, "age": age_buckets}}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

# ✅ 12. 시간대별 건수 (기본: 오늘)
# GET /consult/hourly-stats?date=20250818
# 응답: { ok: true, data: { labels:["00",..,"23"], counts:[..], total: N } }
@consult_bp.route('/hourly-stats', methods=['GET'])
def hourly_stats():
    try:
        date_str = request.args.get('date')   # 'YYYYMMDD' 또는 'YYYY-MM-DD'
        the_day = _parse_ymd(date_str) or datetime.now().date()

        start = datetime.combine(the_day, datetime.min.time())
        end   = datetime.combine(the_day, datetime.max.time())

        # MySQL: HOUR(datetime)으로 그룹핑
        rows = (
            db.session.query(
                func.hour(ConsultRequest.request_datetime).label('hh'),
                func.count(ConsultRequest.id)
            )
            .filter(
                ConsultRequest.request_datetime >= start,
                ConsultRequest.request_datetime <= end
            )
            .group_by('hh')
            .all()
        )

        # 0~23까지 채우기
        by_hour = {int(h): int(c) for h, c in rows}
        labels = [f'{h:02d}' for h in range(24)]
        counts = [by_hour.get(h, 0) for h in range(24)]
        total = sum(counts)

        return jsonify({
            'ok': True,
            'data': {
                'labels': labels,
                'counts': counts,
                'total': total,
            }
        }), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# ✅ 13. 날짜 기준 사진(원본 이미지) 리스트
# GET /consult/images?date=2025-08-18&limit=12&offset=0
# 응답: { ok:true, data:[{id,user_id,image_path,image_url,request_datetime,is_replied}], total:N }
@consult_bp.route('/images', methods=['GET'])
def images_by_date():
    try:
        date_str = request.args.get('date')
        the_day = _parse_ymd(date_str) or datetime.now().date()

        limit = max(1, min(int(request.args.get('limit', 12)), 200))
        offset = max(0, int(request.args.get('offset', 0)))

        start = datetime.combine(the_day, datetime.min.time())
        end   = datetime.combine(the_day, datetime.max.time())

        base = (
            ConsultRequest.query
            .filter(
                ConsultRequest.request_datetime >= start,
                ConsultRequest.request_datetime <= end,
                ConsultRequest.is_requested == 'Y'
            )
            .order_by(ConsultRequest.request_datetime.desc())
        )

        total = base.count()
        rows = base.limit(limit).offset(offset).all()

        host = (current_app.config.get('INTERNAL_BASE_URL') or '').rstrip('/')
        data = []
        for r in rows:
            # image_path가 '/images/...' 형태라면 접두사만 붙여 완전한 URL 생성
            path = r.image_path or ''
            image_url = f'{host}{path}' if path.startswith('/') else f'{host}/{path}'
            dt_str = (
                r.request_datetime.strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(r.request_datetime, datetime) else str(r.request_datetime)
            )
            data.append({
                'id': r.id,
                'user_id': r.user_id,
                'image_path': path,
                'image_url': image_url,
                'request_datetime': dt_str,
                'is_replied': r.is_replied,
            })

        return jsonify({'ok': True, 'data': data, 'total': total}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# ✅ 14. 영상 타입 비율 (MySQL ↔ Mongo 매칭)
# GET /consult/video-type-ratio?date=2025-08-18
# 응답: { ok:true, data:{ "X-ray": n1, "구강이미지": n2 }, total: n1+n2 }
@consult_bp.route('/video-type-ratio', methods=['GET'])
def video_type_ratio():
    try:
        date_str = request.args.get('date')
        the_day = _parse_ymd(date_str) or datetime.now().date()

        start = datetime.combine(the_day, datetime.min.time())
        end   = datetime.combine(the_day, datetime.max.time())

        rows = (
            db.session.query(ConsultRequest.user_id, ConsultRequest.image_path)
            .filter(
                ConsultRequest.request_datetime >= start,
                ConsultRequest.request_datetime <= end,
                ConsultRequest.is_requested == 'Y'
            )
            .all()
        )

        # ✅ user_id는 무조건 문자열로, path도 문자열로 확보
        pairs = {(str(uid), str(path or '')) for uid, path in rows if path}

        coll = _get_mongo_collection()

        def _norm_type(val):
            if not val: return None
            s = str(val).strip().lower()
            if s in ('xray', 'x-ray'): return 'xray'
            if s == 'normal': return 'normal'
            return None

        xray_count = 0
        normal_count = 0

        for uid_str, img_path in pairs:
            norm_path = _normalize_path(img_path)           # '/images/.../a.png'
            alt_path1 = '/' + norm_path.lstrip('/')         # 혹시 몰라 보정
            base_name = os.path.basename(norm_path)         # 'a.png'

            # ✅ user_id는 문자열로 비교
            doc = (
                coll.find_one({"user_id": uid_str, "original_image_path": norm_path}) or
                coll.find_one({"user_id": uid_str, "original_image_path": alt_path1}) or
                coll.find_one({"user_id": uid_str, "original_image_path": base_name})
            )

            if not doc:
                # 문서를 못 찾으면 스킵 (임의로 normal 올리지 않음)
                continue

            image_type = None
            if isinstance(doc.get('image_type'), str):
                image_type = doc['image_type']
            elif isinstance(doc.get('metadata'), dict) and isinstance(doc['metadata'].get('image_type'), str):
                image_type = doc['metadata']['image_type']

            t = _norm_type(image_type)
            if t == 'xray':
                xray_count += 1
            elif t == 'normal':
                normal_count += 1
            else:
                continue

        data = {"X-ray": xray_count, "구강이미지": normal_count}
        return jsonify({"ok": True, "data": data, "total": xray_count + normal_count}), 200

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500