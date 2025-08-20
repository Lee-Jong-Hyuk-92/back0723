from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import func
from models.consult_model import ConsultRequest
from models.model import db, User, Doctor
from datetime import datetime, timedelta
import json
from flask_jwt_extended import jwt_required, get_jwt_identity

# ▶ 추가: Mongo 사용
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
# ▶ MongoDB 컬렉션 핸들러
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
# ▶ 경로 정규화 유틸 (호스트 접두사 제거, 쿼리스트링 제거 등)
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_path(p: str) -> str:
    if not p:
        return ''
    p = p.split('?', 1)[0].split('#', 1)[0]
    base = (current_app.config.get('INTERNAL_BASE_URL') or '').rstrip('/')
    if base and p.startswith(base):
        p = p[len(base):]
    if not p.startswith('/'):
        p = '/' + p
    return p

# ─────────────────────────────────────────────────────────────────────────────
# ▶ (중요) Mongo 문서에서 image_type 및 오버레이 경로 추출
#   - image_type: 'xray' 또는 'normal' 로 표준화
#   - normal:  model1, model2, model3
#   - xray:    xmodel1, xmodel2
#   - 다양한 키 패턴 지원 (top-level / overlays.* / results.*)
# ─────────────────────────────────────────────────────────────────────────────
def _extract_overlays_and_type(doc: dict):
    image_type = None

    # 1) image_type 표준화
    if isinstance(doc.get('image_type'), str):
        image_type = doc.get('image_type')
    elif isinstance(doc.get('metadata'), dict) and isinstance(doc['metadata'].get('image_type'), str):
        image_type = doc['metadata'].get('image_type')

    if isinstance(image_type, str):
        s = image_type.strip().lower()
        if s in ('xray', 'x-ray'):
            image_type = 'xray'
        elif s == 'normal':
            image_type = 'normal'
        else:
            image_type = None

    overlays = {}

    # 공통 수집 유틸: top-level, overlays, results 모두 탐색
    def _collect_with_mapping(pairs):
        # top-level
        for key_alias, std_name in pairs:
            val = doc.get(key_alias)
            if isinstance(val, str) and val:
                overlays[std_name] = val
            elif isinstance(val, dict):
                url = val.get('image_url') or val.get('mask_url') or val.get('overlay_url')
                if isinstance(url, str) and url:
                    overlays[std_name] = url
        # nested: overlays.*, results.*
        for container in ('overlays', 'results'):
            cont = doc.get(container)
            if isinstance(cont, dict):
                for key_alias, std_name in pairs:
                    v = cont.get(key_alias)
                    if isinstance(v, str) and v:
                        overlays[std_name] = v
                    elif isinstance(v, dict):
                        url = v.get('image_url') or v.get('mask_url') or v.get('overlay_url')
                        if isinstance(url, str) and url:
                            overlays[std_name] = url

    normal_pairs = [
        ('model1_image_url', 'model1'),
        ('model2_image_url', 'model2'),
        ('model3_image_url', 'model3'),
        ('model1_image_path', 'model1'),
        ('model2_image_path', 'model2'),
        ('model3_image_path', 'model3'),
        ('model1', 'model1'),
        ('model2', 'model2'),
        ('model3', 'model3'),
    ]

    xray_pairs = [
        ('xmodel1_image_url', 'xmodel1'),
        ('xmodel2_image_url', 'xmodel2'),
        ('xmodel1_image_path', 'xmodel1'),
        ('xmodel2_image_path', 'xmodel2'),
        ('xmodel1', 'xmodel1'),
        ('xmodel2', 'xmodel2'),

        # 일부 파이프라인은 xray인데 model1/2 키로 넣기도 함
        ('model1_image_url', 'xmodel1'),
        ('model2_image_url', 'xmodel2'),
        ('model1_image_path', 'xmodel1'),
        ('model2_image_path', 'xmodel2'),
        ('model1', 'xmodel1'),
        ('model2', 'xmodel2'),
    ]

    if image_type == 'normal':
        _collect_with_mapping(normal_pairs)
    elif image_type == 'xray':
        _collect_with_mapping(xray_pairs)
    else:
        _collect_with_mapping(normal_pairs + xray_pairs)

    host = (current_app.config.get('INTERNAL_BASE_URL') or '').rstrip('/')
    fixed_overlays = {}
    for k, v in overlays.items():
        vv = str(v).split('?', 1)[0].split('#', 1)[0]
        if vv.startswith('http://') or vv.startswith('https://'):
            fixed_overlays[k] = vv
        else:
            vv = vv if vv.startswith('/') else '/' + vv
            fixed_overlays[k] = f'{host}{vv}' if host else vv

    return fixed_overlays, image_type

# ✅ 1. 신청 등록
@consult_bp.route('', methods=['POST'])
@jwt_required()
def create_consult():
    data = request.json
    try:
        user_id = data.get('user_id')
        # 프론트가 original_image_url(절대/상대)을 보냄 → 서버 저장은 정규화 상대경로
        raw_path = data.get('original_image_url') or ''
        image_path = _normalize_path(raw_path)

        request_datetime_str = data.get('request_datetime')
        if not request_datetime_str or len(request_datetime_str) < 14:
            raise ValueError("Invalid request_datetime")
        request_datetime = datetime.strptime(request_datetime_str[:14], '%Y%m%d%H%M%S')

        if User.query.filter_by(register_id=user_id).first() is None:
            return jsonify({'error': 'Invalid user_id'}), 400

        existing = ConsultRequest.query.filter_by(user_id=user_id, is_requested='Y', is_replied='N').first()
        if existing:
            return jsonify({'error': '이미 신청 중인 진료가 있습니다.'}), 400

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
    raw_path = data.get('original_image_url')  # 프론트에서 보내주는 절대/상대 경로
    image_path = _normalize_path(raw_path or '')

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

# ✅ 3. 특정 이미지에 대한 신청 상태 조회  ←(request_id/doctor_comment 포함)
@consult_bp.route('/status', methods=['GET'])
def get_consult_status():
    user_id = request.args.get('user_id')
    raw_path = request.args.get('image_path')
    image_path = _normalize_path(raw_path or '')

    if not user_id or not image_path:
        return jsonify({'error': 'Missing parameters'}), 400

    consult = (ConsultRequest.query
               .filter_by(user_id=user_id, image_path=image_path)
               .order_by(ConsultRequest.id.desc())
               .first())

    if consult:
        print(f"[CONSULT STATUS] user_id={user_id}, image_path={image_path} -> "
              f"is_requested={consult.is_requested}, is_replied={consult.is_replied}")
        return jsonify({
            'is_requested': consult.is_requested,
            'is_replied': consult.is_replied,
            'request_id': consult.id,
            'doctor_comment': consult.doctor_comment,
        }), 200

    return jsonify({
        'is_requested': 'N',
        'is_replied': 'N',
        'doctor_comment': None,
        'request_id': None
    }), 200

# ✅ 4. 의사 응답  ←(JWT 기반 의사 식별)
@consult_bp.route('/reply', methods=['POST'])
@jwt_required()
def doctor_reply():
    try:
        identity = get_jwt_identity()  # 토큰의 register_id (의사)
        data = request.get_json() or {}
        request_id = data.get('request_id')
        comment = data.get('comment', '')
        reply_dt_str = data.get('reply_datetime')

        if not request_id:
            return jsonify({'error': 'request_id required'}), 400

        # 토큰 주체가 의사인지 확인
        doctor = Doctor.query.filter_by(register_id=identity).first()
        if not doctor:
            return jsonify({'error': 'Invalid doctor (token)'}), 401

        # 날짜 파싱 (없으면 서버 UTC)
        try:
            reply_dt = datetime.strptime(reply_dt_str[:14], '%Y%m%d%H%M%S') if reply_dt_str else datetime.utcnow()
        except Exception:
            reply_dt = datetime.utcnow()

        consult = ConsultRequest.query.get(request_id)
        if not consult or consult.is_requested != 'Y':
            return jsonify({'error': 'Request not found or not active'}), 400

        consult.doctor_id = doctor.register_id
        consult.doctor_comment = comment
        consult.reply_datetime = reply_dt
        consult.is_replied = 'Y'

        db.session.commit()
        return jsonify({'message': 'Reply submitted', 'request_id': consult.id}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

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

# ✅ 6. 진료 신청 리스트 조회  ← (중요) image_type 포함
@consult_bp.route('/list', methods=['GET'])
def list_consult_requests():
    try:
        consults = (ConsultRequest.query
                    .filter(ConsultRequest.is_requested == 'Y')
                    .order_by(ConsultRequest.request_datetime.desc())
                    .all())

        host = (current_app.config.get('INTERNAL_BASE_URL') or '').rstrip('/')
        coll = _get_mongo_collection()

        result = []
        for consult in consults:
            user = User.query.filter_by(register_id=consult.user_id).first()

            # Mongo 매칭: 정규화 경로로 우선 조회, 보조 키도 함께 시도
            uid_str = str(consult.user_id)
            norm_path = _normalize_path(consult.image_path or '')
            alt_path1 = '/' + norm_path.lstrip('/')
            base_name = os.path.basename(norm_path)

            doc = (
                coll.find_one({"user_id": uid_str, "original_image_path": norm_path}) or
                coll.find_one({"user_id": uid_str, "original_image_path": alt_path1}) or
                coll.find_one({"user_id": uid_str, "original_image_path": base_name})
            )

            image_type = None
            if doc:
                _, image_type = _extract_overlays_and_type(doc)
                # ✅ 디버깅 로그 추가
                print(f"[LIST] user_id={uid_str}, path={norm_path}, "
                      f"mongo_image_type={doc.get('image_type')}, "
                      f"resolved_image_type={image_type}")

            # 프론트에서 인증 헤더 붙여서 직접 호출하므로 상대경로 유지가 안전
            result.append({
                'request_id': consult.id,
                'user_id': consult.user_id,
                'user_name': user.name if user else '',
                'image_path': norm_path,  # 정규화된 상대 경로 저장/반환
                'request_datetime': consult.request_datetime.strftime('%Y-%m-%d %H:%M:%S')
                    if isinstance(consult.request_datetime, datetime) else str(consult.request_datetime),
                'is_replied': consult.is_replied,
                'image_type': image_type  # ★ 추가
            })

        return jsonify({'consults': result}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to fetch consult list', 'details': str(e)}), 500


# ✅ 7. 사용자 진행 중 진료 조회
@consult_bp.route('/active', methods=['GET'])
@jwt_required()
def get_active_consult_request():
    user_id = get_jwt_identity()
    active = (ConsultRequest.query
              .filter_by(user_id=user_id, is_requested='Y', is_replied='N')
              .order_by(ConsultRequest.id.desc())
              .first())
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
        start_date = today - timedelta(days=6)

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

# ✅ 11. 성별·연령대 통계
@consult_bp.route('/demographics', methods=['GET'])
def consult_demographics():
    try:
        subq = (
            db.session.query(ConsultRequest.user_id)
            .filter(ConsultRequest.is_requested == 'Y')
            .distinct()
            .subquery()
        )

        rows = (
            db.session.query(User.register_id, User.gender, User.birth)
            .join(subq, subq.c.user_id == User.register_id)
            .all()
        )

        gender_counts = {"male": 0, "female": 0}
        for _, g, _ in rows:
            if g in ('M', '남', 'm', 'Male', 'male'):
                gender_counts["male"] += 1
            elif g in ('F', '여', 'f', 'Female', 'female'):
                gender_counts["female"] += 1

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

        return jsonify({"ok": True, "data": {"gender": gender_counts, "age": age_buckets}}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

# ✅ 12. 시간대별 건수
@consult_bp.route('/hourly-stats', methods=['GET'])
def hourly_stats():
    try:
        date_str = request.args.get('date')
        the_day = _parse_ymd(date_str) or datetime.now().date()

        start = datetime.combine(the_day, datetime.min.time())
        end   = datetime.combine(the_day, datetime.max.time())

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

# ✅ 13. 날짜 기준 사진 리스트 (오버레이/타입 포함)
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
        coll = _get_mongo_collection()

        data = []
        for r in rows:
            path = r.image_path or ''
            if path.startswith('http://') or path.startswith('https://'):
                image_url = path
            else:
                if not path.startswith('/'):
                    path = '/' + path
                image_url = f'{host}{path}' if host else path

            dt_str = (
                r.request_datetime.strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(r.request_datetime, datetime) else str(r.request_datetime)
            )

            uid_str = str(r.user_id)
            norm_path = _normalize_path(r.image_path or '')
            alt_path1 = '/' + norm_path.lstrip('/')
            base_name = os.path.basename(norm_path)

            doc = (
                coll.find_one({"user_id": uid_str, "original_image_path": norm_path}) or
                coll.find_one({"user_id": uid_str, "original_image_path": alt_path1}) or
                coll.find_one({"user_id": uid_str, "original_image_path": base_name})
            )

            overlays = {}
            image_type = None
            if doc:
                overlays, image_type = _extract_overlays_and_type(doc)
                if image_type == 'normal':
                    allowed = {'model1', 'model2', 'model3'}
                    overlays = {k: v for k, v in overlays.items() if k in allowed}
                elif image_type == 'xray':
                    allowed = {'xmodel1', 'xmodel2'}
                    overlays = {k: v for k, v in overlays.items() if k in allowed}
                else:
                    allowed = {'model1', 'model2', 'model3', 'xmodel1', 'xmodel2'}
                    overlays = {k: v for k, v in overlays.items() if k in allowed}

            item = {
                'id': r.id,
                'user_id': r.user_id,
                'image_path': r.image_path or '',
                'image_url': image_url,
                'request_datetime': dt_str,
                'is_replied': r.is_replied,
                'image_type': image_type,
                'overlays': overlays
            }
            data.append(item)

        return jsonify({'ok': True, 'data': data, 'total': total}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# ✅ 14. 영상 타입 비율 (MySQL ↔ Mongo 매칭)
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
            norm_path = _normalize_path(img_path)
            alt_path1 = '/' + norm_path.lstrip('/')
            base_name = os.path.basename(norm_path)

            doc = (
                coll.find_one({"user_id": uid_str, "original_image_path": norm_path}) or
                coll.find_one({"user_id": uid_str, "original_image_path": alt_path1}) or
                coll.find_one({"user_id": uid_str, "original_image_path": base_name})
            )

            if not doc:
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

# ✅ (추가) 단건 오버레이 조회
@consult_bp.route('/masks', methods=['GET'])
def get_masks_for_image():
    try:
        user_id = request.args.get('user_id')
        raw_path = request.args.get('image_path')
        if not user_id or not raw_path:
            return jsonify({'ok': False, 'error': 'Missing parameters'}), 400

        uid_str = str(user_id)
        norm_path = _normalize_path(raw_path)
        alt_path1 = '/' + norm_path.lstrip('/')
        base_name = os.path.basename(norm_path)

        coll = _get_mongo_collection()
        doc = (
            coll.find_one({"user_id": uid_str, "original_image_path": norm_path}) or
            coll.find_one({"user_id": uid_str, "original_image_path": alt_path1}) or
            coll.find_one({"user_id": uid_str, "original_image_path": base_name})
        )

        if not doc:
            return jsonify({'ok': True, 'overlays': {}, 'image_type': None}), 200

        overlays, image_type = _extract_overlays_and_type(doc)

        if image_type == 'normal':
            allowed = {'model1', 'model2', 'model3'}
            overlays = {k: v for k, v in overlays.items() if k in allowed}
        elif image_type == 'xray':
            allowed = {'xmodel1', 'xmodel2'}
            overlays = {k: v for k, v in overlays.items() if k in allowed}
        else:
            allowed = {'model1', 'model2', 'model3', 'xmodel1', 'xmodel2'}
            overlays = {k: v for k, v in overlays.items() if k in allowed}

        return jsonify({'ok': True, 'overlays': overlays, 'image_type': image_type}), 200
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
