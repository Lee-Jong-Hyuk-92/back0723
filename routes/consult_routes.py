from flask import Blueprint, request, jsonify
from models.consult_model import ConsultRequest
from models.model import db, User, Doctor
from datetime import datetime, timedelta
import json

consult_bp = Blueprint('consult', __name__)

# ✅ 1. 신청 등록 (프론트 전체 필드 수신 대응)
@consult_bp.route('', methods=['POST'])
def create_consult():
    data = request.json
    try:
        user_id = data.get('user_id')
        register_id = data.get('register_id')
        name = data.get('name')
        phone = data.get('phone')
        birth = data.get('birth')
        gender = data.get('gender')
        role = data.get('role')
        inference_result_id = data.get('inference_result_id')
        clinic = data.get('clinic')
        address = data.get('address')
        original_image_url = data.get('original_image_url')
        processed_image_urls = json.dumps(data.get('processed_image_urls', {}))
        model_infos = json.dumps(data.get('model_infos', {}))
        request_datetime = datetime.strptime(data.get('request_datetime')[:14], '%Y%m%d%H%M%S')
    except Exception as e:
        print(f"❌ 데이터 파싱 실패: {e}")
        return jsonify({'error': 'Invalid request format'}), 400

    if User.query.filter_by(register_id=user_id).first() is None:
        return jsonify({'error': 'Invalid user_id'}), 400

    existing = ConsultRequest.query.filter_by(user_id=user_id, is_requested='Y', is_replied='N').first()
    if existing:
        return jsonify({'error': '이미 신청 중인 진료가 있습니다.'}), 400

    try:
        consult = ConsultRequest(
            user_id=user_id,
            register_id=register_id,
            name=name,
            phone=phone,
            birth=birth,
            gender=gender,
            role=role,
            inference_result_id=inference_result_id,
            clinic=clinic,
            address=address,
            original_image_url=original_image_url,
            processed_image_urls=processed_image_urls,
            model_infos=model_infos,
            request_datetime=request_datetime,
            is_requested='Y',
            is_replied='N'
        )
        db.session.add(consult)
        db.session.commit()
        return jsonify({'message': 'Consultation request created'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Database error: {e}'}), 500

# ✅ 2. 신청 취소
@consult_bp.route('/cancel', methods=['POST'])
def cancel_consult():
    data = request.json
    request_id = data.get('request_id')

    consult = ConsultRequest.query.get(request_id)
    if consult and consult.is_requested == 'Y' and consult.is_replied == 'N':
        db.session.delete(consult)
        db.session.commit()
        return jsonify({'message': 'Request deleted'}), 200

    return jsonify({'error': 'Cannot cancel this request'}), 400

# ✅ 3. 의사 응답
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

# ✅ 4. 통계 조회
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

# ✅ 5. 진료 신청 리스트 조회 (오늘 기준)
@consult_bp.route('/list', methods=['GET'])
def list_consult_requests():
    try:
        today = datetime.now().date()
        start = datetime.combine(today, datetime.min.time())
        end = datetime.combine(today, datetime.max.time())

        consults = ConsultRequest.query.filter(
            ConsultRequest.is_requested == 'Y',
            ConsultRequest.request_datetime >= start,
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
                'request_datetime': consult.request_datetime.strftime('%Y-%m-%d %H:%M:%S') \
                    if isinstance(consult.request_datetime, datetime) else consult.request_datetime,
                'is_replied': consult.is_replied
            })

        return jsonify({'consults': result}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to fetch consult list'}), 500

# ✅ 6. 사용자 진행중 진료 조회
@consult_bp.route('/active', methods=['GET'])
def get_active_consult_request():
    user_id = request.args.get('user_id')
    active = ConsultRequest.query.filter_by(user_id=user_id, is_replied='N') \
        .order_by(ConsultRequest.id.desc()).first()
    if active:
        return jsonify({
            'image_path': active.image_path,
            'request_id': active.id
        }), 200
    return jsonify({'image_path': None, 'request_id': None}), 200