import os
import bcrypt
import random
import string
import smtplib
from flask import Blueprint, request, jsonify
from models.model import db, User, Doctor
from flask_jwt_extended import create_access_token
from email.message import EmailMessage
from dotenv import load_dotenv

auth_bp = Blueprint('auth', __name__)
load_dotenv()  # .env 로드

# ✅ 아이디 중복 체크
@auth_bp.route('/check-username', methods=['GET'])
def check_username_duplicate():
    register_id = request.args.get('username')
    if not register_id:
        return jsonify({"message": "Username parameter is required"}), 400

    user_exists = User.query.filter_by(register_id=register_id).first()
    doctor_exists = Doctor.query.filter_by(register_id=register_id).first()

    if user_exists or doctor_exists:
        return jsonify({"exists": True, "message": "이미 사용 중인 아이디입니다."}), 200
    return jsonify({"exists": False, "message": "사용 가능한 아이디입니다."}), 200


# ✅ 회원가입
@auth_bp.route('/register', methods=['POST'])
def signup():
    data = request.get_json()
    role = data.get('role', 'P')
    register_id = data.get('register_id')
    password = data.get('password')
    name = data.get('name')
    gender = data.get('gender')
    birth = data.get('birth')
    phone = data.get('phone')

    if not all([register_id, password, name, gender, birth, phone]):
        return jsonify({"message": "모든 필드를 입력해야 합니다."}), 400

    user_exists = User.query.filter_by(register_id=register_id).first()
    doctor_exists = Doctor.query.filter_by(register_id=register_id).first()
    if user_exists or doctor_exists:
        return jsonify({"message": "이미 사용 중인 아이디입니다. 다른 아이디를 사용해주세요."}), 409

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    if role == 'D':
        new_user = Doctor(
            register_id=register_id,
            password=hashed_pw.decode('utf-8'),
            name=name, gender=gender, birth=birth, phone=phone, role=role
        )
    else:
        new_user = User(
            register_id=register_id,
            password=hashed_pw.decode('utf-8'),
            name=name, gender=gender, birth=birth, phone=phone, role=role
        )

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "Error registering user", "error": str(e)}), 500

# ✅ 로그인
@auth_bp.route('/login', methods=['POST'])
def login():
    # print("로그인 시작")
    if request.method == 'OPTIONS':
        return '', 200  # preflight 대응
    data = request.get_json()
    role = data.get('role', 'P')
    register_id = data.get('register_id')
    password = data.get('password')

    Model = Doctor if role == 'D' else User
    # print("쿼리 실행")
    user = Model.query.filter_by(register_id=register_id).first()

    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        user_data = {
            "register_id": user.register_id,
            "name": user.name,
            "gender": user.gender,
            "birth": user.birth,
            "phone": user.phone,
            "role": user.role
        }
        user_data["doctor_id" if role == 'D' else "user_id"] = getattr(user, "doctor_id" if role == 'D' else "user_id")

        access_token = create_access_token(
            identity=str(user.register_id),
            additional_claims={"role": user.role}
        )

        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "user": user_data
        }), 200

    return jsonify({"message": "Invalid credentials"}), 401

# ✅ 회원 탈퇴
@auth_bp.route('/delete_account', methods=['DELETE'])
def delete_account():
    data = request.get_json()
    role = data.get('role', 'P')
    register_id = data.get('register_id')
    password = data.get('password')

    if not register_id or not password:
        return jsonify({"message": "아이디와 비밀번호를 모두 입력해주세요."}), 400

    Model = Doctor if role == 'D' else User
    user = Model.query.filter_by(register_id=register_id).first()

    if not user or not bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({"message": "아이디 또는 비밀번호가 잘못되었습니다."}), 401

    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "회원 탈퇴가 완료되었습니다."}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"message": "회원 탈퇴 중 오류가 발생했습니다.", "error": str(e)}), 500

# ✅ 비밀번호 재확인
@auth_bp.route('/reauthenticate', methods=['POST'])
def reauthenticate():
    data = request.get_json()
    register_id = data.get('register_id')
    password = data.get('password')
    role = data.get('role', 'P')

    if not register_id or not password:
        return jsonify({"success": False, "message": "아이디와 비밀번호를 모두 입력해주세요."}), 400

    Model = Doctor if role == 'D' else User
    user = Model.query.filter_by(register_id=register_id).first()

    if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
        return jsonify({"success": True}), 200
    else:
        return jsonify({"success": False, "message": "비밀번호가 일치하지 않습니다."}), 401


# ✅ 프로필 수정
@auth_bp.route('/update-profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    register_id = data.get('register_id')
    name = data.get('name')
    gender = data.get('gender')
    birth = data.get('birth')
    phone = data.get('phone')
    password = data.get('password')
    role = data.get('role', 'P')

    if not all([register_id, name, gender, birth, phone, password]):
        return jsonify({'message': '모든 필드를 입력해주세요.'}), 400

    Model = Doctor if role == 'D' else User
    user = Model.query.filter_by(register_id=register_id).first()

    if not user:
        return jsonify({'message': '해당 사용자를 찾을 수 없습니다.'}), 404

    user.name = name
    user.gender = gender
    user.birth = birth
    user.phone = phone
    user.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    try:
        db.session.commit()
        return jsonify({'message': '프로필이 성공적으로 수정되었습니다.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': '업데이트 중 오류가 발생했습니다.', 'error': str(e)}), 500


# ✅ 아이디 찾기
@auth_bp.route('/find_id', methods=['POST'])
def find_id():
    data = request.get_json()
    name = data.get('name')
    phone = data.get('phone')

    if not name or not phone:
        return jsonify({'message': '이름과 전화번호를 모두 입력해주세요.'}), 400

    user = User.query.filter_by(name=name, phone=phone).first()
    if not user:
        user = Doctor.query.filter_by(name=name, phone=phone).first()

    if user:
        return jsonify({'register_id': user.register_id}), 200
    else:
        return jsonify({'message': '일치하는 사용자 정보를 찾을 수 없습니다.'}), 404


# ✅ 비밀번호 찾기 + 메일 전송
@auth_bp.route('/find_password', methods=['POST'])
def find_password():
    data = request.get_json()
    name = data.get('name')
    phone = data.get('phone')

    if not name or not phone:
        return jsonify({'message': '이름과 전화번호를 모두 입력해주세요.'}), 400

    user = User.query.filter_by(name=name, phone=phone).first()
    if not user:
        user = Doctor.query.filter_by(name=name, phone=phone).first()

    if not user:
        return jsonify({'message': '일치하는 사용자 정보를 찾을 수 없습니다.'}), 404

    # 임시 비밀번호 생성
    temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    hashed_temp_pw = bcrypt.hashpw(temp_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user.password = hashed_temp_pw

    try:
        db.session.commit()

        # 이메일 전송
        email_sender = os.getenv("EMAIL_USER")
        email_password = os.getenv("EMAIL_PASS")

        msg = EmailMessage()
        msg['Subject'] = 'MediTooth 임시 비밀번호 안내'
        msg['From'] = email_sender
        msg['To'] = 'sa4667@naver.com'
        msg.set_content(f"안녕하세요.\n\n임시 비밀번호는 다음과 같습니다:\n\n{temp_password}\n\n로그인 후 반드시 비밀번호를 변경해주세요.")

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(msg)

        return jsonify({'message': '비밀번호 재설정 링크가 이메일로 전송되었습니다.'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': '비밀번호 초기화 중 오류 발생', 'error': str(e)}), 500

# ✅ 의사 이름 조회 API
@auth_bp.route('/doctor-name/<register_id>', methods=['GET'])
def get_doctor_name(register_id):
    doctor = Doctor.query.filter_by(register_id=register_id).first()
    if doctor:
        print(f"🎯 의사 이름 조회 결과: register_id={doctor.register_id}, name={doctor.name}")
        return jsonify({"name": doctor.name}), 200
    return jsonify({"message": "해당 의사를 찾을 수 없습니다."}), 404