# models/consult_model.py
from models.model import db  # ✅ 기존 db 인스턴스 사용
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship

class ConsultRequest(db.Model):
    __tablename__ = 'consult_request'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(80), nullable=False)  # register_id 기준
    image_path = db.Column(db.String(255), nullable=False)
    request_datetime = db.Column(DateTime, nullable=False)
    doctor_id = db.Column(db.String(80), nullable=True)  # register_id
    doctor_comment = db.Column(db.Text, nullable=True)
    reply_datetime = db.Column(DateTime, nullable=True)
    is_requested = db.Column(db.String(1), default='Y')  # Y / N
    is_replied = db.Column(db.String(1), default='N')    # Y / N
