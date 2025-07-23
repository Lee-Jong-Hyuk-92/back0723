from models.model import db  # ✅ app.py에서 init_app 한 인스턴스를 그대로 사용

class Location(db.Model):
    __tablename__ = 'location'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(255), nullable=False)