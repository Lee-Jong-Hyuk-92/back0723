# 가상환경생성
python -m venv venv

# 가상환경 실행
1) powershell TERMINAL 일때
.\venv\Scripts\Activate.ps1

2) Git bash TERMINAL 일때
source venv/Scripts/activate

# requirements.txt 에 적힌대로 라이브러리 설치할때
pip install -r requirements.txt

# 현재 pip install한 모든 라이브러리 requirements.txt 로 만들때
pip freeze > requirements.txt

# 가상환경 종료
deactivate

# 모델 없다고 나올때, ai_model 폴더 안에
https://drive.google.com/drive/folders/1JrZLA9Z2yhq58HVkserV3lqfr9wUAGTI?usp=sharing
이거 두개 넣기

# .env에서 MySQL 설정
MYSQL_PASSWORD= 본인 mysql -u root -p 에 맞는 비밀번호

# MySQL에 toothai라는 database 미리 만들어야함
MYSQL_DB=toothai

# MongoDB 설정
# MongoDB에도 toothai라는 database 미리 만들어야 함
MONGO_URI=mongodb://localhost:27017/
MONGO_DB_NAME=toothai

python app.py 실행
