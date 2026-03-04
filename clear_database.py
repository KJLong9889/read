import os
import sys
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import urllib.parse

# 🔥 加上这段
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed.")
# ===========================
# 1. 环境变量 & 数据库配置
# ===========================

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD_RAW = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_HOST, DB_USER, DB_PASSWORD_RAW, DB_NAME]):
    raise RuntimeError("❌ 数据库环境变量未配置完整")

DB_PASSWORD_ENCODED = urllib.parse.quote_plus(DB_PASSWORD_RAW)
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD_ENCODED}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ===========================
# 2. 引入 ORM Model
# ===========================

# ⚠️ 这里直接复制你原来的 Model 定义，或从公共模块 import
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, DateTime
import datetime
import uuid

Base = declarative_base()

def get_clean_uuid():
    return uuid.uuid4().hex

class TsDatasetAutoFeatureResult(Base):
    __tablename__ = "ts_dataset_auto_feature_result"
    id = Column(String(32), primary_key=True, default=get_clean_uuid)
    task_id = Column(String(32))
    dataset_id = Column(String(32))
    feature_code = Column(String(32))
    feature_name = Column(String(256))
    create_time = Column(DateTime, default=datetime.datetime.now)

class TsAutoFeatureDetailValue(Base):
    __tablename__ = "ts_auto_feature_detail_value"
    id = Column(String(32), primary_key=True, default=get_clean_uuid)
    task_id = Column(String(32))
    relation_id = Column(String(32))
    time = Column(String(32))
    value = Column(String(256))
    create_time = Column(DateTime, default=datetime.datetime.now)

# ===========================
# 3. 清表逻辑
# ===========================

def clear_auto_feature_tables():
    db = SessionLocal()
    try:
        print("🚨 开始清空表数据...")

        # ⚠️ 有外键关系的话，一定要先删 detail，再删 result
        deleted_detail = db.query(TsAutoFeatureDetailValue).delete()
        deleted_result = db.query(TsDatasetAutoFeatureResult).delete()

        db.commit()

        print("✅ 清理完成")
        print(f"   ts_auto_feature_detail_value 删除 {deleted_detail} 行")
        print(f"   ts_dataset_auto_feature_result 删除 {deleted_result} 行")

    except Exception as e:
        db.rollback()
        print("❌ 清理失败，已回滚")
        raise e
    finally:
        db.close()

if __name__ == "__main__":
    clear_auto_feature_tables()
