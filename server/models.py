from sqlalchemy import Column, Integer, String, DateTime
from server.database import Base
from datetime import datetime

class UserFile(Base):
    __tablename__ = "user_files"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String, index=True)
    custom_name = Column(String, index=True) 
    filename = Column(String) 
    uploaded_at = Column(DateTime, default=datetime.utcnow)
