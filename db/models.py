# models/caption_job.py
from sqlalchemy import Column, String, Text
from database import Base

class CaptionJob(Base):
    __tablename__ = "caption_jobs"
    id = Column(String, primary_key=True)
    status = Column(String)
    video_path = Column(Text)
    output_path = Column(Text)
    error = Column(Text, nullable=True)
