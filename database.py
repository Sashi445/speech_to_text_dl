# # database.py
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, scoped_session
# from sqlalchemy.ext.declarative import declarative_base

# # SQLite engine (can replace with PostgreSQL later)
# SQLALCHEMY_DATABASE_URL = "sqlite:///caption_jobs.db"

# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

# Base = declarative_base()
