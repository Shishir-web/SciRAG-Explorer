import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from db.models import Base

load_dotenv()

def init_db():
    engine = create_engine(os.getenv("POSTGRES__URL"))
    Base.metadata.create_all(engine)
    print("Tables created successfully.")

if __name__ == "__main__":
    init_db()