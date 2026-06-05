import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from db.models import Base
from db.chunk_models import Chunk  # noqa: F401


def init_db():
    engine = create_engine(os.environ["POSTGRES_URL"])

    # Enable pgvector extension before creating tables
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    Base.metadata.create_all(engine)
    print("Tables created successfully.")


if __name__ == "__main__":
    init_db()