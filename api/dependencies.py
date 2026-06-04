import os
from functools import lru_cache
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

# Rate limiter - 30 requests/minutes per IP
limiter = Limiter(key_func=get_remote_address)

@lru_cache(maxsize=1)
def get_engine():
    return create_engine(
        os.environ["POSTGRES_URL"],
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True, # verify connections are alive
    )

def check_db_health() -> str:
    """returns 'ok' or an error message string."""
    try:
        engine = get_engine()
        with Session(engine) as session:
            session.execute(text("SELECT 1"))
        return "ok"
    except Exception as e:
        return f"Error: {str(e)}"