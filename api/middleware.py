import time
import uuid
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("scirag")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request,
                       call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start      = time.perf_counter()

        request.state.request_id = request_id

        logger.info(
            f"[{request_id}] {request.method} {request.url.path}"
        )

        response = await call_next(request)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[{request_id}] {response.status_code} ({elapsed:.1f}ms)"
        )

        response.headers["X-Request-ID"]       = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response