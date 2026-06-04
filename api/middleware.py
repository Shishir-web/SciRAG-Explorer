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
    """
    Logs every request with a unique request_id, method, path,
    status code, and response time. Essential for LLMOps debugging.
    """
    async def dispatch(self, request: Request,
                       call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start      = time.perf_counter()

        # Attach request_id so route handlers can refernce it
        request.state.request_id = request_id

        logging.info(
            f"[{request_id}] -> {request.moethod} {request.url.path}"
        )

        response = await call_next(request)

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            f"[{request_id}] <- {response.status_code}"
            f"({elapsed:1f}ms)"
        )

        # Expose request_id in response headers for tracing
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed:.1f}"
        return response