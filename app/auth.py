from fastapi import Header, HTTPException

from app.config import ADMIN_API_KEY


def require_admin(x_admin_key: str = Header(..., alias="X-Admin-Key")) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Admin access is not configured. Set ADMIN_API_KEY in the environment.",
        )
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin credentials.")
