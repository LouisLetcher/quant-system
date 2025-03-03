from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.auth.auth_service import AuthService  # Hypothetical authentication module

router = APIRouter(prefix="/user", tags=["User"])


class UserRegisterRequest(BaseModel):
    username: str
    password: str
    email: str


class UserLoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register_user(request: UserRegisterRequest):
    """Registers a new user."""
    try:
        user_id = AuthService.register_user(request.username, request.password, request.email)
        return {"status": "success", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User registration failed: {str(e)}")


@router.post("/login")
async def login_user(request: UserLoginRequest):
    """Logs in an existing user and returns an authentication token."""
    try:
        token = AuthService.authenticate_user(request.username, request.password)
        return {"status": "success", "token": token}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Login failed: {str(e)}")