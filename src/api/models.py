"""
API models for the dashboard.
"""
from typing import Optional
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class LogoutResponse(BaseModel):
    message: str