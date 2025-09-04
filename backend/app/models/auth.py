from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    """User roles for authorization"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"

class UserStatus(str, Enum):
    """User account status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class UserCreate(BaseModel):
    """Model for user registration"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")
    first_name: str = Field(..., min_length=1, max_length=50, description="First name")
    last_name: str = Field(..., min_length=1, max_length=50, description="Last name")
    date_of_birth: datetime = Field(..., description="Date of birth")
    gender: str = Field(..., description="Gender (male/female/other)")
    phone: Optional[str] = Field(None, description="Phone number")
    role: UserRole = Field(UserRole.PATIENT, description="User role")
    emergency_contact: Optional[dict] = Field(None, description="Emergency contact information")

class UserLogin(BaseModel):
    """Model for user login"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

class UserUpdate(BaseModel):
    """Model for user profile updates"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50, description="First name")
    last_name: Optional[str] = Field(None, min_length=1, max_length=50, description="Last name")
    phone: Optional[str] = Field(None, description="Phone number")
    emergency_contact: Optional[dict] = Field(None, description="Emergency contact information")
    preferences: Optional[dict] = Field(None, description="User preferences")

class PasswordChange(BaseModel):
    """Model for password change"""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")

class PasswordReset(BaseModel):
    """Model for password reset request"""
    email: EmailStr = Field(..., description="User email address")

class PasswordResetConfirm(BaseModel):
    """Model for password reset confirmation"""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password (min 8 characters)")

class UserProfile(BaseModel):
    """Model for user profile response"""
    user_id: str = Field(..., description="User ID")
    email: EmailStr = Field(..., description="User email address")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    date_of_birth: datetime = Field(..., description="Date of birth")
    gender: str = Field(..., description="Gender")
    phone: Optional[str] = Field(None, description="Phone number")
    role: UserRole = Field(..., description="User role")
    status: UserStatus = Field(..., description="User status")
    emergency_contact: Optional[dict] = Field(None, description="Emergency contact information")
    preferences: Optional[dict] = Field(None, description="User preferences")
    created_at: datetime = Field(..., description="Account creation date")
    last_login: Optional[datetime] = Field(None, description="Last login date")
    is_verified: bool = Field(False, description="Email verification status")

class TokenResponse(BaseModel):
    """Model for JWT token response"""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: UserProfile = Field(..., description="User profile information")

class RefreshToken(BaseModel):
    """Model for token refresh"""
    refresh_token: str = Field(..., description="JWT refresh token")

class UserSession(BaseModel):
    """Model for user session tracking"""
    session_id: str = Field(..., description="Session ID")
    user_id: str = Field(..., description="User ID")
    device_info: Optional[dict] = Field(None, description="Device information")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    is_active: bool = Field(True, description="Session active status")

class EmailVerification(BaseModel):
    """Model for email verification"""
    token: str = Field(..., description="Verification token")

class TwoFactorAuth(BaseModel):
    """Model for two-factor authentication"""
    code: str = Field(..., min_length=6, max_length=6, description="2FA code")
    remember_device: bool = Field(False, description="Remember this device")

class UserPermissions(BaseModel):
    """Model for user permissions"""
    user_id: str = Field(..., description="User ID")
    permissions: List[str] = Field(default_factory=list, description="List of permissions")
    roles: List[str] = Field(default_factory=list, description="List of roles")

class AuditLog(BaseModel):
    """Model for audit logging"""
    log_id: str = Field(..., description="Log ID")
    user_id: str = Field(..., description="User ID")
    action: str = Field(..., description="Action performed")
    resource: str = Field(..., description="Resource accessed")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    timestamp: datetime = Field(..., description="Action timestamp")
    details: Optional[dict] = Field(None, description="Additional details")
    success: bool = Field(..., description="Action success status")
