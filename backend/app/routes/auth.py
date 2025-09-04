from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import uuid

from models.auth import (
    UserCreate, UserLogin, UserProfile, UserUpdate, PasswordChange,
    PasswordReset, PasswordResetConfirm, EmailVerification, RefreshToken,
    TokenResponse
)
from services.auth_service import AuthService

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])
security = HTTPBearer()

# Initialize auth service
auth_service = AuthService()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserProfile:
    """Dependency to get current authenticated user"""
    try:
        return await auth_service.get_current_user(credentials.credentials)
    except HTTPException:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, request: Request):
    """Register a new user account"""
    try:
        # Get client IP and user agent
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Register user
        user_profile = await auth_service.register_user(user_data)
        
        # Log audit event
        await auth_service.log_audit_event(
            user_id=user_profile.user_id,
            action="user_registration",
            resource="auth",
            ip_address=ip_address,
            user_agent=user_agent,
            details={"email": user_data.email},
            success=True
        )
        
        return user_profile
    except HTTPException:
        raise
    except Exception as e:
        # Log audit event for failure
        await auth_service.log_audit_event(
            user_id="unknown",
            action="user_registration",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": user_data.email, "error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin, request: Request):
    """Authenticate user and return JWT tokens"""
    try:
        # Get client IP and user agent
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Login user
        token_response = await auth_service.login_user(login_data, ip_address, user_agent)
        
        # Log audit event
        await auth_service.log_audit_event(
            user_id=token_response.user.user_id,
            action="user_login",
            resource="auth",
            ip_address=ip_address,
            user_agent=user_agent,
            details={"email": login_data.email},
            success=True
        )
        
        return token_response
    except HTTPException:
        # Log failed login attempt
        await auth_service.log_audit_event(
            user_id="unknown",
            action="user_login",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": login_data.email},
            success=False
        )
        raise
    except Exception as e:
        # Log audit event for failure
        await auth_service.log_audit_event(
            user_id="unknown",
            action="user_login",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": login_data.email, "error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate user"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_data: RefreshToken, request: Request):
    """Refresh access token using refresh token"""
    try:
        # Get client IP and user agent
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")
        
        # Refresh token
        token_response = await auth_service.refresh_token(refresh_data.refresh_token)
        
        # Log audit event
        await auth_service.log_audit_event(
            user_id=token_response.user.user_id,
            action="token_refresh",
            resource="auth",
            ip_address=ip_address,
            user_agent=user_agent,
            success=True
        )
        
        return token_response
    except HTTPException:
        # Log failed refresh attempt
        await auth_service.log_audit_event(
            user_id="unknown",
            action="token_refresh",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            success=False
        )
        raise
    except Exception as e:
        # Log audit event for failure
        await auth_service.log_audit_event(
            user_id="unknown",
            action="token_refresh",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )

@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout_user(request: Request, current_user: UserProfile = Depends(get_current_user)):
    """Logout user by deactivating session"""
    try:
        # Get session ID from token
        credentials = await security(request)
        payload = auth_service.verify_token(credentials.credentials)
        session_id = payload.get("session_id")
        
        if session_id:
            # Logout user
            success = await auth_service.logout_user(session_id)
            
            # Log audit event
            await auth_service.log_audit_event(
                user_id=current_user.user_id,
                action="user_logout",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"session_id": session_id},
                success=success
            )
            
            return {"message": "Successfully logged out"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session"
            )
    except HTTPException:
        raise
    except Exception as e:
        # Log audit event for failure
        await auth_service.log_audit_event(
            user_id=current_user.user_id,
            action="user_logout",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout user"
        )

@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: UserProfile = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@router.put("/me", response_model=UserProfile)
async def update_user_profile(
    update_data: UserUpdate,
    current_user: UserProfile = Depends(get_current_user),
    request: Request = None
):
    """Update current user profile"""
    try:
        # Update user profile
        updated_profile = await auth_service.update_user_profile(current_user.user_id, update_data)
        
        # Log audit event
        if request:
            await auth_service.log_audit_event(
                user_id=current_user.user_id,
                action="profile_update",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"updated_fields": list(update_data.dict(exclude_unset=True).keys())},
                success=True
            )
        
        return updated_profile
    except HTTPException:
        raise
    except Exception as e:
        # Log audit event for failure
        if request:
            await auth_service.log_audit_event(
                user_id=current_user.user_id,
                action="profile_update",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"error": str(e)},
                success=False
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    password_data: PasswordChange,
    current_user: UserProfile = Depends(get_current_user),
    request: Request = None
):
    """Change user password"""
    try:
        # Change password
        success = await auth_service.change_password(
            current_user.user_id,
            password_data.current_password,
            password_data.new_password
        )
        
        if success:
            # Log audit event
            if request:
                await auth_service.log_audit_event(
                    user_id=current_user.user_id,
                    action="password_change",
                    resource="auth",
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    success=True
                )
            
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to change password"
            )
    except HTTPException:
        raise
    except Exception as e:
        # Log audit event for failure
        if request:
            await auth_service.log_audit_event(
                user_id=current_user.user_id,
                action="password_change",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"error": str(e)},
                success=False
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(reset_data: PasswordReset, request: Request):
    """Request password reset"""
    try:
        # TODO: Implement password reset functionality
        # For now, just return success message
        await auth_service.log_audit_event(
            user_id="unknown",
            action="password_reset_request",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": reset_data.email},
            success=True
        )
        
        return {"message": "Password reset email sent (if email exists)"}
    except Exception as e:
        await auth_service.log_audit_event(
            user_id="unknown",
            action="password_reset_request",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"email": reset_data.email, "error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        )

@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(reset_data: PasswordResetConfirm, request: Request):
    """Reset password using token"""
    try:
        # TODO: Implement password reset confirmation
        # For now, just return success message
        await auth_service.log_audit_event(
            user_id="unknown",
            action="password_reset_confirm",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"token": reset_data.token[:10] + "..."},  # Log partial token for security
            success=True
        )
        
        return {"message": "Password reset successfully"}
    except Exception as e:
        await auth_service.log_audit_event(
            user_id="unknown",
            action="password_reset_confirm",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"token": reset_data.token[:10] + "...", "error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )

@router.post("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(verification_data: EmailVerification, request: Request):
    """Verify user email address"""
    try:
        # Verify email
        success = await auth_service.verify_email(verification_data.token)
        
        if success:
            await auth_service.log_audit_event(
                user_id="unknown",
                action="email_verification",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"token": verification_data.token[:10] + "..."},
                success=True
            )
            
            return {"message": "Email verified successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to verify email"
            )
    except HTTPException:
        raise
    except Exception as e:
        await auth_service.log_audit_event(
            user_id="unknown",
            action="email_verification",
            resource="auth",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            details={"token": verification_data.token[:10] + "...", "error": str(e)},
            success=False
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email"
        )

@router.post("/resend-verification", status_code=status.HTTP_200_OK)
async def resend_verification_email(
    current_user: UserProfile = Depends(get_current_user),
    request: Request = None
):
    """Resend email verification"""
    try:
        # TODO: Implement resend verification functionality
        # For now, just return success message
        if request:
            await auth_service.log_audit_event(
                user_id=current_user.user_id,
                action="resend_verification",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"email": current_user.email},
                success=True
            )
        
        return {"message": "Verification email sent"}
    except Exception as e:
        if request:
            await auth_service.log_audit_event(
                user_id=current_user.user_id,
                action="resend_verification",
                resource="auth",
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                details={"email": current_user.email, "error": str(e)},
                success=False
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend verification email"
        )
