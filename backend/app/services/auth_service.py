import os
import jwt
import bcrypt
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from models.auth import (
    UserCreate, UserLogin, UserProfile, UserUpdate, UserRole, UserStatus,
    TokenResponse, UserSession, AuditLog
)
from services.database import get_database

class AuthService:
    """Authentication service for JWT-based user management"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        self.refresh_token_expire_days = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        
        # Email configuration
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME")
        self.smtp_password = os.getenv("SMTP_PASSWORD")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@healthaitwin.com")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def register_user(self, user_data: UserCreate) -> UserProfile:
        """Register a new user"""
        db = get_database()
        
        # Check if user already exists
        existing_user = await db.users.find_one({"email": user_data.email.lower()})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create user document
        user_id = str(uuid.uuid4())
        hashed_password = self.hash_password(user_data.password)
        
        user_doc = {
            "_id": user_id,
            "email": user_data.email.lower(),
            "password_hash": hashed_password,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "date_of_birth": user_data.date_of_birth,
            "gender": user_data.gender,
            "phone": user_data.phone,
            "role": user_data.role.value,
            "status": UserStatus.PENDING.value,
            "emergency_contact": user_data.emergency_contact,
            "preferences": {},
            "created_at": datetime.utcnow(),
            "last_login": None,
            "is_verified": False,
            "verification_token": secrets.token_urlsafe(32),
            "reset_token": None,
            "reset_token_expires": None
        }
        
        # Insert user into database
        await db.users.insert_one(user_doc)
        
        # Send verification email
        await self.send_verification_email(user_data.email, user_doc["verification_token"])
        
        # Return user profile
        return UserProfile(
            user_id=user_id,
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            date_of_birth=user_data.date_of_birth,
            gender=user_data.gender,
            phone=user_data.phone,
            role=user_data.role,
            status=UserStatus.PENDING,
            emergency_contact=user_data.emergency_contact,
            preferences={},
            created_at=user_doc["created_at"],
            last_login=None,
            is_verified=False
        )
    
    async def login_user(self, login_data: UserLogin, ip_address: str = None, user_agent: str = None) -> TokenResponse:
        """Authenticate user and return JWT tokens"""
        db = get_database()
        
        # Find user by email
        user = await db.users.find_one({"email": login_data.email.lower()})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Verify password
        if not self.verify_password(login_data.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password"
            )
        
        # Check if user is active
        if user["status"] != UserStatus.ACTIVE.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is not active"
            )
        
        # Update last login
        await db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        
        # Create session
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "user_id": user["_id"],
            "device_info": {"ip_address": ip_address, "user_agent": user_agent},
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "is_active": True
        }
        await db.user_sessions.insert_one(session_data)
        
        # Create tokens
        token_data = {
            "sub": user["_id"],
            "email": user["email"],
            "role": user["role"],
            "session_id": session_id
        }
        
        access_token = self.create_access_token(token_data)
        refresh_token = self.create_refresh_token(token_data)
        
        # Create user profile
        user_profile = UserProfile(
            user_id=user["_id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            date_of_birth=user["date_of_birth"],
            gender=user["gender"],
            phone=user.get("phone"),
            role=UserRole(user["role"]),
            status=UserStatus(user["status"]),
            emergency_contact=user.get("emergency_contact"),
            preferences=user.get("preferences", {}),
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            is_verified=user.get("is_verified", False)
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60,
            user=user_profile
        )
    
    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Refresh access token using refresh token"""
        # Verify refresh token
        payload = self.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        db = get_database()
        
        # Check if user still exists
        user = await db.users.find_one({"_id": payload["sub"]})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Check if session is still active
        session = await db.user_sessions.find_one({
            "session_id": payload["session_id"],
            "is_active": True
        })
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired"
            )
        
        # Update session activity
        await db.user_sessions.update_one(
            {"session_id": payload["session_id"]},
            {"$set": {"last_activity": datetime.utcnow()}}
        )
        
        # Create new access token
        token_data = {
            "sub": user["_id"],
            "email": user["email"],
            "role": user["role"],
            "session_id": payload["session_id"]
        }
        
        new_access_token = self.create_access_token(token_data)
        
        # Create user profile
        user_profile = UserProfile(
            user_id=user["_id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            date_of_birth=user["date_of_birth"],
            gender=user["gender"],
            phone=user.get("phone"),
            role=UserRole(user["role"]),
            status=UserStatus(user["status"]),
            emergency_contact=user.get("emergency_contact"),
            preferences=user.get("preferences", {}),
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            is_verified=user.get("is_verified", False)
        )
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60,
            user=user_profile
        )
    
    async def logout_user(self, session_id: str) -> bool:
        """Logout user by deactivating session"""
        db = get_database()
        
        result = await db.user_sessions.update_one(
            {"session_id": session_id},
            {"$set": {"is_active": False, "last_activity": datetime.utcnow()}}
        )
        
        return result.modified_count > 0
    
    async def get_current_user(self, token: str) -> UserProfile:
        """Get current user from JWT token"""
        payload = self.verify_token(token)
        
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        db = get_database()
        user = await db.users.find_one({"_id": payload["sub"]})
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return UserProfile(
            user_id=user["_id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            date_of_birth=user["date_of_birth"],
            gender=user["gender"],
            phone=user.get("phone"),
            role=UserRole(user["role"]),
            status=UserStatus(user["status"]),
            emergency_contact=user.get("emergency_contact"),
            preferences=user.get("preferences", {}),
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            is_verified=user.get("is_verified", False)
        )
    
    async def update_user_profile(self, user_id: str, update_data: UserUpdate) -> UserProfile:
        """Update user profile"""
        db = get_database()
        
        # Build update document
        update_doc = {}
        if update_data.first_name is not None:
            update_doc["first_name"] = update_data.first_name
        if update_data.last_name is not None:
            update_doc["last_name"] = update_data.last_name
        if update_data.phone is not None:
            update_doc["phone"] = update_data.phone
        if update_data.emergency_contact is not None:
            update_doc["emergency_contact"] = update_data.emergency_contact
        if update_data.preferences is not None:
            update_doc["preferences"] = update_data.preferences
        
        if not update_doc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Update user
        result = await db.users.update_one(
            {"_id": user_id},
            {"$set": update_doc}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get updated user
        user = await db.users.find_one({"_id": user_id})
        
        return UserProfile(
            user_id=user["_id"],
            email=user["email"],
            first_name=user["first_name"],
            last_name=user["last_name"],
            date_of_birth=user["date_of_birth"],
            gender=user["gender"],
            phone=user.get("phone"),
            role=UserRole(user["role"]),
            status=UserStatus(user["status"]),
            emergency_contact=user.get("emergency_contact"),
            preferences=user.get("preferences", {}),
            created_at=user["created_at"],
            last_login=user.get("last_login"),
            is_verified=user.get("is_verified", False)
        )
    
    async def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change user password"""
        db = get_database()
        
        # Get user
        user = await db.users.find_one({"_id": user_id})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not self.verify_password(current_password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_password_hash = self.hash_password(new_password)
        
        # Update password
        result = await db.users.update_one(
            {"_id": user_id},
            {"$set": {"password_hash": new_password_hash}}
        )
        
        return result.modified_count > 0
    
    async def send_verification_email(self, email: str, token: str) -> bool:
        """Send email verification"""
        if not all([self.smtp_username, self.smtp_password]):
            # In development, just log the token
            print(f"Verification token for {email}: {token}")
            return True
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = email
            msg['Subject'] = "Verify your Health AI Twin account"
            
            body = f"""
            Welcome to Health AI Twin!
            
            Please click the following link to verify your email address:
            http://localhost:3000/verify-email?token={token}
            
            If you didn't create an account, please ignore this email.
            
            Best regards,
            Health AI Twin Team
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Error sending verification email: {e}")
            return False
    
    async def verify_email(self, token: str) -> bool:
        """Verify user email"""
        db = get_database()
        
        # Find user with verification token
        user = await db.users.find_one({"verification_token": token})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token"
            )
        
        # Update user status
        result = await db.users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "is_verified": True,
                    "status": UserStatus.ACTIVE.value,
                    "verification_token": None
                }
            }
        )
        
        return result.modified_count > 0
    
    async def log_audit_event(self, user_id: str, action: str, resource: str, 
                            ip_address: str = None, user_agent: str = None, 
                            details: dict = None, success: bool = True) -> None:
        """Log audit event"""
        db = get_database()
        
        audit_log = {
            "log_id": str(uuid.uuid4()),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow(),
            "details": details or {},
            "success": success
        }
        
        await db.audit_logs.insert_one(audit_log)
