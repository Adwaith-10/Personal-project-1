import os
import uuid
from typing import Optional, BinaryIO
import logging
from app.config import settings
from datetime import datetime
import aiofiles
import mimetypes

logger = logging.getLogger(__name__)


class StorageService:
    def __init__(self):
        self.storage_type = settings.STORAGE_TYPE
        self.local_path = settings.LOCAL_STORAGE_PATH
        self.s3_bucket = settings.S3_BUCKET
        self.s3_access_key = settings.S3_ACCESS_KEY
        self.s3_secret_key = settings.S3_SECRET_KEY
        self.s3_endpoint = settings.S3_ENDPOINT_URL
        
        # Ensure local storage directory exists
        if self.storage_type == "local":
            os.makedirs(self.local_path, exist_ok=True)
    
    async def save_image(self, file_content: BinaryIO, filename: str = None, 
                        user_id: str = None) -> str:
        """
        Save an uploaded image file
        
        Args:
            file_content: File content as file-like object
            filename: Original filename (optional)
            user_id: User ID for organizing files (optional)
            
        Returns:
            Path/URL to the saved image
        """
        try:
            # Generate unique filename
            if not filename:
                filename = f"image_{uuid.uuid4().hex}.jpg"
            else:
                # Extract extension and generate unique name
                name, ext = os.path.splitext(filename)
                if not ext:
                    ext = ".jpg"
                filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            
            # Create user-specific directory
            if user_id:
                user_dir = os.path.join(self.local_path, user_id)
                os.makedirs(user_dir, exist_ok=True)
                file_path = os.path.join(user_dir, filename)
            else:
                file_path = os.path.join(self.local_path, filename)
            
            # Save file based on storage type
            if self.storage_type == "local":
                return await self._save_local(file_content, file_path)
            elif self.storage_type == "s3":
                return await self._save_s3(file_content, filename, user_id)
            elif self.storage_type == "minio":
                return await self._save_minio(file_content, filename, user_id)
            else:
                raise ValueError(f"Unsupported storage type: {self.storage_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to save image: {e}")
            raise e
    
    async def _save_local(self, file_content: BinaryIO, file_path: str) -> str:
        """Save file to local storage"""
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                # Read file content in chunks
                chunk_size = 8192
                while True:
                    chunk = file_content.read(chunk_size)
                    if not chunk:
                        break
                    await f.write(chunk)
            
            logger.info(f"✅ Image saved locally: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"❌ Local save failed: {e}")
            raise e
    
    async def _save_s3(self, file_content: BinaryIO, filename: str, user_id: str = None) -> str:
        """Save file to S3"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            # Create S3 client
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                endpoint_url=self.s3_endpoint if self.s3_endpoint else None
            )
            
            # Generate S3 key
            if user_id:
                s3_key = f"users/{user_id}/{filename}"
            else:
                s3_key = f"uploads/{filename}"
            
            # Upload file
            file_content.seek(0)  # Reset file pointer
            s3_client.upload_fileobj(file_content, self.s3_bucket, s3_key)
            
            # Generate URL
            if self.s3_endpoint:
                # Custom endpoint (e.g., MinIO)
                url = f"{self.s3_endpoint}/{self.s3_bucket}/{s3_key}"
            else:
                # AWS S3
                url = f"https://{self.s3_bucket}.s3.amazonaws.com/{s3_key}"
            
            logger.info(f"✅ Image saved to S3: {url}")
            return url
            
        except ImportError:
            logger.error("❌ boto3 not installed for S3 storage")
            raise RuntimeError("S3 storage requires boto3 package")
        except Exception as e:
            logger.error(f"❌ S3 save failed: {e}")
            raise e
    
    async def _save_minio(self, file_content: BinaryIO, filename: str, user_id: str = None) -> str:
        """Save file to MinIO (S3-compatible)"""
        try:
            from minio import Minio
            from minio.error import S3Error
            
            # Create MinIO client
            minio_client = Minio(
                self.s3_endpoint.replace("http://", "").replace("https://", ""),
                access_key=self.s3_access_key,
                secret_key=self.s3_secret_key,
                secure=self.s3_endpoint.startswith("https://")
            )
            
            # Ensure bucket exists
            if not minio_client.bucket_exists(self.s3_bucket):
                minio_client.make_bucket(self.s3_bucket)
            
            # Generate object name
            if user_id:
                object_name = f"users/{user_id}/{filename}"
            else:
                object_name = f"uploads/{filename}"
            
            # Upload file
            file_content.seek(0)  # Reset file pointer
            minio_client.put_object(
                self.s3_bucket,
                object_name,
                file_content,
                length=-1,  # Unknown length
                content_type=mimetypes.guess_type(filename)[0] or "application/octet-stream"
            )
            
            # Generate URL
            url = f"{self.s3_endpoint}/{self.s3_bucket}/{object_name}"
            
            logger.info(f"✅ Image saved to MinIO: {url}")
            return url
            
        except ImportError:
            logger.error("❌ minio package not installed")
            raise RuntimeError("MinIO storage requires minio package")
        except Exception as e:
            logger.error(f"❌ MinIO save failed: {e}")
            raise e
    
    async def delete_image(self, image_path: str) -> bool:
        """
        Delete an image file
        
        Args:
            image_path: Path/URL to the image
            
        Returns:
            True if deletion was successful
        """
        try:
            if self.storage_type == "local":
                return await self._delete_local(image_path)
            elif self.storage_type in ["s3", "minio"]:
                return await self._delete_remote(image_path)
            else:
                logger.warning(f"Unknown storage type: {self.storage_type}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete image: {e}")
            return False
    
    async def _delete_local(self, file_path: str) -> bool:
        """Delete local file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"✅ Local image deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Local delete failed: {e}")
            return False
    
    async def _delete_remote(self, image_url: str) -> bool:
        """Delete remote file (S3/MinIO)"""
        try:
            # Extract bucket and key from URL
            if "/" in image_url:
                parts = image_url.split("/")
                if len(parts) >= 3:
                    bucket = parts[-2]
                    key = "/".join(parts[-1:])
                    
                    if self.storage_type == "s3":
                        import boto3
                        s3_client = boto3.client(
                            's3',
                            aws_access_key_id=self.s3_access_key,
                            aws_secret_access_key=self.s3_secret_key,
                            endpoint_url=self.s3_endpoint if self.s3_endpoint else None
                        )
                        s3_client.delete_object(Bucket=bucket, Key=key)
                        
                    elif self.storage_type == "minio":
                        from minio import Minio
                        minio_client = Minio(
                            self.s3_endpoint.replace("http://", "").replace("https://", ""),
                            access_key=self.s3_access_key,
                            secret_key=self.s3_secret_key,
                            secure=self.s3_endpoint.startswith("https://")
                        )
                        minio_client.remove_object(bucket, key)
                    
                    logger.info(f"✅ Remote image deleted: {image_url}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Remote delete failed: {e}")
            return False
    
    def get_image_url(self, image_path: str) -> str:
        """
        Get public URL for an image
        
        Args:
            image_path: Local file path or storage path
            
        Returns:
            Public URL for the image
        """
        try:
            if self.storage_type == "local":
                # For local storage, return relative path
                # In production, you'd want to serve these through a web server
                return f"/uploads/{os.path.basename(image_path)}"
            else:
                # For remote storage, return the full URL
                return image_path
                
        except Exception as e:
            logger.error(f"❌ Failed to get image URL: {e}")
            return image_path
    
    def cleanup_old_files(self, days_old: int = 30) -> int:
        """
        Clean up old files from local storage
        
        Args:
            days_old: Delete files older than this many days
            
        Returns:
            Number of files deleted
        """
        if self.storage_type != "local":
            return 0
            
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 3600)
            
            deleted_count = 0
            
            for root, dirs, files in os.walk(self.local_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)
                    
                    if file_time < cutoff_time:
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete old file {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"✅ Cleaned up {deleted_count} old files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 0


# Global instance
storage_service = StorageService()
