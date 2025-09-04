#!/usr/bin/env python3
"""
Fix SSL certificate issues for PyTorch model downloads
"""

import ssl
import urllib.request
import os

def fix_ssl():
    """Fix SSL certificate verification issues"""
    print("🔧 Fixing SSL certificate issues...")
    
    # Create unverified SSL context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Set environment variables
    os.environ['PYTHONHTTPSVERIFY'] = '0'
    
    print("✅ SSL context fixed for PyTorch model downloads")

if __name__ == "__main__":
    fix_ssl()



