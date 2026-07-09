#!/usr/bin/env python3
"""Script to verify network connectivity from a remote worker to the main server services."""

import os
import sys

# Load environment variables
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
CALLBACK_URL = os.environ.get("BACKEND_CALLBACK_URL", "http://localhost:8080/tlhub/api/internal/jobs/callback")
INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN", "")

print("==================================================")
print("🔍 REMOTE WORKER CONNECTIVITY TESTER")
print("==================================================")
print(f"Redis Host:       {REDIS_HOST}:{REDIS_PORT}")
print(f"MinIO Endpoint:   {MINIO_ENDPOINT}")
print(f"Callback URL:     {CALLBACK_URL}")
print("==================================================")

errors = 0

# 1. Test Valkey / Redis Connectivity
print("\n[1/3] Testing Valkey/Redis connection...")
try:
    import redis
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_timeout=5)
    if client.ping():
        print("✅ Redis: Connected successfully!")
    else:
        print("❌ Redis: Ping failed (returned False)")
        errors += 1
except ImportError:
    print("⚠️  Redis package not installed. Skipping test.")
except Exception as e:
    print(f"❌ Redis: Connection failed: {e}")
    errors += 1

# 2. Test MinIO Connectivity
print("\n[2/3] Testing MinIO connection...")
try:
    from minio import Minio
    # Strip protocol helper if included by accident
    endpoint = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    minio_client = Minio(
        endpoint,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    # Check if we can list buckets
    buckets = minio_client.list_buckets()
    print(f"✅ MinIO: Connected successfully! Found {len(buckets)} bucket(s).")
except ImportError:
    print("⚠️  Minio package not installed. Skipping test.")
except Exception as e:
    print(f"❌ MinIO: Connection failed: {e}")
    print("👉 Hint: Ensure port 9000 is exposed on the main server's docker-compose.")
    errors += 1

# 3. Test Spring Boot Backend Connectivity
print("\n[3/3] Testing Backend API connection...")
try:
    import requests
    # Deriving actuator health URL
    base_url = CALLBACK_URL.split("/api/")[0]
    health_url = f"{base_url}/actuator/health"
    print(f"Pinging health actuator: {health_url}")
    
    headers = {"X-Internal-Token": INTERNAL_API_TOKEN} if INTERNAL_API_TOKEN else {}
    res = requests.get(health_url, headers=headers, timeout=5)
    if res.status_code == 200:
        print(f"✅ Backend: Connected successfully! Response: {res.json()}")
    else:
        print(f"❌ Backend: Returned status code {res.status_code}. Response: {res.text}")
        errors += 1
except ImportError:
    print("⚠️  Requests package not installed. Skipping test.")
except Exception as e:
    print(f"❌ Backend: Connection failed: {e}")
    errors += 1

print("\n==================================================")
if errors == 0:
    print("🎉 ALL TESTS PASSED! Your remote worker can connect successfully.")
    sys.exit(0)
else:
    print(f"⚠️  {errors} TEST(S) FAILED. Please review the errors and hints above.")
    sys.exit(1)
