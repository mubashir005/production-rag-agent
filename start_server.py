#!/usr/bin/env python3
"""
Production server startup script with optimized settings for large file uploads.
This script configures uvicorn with proper limits for handling large documents.

Usage:
    python start_server.py
"""
import os
import uvicorn

if __name__ == "__main__":
    # Get port from environment variable (Railway, Render, etc.) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    
    # Configure uvicorn with settings optimized for large file uploads
    uvicorn.run(
        "app.server:app",
        host="0.0.0.0",
        port=port,
        # Increase timeout for large file uploads (5 minutes)
        timeout_keep_alive=300,
        # Set high limit for concurrent connections
        limit_concurrency=1000,
        # No limit on number of requests (set to 0)
        limit_max_requests=0,
        # Log level
        log_level="info",
        # Disable access log for better performance (enable for debugging)
        access_log=True,
        # Use automatic reload in development only
        reload=os.environ.get("ENV") == "development",
    )
