#!/usr/bin/env python3
"""
ImpViz Trading Analytics API Server
Start the FastAPI backend server
"""

import uvicorn
import sys
from pathlib import Path

def main():
    """Start the API server"""
    print("=" * 60)
    print("🚀 ImpViz Trading Analytics API Backend")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("api_server.py").exists():
        print("❌ Error: Please run this script from the API directory")
        print("   Current directory should contain: api_server.py")
        sys.exit(1)
    
    print("🔄 Starting FastAPI server...")
    print("📡 API will be available at: http://127.0.0.1:8000")
    print("📚 API docs available at: http://127.0.0.1:8000/docs")
    print("🔍 ReDoc available at: http://127.0.0.1:8000/redoc")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "api_server:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 API server stopped")
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()