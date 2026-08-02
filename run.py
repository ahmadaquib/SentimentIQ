"""
SentimentIQ Unified Launcher Script

Launches both FastAPI Backend and Streamlit Frontend concurrently.
Gracefully handles shutdown on Ctrl+C (KeyboardInterrupt).
"""

import sys
import os
import time
import subprocess

def main():
    print("=" * 60)
    print("🚀 Starting SentimentIQ Application Platform")
    print("=" * 60)

    python_executable = sys.executable

    # Command 1: Backend Uvicorn Server
    backend_cmd = [
        python_executable, "-m", "uvicorn", "backend.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]

    # Command 2: Frontend Streamlit Web App
    frontend_cmd = [
        python_executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "127.0.0.1"
    ]

    print(f"[Launcher] Starting FastAPI Backend on http://127.0.0.1:8000...")
    backend_proc = subprocess.Popen(backend_cmd)

    time.sleep(2)

    print(f"[Launcher] Starting Streamlit Frontend on http://127.0.0.1:8501...")
    frontend_proc = subprocess.Popen(frontend_cmd)

    print("\n" + "=" * 60)
    print("✅ SentimentIQ is running!")
    print("   • API Swagger Docs: http://127.0.0.1:8000/docs")
    print("   • Streamlit Web App: http://127.0.0.1:8501")
    print("   Press Ctrl+C to terminate both servers.")
    print("=" * 60 + "\n")

    try:
        while True:
            time.sleep(1)
            # Check if any process terminated unexpectedly
            if backend_proc.poll() is not None:
                print("[Launcher] Warning: Backend process exited.")
                break
            if frontend_proc.poll() is not None:
                print("[Launcher] Warning: Frontend process exited.")
                break
    except KeyboardInterrupt:
        print("\n[Launcher] Shutting down SentimentIQ servers...")
    finally:
        if backend_proc.poll() is None:
            backend_proc.terminate()
            backend_proc.wait()
        if frontend_proc.poll() is None:
            frontend_proc.terminate()
            frontend_proc.wait()
        print("[Launcher] Shutdown complete.")

if __name__ == "__main__":
    main()
