"""
Run Surveillance System with Dashboard
"""

import subprocess
import sys
import threading
import time

def run_surveillance():
    """Run main surveillance system"""
    subprocess.run([sys.executable, "main.py"])

def run_dashboard():
    """Run Streamlit dashboard"""
    subprocess.run([sys.executable, "-m", "streamlit", "run", "interface/dashboard.py"])

if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Surveillance System")
    print("=" * 60)
    
    # Start surveillance in background thread
    surv_thread = threading.Thread(target=run_surveillance, daemon=True)
    surv_thread.start()
    
    time.sleep(3)  # Wait for system to initialize
    
    # Start dashboard (blocks until closed)
    print("\n🚀 Starting Dashboard...")
    print("Open http://localhost:8501 in your browser")
    print("=" * 60)
    
    run_dashboard()