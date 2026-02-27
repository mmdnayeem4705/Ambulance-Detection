"""
Startup script for Flask web app
Opens browser automatically after starting the server
"""
import webbrowser
import threading
import time
from app import app

def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask app
    print("\n" + "="*60)
    print("Starting Ambulance Detection Web App...")
    print("="*60)
    print("Browser will open automatically!")
    print("If not, manually go to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
