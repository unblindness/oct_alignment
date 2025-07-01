# main.py

"""
Multi-volume-alignment-average tool by Taehoon Kim
"""

import tkinter as tk
from tkinter import messagebox
import traceback
from datetime import datetime

# ui.py에서 메인 앱 클래스를 가져옴
from ui import SimpleOCTApp

def main():
    """run application"""
    try:
        app = SimpleOCTApp()
        app.run()
    except Exception as e:
        # Fallback mechanism to handle failures during application startup
        root = tk.Tk()
        root.withdraw()
        error_msg = f"Application failed to start:\n{str(e)}\n\nFull error:\n{traceback.format_exc()}"
        messagebox.showerror("Critical Error", error_msg)
        try:
            with open('error_log.txt', 'w') as f:
                f.write(f"Error occurred at {datetime.now()}\n")
                f.write(error_msg)
        except Exception as log_e:
            messagebox.showerror("Logging Error", f"Failed to write to error_log.txt:\n{log_e}")


if __name__ == "__main__":
    main()