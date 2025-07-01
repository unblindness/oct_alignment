# ui.py
"""
Multi-volume-alignment-average tool by Taehoon Kim
"""

"""
Tkinter based GUI
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from datetime import datetime

# # Import necessary components from local modules
from config import CONFIG
import processing


class SimpleOCTApp:
    def __init__(self):
        self.root = tk.Tk()

        try:
            self.root.tk.call('tk', 'appname', 'OCT Alignment Tool')
            self.root.tk.call('::tk::unsupported::MacWindowStyle', 'style', self.root._w, 'document')
        except Exception:
            pass  # macOS-specific style; ignored on other operating systems

        self.root.title("OCT Volume Alignment Tool V1.0 by Taehoon Kim")
        self.root.geometry("650x750")
        self.root.configure(bg='white')
        self.root.option_add('*Background', 'white')
        self.root.option_add('*Foreground', 'black')
        self.root.resizable(True, True)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Custom.TFrame', background='white')
        self.style.configure('Custom.TLabel', background='white', foreground='black')
        self.style.configure('Custom.TButton', background='#007AFF', foreground='white')

        self.setup_ui()

    def setup_ui(self):
        main_container = tk.Frame(self.root, bg='white', highlightbackground='gray', highlightthickness=1)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        title_frame = tk.Frame(main_container, bg='white', height=60)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        title_frame.pack_propagate(False)
        title_label = tk.Label(title_frame, text="OCT Volume Alignment Tool", font=('Arial', 20, 'bold'), bg='white',
                               fg='black')
        title_label.pack(expand=True)

        inst_frame = tk.Frame(main_container, bg='white')
        inst_frame.pack(fill=tk.X, pady=(0, 20))
        instructions = tk.Label(inst_frame,
                                text="This tool will align and average your OCT volumes.\n" + "Adjust settings below, then click 'Start Processing'.",
                                font=('Arial', 12), bg='white', fg='#333333', justify=tk.CENTER)
        instructions.pack()

        settings_frame = tk.LabelFrame(main_container, text=" Settings ", bg='white', fg='black',
                                       font=('Arial', 14, 'bold'), relief='solid', bd=2, labelanchor='n')
        settings_frame.pack(fill=tk.X, pady=(0, 20), padx=5)
        settings_inner = tk.Frame(settings_frame, bg='white')
        settings_inner.pack(fill=tk.X, padx=15, pady=15)

        self.config_vars = {}
        self.create_setting_row(settings_inner, "Rows to trim from top:", 'rows_to_trim', CONFIG['rows_to_trim'],
                                var_type=tk.IntVar)
        self.create_setting_row(settings_inner, "Zero padding size:", 'pad_size', CONFIG['pad_size'],
                                var_type=tk.IntVar)
        self.create_setting_row(settings_inner, "Batch size (volumes per batch):", 'batch_size', CONFIG['batch_size'],
                                var_type=tk.IntVar)
        self.create_setting_row(settings_inner, "En Face Clip Limit:", 'clipLimit', CONFIG['clipLimit'],
                                var_type=tk.DoubleVar)

        button_frame = tk.Frame(main_container, bg='white', height=80)
        button_frame.pack(fill=tk.X, pady=20)
        button_frame.pack_propagate(False)
        self.start_button = tk.Button(button_frame, text="Start Processing", command=self.start_processing,
                                      bg='#007AFF', fg='white', font=('Arial', 16, 'bold'), padx=40, pady=20,
                                      relief='raised', bd=3, cursor='hand2', activebackground='#0056b3',
                                      activeforeground='white')
        self.start_button.pack(expand=True)

        progress_frame = tk.LabelFrame(main_container, text=" Progress ", bg='white', fg='black',
                                       font=('Arial', 14, 'bold'), relief='solid', bd=2, labelanchor='n')
        progress_frame.pack(fill=tk.X, pady=(0, 20), padx=5)
        progress_inner = tk.Frame(progress_frame, bg='white')
        progress_inner.pack(fill=tk.X, padx=15, pady=15)

        self.status_var = tk.StringVar(value="Ready to process files...")
        self.status_label = tk.Label(progress_inner, textvariable=self.status_var, bg='white', fg='black',
                                     font=('Arial', 11))
        self.status_label.pack(pady=(0, 10))

        progress_container = tk.Frame(progress_inner, bg='white')
        progress_container.pack(fill=tk.X, pady=(0, 10))
        self.progress_bar = tk.Frame(progress_container, bg='#d0d0d0', height=25, relief='sunken', bd=2)
        self.progress_bar.pack(fill=tk.X)
        self.progress_fill = tk.Frame(self.progress_bar, bg='#e0e0e0', height=21)
        self.progress_fill.place(x=2, y=2, width=0, height=21)
        self.progress_text = tk.Label(progress_inner, text="0%", bg='white', fg='black', font=('Arial', 11, 'bold'))
        self.progress_text.pack()

        log_frame = tk.LabelFrame(main_container, text=" Processing Log ", bg='white', fg='black',
                                  font=('Arial', 14, 'bold'), relief='solid', bd=2, labelanchor='n')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        log_inner = tk.Frame(log_frame, bg='white')
        log_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_frame = tk.Frame(log_inner, bg='white')
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(text_frame, height=12, wrap=tk.WORD, bg='white', fg='black', font=('Arial', 10),
                                relief='sunken', bd=2, insertbackground='black')
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview, bg='white',
                                 troughcolor='#f0f0f0')
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.root.update()

    def create_setting_row(self, parent, label_text, var_name, default_value, var_type=tk.IntVar):
        row_frame = tk.Frame(parent, bg='white')
        row_frame.pack(fill=tk.X, pady=8)
        label = tk.Label(row_frame, text=label_text, bg='white', fg='black', font=('Arial', 12))
        label.pack(side=tk.LEFT)
        self.config_vars[var_name] = var_type(value=default_value)
        entry = tk.Entry(row_frame, textvariable=self.config_vars[var_name], width=12, font=('Arial', 12), bg='white',
                         fg='black', relief='solid', bd=2, insertbackground='black')
        entry.pack(side=tk.RIGHT)

    def update_progress(self, percentage, message=None):
        if message:
            self.status_var.set(message)

        bar_width = self.progress_bar.winfo_width()
        if bar_width > 4:
            fill_width = max(0, int((bar_width - 4) * percentage / 100))
            self.progress_fill.config(width=fill_width)

        self.progress_text.config(text=f"{percentage:.1f}%")

        if percentage >= 100:
            self.progress_fill.config(bg='#28a745')
        elif percentage > 0:
            self.progress_fill.config(bg='#007AFF')
        else:
            self.progress_fill.config(bg='#e0e0e0')
        self.root.update_idletasks()

    def reset_progress(self):
        self.update_progress(0, "Ready to process files...")

    def log_message(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def start_processing(self):
        try:
            for key, var in self.config_vars.items():
                CONFIG[key] = var.get()
            self.log_message(f"Using settings: {CONFIG}")

            self.start_button.config(state='disabled', text='Processing...', bg='gray')
            self.reset_progress()
            self.update_progress(0, "Selecting files...")

            filepaths = filedialog.askopenfilenames(title='Select OCT TIFF files to align',
                                                    filetypes=[('TIFF files', '*.tif *.tiff')])
            if not filepaths:
                self.log_message("No input files selected.")
                self.start_button.config(state='normal', text='Start Processing', bg='#007AFF')
                return

            self.log_message(f"Selected {len(filepaths)} files.")
            self.update_progress(5, "Selecting output directory...")

            output_dir = filedialog.askdirectory(title='Select output directory')
            if not output_dir:
                self.log_message("No output directory selected.")
                self.start_button.config(state='normal', text='Start Processing', bg='#007AFF')
                return

            self.log_message(f"Output directory: {output_dir}")
            self.update_progress(10, "Starting processing...")

            # ---  Call core logic  ---
            # Pass self.log_message and self.update_progress as callbacks
            processing.process_all_volumes(filepaths, output_dir,
                                           log_callback=self.log_message,
                                           progress_callback=self.update_progress)

            self.log_message("Processing completed successfully!")
            self.update_progress(100, "Processing complete!")
            messagebox.showinfo("Success", f"Processing complete!\nResults saved to: {output_dir}")

        except Exception as e:
            self.update_progress(0, "Error occurred")
            self.log_message(f"ERROR: {str(e)}")
            messagebox.showerror("Processing Error", f"An error occurred:\n{str(e)}")
        finally:
            self.start_button.config(state='normal', text='Start Processing', bg='#007AFF')
            if "complete" not in self.status_var.get().lower():
                self.reset_progress()

    def run(self):
        self.root.mainloop()