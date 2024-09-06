import customtkinter as ctk
from tkinter import filedialog, messagebox
import subprocess
import os
import json

# Path to the settings file
SETTINGS_FILE = "last_settings.json"

# Function to save settings to a file
def save_settings():
    settings = {
        "bits_threshold": threshold_entry.get(),
        "model_path": model_entry.get(),
        "sequences_dir": sequences_entry.get(),
        "annotations_dir": annotations_entry.get(),
        "results_dir": results_entry.get(),
        "fps_file": fps_entry.get()
    }
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

# Function to load settings from a file
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Function to choose a file
def choose_file(entry):
    filename = filedialog.askopenfilename(
        initialdir="/home/",
        title="Select File"
    )
    if filename:
        entry.delete(0, ctk.END)
        entry.insert(0, filename)
    save_settings()

# Function to choose a directory
def choose_dir(entry):
    dirname = filedialog.askdirectory(
        initialdir="/home/",
        title="Select a Directory"
    )
    if dirname:
        if not dirname.endswith('/'):
            dirname += '/'
        entry.delete(0, ctk.END)
        entry.insert(0, dirname)
    save_settings()

# Function to run the tracker
def run_tracker():
    bits_threshold = threshold_entry.get()
    model_path = model_entry.get()
    sequences_dir = sequences_entry.get()
    annotations_dir = annotations_entry.get()
    results_dir = results_entry.get()
    fps_file = fps_entry.get()

    if not all([bits_threshold, model_path, sequences_dir, annotations_dir, results_dir, fps_file]):
        messagebox.showerror("Error", "Please fill out all fields")
        return

    if not os.path.isfile("./build/tracker"):
        messagebox.showerror("Error", "Tracker executable not found")
        return

    save_settings()

    try:
        command = f'./build/tracker "{model_path}" {bits_threshold} "{sequences_dir}" "{annotations_dir}" "{results_dir}" "{fps_file}" 2>nul'
        result = subprocess.run(command, shell=True)
        
        if result.returncode == 0:
            messagebox.showinfo("Success", "Tracker executed successfully")
        else:
            messagebox.showerror("Error", f"Tracker execution failed: {result.stderr}")
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run tracker: {e}")

# Function to populate fields with last saved settings
def populate_fields():
    settings = load_settings()

    if settings:
        threshold_entry.insert(0, settings.get("bits_threshold", ""))
        model_entry.insert(0, settings.get("model_path", ""))
        sequences_entry.insert(0, settings.get("sequences_dir", ""))
        annotations_entry.insert(0, settings.get("annotations_dir", ""))
        results_entry.insert(0, settings.get("results_dir", ""))
        fps_entry.insert(0, settings.get("fps_file", ""))

# Create the interface
ctk.set_appearance_mode("dark")  # Modes: "light", "dark"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

root = ctk.CTk()
root.title("Tracker GUI")

root.geometry("700x400")
root.resizable(False, False)

# Input fields for each option
ctk.CTkLabel(root, text="Bits Threshold:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
threshold_entry = ctk.CTkEntry(root, width=350)
threshold_entry.grid(row=0, column=1, padx=10, pady=10)

ctk.CTkLabel(root, text="Model Path:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
model_entry = ctk.CTkEntry(root, width=350)
model_entry.grid(row=1, column=1, padx=10, pady=10)
ctk.CTkButton(root, text="Choose File", command=lambda: choose_file(model_entry)).grid(row=1, column=2, padx=10, pady=10)

ctk.CTkLabel(root, text="Sequences Directory:").grid(row=2, column=0, padx=10, pady=10, sticky="w")
sequences_entry = ctk.CTkEntry(root, width=350)
sequences_entry.grid(row=2, column=1, padx=10, pady=10)
ctk.CTkButton(root, text="Choose Folder", command=lambda: choose_dir(sequences_entry)).grid(row=2, column=2, padx=10, pady=10)

ctk.CTkLabel(root, text="Annotations Directory:").grid(row=3, column=0, padx=10, pady=10, sticky="w")
annotations_entry = ctk.CTkEntry(root, width=350)
annotations_entry.grid(row=3, column=1, padx=10, pady=10)
ctk.CTkButton(root, text="Choose Folder", command=lambda: choose_dir(annotations_entry)).grid(row=3, column=2, padx=10, pady=10)

ctk.CTkLabel(root, text="Results Directory:").grid(row=4, column=0, padx=10, pady=10, sticky="w")
results_entry = ctk.CTkEntry(root, width=350)
results_entry.grid(row=4, column=1, padx=10, pady=10)
ctk.CTkButton(root, text="Choose Folder", command=lambda: choose_dir(results_entry)).grid(row=4, column=2, padx=10, pady=10)

ctk.CTkLabel(root, text="FPS File:").grid(row=5, column=0, padx=10, pady=10, sticky="w")
fps_entry = ctk.CTkEntry(root, width=350)
fps_entry.grid(row=5, column=1, padx=10, pady=10)
ctk.CTkButton(root, text="Choose File", command=lambda: choose_file(fps_entry)).grid(row=5, column=2, padx=10, pady=10)

# Button to run the tracker
ctk.CTkButton(root, text="Run Tracker", command=run_tracker).grid(row=6, column=1, pady=30)

# Populate fields with last used settings
populate_fields()

root.mainloop()
