import os
import sys
import yaml
import subprocess
import customtkinter as ctk
from tkinter import filedialog, messagebox, Toplevel
import threading
import queue
import time
import glob
from PIL import Image, ImageTk

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: system (default), light, dark
ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

class TerminalRedirect:
    def __init__(self, text_widget, queue):
        self.text_widget = text_widget
        self.queue = queue

    def write(self, string):
        self.queue.put(string)

    def flush(self):
        pass

class AIToolkitGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window
        self.title("AI Toolkit Configuration Interface")
        self.geometry("1650x1175")
        self.minsize(1400, 900)
        
        # Initialize variables
        self.saved_configs = []
        self.current_config = {}
        self.output_queue = queue.Queue()
        self.current_samples_dir = None
        self.image_data = {}
        
        # Create the main frame layout
        self.create_layout()
        
        # Load saved configurations
        self.load_saved_configs()
        
        # Start the queue processing
        self.process_queue()

    def create_layout(self):
        # Create main sections
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create a tabview for the right side
        self.right_tabview = ctk.CTkTabview(self)
        self.right_tabview.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Add tabs
        self.terminal_tab = self.right_tabview.add("Terminal")
        self.samples_tab = self.right_tabview.add("Samples")
        
        # Set default tab
        self.right_tabview.set("Terminal")
        
        # Left panel - Form
        self.left_panel = ctk.CTkFrame(self)
        self.left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Configure left panel
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_columnconfigure(1, weight=3)
        
        # Create form elements
        self.create_form()
        
        # Configure terminal tab
        self.terminal_tab.grid_columnconfigure(0, weight=1)
        self.terminal_tab.grid_rowconfigure(0, weight=0)
        self.terminal_tab.grid_rowconfigure(1, weight=1)
        
        # Create terminal section
        self.create_terminal()
        
        # Configure samples tab
        self.samples_tab.grid_columnconfigure(0, weight=1)
        self.samples_tab.grid_rowconfigure(0, weight=0)
        self.samples_tab.grid_rowconfigure(1, weight=1)
        
        # Create samples section
        self.create_samples_viewer()

    def create_form(self):
        # Title label
        title_label = ctk.CTkLabel(self.left_panel, text="AI Toolkit Configuration", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Saved YAML files dropdown
        saved_label = ctk.CTkLabel(self.left_panel, text="Saved Configurations:")
        saved_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")
        
        self.saved_config_var = ctk.StringVar()
        self.saved_config_dropdown = ctk.CTkOptionMenu(
            self.left_panel, 
            values=["New Configuration"], 
            variable=self.saved_config_var,
            command=self.on_config_selected
        )
        self.saved_config_dropdown.grid(row=1, column=1, padx=20, pady=(10, 5), sticky="ew")
        
        # Header: Settings section
        settings_label = ctk.CTkLabel(self.left_panel, text="Header Settings", font=ctk.CTkFont(size=16, weight="bold"))
        settings_label.grid(row=2, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        
        # Form fields
        current_row = 3
        
        # Line 5: Name
        self.create_form_field("Name:", "name", current_row)
        current_row += 1
        
        # Line 9: Training folder
        self.create_form_field("Lora Save Folder:", "training_folder", current_row, browse=True)
        current_row += 1
        
        # Line 14: Trigger word
        self.create_form_field("Trigger Word:", "trigger_word", current_row)
        current_row += 1
        
        # Line 21: Save every
        self.create_form_field("Save Every (steps):", "save_every", current_row, is_numeric=True)
        current_row += 1
        
        # Line 22: Max step saves to keep
        self.create_form_field("Max Step Saves to Keep:", "max_step_saves_to_keep", current_row, is_numeric=True)
        current_row += 1
        
        # Line 33: Folder path
        self.create_form_field("Dataset Folder Path:", "folder_path", current_row, browse=True)
        current_row += 1
        
        # Line 41: Steps
        self.create_form_field("Training Steps:", "steps", current_row, is_numeric=True)
        current_row += 1
        
        # Line 71: Sample every
        self.create_form_field("Sample Every (steps):", "sample_every", current_row, is_numeric=True)
        current_row += 1
        
        # Creator name and version
        self.create_form_field("Creator Name:", "creator_name", current_row)
        current_row += 1
        
        self.create_form_field("Lora Version:", "lora_version", current_row)
        current_row += 1
        
        # Prompts header
        prompts_label = ctk.CTkLabel(self.left_panel, text="Sample Prompts", font=ctk.CTkFont(size=16, weight="bold"))
        prompts_label.grid(row=current_row, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="w")
        current_row += 1
        
        # Line 74: Prompts
        self.prompt_entries = []
        for i in range(5):
            prompt_label = ctk.CTkLabel(self.left_panel, text=f"Prompt {i+1}:")
            prompt_label.grid(row=current_row, column=0, padx=20, pady=5, sticky="w")
            
            prompt_entry = ctk.CTkTextbox(self.left_panel, height=80)  # Doubled height
            prompt_entry.grid(row=current_row, column=1, padx=20, pady=5, sticky="ew")
            self.prompt_entries.append(prompt_entry)
            current_row += 1
        
        # Action buttons
        button_frame = ctk.CTkFrame(self.left_panel)
        button_frame.grid(row=current_row, column=0, columnspan=2, padx=20, pady=20, sticky="ew")
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        save_button = ctk.CTkButton(button_frame, text="Save", command=self.save_config)
        save_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        start_button = ctk.CTkButton(button_frame, text="Start", command=self.start_training)
        start_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    def create_form_field(self, label_text, field_name, row, browse=False, is_numeric=False):
        label = ctk.CTkLabel(self.left_panel, text=label_text)
        label.grid(row=row, column=0, padx=20, pady=5, sticky="w")
        
        if browse:
            frame = ctk.CTkFrame(self.left_panel)
            frame.grid(row=row, column=1, padx=20, pady=5, sticky="ew")
            frame.grid_columnconfigure(0, weight=3)
            frame.grid_columnconfigure(1, weight=1)
            
            entry = ctk.CTkEntry(frame)
            entry.grid(row=0, column=0, padx=(0, 10), pady=0, sticky="ew")
            
            browse_btn = ctk.CTkButton(
                frame, 
                text="Browse", 
                width=80, 
                command=lambda e=entry, fn=field_name: self.browse_folder(e, fn)
            )
            browse_btn.grid(row=0, column=1, padx=0, pady=0)
            
            setattr(self, f"{field_name}_entry", entry)
        else:
            entry = ctk.CTkEntry(self.left_panel)
            entry.grid(row=row, column=1, padx=20, pady=5, sticky="ew")
            setattr(self, f"{field_name}_entry", entry)

    def create_terminal(self):
        # Terminal header
        terminal_label = ctk.CTkLabel(
            self.terminal_tab, 
            text="Terminal Output", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        terminal_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Terminal output text area
        self.terminal_output = ctk.CTkTextbox(self.terminal_tab, wrap="word")
        self.terminal_output.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.terminal_output.configure(state="disabled")

    def create_samples_viewer(self):
        # Samples header frame
        samples_header_frame = ctk.CTkFrame(self.samples_tab)
        samples_header_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        samples_header_frame.grid_columnconfigure(0, weight=1)
        samples_header_frame.grid_columnconfigure(1, weight=0)
        
        # Samples header
        samples_label = ctk.CTkLabel(
            samples_header_frame, 
            text="Sample Images", 
            font=ctk.CTkFont(size=16, weight="bold")
        )
        samples_label.grid(row=0, column=0, padx=0, pady=0, sticky="w")
        
        # Refresh button
        refresh_button = ctk.CTkButton(
            samples_header_frame,
            text="Refresh",
            width=100,
            command=self.refresh_samples
        )
        refresh_button.grid(row=0, column=1, padx=0, pady=0, sticky="e")
        
        # Create a frame for the samples
        self.samples_frame = ctk.CTkScrollableFrame(self.samples_tab)
        self.samples_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")
        
        # Initial message
        self.no_samples_label = ctk.CTkLabel(
            self.samples_frame,
            text="No sample images found. Select a configuration and click 'Refresh'.",
            font=ctk.CTkFont(size=14)
        )
        self.no_samples_label.pack(pady=20)
        
        # Initialize image containers
        self.image_labels = []

    def refresh_samples(self):
        # Clear current images
        for label in self.image_labels:
            label.destroy()
        self.image_labels = []
        self.image_data = {}
        
        # Get the current configuration name
        config_name = self.name_entry.get()
        
        if not config_name:
            # If no configuration is selected, show message
            self.no_samples_label.configure(text="No configuration selected. Please select or create a configuration.")
            self.no_samples_label.pack(pady=20)
            return
            
        # Construct the samples directory path
        base_dir = self.training_folder_entry.get()
        if not base_dir:
            base_dir = "output"  # Default directory
            
        samples_dir = os.path.join(base_dir, config_name, "samples")
        self.current_samples_dir = samples_dir
        
        # Check if directory exists
        if not os.path.exists(samples_dir):
            self.no_samples_label.configure(text=f"No samples directory found at: {samples_dir}")
            self.no_samples_label.pack(pady=20)
            return
            
        # Find image files
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(glob.glob(os.path.join(samples_dir, ext)))
            
        # Sort by modification time (newest first)
        image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if not image_files:
            self.no_samples_label.configure(text=f"No sample images found in: {samples_dir}")
            self.no_samples_label.pack(pady=20)
            return
            
        # Hide the no samples label
        self.no_samples_label.pack_forget()
        
        # Get number of prompts (for grid layout)
        num_prompts = sum(1 for p in self.prompt_entries if p.get("0.0", "end-1c").strip())
        num_prompts = max(1, min(5, num_prompts))  # Ensure between 1 and 5
        
        # Create a grid frame to hold the images
        grid_frame = ctk.CTkFrame(self.samples_frame)
        grid_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure grid columns
        for i in range(num_prompts):
            grid_frame.grid_columnconfigure(i, weight=1)
        
        # Display images in a grid
        row, col = 0, 0
        for i, img_path in enumerate(image_files):
            try:
                # Create frame for each image with caption
                img_frame = ctk.CTkFrame(grid_frame)
                img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Load and resize image
                img = Image.open(img_path)
                # Store original image
                self.image_data[img_path] = img.copy()
                
                # Calculate aspect ratio and resize to fit
                width, height = img.size
                max_width = 250  # Maximum thumbnail width
                if width > max_width:
                    ratio = max_width / width
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to Tkinter PhotoImage
                tk_img = ImageTk.PhotoImage(img)
                
                # Image label with click handler
                img_label = ctk.CTkLabel(img_frame, image=tk_img, text="")
                img_label.image = tk_img  # Keep a reference to prevent garbage collection
                img_label.bind("<Button-1>", lambda e, path=img_path: self.show_image_popup(path, self.image_data[path]))
                img_label.pack(pady=5)
                
                # Caption with short filename
                filename = os.path.basename(img_path)
                if len(filename) > 25:  # Truncate long filenames
                    filename = filename[:20] + "..." + filename[-5:]
                caption_label = ctk.CTkLabel(img_frame, text=filename)
                caption_label.pack(pady=2)
                
                # Add to list for later cleanup
                self.image_labels.append(img_frame)
                
                # Update grid position
                col += 1
                if col >= num_prompts:
                    col = 0
                    row += 1
                
                # Limit to 20 images to prevent performance issues
                if i >= 19:
                    more_label = ctk.CTkLabel(self.samples_frame, 
                                            text=f"+ {len(image_files) - 20} more images in {samples_dir}")
                    more_label.pack(pady=10)
                    self.image_labels.append(more_label)
                    break
                    
            except Exception as e:
                error_label = ctk.CTkLabel(self.samples_frame, 
                                        text=f"Error loading {os.path.basename(img_path)}: {str(e)}")
                error_label.pack(pady=5)
                self.image_labels.append(error_label)
        
        # Add the grid frame to cleanup list
        self.image_labels.append(grid_frame)
    
    def show_image_popup(self, img_path, original_img):
        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Image Viewer")
        
        # Set a reasonable size for the popup window
        screen_width = popup.winfo_screenwidth()
        screen_height = popup.winfo_screenheight()
        
        # Calculate dimensions - use up to 80% of screen but maintain aspect ratio
        img_width, img_height = original_img.size
        
        max_width = int(screen_width * 0.8)
        max_height = int(screen_height * 0.8)
        
        # Calculate scaled dimensions maintaining aspect ratio
        if img_width > max_width or img_height > max_height:
            width_ratio = max_width / img_width
            height_ratio = max_height / img_height
            scale_ratio = min(width_ratio, height_ratio)
            
            new_width = int(img_width * scale_ratio)
            new_height = int(img_height * scale_ratio)
        else:
            new_width = img_width
            new_height = img_height
            
        # Resize image
        img_resized = original_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)
        
        # Calculate window dimensions with some padding
        window_width = new_width + 40
        window_height = new_height + 100
        
        # Position the window in the center of the screen
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        
        popup.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
        
        # Create label for the image
        img_label = ctk.CTkLabel(popup, image=img_tk, text="")
        img_label.image = img_tk  # Keep reference to prevent garbage collection
        img_label.pack(pady=10, padx=10)
        
        # Add filename label
        filename_label = ctk.CTkLabel(popup, text=os.path.basename(img_path))
        filename_label.pack(pady=5)
        
        # Add close button
        close_button = ctk.CTkButton(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=10)
        
        # Make the popup modal
        popup.transient(self)
        popup.grab_set()
        
        # Wait for the window to be closed
        self.wait_window(popup)

    def load_saved_configs(self):
        config_dir = os.path.join(os.getcwd(), "config")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            
        self.saved_configs = ["New Configuration"]
        
        for file in os.listdir(config_dir):
            if file.endswith(".yaml") or file.endswith(".yml"):
                self.saved_configs.append(file)
                
        self.saved_config_dropdown.configure(values=self.saved_configs)
        self.saved_config_dropdown.set(self.saved_configs[0])

    def on_config_selected(self, selection):
        if selection == "New Configuration":
            self.clear_form()
            return
            
        config_path = os.path.join(os.getcwd(), "config", selection)
        
        try:
            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                
            # Extract values from YAML and populate form
            self.populate_form(config_data)
            
            # Check for samples after loading a configuration
            self.right_tabview.set("Samples")
            self.refresh_samples()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")

    def populate_form(self, config_data):
        # Clear the form first
        self.clear_form()
        
        # Store the current config
        self.current_config = config_data
        
        # Extract values and set form fields
        
        # Basic fields
        self.name_entry.insert(0, config_data.get("config", {}).get("name", ""))
        
        process_config = config_data.get("config", {}).get("process", [{}])[0]
        
        self.training_folder_entry.insert(0, process_config.get("training_folder", ""))
        self.trigger_word_entry.insert(0, process_config.get("trigger_word", ""))
        
        save_config = process_config.get("save", {})
        self.save_every_entry.insert(0, str(save_config.get("save_every", "")))
        self.max_step_saves_to_keep_entry.insert(0, str(save_config.get("max_step_saves_to_keep", "")))
        
        # Dataset folder path
        datasets = process_config.get("datasets", [{}])
        if datasets and len(datasets) > 0:
            self.folder_path_entry.insert(0, datasets[0].get("folder_path", ""))
        
        # Training steps
        train_config = process_config.get("train", {})
        self.steps_entry.insert(0, str(train_config.get("steps", "")))
        
        # Sample every
        sample_config = process_config.get("sample", {})
        self.sample_every_entry.insert(0, str(sample_config.get("sample_every", "")))
        
        # Creator name and version
        meta_config = config_data.get("meta", {})
        creator_name = meta_config.get("name", "").replace("[name]", "").replace("[", "").replace("]", "")
        self.creator_name_entry.insert(0, creator_name if creator_name else "User")
        self.lora_version_entry.insert(0, meta_config.get("version", ""))
        
        # Prompts
        prompts = sample_config.get("prompts", [])
        for i, prompt in enumerate(prompts):
            if i < len(self.prompt_entries):
                self.prompt_entries[i].insert("0.0", prompt)

    def clear_form(self):
        # Clear all entry fields
        for attr in dir(self):
            if attr.endswith("_entry") and hasattr(self, attr):
                entry = getattr(self, attr)
                if isinstance(entry, ctk.CTkEntry):
                    entry.delete(0, "end")
                    # Set default User for creator name
                    if attr == "creator_name_entry":
                        entry.insert(0, "User")
        
        # Clear all prompt text boxes
        for prompt_entry in self.prompt_entries:
            prompt_entry.delete("0.0", "end")
            
        # Reset current config
        self.current_config = {}

    def browse_folder(self, entry_widget, field_name):
        if field_name == "folder_path":
            # Browse for a folder
            folder_path = filedialog.askdirectory()
            if folder_path:
                entry_widget.delete(0, "end")
                entry_widget.insert(0, folder_path)
        else:
            # Browse for a folder
            folder_path = filedialog.askdirectory()
            if folder_path:
                entry_widget.delete(0, "end")
                entry_widget.insert(0, folder_path)

    def get_form_data(self):
        # Create a dictionary structure that matches the YAML structure
        config_data = {
            "job": "extension",
            "config": {
                "name": self.name_entry.get(),
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": self.training_folder_entry.get(),
                        "performance_log_every": 5000,
                        "device": "cuda:0",
                        "trigger_word": self.trigger_word_entry.get(),
                        "network": {
                            "type": "lora",
                            "linear": 16,
                            "linear_alpha": 16
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": int(self.save_every_entry.get()) if self.save_every_entry.get() else 500,
                            "max_step_saves_to_keep": int(self.max_step_saves_to_keep_entry.get()) if self.max_step_saves_to_keep_entry.get() else 100,
                            "push_to_hub": False
                        },
                        "datasets": [
                            {
                                "folder_path": self.folder_path_entry.get(),
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": True,
                                "resolution": [512, 768, 1024]
                            }
                        ],
                        "train": {
                            "batch_size": 1,
                            "steps": int(self.steps_entry.get()) if self.steps_entry.get() else 4000,
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": True,
                            "noise_scheduler": "flowmatch",
                            "optimizer": "adamw8bit",
                            "lr": 1e-4,
                            "ema_config": {
                                "use_ema": True,
                                "ema_decay": 0.99
                            },
                            "dtype": "bf16"
                        },
                        "model": {
                            "name_or_path": "black-forest-labs/FLUX.1-dev",
                            "is_flux": True,
                            "quantize": True
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": int(self.sample_every_entry.get()) if self.sample_every_entry.get() else 200,
                            "width": 1024,
                            "height": 1536,
                            "prompts": self.get_prompts(),
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 4,
                            "sample_steps": 20
                        }
                    }
                ]
            },
            "meta": {
                "name": f"[{self.creator_name_entry.get()}]",
                "version": self.lora_version_entry.get()
            }
        }
        
        # Merge with existing config to preserve unspecified fields
        if self.current_config:
            # Only merge if we loaded an existing config
            return self.merge_configs(self.current_config, config_data)
        
        return config_data

    def get_prompts(self):
        prompts = []
        for prompt_entry in self.prompt_entries:
            prompt_text = prompt_entry.get("0.0", "end-1c").strip()
            if prompt_text:
                prompts.append(prompt_text)
        return prompts

    def merge_configs(self, original, new):
        # Helper function to recursively merge dictionaries, preserving values from original
        # that aren't explicitly set in the new config
        result = original.copy()
        
        for key, value in new.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self.merge_configs(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # For lists, we need to handle special cases
                    if key == "process" and len(value) > 0 and len(result[key]) > 0:
                        # Merge process items
                        result[key][0] = self.merge_configs(result[key][0], value[0])
                    elif key == "datasets" and len(value) > 0 and len(result[key]) > 0:
                        # Merge datasets
                        result[key][0] = self.merge_configs(result[key][0], value[0])
                    elif key == "prompts":
                        # Replace prompts completely
                        result[key] = value
                    else:
                        # Default behavior for other lists
                        result[key] = value
                else:
                    # For regular values, use the new one if it's set
                    if isinstance(value, str) and not value.strip():
                        # Keep original if new is empty
                        pass
                    else:
                        result[key] = value
            else:
                result[key] = value
                
        return result

    def save_config(self):
        try:
            # Get the form data
            config_data = self.get_form_data()
            
            # Get the filename from the name field
            filename = f"{config_data['config']['name']}.yaml"
            
            # Ensure config directory exists
            config_dir = os.path.join(os.getcwd(), "config")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            # Build the file path
            file_path = os.path.join(config_dir, filename)
            
            # Save the YAML file
            with open(file_path, 'w') as file:
                # Use custom yaml representation to preserve formatting
                class NoAliasDumper(yaml.SafeDumper):
                    def ignore_aliases(self, data):
                        return True
                
                yaml.dump(config_data, file, default_flow_style=False, sort_keys=False, Dumper=NoAliasDumper)
                
            messagebox.showinfo("Success", f"Configuration saved as {filename}")
            
            # Refresh the saved configs list
            self.load_saved_configs()
            self.saved_config_dropdown.set(filename)
            
            # Check for samples directory and refresh if needed
            self.refresh_samples()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def start_training(self):
        try:
            # Save the current configuration first
            self.save_config()
            
            # Get the filename
            filename = f"{self.name_entry.get()}.yaml"
            file_path = os.path.join("config", filename)
            
            # Clear the terminal output
            self.terminal_output.configure(state="normal")
            self.terminal_output.delete("0.0", "end")
            self.terminal_output.configure(state="disabled")
            
            # Run the training process in a separate thread
            threading.Thread(target=self.run_training, args=(file_path,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")

    def run_training(self, config_path):
        try:
            # Get just the filename from the path
            config_filename = os.path.basename(config_path)
            
            # Add terminal output
            self.log_to_terminal(f"Starting training with configuration: {config_filename}\n")
            self.log_to_terminal("Activating virtual environment...\n")
            
            # Switch to Terminal tab to show output
            self.right_tabview.set("Terminal")
            
            # Create and use a batch file to run the training
            batch_file = "launch-trainer.bat"
            
            if not os.path.exists(batch_file):
                # Create the batch file if it doesn't exist
                with open(batch_file, 'w') as f:
                    f.write("@echo off\n")
                    f.write("echo Starting AI Toolkit training...\n")
                    f.write("call venv\\Scripts\\activate.bat\n")
                    f.write("python run.py %1\n")
                    f.write("pause\n")
            
            # Run the batch file with the config path as argument
            process = subprocess.Popen(
                [batch_file, config_path],
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Read and display output
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                self.log_to_terminal(line)
                
            # Wait for process to complete
            process.wait()
            
            if process.returncode == 0:
                self.log_to_terminal("\nTraining completed successfully!")
            else:
                self.log_to_terminal(f"\nTraining exited with error code: {process.returncode}")
                
        except Exception as e:
            self.log_to_terminal(f"\nError during training: {str(e)}")

    def log_to_terminal(self, text):
        self.output_queue.put(text)

    def process_queue(self):
        try:
            while not self.output_queue.empty():
                message = self.output_queue.get_nowait()
                self.terminal_output.configure(state="normal")
                self.terminal_output.insert("end", message)
                self.terminal_output.see("end")
                self.terminal_output.configure(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.after(100, self.process_queue)

if __name__ == "__main__":
    app = AIToolkitGUI()
    app.mainloop()
