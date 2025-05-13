"""GUI for selecting feeders."""
import cv2
import json
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import argparse


class SelectFeedersGUI:
    def __init__(self, root, video_path=None):
        self.root = root
        self.root.title("Feeder Selection GUI")
        self.root.geometry("800x800")

        self.video_path = video_path
        if not os.path.exists(self.video_path):
            self.show_video_error()
            return

        video_dir = os.path.dirname(self.video_path)
        self.json_file = os.path.join(video_dir, "feeder_locations.json")

        self.boxes = []
        self.current_box = None
        self.drawing = False
        self.start_point = (0, 0)
        self.feeder_name = tk.StringVar()

        self.load_boxes()
        self.load_video_frame()
        self.create_ui()

    def show_video_error(self):
        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Error: Video file not found", font=("Arial", 16, "bold")).pack(pady=10)
        ttk.Label(frame, text=f"Missing file:\n{self.video_path}", justify="center").pack(pady=10)
        ttk.Button(frame, text="Select Video", command=self.select_video_file).pack(pady=10)
        ttk.Button(frame, text="Exit", command=self.root.destroy).pack(pady=5)

    def select_video_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*"))
        )
        if file_path:
            self.video_path = file_path
            self.json_file = os.path.join(os.path.dirname(file_path), "feeder_locations.json")
            for widget in self.root.winfo_children():
                widget.destroy()
            self.__init__(self.root, self.video_path)

    def load_boxes(self):
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, 'r') as f:
                    self.boxes = json.load(f)
            except json.JSONDecodeError:
                messagebox.showerror("Error", f"Could not parse {self.json_file}")
                self.boxes = []

    def load_video_frame(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            messagebox.showerror("Error", "Unable to read video.")
            self.root.destroy()
            return

        self.original_frame = frame
        h, w = frame.shape[:2]

        # Tighter bounds for GUI-friendliness
        max_width, max_height = 800, 500
        self.scale_factor = min(max_width / w, max_height / h, 1.0)

        new_size = (int(w * self.scale_factor), int(h * self.scale_factor))
        self.frame = cv2.resize(frame, new_size)
        self.preview = self.frame.copy()


    def create_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom_frame = ttk.Frame(self.root, height=280)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        bottom_frame.pack_propagate(False)

        self.canvas = tk.Canvas(top_frame, bg="gray", height=self.frame.shape[0])
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.update_display()

        ttk.Label(bottom_frame, text="Click and drag on the video above to draw a bounding box around each feeder you want to label."
, font=("Arial", 12, "bold")).pack(pady=5)

        paths_frame = ttk.Frame(bottom_frame)
        paths_frame.pack(fill=tk.X, padx=5)

        for label, value in [("Video Path:", self.video_path), ("JSON Output:", self.json_file)]:
            row = ttk.Frame(paths_frame)
            row.pack(fill=tk.X, expand=True, padx=5, pady=2)

            ttk.Label(row, text=label, width=12, anchor="w").pack(side=tk.LEFT)
            entry = ttk.Entry(row, font=("Arial", 9))
            entry.insert(0, value)
            entry.config(state="readonly")
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)


        entry_frame = ttk.Frame(bottom_frame)
        entry_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(entry_frame, text="Feeder Name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(entry_frame, textvariable=self.feeder_name, font=("Arial", 10), width=30).pack(side=tk.LEFT, padx=5)

        button_frame = ttk.Frame(bottom_frame)
        button_frame.pack(fill=tk.X, padx=5)

        for text, command in [
            ("Add Feeder", self.add_feeder),
            ("Clear Current", self.clear_current),
            ("Save All", self.save_boxes),
            ("Reset All", self.reset_all)
        ]:
            ttk.Button(button_frame, text=text, command=command).pack(side=tk.LEFT, padx=5, pady=5)

        list_frame = ttk.LabelFrame(bottom_frame, text="Defined Feeders")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.listbox = tk.Listbox(list_frame, font=("Courier", 10))
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status_var = tk.StringVar(value=f"Loaded {len(self.boxes)} feeders")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        self.update_listbox()
        self.root.bind("<Return>", lambda e: self.add_feeder())
        self.root.bind("<Escape>", lambda e: self.clear_current())

    def update_display(self):
        self.preview = self.frame.copy()

        for box in self.boxes:
            x1, y1, x2, y2 = box["bbox"]
            cv2.rectangle(self.preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.preview, box["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(self.preview, (x1, y1), (x2, y2), (255, 0, 0), 2)

        rgb_image = cv2.cvtColor(self.preview, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        self.tk_image = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def update_listbox(self):
        self.listbox.delete(0, tk.END)
        scale = 1 / self.scale_factor
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = [int(coord * scale) for coord in box["bbox"]]
            self.listbox.insert(tk.END, f"{i + 1}. {box['label']} - [{x1}, {y1}, {x2}, {y2}]")
        self.status_var.set(f"Total feeders: {len(self.boxes)}")

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_point = (event.x, event.y)
        self.current_box = [event.x, event.y, event.x, event.y]
        self.update_display()

    def on_mouse_move(self, event):
        if self.drawing:
            self.current_box[2], self.current_box[3] = event.x, event.y
            self.update_display()

    def on_mouse_up(self, event):
        if self.drawing:
            self.drawing = False
            x1, y1, x2, y2 = self.current_box
            self.current_box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            if not self.feeder_name.get():
                self.feeder_name.set(f"Feeder {len(self.boxes) + 1}")
            self.update_display()

    def add_feeder(self):
        if not self.current_box:
            messagebox.showwarning("Warning", "Draw a box first.")
            return

        name = self.feeder_name.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Enter a name.")
            return

        self.boxes.append({"bbox": self.current_box.copy(), "label": name})
        self.current_box = None
        self.feeder_name.set("")
        self.update_display()
        self.update_listbox()
        self.save_boxes()

    def clear_current(self):
        self.current_box = None
        self.update_display()

    def reset_all(self):
        if messagebox.askyesno("Confirm", "Reset all feeders?"):
            self.boxes.clear()
            self.current_box = None
            self.feeder_name.set("")
            self.update_display()
            self.update_listbox()
            self.save_boxes()

    def save_boxes(self):
        try:
            os.makedirs(os.path.dirname(self.json_file), exist_ok=True)
            scale = 1 / self.scale_factor
            data = [
                {"label": box["label"], "bbox": [int(x * scale) for x in box["bbox"]]}
                for box in self.boxes
            ]
            with open(self.json_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_var.set(f"Saved {len(self.boxes)} feeders.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Feeder Selection GUI")
    parser.add_argument("--video", "-v", type=str, help="Path to the video file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    root = tk.Tk()
    SelectFeedersGUI(root, video_path=args.video)
    root.mainloop()
