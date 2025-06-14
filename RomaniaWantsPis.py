import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import time

THEME_COLOR_DARK = '#FF1493'
THEME_COLOR_LIGHT = '#FF69B4'

class ImageSegmenter:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image could not be loaded. Please check the path.")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.current_method = "binary"
        self.default_params = {
            "binary": {"threshold": 128},
            "adaptive_gaussian": {"block_size": 11, "constant": 2},
            "adaptive_mean": {"block_size": 11, "constant": 2},
            "otsu": {},
            "watershed": {
                "gaussian_ksize": 5,
                "binary_thresh": 0,
                "opening_iter": 2,
                "dilate_iter": 3,
                "dist_thresh": 0.7,
                "kernel_size": 3
            },
            "clustering": {"n_clusters": 3}
        }
        self.current_params = self.default_params.copy()

    def region_based_segmentation(self, method, **params):
        if method == "binary":
            _, binary = cv2.threshold(self.gray, params.get("threshold", 128), 255, cv2.THRESH_BINARY)
        elif method == "adaptive_gaussian":
            binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, params.get("block_size", 11), params.get("constant", 2))
        elif method == "adaptive_mean":
            binary = cv2.adaptiveThreshold(self.gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, params.get("block_size", 11), params.get("constant", 2))
        elif method == "otsu":
            _, binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(self.gray, params.get("threshold", 128), 255, cv2.THRESH_BINARY)
        return binary
    def watershed_segmentation(self, **params):
        blurred = cv2.GaussianBlur(self.gray, (params.get("gaussian_ksize", 5), params.get("gaussian_ksize", 5)), 0)
        _, thresh = cv2.threshold(blurred, params.get("binary_thresh", 0), 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((params.get("kernel_size", 3), params.get("kernel_size", 3)), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=params.get("opening_iter", 2))
        sure_bg = cv2.dilate(opening, kernel, iterations=params.get("dilate_iter", 3))
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, params.get("dist_thresh", 0.7) * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers += 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(self.image, markers)
        result = self.image.copy()
        result[markers == -1] = [255, 0, 0]
        return result

    def clustering_based_segmentation(self, n_clusters=3):
        pixel_values = np.float32(self.image.reshape((-1, 3)))
        scaler = MinMaxScaler()
        normalized_values = scaler.fit_transform(pixel_values)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(normalized_values)
        labels = kmeans.labels_
        centers = np.uint8(scaler.inverse_transform(kmeans.cluster_centers_))
        segmented_image = centers[labels.flatten()].reshape(self.image.shape)
        return segmented_image

class SegmentationGUI:
    def __init__(self, arg_root):
        self.root = arg_root
        self.root.configure(bg=THEME_COLOR_LIGHT)
        self.segmenter = None
        self.current_image = None
        style = ttk.Style()
        style.configure('TFrame', background=THEME_COLOR_LIGHT)
        style.configure('TLabel', background=THEME_COLOR_LIGHT, foreground='black', font=('Arial', 10, 'bold'))
        style.configure('TButton', background=THEME_COLOR_LIGHT, foreground='black', font=('Arial', 10, 'bold'))
        style.configure('TLabelFrame', background=THEME_COLOR_LIGHT, foreground='black', font=('Arial', 10, 'bold'))
        style.configure('TScale', background=THEME_COLOR_LIGHT)
        style.configure('TMenubutton', background=THEME_COLOR_DARK, foreground='white', font=('Arial', 10, 'bold'))

        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.image_panel = ttk.Label(self.main_frame)
        self.image_panel.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.control_panel = ttk.Frame(self.main_frame, width=300)
        self.control_panel.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.load_btn = ttk.Button(self.control_panel, text="Load Image", command=self.load_image)
        self.load_btn.pack(fill=tk.X, pady=5)

        self.method_var = tk.StringVar(value="binary")
        ttk.Label(self.control_panel, text="Segmentation Method:").pack(anchor=tk.W)
        methods = ["binary", "adaptive_gaussian", "adaptive_mean", "otsu", "watershed", "clustering"]
        self.method_menu = ttk.OptionMenu(self.control_panel, self.method_var, "binary", *methods, command=lambda _: self.update_controls())
        self.method_menu.pack(fill=tk.X, pady=5)

        self.params_frame = ttk.LabelFrame(self.control_panel, text="Parameters")
        self.params_frame.pack(fill=tk.X, pady=5)

        self.btn_frame = ttk.Frame(self.control_panel)
        self.btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(self.btn_frame, text="Update", command=self.update_image).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(self.btn_frame, text="Reset", command=self.reset_params).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(self.btn_frame, text="Save", command=self.save_image).pack(side=tk.LEFT, expand=True, padx=2)

        self.sliders = {}
        self.update_controls()
        self.root.bind('<Configure>', self.on_window_resize)
        self.root.state('zoomed')
        self.last_resize_time = 0

    def on_window_resize(self, event):
        if event.widget == self.root:
            if time.time() - self.last_resize_time > 0.1:
                if hasattr(self, 'current_image') and self.current_image is not None:
                    self.update_image_display()
                self.last_resize_time = time.time()

    def update_image_display(self):
        if self.current_image is None:
            return
        img_pil = Image.fromarray(self.current_image)
        control_panel_width = 350
        min_margin = 20
        window_width = max(1, self.root.winfo_width() - control_panel_width)
        window_height = max(1, self.root.winfo_height() - min_margin)
        aspect_ratio = img_pil.width / img_pil.height if img_pil.height else 1.0
        if window_width / window_height > aspect_ratio:
            new_width, new_height = int(window_height * aspect_ratio), window_height
        else:
            new_width, new_height = window_width, int(window_width / aspect_ratio)
        img_resized = img_pil.resize((max(1, new_width), max(1, new_height)), Image.LANCZOS)
        self.image_panel.image = ImageTk.PhotoImage(img_resized)
        self.image_panel.config(image=self.image_panel.image)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            try:
                self.segmenter = ImageSegmenter(file_path)
                self.reset_params()
                self.update_image()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")

    def update_controls(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.sliders = {}
        if self.segmenter is None:
            return
        method = self.method_var.get()
        params = self.segmenter.current_params[method]
        if method == "binary":
            self.add_slider("Threshold", 0, 255, params.get("threshold", 128))
        elif method == "clustering":
            self.add_slider("Clusters", 2, 10, params.get("n_clusters", 3))
        elif method == "watershed":
            self.add_slider("Gaussian Kernel", 1, 15, params.get("gaussian_ksize", 5), step=2)
            self.add_slider("Binary Threshold", 0, 255, params.get("binary_thresh", 0))
            self.add_slider("Opening Iterations", 1, 10, params.get("opening_iter", 2))
            self.add_slider("Dilate Iterations", 1, 10, params.get("dilate_iter", 3))
            self.add_slider("Distance Thresh %", 10, 90, int(params.get("dist_thresh", 0.7) * 100))
            self.add_slider("Morph Kernel Size", 1, 15, params.get("kernel_size", 3), step=2)
        elif method in ["adaptive_gaussian", "adaptive_mean"]:
            self.add_slider("Block Size", 3, 31, 11, step=2)
            self.add_slider("Constant", 0, 10, 2)
        elif method == "otsu":
            ttk.Label(self.params_frame, text="Otsu's method uses automatic thresholding.\nNo parameters required.").pack()

    def add_slider(self, name, min_val, max_val, default_val, step=1):
        frame = ttk.Frame(self.params_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=f"{name}:").pack(side=tk.LEFT)
        val = tk.StringVar(value=str(default_val))
        ttk.Label(frame, textvariable=val, width=4).pack(side=tk.RIGHT)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, value=default_val, orient=tk.HORIZONTAL,
                           command=lambda v, tv=val, s=step: tv.set(str(int(float(v)) // 2 * 2 + 1 if s > 1 else int(float(v)))), length=150)
        slider.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        self.sliders[name.lower().replace(" ", "_")] = (slider, val)

    def get_params(self):
        method = self.method_var.get()
        p = self.sliders
        if method == "binary":
            return {"threshold": int(float(p["threshold"][1].get()))}
        if method == "clustering":
            return {"n_clusters": int(float(p["clusters"][1].get()))}
        if method == "watershed":
            return {
                "gaussian_ksize": int(float(p["gaussian_kernel"][1].get()) // 2 * 2 + 1),
                "binary_thresh": int(float(p["binary_threshold"][1].get())),
                "opening_iter": int(float(p["opening_iterations"][1].get())),
                "dilate_iter": int(float(p["dilate_iterations"][1].get())),
                "dist_thresh": float(p["distance_thresh_%"][1].get()) / 100,
                "kernel_size": int(float(p["morph_kernel_size"][1].get()) // 2 * 2 + 1)
            }
        if method in ["adaptive_gaussian", "adaptive_mean"]:
            return {
                "block_size": max(3, int(float(p["block_size"][1].get()) // 2 * 2 + 1)),
                "constant": int(float(p["constant"][1].get()))
            }
        return {}

    def reset_params(self):
        if self.segmenter:
            method = self.method_var.get()
            self.segmenter.current_params[method] = self.segmenter.default_params[method].copy()
            self.update_controls()
            self.update_image()

    def update_image(self):
        if not self.segmenter:
            return
        method = self.method_var.get()
        try:
            params = self.get_params()
            if method in ["binary", "adaptive_gaussian", "adaptive_mean"]:
                segmented = self.segmenter.region_based_segmentation(method, **params)
            elif method == "otsu":
                segmented = self.segmenter.region_based_segmentation(method)
            elif method == "watershed":
                segmented = self.segmenter.watershed_segmentation(**params)
            elif method == "clustering":
                segmented = self.segmenter.clustering_based_segmentation(**params)
            if segmented.ndim == 2 and segmented.dtype == np.int32:
                segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                img_display = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)
            else:
                img_display = segmented
            self.current_image = img_display
            self.segmenter.current_params[method] = params
            self.update_image_display()
        except Exception as e:
            print(f"Error during segmentation: {e}")
            messagebox.showerror("Error", f"Segmentation failed: {str(e)}")
    def save_image(self):
        if self.current_image is None:
            messagebox.showinfo("No Image", "No segmented image to save.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Image", "*.png"),
                                                            ("JPEG Image", "*.jpg"),
                                                            ("BMP Image", "*.bmp")])
        if file_path:
            try:
                img_bgr = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, img_bgr)
                messagebox.showinfo("Saved", f"Image saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Interactive Image Segmentation")
    gui = SegmentationGUI(root)
    root.mainloop()