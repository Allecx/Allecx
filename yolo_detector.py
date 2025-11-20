import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import time


class YOLODetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv5目标检测软件")
        self.root.geometry("1200x800")
        
        # 初始化变量
        self.model_path = None
        self.model = None
        self.cap = None
        self.is_running = False
        self.current_image = None
        self.current_video_path = None
        self.video_cap = None
        self.playing_video = False
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 模型选择
        ttk.Label(control_frame, text="模型路径:").pack(anchor=tk.W)
        model_frame = ttk.Frame(control_frame)
        model_frame.pack(fill=tk.X, pady=5)
        self.model_path_var = tk.StringVar()
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var)
        self.model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="浏览", command=self.browse_model).pack(side=tk.RIGHT, padx=(5, 0))
        
        # 加载模型按钮
        ttk.Button(control_frame, text="加载模型", command=self.load_model).pack(fill=tk.X, pady=5)
        
        # 功能选择
        ttk.Label(control_frame, text="功能选择:").pack(anchor=tk.W, pady=(10, 5))
        self.func_var = tk.StringVar(value="image")
        
        ttk.Radiobutton(control_frame, text="图片检测", variable=self.func_var, value="image").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="摄像头检测", variable=self.func_var, value="camera").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="视频检测", variable=self.func_var, value="video").pack(anchor=tk.W)
        
        # 图片检测相关控件
        self.image_frame = ttk.Frame(control_frame)
        self.image_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.image_frame, text="图片路径:").pack(anchor=tk.W)
        img_frame = ttk.Frame(self.image_frame)
        img_frame.pack(fill=tk.X)
        self.image_path_var = tk.StringVar()
        ttk.Entry(img_frame, textvariable=self.image_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(img_frame, text="浏览", command=self.browse_image).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(self.image_frame, text="开始检测", command=self.detect_image).pack(fill=tk.X, pady=(5, 0))
        
        # 摄像头相关控件
        self.camera_frame = ttk.Frame(control_frame)
        self.camera_frame.pack(fill=tk.X, pady=5)
        self.camera_frame.pack_forget()  # 默认隐藏
        ttk.Button(self.camera_frame, text="打开摄像头", command=self.open_camera).pack(fill=tk.X, pady=5)
        ttk.Button(self.camera_frame, text="关闭摄像头", command=self.close_camera).pack(fill=tk.X, pady=5)
        
        # 视频检测相关控件
        self.video_frame = ttk.Frame(control_frame)
        self.video_frame.pack(fill=tk.X, pady=5)
        self.video_frame.pack_forget()  # 默认隐藏
        ttk.Label(self.video_frame, text="视频路径:").pack(anchor=tk.W)
        vid_frame = ttk.Frame(self.video_frame)
        vid_frame.pack(fill=tk.X)
        self.video_path_var = tk.StringVar()
        ttk.Entry(vid_frame, textvariable=self.video_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(vid_frame, text="浏览", command=self.browse_video).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(self.video_frame, text="开始检测", command=self.detect_video).pack(fill=tk.X, pady=(5, 0))
        ttk.Button(self.video_frame, text="暂停/继续", command=self.toggle_video_play).pack(fill=tk.X, pady=5)
        
        # 功能选择事件绑定
        self.func_var.trace('w', self.on_func_change)
        
        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="检测结果", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建Canvas用于显示图像
        self.canvas = tk.Canvas(display_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 进度条（用于视频检测）
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(display_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.progress_bar.pack_forget()
        
        # 信息显示区域
        info_frame = ttk.LabelFrame(main_frame, text="检测信息", padding=10)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=6)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
    def on_func_change(self, *args):
        # 根据选择的功能显示/隐藏相关控件
        func = self.func_var.get()
        if func == "image":
            self.image_frame.pack(fill=tk.X, pady=5)
            self.camera_frame.pack_forget()
            self.video_frame.pack_forget()
        elif func == "camera":
            self.image_frame.pack_forget()
            self.camera_frame.pack(fill=tk.X, pady=5)
            self.video_frame.pack_forget()
        elif func == "video":
            self.image_frame.pack_forget()
            self.camera_frame.pack_forget()
            self.video_frame.pack(fill=tk.X, pady=5)
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(
            title="选择YOLOv5模型文件",
            filetypes=[("PyTorch模型文件", "*.pt"), ("所有文件", "*.*")]
        )
        if file_path:
            self.model_path_var.set(file_path)
            self.model_path = file_path
    
    def browse_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        if file_path:
            self.image_path_var.set(file_path)
    
    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.video_path_var.set(file_path)
            self.current_video_path = file_path
    
    def load_model(self):
        if not self.model_path:
            messagebox.showerror("错误", "请先选择模型文件")
            return
        
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
            messagebox.showinfo("成功", "模型加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
    
    def detect_image(self):
        if not self.model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        image_path = self.image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("错误", "请选择有效的图片文件")
            return
        
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("无法读取图片")
            
            # 检测
            results = self.model(img)
            
            # 获取检测结果
            img_with_detections = results.render()[0]
            
            # 显示结果
            self.display_image(img_with_detections)
            
            # 显示检测信息
            self.show_detection_info(results)
            
        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {str(e)}")
    
    def open_camera(self):
        if not self.model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("无法打开摄像头")
            
            self.is_running = True
            self.update_camera()
        except Exception as e:
            messagebox.showerror("错误", f"打开摄像头失败: {str(e)}")
    
    def close_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width()//2, 
            self.canvas.winfo_height()//2, 
            text="摄像头已关闭", 
            fill="white", 
            font=("Arial", 20)
        )
    
    def update_camera(self):
        if not self.is_running or not self.cap or not self.cap.isOpened():
            return
        
        ret, frame = self.cap.read()
        if ret:
            # 检测
            results = self.model(frame)
            img_with_detections = results.render()[0]
            
            # 显示结果
            self.display_image(img_with_detections)
            
            # 显示检测信息
            self.show_detection_info(results)
        
        # 继续更新
        if self.is_running:
            self.root.after(10, self.update_camera)
    
    def detect_video(self):
        if not self.model:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("错误", "请选择有效的视频文件")
            return
        
        if self.video_cap and self.video_cap.isOpened():
            self.video_cap.release()
        
        self.video_cap = cv2.VideoCapture(video_path)
        if not self.video_cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return
        
        self.playing_video = True
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.update_video()
    
    def update_video(self):
        if not self.playing_video or not self.video_cap or not self.video_cap.isOpened():
            return
        
        ret, frame = self.video_cap.read()
        if not ret:
            # 视频播放完毕
            self.playing_video = False
            self.progress_bar.pack_forget()
            return
        
        # 检测
        results = self.model(frame)
        img_with_detections = results.render()[0]
        
        # 显示结果
        self.display_image(img_with_detections)
        
        # 显示检测信息
        self.show_detection_info(results)
        
        # 更新进度条
        total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.progress_var.set((current_frame / total_frames) * 100)
        
        # 继续更新
        if self.playing_video:
            self.root.after(30, self.update_video)  # 约30fps
    
    def toggle_video_play(self):
        if self.playing_video:
            self.playing_video = False
            self.progress_bar.pack_forget()
        else:
            if self.current_video_path:
                self.detect_video()
    
    def display_image(self, img):
        # 转换BGR到RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL Image
        img_pil = Image.fromarray(img_rgb)
        
        # 调整大小以适应canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_pil = img_pil.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        # 转换为PhotoImage
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # 显示在canvas上
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        
        # 保持引用以防止被垃圾回收
        self.current_image = img_tk
    
    def show_detection_info(self, results):
        # 清空信息框
        self.info_text.delete(1.0, tk.END)
        
        # 获取检测结果
        detections = results.pandas().xyxy[0]  # 获取pandas格式的结果
        
        if len(detections) > 0:
            info_str = f"检测到 {len(detections)} 个对象:\n\n"
            for i, det in detections.iterrows():
                info_str += f"对象 {i+1}: {det['name']} (置信度: {det['confidence']:.2f})\n"
                info_str += f"  位置: ({int(det['xmin'])}, {int(det['ymin'])}) - ({int(det['xmax'])}, {int(det['ymax'])})\n\n"
        else:
            info_str = "未检测到任何对象"
        
        self.info_text.insert(tk.END, info_str)
    
    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        if self.video_cap:
            self.video_cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()