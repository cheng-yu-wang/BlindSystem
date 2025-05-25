import pyaudio
from vosk import Model, KaldiRecognizer
import json
import threading
import os
import sys
from pathlib import Path
import cv2
import torch
import base64
import time
import queue
import openai
import pyttsx3
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from models.common import DetectMultiBackend
from utils.datasets import LoadStreams
#from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

#定义豆包的配置信息
class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 配置信息
            cls._instance.api_key = "5be20e09-db0b-488d-a27c-c9ecf8b9f74e"
            cls._instance.api_url = "https://ark.cn-beijing.volces.com/api/v3"
            cls._instance.model = "ep-20250421183159-zjph7"
        return cls._instance

#定义语音播报类
class SpeechEngine:
    """线程安全的语音播报引擎"""

    def __init__(self):
        self.engine = pyttsx3.init()    #初始化语音引擎
        self.engine.setProperty('rate', 250)    #设置语速为180
        self.engine.setProperty('volume', 1.0)  #设置音量
        self.queue = queue.Queue()  #初始化任务队列
        self.running = True     #用于标识语音引擎是否正在运行
        self.lock = threading.Lock()    #创建线程锁，使在同一时间只有一个关键代码段执行
        self.currently_speaking=False   #标识此时是否有语音播报，当有请求时可终止播报
        self.thread = threading.Thread(target=self._run_engine) #创建后台线程，用于运行后台语音引擎
        self.thread.daemon = True   #设置线程为守护线程，主线程结束时自动退出
        self.thread.start()     #启动后台线程

    def _run_engine(self):
        """语音引擎后台线程"""
        while self.running:
            try:
                with self.lock:
                    if not self.queue.empty():
                        text = self.queue.get_nowait()  #从队列中获取播报的文本，不会阻塞

                        #文本不为空，开始播报
                        if text:
                            self.currently_speaking = True
                            self.engine.say(text)
                            self.engine.runAndWait()
                            self.currently_speaking = False
                        else:
                            self.engine.stop()
            except Exception as err:
                print(f"语音播报出错: {err}")
            time.sleep(0.3) #每次循环完后暂停0.3s避免过度占用cpu

    def speak(self, text, priority=False):
        """线程安全的语音播报"""
        with self.lock:
            if priority:
                # 中断当前播报并清空队列
                if self.currently_speaking:
                    self.queue.queue.clear()
                    self.queue.put("")  # 发送中断信号,通过空文本触发engine.stop()
                self.queue.put(text)
            else:
                self.queue.put(text)

    def stop_all(self):
        """强制停止所有播报"""
        with self.lock:
            if self.currently_speaking:
                self.queue.queue.clear()
                self.queue.put("")
                self.engine.stop()  #终止底层语音引擎

    def stop(self):
        """停止语音引擎"""
        self.running = False
        self.thread.join()  #等到线程退出
        self.engine.stop()


class VoiceWakeup:
    """语音唤醒模块"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.wake_word = "你好"   #设置指令模式唤醒词
        self.interrupt_keywords=['你好']  #设置中断词
        self.listening = False  #是否处于指令监听
        self.last_activity_time = 0 #上次活动时间
        self.wake_timeout = 5  #无活动超时时间(秒)
        self.waiting_for_response = False   #是否正在等待系统响应
        self.need_wake_word = True  # 标记是否需要唤醒词
        self.last_command_time=0    #上次用户指令时间
        self.exit_timer=None    #超时定时器

        # 初始化语音识别模型
        try:
            model_path = "models/vosk-model-cn-0.22"    #语音模型
            self.model = Model(model_path)
            print("语音模型加载完成")
            self.audio = pyaudio.PyAudio()  #语音初始化
            self.stream = self.audio.open(
                format=pyaudio.paInt16, #语音格式
                channels=1, #单声道
                rate=16000, #采样率
                input=True,
                frames_per_buffer=4000  #缓存区大小
            )
        except Exception as e:
            print(f"语音识别模型加载失败: {e}")

        self.recognizer = KaldiRecognizer(self.model, 16000)    #初始化识别器

        # 启动语音唤醒线程（后台持续监听）
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop)
        self.thread.daemon = True   #主线程退出时自动终止
        self.thread.start()

    def _listen_loop(self):
        """语音唤醒监听循环"""
        print('开始语音唤醒监听')
        while self.running:
            data = self.stream.read(4000)   #读取音频数据（每次4000帧）

            #检查是否有完整语音结果
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()   #返回识别器的识别结果
                text = json.loads(result)['text'].replace(' ', '')  #提取文本

                # 优先处理中断关键词
                if any(keyword in text for keyword in self.interrupt_keywords):
                    print("[系统] 检测到中断指令")
                    self.main_window.speech_engine.stop_all()   #停止当前语音播报

                    # 如果已经在指令模式，重置状态
                    if self.listening:
                        self._exit_command_mode()

                    # 进入新的指令模式
                    self._enter_command_mode()
                    continue

                if not self.listening:
                    # 检测唤醒词
                    if self.wake_word in text and self.need_wake_word:
                        self._enter_command_mode()  #唤醒成功，进入指令模式
                else:
                    # 处理用户指令
                    if text and not self.waiting_for_response:
                        print(f"[用户指令] {text}")
                        self.main_window.process_user_command(text) #传递指令到主窗口
                        self.last_activity_time = time.time()   #更新活动时间
                        self.waiting_for_response = True    #标记等待响应

                    # 检查是否超时
                    if not self.waiting_for_response and time.time() - self.last_activity_time > self.wake_timeout:
                        self._exit_command_mode()   #超时退出指令模式

    def _enter_command_mode(self):
        """进入指令模式"""

        #取消之前的超时定时器
        #is_alive()用于检查进程是否正在进行
        if self.exit_timer and self.exit_timer.is_alive():
            self.exit_timer.cancel()
        self.listening = True   #开启指令监听
        self.need_wake_word = False     #进入指令模式后无需再次唤醒
        self.last_activity_time = time.time()   #记录活动时间
        self.last_command_time = time.time()  # 记录最后指令时间
        print("[系统响应] 你好，请问有什么可以帮到您？")
        self.main_window.pause_environment_report(True) #暂停环境播报
        self.main_window.speech_engine.speak("你好，请问有什么可以帮到您？", priority=True)

    def _exit_command_mode(self):
        """退出指令模式"""
        current_time = time.time()

        # 重置状态，移除原有条件判断，直接执行退出逻辑
        self.listening = False
        self.waiting_for_response = False
        self.need_wake_word = True
        print(f"[系统响应] 已退出指令模式（当前时间: {current_time}，最后活动时间: {self.last_activity_time}，最后指令时间: {self.last_command_time}）")
        self.main_window.pause_environment_report(False)    #恢复环境检测播报
        self.main_window.speech_engine.speak("返回环境检测模式")

    def command_response_received(self):
        """豆包响应完成时调用"""
        self.waiting_for_response = False   #用于标记是否完成
        self.last_activity_time = time.time()   #更新活动时间，获取的是当前时间的时间戳
        self.last_command_time = time.time()    #更新指令时间

        # 取消之前的定时器
        if self.exit_timer and self.exit_timer.is_alive():
            self.exit_timer.cancel()

        # 创建新定时器
        self.exit_timer = threading.Timer(self.wake_timeout, self._check_timeout)   #第一个参数是延迟时间，第二个参数要执行的函数对象
        self.exit_timer.start()

    def _check_timeout(self):
        """检查是否超时--由定时器触发"""
        current_time = time.time()
        print(f"[调试] 超时检查，当前时间: {current_time}，最后活动: {self.last_activity_time}，最后指令: {self.last_command_time}")
        if current_time - self.last_activity_time >= self.wake_timeout and current_time - self.last_command_time >= self.wake_timeout:
            print("[调试] 已超时，正在退出指令模式...")
            self._exit_command_mode()

    def stop(self):
        """停止语音唤醒--清理资源并终止线程"""
        if self.exit_timer and self.exit_timer.is_alive():
            self.exit_timer.cancel()
        self.running = False
        self.thread.join()  #等待线程退出
        self.stream.stop_stream()   #暂停音频流的数据传输，用于暂停录音或播放
        self.stream.close()     #关闭音频流并释放与之关联的资源
        self.audio.terminate()  #终止PyAudio的运行并释放所有系统资源

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.is_environment_speaking = None #是否正在进行环境播报
        self.current_objects = None #当前检测到的物体列表
        self.current_frame = None   #当前视频帧
        self.stop_btn = None    #停止按钮
        self.start_btn = None   #开始按钮
        self.status_label = None    #状态标签
        self.vid_img = None #视频显示区域
        self.setWindowTitle('盲人辅助导航系统')
        self.resize(800, 600)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))

        # 初始化配置
        self.environment_check_interval = 10  # 环境检测间隔(秒)
        self.last_analysis_time = 0 #上次分析环境的时间
        self.next_environment_check_time = time.time()  #下次环境检测时间
        self.environment_report_paused = False  # 环境播报暂停标志

        self.output_size = 480  #显示窗口大小
        self.device = select_device('cpu')  #计算设备选择
        self.vid_source = '0'   #摄像头选择
        self.stopEvent = threading.Event()  #创建一个线程事件对象
        self.stopEvent.clear()  #清除对象的标志设置
        self.analysis_interval = 10  #分析间隔(秒)

        # 距离计算参数--基于针孔相机模型
        self.focal_length = 200  # 摄像头焦距（需要根据实际情况校准）
        self.average_heights = {
            'person': 1.7,  # 行人平均高度(米)
            'car': 1.5,  # 汽车平均高度(米)
            'truck': 2.5,  # 卡车平均高度(米)
            'bus': 3.0,  # 公交车平均高度(米)
            'tree': 3.0,  # 树平均高度(米)
            'dog': 0.5,  # 狗平均高度(米)
            'cat': 0.3,  # 猫平均高度(米)
            'chair': 0.9,  # 椅子平均高度(米)
            'bench': 0.9,  # 长凳平均高度(米)
        }
        self.default_height = 1.0  #默认物体高度(米)

        # 加载YOLOv5模型
        self.model = self.model_load(
            weights="E:/pythonWorkspace/BlindSystem/yolov5-mask-42-master/pretrained/yolov5s.pt",device=self.device)

        # 初始化豆包API配置
        self.config = Config()
        self.api_queue = queue.Queue(maxsize=1) #api请求队列
        self.running = True #主循环运行标志

        # 初始化语音引擎
        self.speech_engine = SpeechEngine()

        # 初始化语音唤醒模块
        self.voice_wakeup = VoiceWakeup(self)

        # 启动API处理线程
        self.api_thread = threading.Thread(target=self._process_api_requests)
        self.api_thread.daemon = True
        self.api_thread.start()

        # 创建界面元素
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        # 视频显示区域
        self.vid_img = QLabel()
        self.vid_img.setAlignment(Qt.AlignCenter)
        self.vid_img.setMinimumSize(640, 480)

        # 控制按钮
        self.start_btn = QPushButton("开始检测")
        self.stop_btn = QPushButton("停止检测")
        self.start_btn.clicked.connect(self.open_cam)
        self.stop_btn.clicked.connect(self.close_vid)
        self.stop_btn.setEnabled(False)

        # 状态标签
        self.status_label = QLabel("系统就绪")
        self.status_label.setAlignment(Qt.AlignCenter)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.vid_img)
        layout.addWidget(self.status_label)

        # 按钮布局
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        # 设置样式
        self.setStyleSheet("""
            QPushButton {
                min-width: 100px;
                min-height: 30px;
                font-size: 14px;
            }
            QLabel {
                font-size: 16px;
            }
        """)

    def pause_environment_report(self, pause):
        """暂停或恢复环境播报"""
        self.environment_report_paused = pause
        print(f"[调试] 环境检测状态: {'暂停' if pause else '恢复'}")

        #指令模式下，停止环境检测
        if pause:
            self.status_label.setText("指令模式 - 等待用户指令...")
            self.last_analysis_time = time.time()
            self.next_environment_check_time = time.time() + self.environment_check_interval    #设置下次环境检测时间
        else:
            self.status_label.setText("检测中 - 正在分析环境...")
            # 立即触发一次环境检测
            self.next_environment_check_time = time.time()

    def process_user_command(self, command):
        """处理用户语音指令"""
        # 检查是否是中断指令
        if any(keyword in command for keyword in self.voice_wakeup.interrupt_keywords):
            print("[系统] 检测到唤醒指令")
            self.speech_engine.stop_all()
            self.voice_wakeup.command_response_received()   #重置唤醒后的系统状态
            return

        if not command or len(command.strip()) < 2:
            print("[系统] 空指令，忽略")
            self.voice_wakeup.command_response_received()
            return

        print("[系统响应] 正在处理您的请求...")
        self.speech_engine.stop_all()  # 确保中断之前的播报

        try:
            print("[系统] 正在调用API获取响应...")
            response = self._get_command_response(command)
            if response:
                print(f"[系统响应] 从API获取到内容: {response}")
                self.speech_engine.speak(response, priority=True)   #优先处理用户请求
            else:
                print("[系统] API返回空响应")
                self.speech_engine.speak("抱歉，我没有理解您的意思", priority=True)
        except Exception as err:
            print(f"[系统错误] API调用出错: {err}")
            self.speech_engine.speak("指令处理出错，请重试", priority=True)
        finally:
            self.voice_wakeup.command_response_received()

    def _get_command_response(self, command):
        """获取豆包对指令的响应"""
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.api_url
        )

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"用户指令: {command}\n"}]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content

    @torch.no_grad()    #禁用梯度计算，提高推理速度
    def model_load(self, weights="", device='', half=False, dnn=False):
        """加载YOLOv5模型"""
        device = select_device(device)  #选择计算设备
        model = DetectMultiBackend(weights, device=device, dnn=dnn) #初始化多后端模型

        # 获取部分属性，stride:模型的步长，用于表示特征图与输入图像之间的缩放比例，name:类别名称，pt:表示是否使用PyTorch后端，
        # jit:模型是否使用PyTorch的JIT编译，onnx:模型是否使用ONNX格式
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        if pt:
            model.model.half() if half else model.model.float()
        print("YOLOv5模型加载完成!")
        return model

    def open_cam(self):
        """启动摄像头检测"""
        self.start_btn.setEnabled(False)    #禁用开始检测按钮
        self.stop_btn.setEnabled(True)      #启用停止检测按钮
        self.status_label.setText("正在检测中...")
        self.speech_engine.speak("系统启动，开始检测周围环境")

        # 启动检测线程（避免阻塞UI线程）
        th = threading.Thread(target=self.detect_vid)
        th.start()

    def detect_vid(self):
        """视频检测主循环（带距离计算）"""
        model = self.model  #获取已加载YOLOv5模型
        output_size = self.output_size  #获取显示窗口的大小
        source = str(self.vid_source)   #获取摄像头

        # 初始化摄像头
        dataset = LoadStreams(source, img_size=[640, 640], stride=model.stride, auto=model.pt)

        for path, im, im0s, vid_cap, s in dataset:
            if self.stopEvent.is_set(): #检查请求是否停止
                break

            # 准备输入数据
            im = torch.from_numpy(im).to(self.device)   #转换为Tensor并移至设备
            im = im.float() / 255.0 #归一化处理
            if len(im.shape) == 3:  #确保输入维度正确
                im = im[None]   #添加批次维度

            #模型推理
            pred = model(im, augment=False, visualize=False)    #前向传播
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000) #非极大值抑制

            # 处理检测结果
            for i, det in enumerate(pred):  #遍历每个检测结果
                p, im0 = path[i], im0s[i].copy()    #获取原始图像
                annotator = Annotator(im0, line_width=3, example=str(model.names))  #初始化标注器

                detected_objects = []   #存储检测到的物体信息
                if len(det):
                    #调整边界框坐标以匹配原始图像尺寸
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    #边界框坐标、置信度、类别
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)    #转换为整数类别ID
                        label_name = model.names[c] #获取类别名称

                        # 计算物体高度(像素)
                        box_height = xyxy[3] - xyxy[1]

                        # 获取物体平均高度，用于计算距离
                        obj_height = self.average_heights.get(label_name, self.default_height)

                        # 基于针孔相机计算距离，距离=（物体实际高度x相机焦距）/物体像素高度
                        distance = (obj_height * self.focal_length) / box_height

                        # 计算检测框中心坐标，用于定位
                        box_center_x = (xyxy[0] + xyxy[2]) / 2
                        box_center_y = (xyxy[1] + xyxy[3]) / 2

                        # 在构建标注标签，包括类别、置信度和距离
                        label = f'{label_name} {conf:.2f} {distance:.1f}m'
                        annotator.box_label(xyxy, label, color=colors(c, True)) #在图像上绘制边界框和标签

                        # 保存检测到的物体信息
                        detected_objects.append({
                            'name': label_name,
                            'distance': distance,
                            'position': (box_center_x, box_center_y)
                        })

                # 更新显示画面
                frame = annotator.result()  #获取标注后的图像
                self.update_display(frame)  #更新UI显示

                # 保存当前帧和物体信息用于环境分析
                self.current_frame = frame.copy()
                self.current_objects = detected_objects

                # 定时分析（如果环境播报没有被暂停）
                current_time = time.time()
                if not self.environment_report_paused and current_time - self.last_analysis_time >= self.environment_check_interval:
                    self.analyze_current_frame()    #分析当前帧
                    self.last_analysis_time = current_time #更新最后的分析时间

        self.reset_vid()    #重置视频检测状态

    def update_display(self, frame):
        """更新显示画面"""
        #调整图像大小以 适应显示窗口
        resize_scale = self.output_size / frame.shape[0]
        frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        #保存图像到临时文件（用于PyQt显示）
        cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)

        # 在主线程中更新UI
        self.vid_img.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))

    def analyze_current_frame(self):
        """分析当前帧并准备API请求"""
        if not hasattr(self, 'current_frame') or self.current_frame is None or self.api_queue.full():
            return

        try:
            #将当前帧转换为Base64编码（用于API传输）
            _, img_encoded = cv2.imencode('.jpg', self.current_frame)
            img_bytes = img_encoded.tobytes()  # 将 ndarray 转换为 bytes
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # 构建物体信息提示文本
            objects_info = "检测到以下物体: "
            if hasattr(self, 'current_objects') and self.current_objects:
                for obj in self.current_objects:
                    objects_info += f"{obj['name']}(距离:{obj['distance']:.1f}米), "
                objects_info = objects_info.rstrip(", ")
            else:
                objects_info = "未检测到障碍物"

            # 放入队列
            self.api_queue.put((img_base64, objects_info))
            self.status_label.setText("正在分析环境...")

        except Exception as err:
            print(f"分析失败: {err}")
            self.status_label.setText(f"分析出错: {str(err)}")

    def _get_scene_description(self, image_base64, objects_info=""):
        """使用豆包API获取场景描述"""
        try:
            client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_url
            )

            # 修改提示词，要求API只在需要行动时才给出建议
            response = client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{objects_info}\n"
                                                     "这张图片已经用YOLOv5进行了目标检测，并计算了物体距离。"
                                                     "请用简洁的语言描述检测到的环境，侧重描述被圈出的物体，对其他事物可适当描述，"
                                                     "如果障碍物对其行动有影响再给出对盲人的具体行动建议。例如：'正前方2-3米处有一个人，建议稍向右偏转绕行'"},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                        ]
                    }
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as err:
            print(f"API调用失败: {err}")
            return None

    def _process_api_requests(self):
        alert_keywords = ['避让', '绕行', '左转', '右转', '小心', '注意', '危险','信号灯','公交车站','商城','岔路口','红绿灯']

        while self.running:
            try:
                # 如果处于指令模式，则跳过环境检测
                if self.voice_wakeup.listening:
                    time.sleep(0.5)
                    continue

                current_time = time.time()

                # 检查是否可以开始新的环境检测
                if current_time >= self.next_environment_check_time and not self.api_queue.full():
                    #检查当前帧是否存在
                    if not hasattr(self, 'current_frame') or self.current_frame is None:
                        time.sleep(0.1)
                        continue

                    # 获取最新帧进行分析
                    _, img_encoded = cv2.imencode('.jpg', self.current_frame)
                    img_bytes = img_encoded.tobytes()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

                    # 构建检测物体信息文本
                    objects_info = "检测到以下物体: "
                    if hasattr(self, 'current_objects') and self.current_objects:
                        objects_info += ", ".join([f"{obj['name']}({obj['distance']:.1f}米)"
                                                   for obj in self.current_objects])
                    else:
                        objects_info = "未检测到障碍物"

                    # 放入队列
                    self.api_queue.put((img_base64, objects_info))
                    self.status_label.setText("正在分析环境...")
                    self.next_environment_check_time = current_time + self.environment_check_interval  # 设置下次检测时间

                # 处理API响应
                if not self.api_queue.empty():
                    img_base64, objects_info = self.api_queue.get_nowait()
                    description = self._get_scene_description(img_base64, objects_info)
                    if description:
                        print("\n环境分析结果:", description)

                        # 检查是否需要语音提醒
                        if any(keyword in description for keyword in alert_keywords):
                            self.speech_engine.speak(description)

            except Exception as err:
                print(f"处理出错: {err}")
                time.sleep(1)

    def update_status(self, message):
        """更新状态标签"""
        self.status_label.setText(message)

    def reset_vid(self):
        """重置视频检测"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("系统就绪")
        self.stopEvent.clear()  #清除停止标志

    def close_vid(self):
        """停止视频检测"""
        self.stopEvent.set()    #设置停止标志
        self.speech_engine.speak("检测已停止")
        self.reset_vid()    #重置UI状态

    def closeEvent(self, event):
        """关闭窗口事件"""
        self.running = False    #停止API处理线程
        self.stopEvent.set()    #停止视频检查线程
        self.speech_engine.stop()   #停止语音引擎
        self.voice_wakeup.stop()    #停止语音唤醒模块

        reply = QMessageBox.question(
            self,
            '退出',
            "确定要退出系统吗?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            self.running=True

    def on_speech_finished(self):
        """语音播报完成回调"""
        # 如果是环境播报完成，设置下次检测时间
        if hasattr(self, 'is_environment_speaking') and self.is_environment_speaking:
            self.is_environment_speaking = False
            self.next_environment_check_time = time.time() + 5


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(10)
    app.setFont(font, None)

    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())