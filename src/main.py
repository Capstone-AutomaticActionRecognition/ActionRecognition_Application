from pyautogui import press, typewrite, hotkey, moveTo, position
import tkinter as tk
import numpy as np
import cv2
import os
from typing import Tuple
import io
import time
import torch
from PIL import Image, ImageTk
import tvm.contrib.graph_runtime as graph_runtime
from mobilenet_v2_tsm import MobileNetV2
import onnx
import tvm
import tvm.relay
import torch.onnx
from typing import Tuple
import torchvision
from helpers import *


SOFTMAX_THRES = 0
HISTORY_LOGIT = True
n_still_frame = 0
WINDOW_NAME = 'Video Gesture Recognition'

categories = [
    "Doing other things",  # 0
    "Drumming Fingers",  # 1
    "No gesture",  # 2
    "Pulling Hand In",  # 3
    "Pulling Two Fingers In",  # 4
    "Pushing Hand Away",  # 5
    "Pushing Two Fingers Away",  # 6
    "Rolling Hand Backward",  # 7
    "Rolling Hand Forward",  # 8
    "Shaking Hand",  # 9
    "Sliding Two Fingers Down",  # 10
    "Sliding Two Fingers Left",  # 11
    "Sliding Two Fingers Right",  # 12
    "Sliding Two Fingers Up",  # 13
    "Stop Sign",  # 14
    "Swiping Down",  # 15
    "Swiping Left",  # 16
    "Swiping Right",  # 17
    "Swiping Up",  # 18
    "Thumb Down",  # 19
    "Thumb Up",  # 20
    "Turning Hand Clockwise",  # 21
    "Turning Hand Counterclockwise",  # 22
    "Zooming In With Full Hand",  # 23
    "Zooming In With Two Fingers",  # 24
    "Zooming Out With Full Hand",  # 25
    "Zooming Out With Two Fingers"  # 26
]

class Application():
    def __init__(self):
        self.WINDOW_NAME = 'Video Gesture Recognition'
        self.t = None
        self.index = 0
        
        self.transform = get_transform()
        self.executor, self.ctx = get_executor()
        self.buffer = (
            tvm.nd.empty((1, 3, 56, 56), ctx=self.ctx),
            tvm.nd.empty((1, 4, 28, 28), ctx=self.ctx),
            tvm.nd.empty((1, 4, 28, 28), ctx=self.ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 8, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 12, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 12, 14, 14), ctx=self.ctx),
            tvm.nd.empty((1, 20, 7, 7), ctx=self.ctx),
            tvm.nd.empty((1, 20, 7, 7), ctx=self.ctx)
        )
        self.idx = 0
        self.history = [2]
        self.history_logit = []
        self.history_timing = []
        self.i_frame = -1
        

        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=600, height=300)
        self.canvas.grid(columnspan=7)
        self.titleText = tk.Label(self.root, text="Automatic Action Recognition")
        self.titleText.grid(columnspan=3, column=2, row=0)
        self.camLabel = tk.Label(self.root)
        self.camLabel.grid(columnspan=3, column=2, row=1)
        self.cap = cv2.VideoCapture(0)
        self.runApp()
        self.root.mainloop()


    def runApp(self):
            self.i_frame +=1
            _, frame = self.cap.read()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camLabel.imgtk = imgtk
            self.camLabel.configure(image=imgtk)
            if self.i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
                t1 = time.time()
                img_tran = self.transform([Image.fromarray(frame).convert('RGB')])
                input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
                img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=self.ctx)
                inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + self.buffer
                outputs = self.executor(inputs)
                self.feat, self.buffer = outputs[0], outputs[1:]
                assert isinstance(self.feat, tvm.nd.NDArray)
                
                if SOFTMAX_THRES > 0:
                    feat_np = self.feat.asnumpy().reshape(-1)
                    feat_np -= feat_np.max()
                    softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                    print(max(softmax))
                    if max(softmax) > SOFTMAX_THRES:
                        idx_ = np.argmax(self.feat.asnumpy(), axis=1)[0]
                    else:
                        idx_ = self.idx
                else:
                    idx_ = np.argmax(self.feat.asnumpy(), axis=1)[0]

                if HISTORY_LOGIT:
                    self.history_logit.append(self.feat.asnumpy())
                    self.history_logit = self.history_logit[-12:]
                    avg_logit = sum(self.history_logit)
                    idx_ = np.argmax(avg_logit, axis=1)[0]

                self.idx, self.history = process_output(idx_, self.history)

                t2 = time.time()
                print(f"{self.index} {categories[self.idx]}")


                current_time = t2 - t1

            if self.t is None:
                self.t = time.time()
            else:
                nt = time.time()
                self.index += 1
                self.t = nt
            self.camLabel.after(1,self.runApp)


    """
    def run(self):
        cap = cv2.VideoCapture(0)
        
        # set a lower resolution for speed up
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # env variables
        while True:
            self.i_frame += 1
            _, img = cap.read()  # (480, 640, 3) 0 ~ 255
            if self.i_frame % 2 == 0:  # skip every other frame to obtain a suitable frame rate
                t1 = time.time()
                img_tran = transform([Image.fromarray(img).convert('RGB')])
                input_var = torch.autograd.Variable(img_tran.view(1, 3, img_tran.size(1), img_tran.size(2)))
                img_nd = tvm.nd.array(input_var.detach().numpy(), ctx=self.ctx)
                inputs: Tuple[tvm.nd.NDArray] = (img_nd,) + buffer
                outputs = self.executor(inputs)
                feat, buffer = outputs[0], outputs[1:]
                assert isinstance(feat, tvm.nd.NDArray)
                
                if SOFTMAX_THRES > 0:
                    feat_np = feat.asnumpy().reshape(-1)
                    feat_np -= feat_np.max()
                    softmax = np.exp(feat_np) / np.sum(np.exp(feat_np))

                    print(max(softmax))
                    if max(softmax) > SOFTMAX_THRES:
                        idx_ = np.argmax(feat.asnumpy(), axis=1)[0]
                    else:
                        idx_ = self.idx
                else:
                    idx_ = np.argmax(feat.asnumpy(), axis=1)[0]

                if HISTORY_LOGIT:
                    self.history_logit.append(feat.asnumpy())
                    self.history_logit = self.history_logit[-12:]
                    avg_logit = sum(self.history_logit)
                    idx_ = np.argmax(avg_logit, axis=1)[0]

                self.idx, self.history = process_output(idx_, self.history)

                t2 = time.time()
                print(f"{self.index} {categories[self.idx]}")


                current_time = t2 - t1


            if self.t is None:
                self.t = time.time()
            else:
                nt = time.time()
                self.index += 1
                self.t = nt

            
            
            currentMouseX, currentMouseY = position()
            if self.idx ==  15:
                #if using the swipe down gesture
                moveTo(currentMouseX, currentMouseY + 5)
            elif self.idx == 16:
                #left
                moveTo(currentMouseX - 5, currentMouseY)
            elif self.idx == 17:
                #right
                moveTo(currentMouseX + 5, currentMouseY)
            elif self.idx == 18:
                #up
                moveTo(currentMouseX, currentMouseY - 5)
    """


app = Application()




