from ast import Lambda
from math import floor
from cv2 import createButton
import pyautogui
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
        self.current_time = 1


        # Very Crude 1st attempt at GUI -- need to organize

        self.webcamDimensions = (356, 200)
        

        self.root = tk.Tk()
        self.root.geometry("1920x1080")
        self.appX = 1920
        self.appY = 1080
        self.app = tk.Frame(self.root, width=self.appX, height=self.appY, bg='white')
        self.app.place(x=00,y=0)

        
        self.moveCamFactor = 50
        self.ableToExecute = True
        self.LeftClickFlag = False

        self.camFrameX = 25
        self.camFrameY = 275
        self.camFrame = tk.Frame(self.app, width=self.webcamDimensions[0], height=self.webcamDimensions[1])
        self.camFrame.place(x=self.camFrameX, y=self.camFrameY)

        self.camLabel = tk.Label(self.camFrame)
        self.camLabel.pack()
        self.cam_stats = tk.Label(self.camFrame, text=str(1/self.current_time))
        self.cam_stats.pack()

        self.TopBar = tk.Frame(self.app, width=self.appX, height=100, bg='green')        
        self.TopBar.place(x=0,y=0)
        self.ToggleCamMovement = tk.Button(self.TopBar, text="ToggleCamMovement", bg='black', fg='white', command=self.toggle_canExecute)
        self.ToggleCamMovement.place(x=25, y=25)


        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.runApp()
        self.root.mainloop()


    def toggle_canExecute(self):
        if self.ableToExecute:
            self.ableToExecute = False
        else:
            self.ableToExecute = True
    
    def runApp(self):
            #Get Camera Information for camLabel widget and each
            #frame for the feature section below
            self.i_frame +=1
            _, frame = self.cap.read()
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = img.resize(self.webcamDimensions, resample=1)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camLabel.imgtk = imgtk
            self.camLabel.configure(image=imgtk)



            #Feature Mapping and Extraction
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
                #print(f"{self.index} {categories[self.idx]}")

                self.current_time = (t2 - t1)
                self.cam_stats.configure(text=str(floor(1/self.current_time))+" Frames/second")
            #counting section for the print of index and categories
            if self.t is None:
                self.t = time.time()
            else:
                nt = time.time()
                self.index += 1
                self.t = nt

            #controller -- maybe need a fix for the delay
            if self.ableToExecute:
                if self.idx == 14:
                    pyautogui.click(button="left", clicks=1)
                elif self.idx == 15:
                    #swipe down                
                    self.camFrame.place(x=self.camFrameX,y=(self.camFrameY+self.moveCamFactor)) 
                elif self.idx == 18:
                    #swipe up
                    self.camFrame.place(x=self.camFrameX,y=(self.camFrameY-self.moveCamFactor)) 
            if not self.LeftClickFlag:
                if self.idx == 14:
                    self.LeftClickFlag = True
                    #DO THE ACTION
                    pyautogui.click(button='left')
            
            self.camLabel.after(1,self.runApp)

app = Application()




