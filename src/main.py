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
        self.history = [1,2,3]
        self.history_logit = []
        self.history_timing = []
        self.i_frame = -1
        self.current_time = 1


        # Very Crude 1st attempt at GUI -- need to organize

        self.webcamDimensions = (356, 200)
        

        self.root = tk.Tk()
        self.root.geometry("960x600")
        self.root.title("Automatic Action Recognition")
        self.appX = 1920
        self.appY = 1080
        self.app = tk.Frame(self.root, width=self.appX, height=self.appY, bg='white')
        self.app.place(x=00,y=0)

        
        self.moveCamFactor = 50
        self.gestureFlag = True
        self.previousAction = ""

        self.camFrameX = 25
        self.camFrameY = 275
        self.camFrame = tk.Frame(self.app, width=self.webcamDimensions[0], height=self.webcamDimensions[1])
        self.camFrame.place(x=self.camFrameX, y=self.camFrameY)

        self.camLabel = tk.Label(self.camFrame)
        self.camLabel.pack()
        self.cam_stats = tk.Label(self.camFrame, text=str(1/self.current_time))
        self.cam_stats.pack()
        self.cam_gesture = tk.Label(self.camFrame, text=str({categories[self.idx]}))
        self.cam_gesture.pack()

        self.TopBar = tk.Frame(self.app, width=self.appX, height=100, bg='green')        
        self.TopBar.place(x=0,y=0)
        self.ToggleCamMovement = tk.Button(self.TopBar, text="ToggleMovement", bg='black', fg='white', command=self.toggle_canExecute)
        self.ToggleCamMovement.place(x=25, y=25)

        self.enableGestureListRecord = tk.Button(self.TopBar, text="Record Gesture List", bg='black', fg='white', command=self.record_gesture_list)
        self.enableGestureListRecord.place(x=200, y=25)
        
        self.recordGesturesRight = False
        self.recordGesturesLeft = False

        self.variable = tk.StringVar(self.app)
        self.variable.set("Select an action:")
        self.action_list = ["Go Right", "Go Left"]
        self.dropDown = tk.OptionMenu(self.app, self.variable, *self.action_list)
        self.dropDown.place(x= 200, y= 125)

        self.variableGesture = tk.StringVar(self.app)
        self.variableGesture.set("Select a Gesture:")
        self.gesture_list = ["Swipe Left", "Swipe Right"]
        self.dropDown_gestures = tk.OptionMenu(self.app, self.variableGesture, *self.gesture_list)
        self.dropDown_gestures.place(x=750, y=380)
        self.dropDown_gestures.place_forget()

        self.rightActions = [12,17]
        self.leftActions = [11,16]

        self.ableToExecute = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.runApp()
        self.root.mainloop()

    def record_gesture_list(self):
        current_action = self.variable.get()
        if current_action == "Go Left":
            if self.recordGesturesLeft == False:
                self.recordGesturesLeft = True
            else:
                self.recordGesturesLeft = False
            return
        elif current_action == "Go Right":
            if self.recordGesturesRight == False:
                self.recordGesturesRight = True
            else:
                self.recordGesturesRight = False
            return
        
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
            self.cam_gesture.config(text=str({categories[self.idx]}))



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
                print(f"{self.index} {categories[self.idx]}")

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
            
            if self.idx == 14 and self.ableToExecute:
                pyautogui.click(button="left", clicks=1)
            elif self.idx == 15 and self.ableToExecute:
                #swipe down                
                self.camFrame.place(x=self.camFrameX,y=(self.camFrameY+self.moveCamFactor)) 
            elif self.idx == 18 and self.ableToExecute:
                #swipe up
                self.camFrame.place(x=self.camFrameX,y=(self.camFrameY-self.moveCamFactor)) 
            elif self.idx in self.leftActions and self.previousAction not in self.leftActions:
                pyautogui.press('left')
            elif self.idx in self.rightActions and self.previousAction not in self.rightActions:
                pyautogui.press('right')
            self.previousAction = self.idx 
            self.camLabel.after(1,self.runApp)

app = Application()




