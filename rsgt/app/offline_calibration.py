import csv
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
import datetime
import argparse

import cv2
import dlib
import numpy as np
import wx
import wx.lib.newevent

from ..config import config as configuration
from ..eye import eye_filter, eyedata
from ..face import facedata, get_face_boxes, get_face_landmarks
from ..screen import screen
from ..util import (LM_calibration, calc_calibration_results,
                    dlgAskopenfilename, dlgAsksaveasfilename, dlgAskyesno,
                    dlgShowerror, dlgShowinfo)
from ..iris_detectors import get_iris_detector

module_dir = Path(__file__).parent.parent
debug_mode = True

def str2points(s):
    p = np.array(tuple(map(float,s[1:-1].split(','))))
    return p

ID_OPEN_CAMERA_CONFIG = wx.NewIdRef()
ID_OPEN_FACE_CONFIG = wx.NewIdRef()
ID_OPEN_CALINFO = wx.NewIdRef()
ID_SAVE_CALINFO = wx.NewIdRef()
ID_RUN_CAL = wx.NewIdRef()

menu_items_all = [
    ID_OPEN_CAMERA_CONFIG,
    ID_OPEN_FACE_CONFIG,
    ID_OPEN_CALINFO,
    ID_SAVE_CALINFO,
    ID_RUN_CAL
]

class calibrationDialog(wx.Dialog):
    def __init__(self, parent):
        super(calibrationDialog, self).__init__(parent=parent, id=wx.ID_ANY, title='Running calibration...')

        self.parent = parent
        self.fitting_param = None
        self.results = None

        self.mediapanel = wx.Panel(self,wx.ID_ANY)
        self.camera_view = camera_view(self.mediapanel, wx.ID_ANY, wx.Bitmap(self.parent.camera_view_width,self.parent.camera_view_height))
        mediasizer = wx.BoxSizer()
        mediasizer.Add(self.camera_view)
        self.mediapanel.SetSizer(mediasizer)

        self.results_text = wx.TextCtrl(self, wx.ID_ANY, '\n\n\n', style=wx.TE_MULTILINE|wx.TE_READONLY)

        self.buttonpanel = wx.Panel(self,wx.ID_ANY)
        self.message_text = wx.StaticText(self.buttonpanel,wx.ID_ANY,'-/- calibration points  -/- frames          ')
        self.button_save = wx.Button(self.buttonpanel,wx.ID_ANY,"Save")
        self.button_cancel = wx.Button(self.buttonpanel,wx.ID_CANCEL,"Cancel")
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.message_text,0,wx.RIGHT, 10)
        sizer.Add(self.button_save,0,wx.ALIGN_CENTER)
        sizer.Add(self.button_cancel,0,wx.ALIGN_CENTER)
        self.buttonpanel.SetSizer(sizer)

        self.button_save.Enable(False)
        self.button_save.Bind(wx.EVT_BUTTON, self.save)

        mainsizer = wx.BoxSizer(wx.VERTICAL)
        mainsizer.Add(self.mediapanel,4,wx.EXPAND)
        mainsizer.Add(self.results_text,0,wx.EXPAND)
        mainsizer.Add(self.buttonpanel,0,wx.ALIGN_RIGHT|wx.ALL, 10)
        self.SetSizer(mainsizer)
        self.Fit()

        self.running = True
        self.thread = threading.Thread(target=self.calibration_loop)
        self.thread.start()

        self.Show()

    def calibration_loop(self):
        cap = cv2.VideoCapture(self.parent.moviefile)
        calibration_data = []
        face_rvec = None
        face_tvec = None

        num_calibration_points = self.parent.calpoint_listbox.GetItemCount()

        for calpoint_idx in range(num_calibration_points):
            fidx = int(self.parent.calpoint_listbox.GetItem(calpoint_idx, 0).GetText())
            uidx = int(self.parent.calpoint_listbox.GetItem(calpoint_idx, 1).GetText())
            calibration_sample_point = str2points(self.parent.calpoint_listbox.GetItem(calpoint_idx, 2).GetText())

            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,fidx-1))
            face_rvec = None
            face_tvec = None

            for fcount in range(uidx-fidx):
                if not self.running:
                    return

                _, frame = cap.read()

                try:
                    frame_mono = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                except:
                    print('{},{}'.format(fcount,fidx+uidx))
                    continue

                self.message_text.SetLabel('{}/{} calibration points  {}/{} frames'.format(calpoint_idx+1, num_calibration_points, fcount+1, uidx-fidx))

                #if debug_mode:
                #    self.downscaling_factor = 0.25

                if self.parent.downscaling_factor == 1.0: # original size
                    dets, _ = get_face_boxes(frame_mono, engine='dlib_hog')
                else: # downscale camera image
                    dets, _ = get_face_boxes(cv2.resize(frame_mono, None, fx=self.parent.downscaling_factor, fy=self.parent.downscaling_factor), engine='dlib_hog') # detections, scores, weight_indices
                    inv = 1.0/self.parent.downscaling_factor
                    # recover rectangle size
                    for i in range(len(dets)):
                        dets[i] = dlib.rectangle(int(dets[i].left()*inv), int(dets[i].top()*inv),
                                                int(dets[i].right()*inv), int(dets[i].bottom()*inv))

                detect_face = False
                if self.parent.area_of_interest is None:
                    if len(dets) > 0:
                        detect_face = True
                        target_idx = 0
                else:
                    for target_idx in range(len(dets)):
                        if self.parent.area_of_interest.contains(dets[target_idx]):
                            detect_face = True
                            break
                
                if detect_face: # face is found
                    # only first face is used
                    landmarks = get_face_landmarks(frame_mono, dets[target_idx])
                    
                    # create facedata
                    face = facedata(landmarks, camera_matrix=self.parent.camera_matrix, face_model=self.parent.face_model,
                        eye_params=self.parent.eye_params, prev_rvec=face_rvec, prev_tvec=face_tvec)

                    # create eyedata
                    left_eye = eyedata(frame_mono, landmarks, eye='L', iris_detector=self.parent.iris_detector)
                    right_eye = eyedata(frame_mono, landmarks, eye='R', iris_detector=self.parent.iris_detector)

                    if not (left_eye.detected and right_eye.detected):
                        # Eyes are too close to the edges of the image
                        detect_face = False

                    # save previous rvec and tvec
                    face_rvec = face.rotation_vector
                    face_tvec = face.translation_vector

                    # draw results
                    face.draw_marker(frame)
                    face.draw_eyelids_landmarks(frame)
                    if left_eye.detected:
                        left_eye.draw_marker(frame)
                    if right_eye.detected:
                        right_eye.draw_marker(frame)

                im = cv2.resize(frame, (int(frame.shape[1]*self.parent.camera_view_scale),int(frame.shape[0]*self.parent.camera_view_scale)))
                bmp = wx.Bitmap.FromBuffer(im.shape[1], im.shape[0], cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                self.camera_view.SetBitmap(bmp)

                if detect_face and (not left_eye.blink) and (not right_eye.blink):
                    calibration_data.append((calibration_sample_point, face, left_eye, right_eye))
        
        # run fitting
        fitting_param = LM_calibration(calibration_data, self.parent.screen)
        results = calc_calibration_results(calibration_data, self.parent.screen, fitting_param)

        # if dialog has already been closed, return
        if not self.running:
            return

        # draw results
        canvas = np.zeros((self.parent.camera_view_height,self.parent.camera_view_width,3),dtype=np.uint8)
        data = np.array([float(x) for x in results[3].split(',')],dtype=np.int64).reshape(-1,6)

        #outliers
        #error_l = np.linalg.norm(data[:,2:4]-data[:,:2], axis=1)
        #error_r = np.linalg.norm(data[:,4:6]-data[:,:2], axis=1)
        #out_l = np.where(error_l > 3*np.std(error_l))[0]
        #out_r = np.where(error_r > 3*np.std(error_r))[0]

        txmin, txmax = (1.25*min(data[:,0]),1.25*max(data[:,0]))
        tymin, tymax = (1.25*min(data[:,1]),1.25*max(data[:,1]))
        txcenter, tycenter = ((txmax+txmin)/2, (tymax+tymin)/2)
        scale = (tymax-tymin)/self.parent.camera_view_height
        if (txmax-txmin)*scale > self.parent.camera_view_width:
            scale = (txmax-txmin)/self.parent.camera_view_width
        for i in range(3):
            data[:,2*i]   = (data[:,2*i]  -txcenter)/scale + self.parent.camera_view_width/2
            data[:,2*i+1] = (data[:,2*i+1]-tycenter)/scale + self.parent.camera_view_height/2
        for idx, d in enumerate(data):
            #outliers
            #wl = 3 if idx in out_l else 1
            #wr = 3 if idx in out_r else 1
            wl = wr = 1
            cv2.line(canvas,d[0:2],d[2:4],(0,255,0),wl) # left eye: green
            cv2.line(canvas,d[0:2],d[4:6],(255,0,0),wr) # right eye: red
        bmp = wx.Bitmap.FromBuffer(canvas.shape[1], canvas.shape[0], canvas)#cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        self.camera_view.SetBitmap(bmp)

        self.fitting_param = fitting_param
        self.results = results

        results_msg = "Precision: {}\nAccuracy: {}\nMax error: {}".format(*results[:3])
        self.results_text.SetValue(results_msg)

        self.button_save.Enable(True)
        #wx.CallAfter(self.Close)


    def save(self, event):
        filename = dlgAsksaveasfilename(self, filetypes='Fitting results (*.npz)|*.npz')
        if filename != '':
            if self.parent.area_of_interest is not None:
                aoi = (self.parent.area_of_interest.left(), self.parent.area_of_interest.top(),
                        self.parent.area_of_interest.right(), self.parent.area_of_interest.bottom())
            else:
                aoi = (np.nan, np.nan, np.nan, np.nan)
            np.savez(filename,
                fitting_param=self.fitting_param, 
                precision=self.results[0],
                accuracy=self.results[1],
                max_error=self.results[2],
                results_detail=self.results[3],
                area_of_interest = aoi)
        


class dlgEditCalPoint(wx.Dialog):
    def __init__(self, parent, data, id=wx.ID_ANY):
        super(dlgEditCalPoint, self).__init__(parent=parent, id=id, title='Edit Calibration Point')
        self.mainWindow = parent
        self.update  = False

        if data is None:
            val_from = ''
            val_until = ''
            val_point = ''
        else:
            val_from = data[0]
            val_until = data[1]
            val_point = data[2]

        editPanel = wx.Panel(self, id=wx.ID_ANY)
        self.tcFrom = wx.TextCtrl(editPanel, wx.ID_ANY, val_from, size=(80,-1))
        self.tcUntil = wx.TextCtrl(editPanel, wx.ID_ANY, val_until, size=(80,-1))
        self.tcPoint = wx.TextCtrl(editPanel, wx.ID_ANY, val_point, size=(200,-1))

        box = wx.FlexGridSizer(2, 3, 0, 0)
        box.Add(wx.StaticText(editPanel, wx.ID_ANY, 'From'), flag=wx.LEFT|wx.RIGHT, border=5)
        box.Add(wx.StaticText(editPanel, wx.ID_ANY, 'Until'), flag=wx.LEFT|wx.RIGHT, border=5)
        box.Add(wx.StaticText(editPanel, wx.ID_ANY, 'Point'), flag=wx.LEFT|wx.RIGHT, border=5)
        box.Add(self.tcFrom, flag=wx.LEFT|wx.RIGHT, border=5)
        box.Add(self.tcUntil, flag=wx.LEFT|wx.RIGHT, border=5)
        box.Add(self.tcPoint, flag=wx.LEFT|wx.RIGHT, border=5)
        editPanel.SetSizer(box)
        editPanel.SetAutoLayout(True)

        buttonPanel = wx.Panel(self, id=wx.ID_ANY)
        okButton = wx.Button(buttonPanel, wx.ID_OK, 'OK')
        cancelButton = wx.Button(buttonPanel, wx.ID_CANCEL, 'Cancel')
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(okButton)
        hbox.Add(cancelButton)
        buttonPanel.SetSizer(hbox)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(editPanel, flag=wx.EXPAND|wx.ALL, border=5)
        vbox.Add(buttonPanel, flag=wx.ALIGN_RIGHT)
        self.SetSizerAndFit(vbox)

    def set_current_from(self,event):
        idx = self.mainWindow.slider.GetValue()
        self.tcFrom.SetValue(idx)

    def set_current_until(self,event):
        idx = self.mainWindow.slider.GetValue()
        self.tcUntil.SetValue(idx)

class camera_view(wx.StaticBitmap):
    def __init__(self, *args, **kwargs):
        super(camera_view, self).__init__(*args, **kwargs)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        try:
            image = self.GetBitmap()
            if not image:
                return
            dc = wx.AutoBufferedPaintDC(self)
            dc.Clear()
            dc.DrawBitmap(image, 0, 0, True)
        except:
            pass

class offline_calibration_app(wx.Frame):

    NewImageEvent, EVT_NEWIMAGE = wx.lib.newevent.NewEvent()

    def __init__(self, config, movie, calinfo, iris_detector=None):
        super(offline_calibration_app, self).__init__(parent=None,id=wx.ID_ANY,title="Offline calibration")

        self.calinfofile = calinfo
        self.moviefile = movie
        self.config = config
        self.cap = None

        self.orig_img = None

        self.camera_view_width = 600
        self.camera_view_height = 340
        self.camera_view_scale = 1.0

        self.camera_matrix = config.camera_matrix
        self.downscaling_factor = config.downscaling_factor

        self.screen = screen()
        self.screen.set_parameters(
            config.screen_width/conf.screen_h_res, 
            config.screen_rot,
            config.screen_offset)

        self.face_model = config.face_model
        self.eye_params = config.eye_params

        if iris_detector is None:
            raise RuntimeError('Offilne_Calibration: iris_detector must be specified.')
        self.iris_detector = iris_detector

        self.area_of_interest =  None
        self.updating_aoi = False
        self.aoi_p0 = None

        ### Menu ###
        self.menu_bar = wx.MenuBar()
        self.menu_file = wx.Menu()
        self.menu_movie = wx.Menu()
        self.menu_cal = wx.Menu()
        self.menu_bar.Append(self.menu_file,'File')
        self.menu_bar.Append(self.menu_movie,'Movie')
        self.menu_bar.Append(self.menu_cal,'Calibration')

        self.menu_file.Append(ID_OPEN_CAMERA_CONFIG, 'Open Camera Config')
        self.menu_file.Append(ID_OPEN_FACE_CONFIG, 'Open Face Config')
        self.menu_file.Append(wx.ID_CLOSE, 'Exit')
        self.Bind(wx.EVT_MENU, self.open_camera_config, id=ID_OPEN_CAMERA_CONFIG)
        self.Bind(wx.EVT_MENU, self.open_face_config, id=ID_OPEN_FACE_CONFIG)
        self.Bind(wx.EVT_MENU, self.exit, id=wx.ID_CLOSE)

        self.menu_movie.Append(wx.ID_OPEN, 'Open Movie')
        self.Bind(wx.EVT_MENU, self.open_movie, id=wx.ID_OPEN)

        self.menu_cal.Append(ID_OPEN_CALINFO, 'Open Calibration Info')
        self.menu_cal.Append(ID_SAVE_CALINFO, 'Save Calibration Info')
        self.menu_cal.AppendSeparator()
        self.menu_cal.Append(ID_RUN_CAL, 'Run Calibration')
        self.Bind(wx.EVT_MENU, self.open_calinfo, id=ID_OPEN_CALINFO)
        self.Bind(wx.EVT_MENU, self.save_calinfo, id=ID_SAVE_CALINFO)
        self.Bind(wx.EVT_MENU, self.run_calibration, id=ID_RUN_CAL)
 
        self.SetMenuBar(self.menu_bar)

        ### Main Panel ###
        mainpanel = wx.Panel(self, wx.ID_ANY)
        self.mediapanel = wx.Panel(mainpanel,wx.ID_ANY)
        self.camera_view = camera_view(self.mediapanel, wx.ID_ANY, wx.Bitmap(self.camera_view_width,self.camera_view_height))
        mediasizer = wx.BoxSizer()
        mediasizer.Add(self.camera_view)
        self.mediapanel.SetSizer(mediasizer)
        self.slider = wx.Slider(mainpanel, wx.ID_ANY, 0, 0, 1000,style=wx.SL_HORIZONTAL|wx.SL_AUTOTICKS|wx.SL_MIN_MAX_LABELS)
        self.slider_label = wx.StaticText(mainpanel, wx.ID_ANY, 'Frame    0 / 0:00:00.000000')
        self.buttonpanel = wx.Panel(mainpanel,wx.ID_ANY)

        self.camera_view.Bind(wx.EVT_LEFT_DOWN, self.camera_view_leftdown)
        self.camera_view.Bind(wx.EVT_LEFT_UP, self.camera_view_leftup)
        self.camera_view.Bind(wx.EVT_LEAVE_WINDOW, self.camera_view_leave)
        self.camera_view.Bind(wx.EVT_LEFT_DCLICK, self.camera_view_leftdclick)
        self.camera_view.Bind(wx.EVT_MOTION, self.camera_view_motion)

        self.playing = False
        self.movie_frames = 0
        self.current_frame = 0
        self.seek_frame = -1

        # Buttons
        self.button_play = wx.Button(self.buttonpanel,wx.ID_ANY,"Play")
        self.button_play.Bind(wx.EVT_BUTTON,self.PlayMedia)

        self.button_stepback1f = wx.Button(self.buttonpanel,wx.ID_ANY,"-1")
        self.button_stepback1f.Bind(wx.EVT_BUTTON,self.stepback_1f)
        self.button_step1f = wx.Button(self.buttonpanel,wx.ID_ANY,"+1")
        self.button_step1f.Bind(wx.EVT_BUTTON,self.step_1f)
        self.cb_detect_face = wx.CheckBox(self.buttonpanel,wx.ID_ANY,"Detect face")

        self.buttonpanel.Enable(False)
        self.slider.Enable(False)

        # Button layout
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.button_play)
        sizer.Add(self.button_stepback1f)
        sizer.Add(self.button_step1f)
        sizer.Add(self.cb_detect_face)
        self.buttonpanel.SetSizer(sizer)

        # Widget layout
        mainsizer = wx.BoxSizer(wx.VERTICAL)
        mainsizer.Add(self.mediapanel,4,wx.EXPAND)
        mainsizer.Add(self.slider,0,wx.EXPAND|wx.ALL, 5)
        mainsizer.Add(self.slider_label,0,wx.EXPAND|wx.LEFT, 20)
        mainsizer.Add(self.buttonpanel,0,wx.EXPAND|wx.ALL, 10)
        mainpanel.SetSizer(mainsizer)

        self.slider.Bind(wx.EVT_SLIDER, self.on_seek)

        ### Sub Panel ###
        subpanel = wx.Panel(self, wx.ID_ANY)
        self.calpoint_listbox = wx.ListCtrl(subpanel, wx.ID_ANY, style=wx.LC_REPORT)
        self.calpoint_listbox.InsertColumn(0, 'From')
        self.calpoint_listbox.InsertColumn(1, 'Until')
        self.calpoint_listbox.InsertColumn(2, 'Calibration Point')

        menus = (
                 ('Set (from)', self.set_from),
                 ('Set (until)', self.set_until),
                 ('Jump (from)', self.jump_from),
                 ('Jump (until)', self.jump_until),
                )
        self.popup_calpoint_listbox = wx.Menu()
        for menu in menus:
            item = self.popup_calpoint_listbox.Append(wx.ID_ANY, menu[0])
            self.Bind(wx.EVT_MENU, menu[1], item)
        self.calpoint_listbox.Bind(wx.EVT_CONTEXT_MENU, self.show_popup_calpoint_listbox)

        subButtonpanel = wx.Panel(subpanel, wx.ID_ANY)
        insertButton = wx.Button(subButtonpanel, wx.ID_ANY, 'Insert')
        deleteButton = wx.Button(subButtonpanel, wx.ID_ANY, 'Delete')
        editButton = wx.Button(subButtonpanel, wx.ID_ANY, 'Edit')
        insertButton.Bind(wx.EVT_BUTTON, self.insert_calpoint)
        deleteButton.Bind(wx.EVT_BUTTON, self.delete_calpoint)
        editButton.Bind(wx.EVT_BUTTON, self.edit_calpoint)
        subButtonsizer = wx.BoxSizer(wx.HORIZONTAL)
        subButtonsizer.Add(insertButton)
        subButtonsizer.Add(deleteButton)
        subButtonsizer.Add(editButton)
        subButtonpanel.SetSizer(subButtonsizer)

        subsizer = wx.BoxSizer(wx.VERTICAL)
        subsizer.Add(subButtonpanel, proportion=0)
        subsizer.Add(self.calpoint_listbox, flag=wx.EXPAND, proportion=1)
        subpanel.SetSizer(subsizer)

        ### Base Panel ###
        basesizer = wx.BoxSizer(wx.HORIZONTAL)
        basesizer.Add(mainpanel,2,wx.EXPAND)
        basesizer.Add(subpanel,1,wx.EXPAND)
        self.SetSizerAndFit(basesizer)

        ### load movie and calinfo if necessary ###
        if self.moviefile is not None:
            self.open_movie(None)
        if self.calinfofile is not None:
            self.open_calinfo(None)

        self.Show(True)

        self.Bind(offline_calibration_app.EVT_NEWIMAGE, self.new_image)
        self.queue = queue.Queue(1)
        self.running = True
        self.thread = threading.Thread(target=self.capture_image)
        self.thread.daemon = True
        self.thread.start()

    def camera_view_leftdown(self,event):
        if not self.updating_aoi:
            self.aoi_p0 = event.GetPosition()
            self.updating_aoi = True

    def camera_view_leftup(self,event):
        if self.updating_aoi:
            p1 = event.GetPosition()
            left = int(min(self.aoi_p0[0],p1[0]) / self.camera_view_scale)
            right = int(max(self.aoi_p0[0],p1[0]) / self.camera_view_scale)
            top = int(min(self.aoi_p0[1],p1[1]) / self.camera_view_scale)
            bottom = int(max(self.aoi_p0[1],p1[1]) / self.camera_view_scale)
            if (right-left) * (bottom-top) != 0:
                self.area_of_interest =  dlib.rectangle(left, top, right, bottom)
            self.updating_aoi = False

            self.new_image(None)

    def camera_view_leave(self,event):
        if self.updating_aoi:
            p1 = event.GetPosition()
            left = int(min(self.aoi_p0[0],p1[0]) / self.camera_view_scale)
            right = int(max(self.aoi_p0[0],p1[0]) / self.camera_view_scale)
            top = int(min(self.aoi_p0[1],p1[1]) / self.camera_view_scale)
            bottom = int(max(self.aoi_p0[1],p1[1]) / self.camera_view_scale)
            if (right-left) * (bottom-top) != 0:
                self.area_of_interest =  dlib.rectangle(left, top, right, bottom)
            self.updating_aoi = False

            self.new_image(None)

    def camera_view_leftdclick(self,event):
        if self.updating_aoi:
            self.updating_aoi = False
        self.aoi_p0 = None
        self.area_of_interest = None

        self.new_image(None)
    
    def camera_view_motion(self,event):
        self.new_image(None)

    def on_seek(self,event):
        self.seek_frame = self.slider.GetValue()

    def PlayMedia(self,event):
        if self.playing:
            self.playing = False
            self.button_play.SetLabel("Play")
        else:
            self.playing = True
            self.button_play.SetLabel("Pause")

    def stepback_1f(self,event):
        if self.playing:
            self.playing = False
            self.button_play.SetLabel("Play")
        # because forward 1 frame by read(), we need to subtract 2
        self.seek_frame = max(self.slider.GetValue()-2,0)

    def step_1f(self,event):
        if self.playing:
            self.playing = False
            self.button_play.SetLabel("Play")
        # because forward 1 frame by read(), we don't need to add 1
        self.seek_frame = min(self.slider.GetValue(),self.movie_frames-1)

    def show_popup_calpoint_listbox(self,event):
        pos = event.GetPosition()
        pos = self.calpoint_listbox.ScreenToClient(pos)
        self.calpoint_listbox.PopupMenu(self.popup_calpoint_listbox, pos)
        event.Skip()

    def insert_calpoint(self,event):
        n = self.calpoint_listbox.GetItemCount()
        if n == 0:
            insert_idx = 0
        else:
            selected = []
            for idx in range(self.calpoint_listbox.GetItemCount()):
                if self.calpoint_listbox.GetItemState(idx, wx.LIST_STATE_SELECTED) != 0:
                    selected.append(idx)
            if len(selected) == 0:
                insert_idx = n
            elif len(selected) > 1:
                dlgShowerror(self, 'Error', 'Select a single entry.')
                return
            else:
                insert_idx = selected[0]

        dlg = dlgEditCalPoint(self, None)
        if dlg.ShowModal() == wx.ID_OK:
            self.calpoint_listbox.InsertItem(insert_idx, dlg.tcFrom.GetValue())
            self.calpoint_listbox.SetItem(insert_idx, 1, dlg.tcUntil.GetValue())
            self.calpoint_listbox.SetItem(insert_idx, 2, dlg.tcPoint.GetValue())

    def delete_calpoint(self,event):
        selected = []
        for idx in range(self.calpoint_listbox.GetItemCount()):
            if self.calpoint_listbox.GetItemState(idx, wx.LIST_STATE_SELECTED) != 0:
                selected.append(idx)
        
        if len(selected) == 0:
            return
        
        selected.reverse()

        for idx in selected:
            self.calpoint_listbox.DeleteItem(idx)

    def edit_calpoint(self,event):
        selected = []
        for idx in range(self.calpoint_listbox.GetItemCount()):
            if self.calpoint_listbox.GetItemState(idx, wx.LIST_STATE_SELECTED) != 0:
                selected.append(idx)
        if len(selected) != 1:
            dlgShowerror(self, 'Error', 'Select a single entry.')
            return

        data = [self.calpoint_listbox.GetItem(selected[0],i).GetText() for i in range(3)]
        dlg = dlgEditCalPoint(self, data)
        if dlg.ShowModal() == wx.ID_OK:
            self.calpoint_listbox.SetItem(selected[0], 0, dlg.tcUntil.GetValue())
            self.calpoint_listbox.SetItem(selected[0], 1, dlg.tcUntil.GetValue())
            self.calpoint_listbox.SetItem(selected[0], 2, dlg.tcPoint.GetValue())

    def open_movie(self, event):
        if event is not None:
            filename = dlgAskopenfilename(self)
            if filename == '':
                return
        else:
            filename = self.moviefile
        
        self.cap = cv2.VideoCapture(filename)
        if self.cap.isOpened():
            self.moviefile = filename
            self.movie_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.movie_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.slider.SetRange(0, self.movie_frames)
            self.buttonpanel.Enable(True)
            self.slider.Enable(True)

            img_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            img_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if img_width != self.config.camera_resolution_h or img_height != self.config.camera_resolution_v:
                dlgShowinfo(self, 'Error',
                    'Resolution defined in the cameara parameter ({:.0f},{:.0f}) does not match with that of this movie ({:.0f},{:.0f})'.format(
                        self.config.camera_resolution_h, self.config.camera_resolution_v, img_width, img_height))
            self.camera_view_scale = self.camera_view_height/img_height
            if img_width*self.camera_view_scale > self.camera_view_width:
                self.camera_view_scale = self.camera_view_width/img_width

            #read and show the first frame
            ret, im = self.cap.read()
            if ret:
                self.orig_img = im
                im = cv2.resize(im, (int(im.shape[1]*self.camera_view_scale),int(im.shape[0]*self.camera_view_scale)))
                bmp = wx.Bitmap.FromBuffer(im.shape[1], im.shape[0], cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                self.camera_view.SetBitmap(bmp)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        else:
            self.cap = None

    def open_camera_config(self,event):
        filename = dlgAskopenfilename(self, filetypes='Camera config (*.cfg)|*.cfg')
        if filename == '':
            return
        
        tmpconfig = configuration()
        try:
            tmpconfig.load_camera_param(filename)
        except:
            dlgShowerror(self, 'Error', 'Cannot open {} as a camera parameter file'.format(filename))
            return
        
        update_cameraview = False
        if self.cap is not None and self.cap.isOpened():
            w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if w != tmpconfig.camera_resolution_h or h != tmpconfig.camera_resolution_v:
                if dlgAskyesno(self, 'Info',
                    'Resolution defined in this cameara parameter file ({:.0f},{:.0f}) does not match with that of the current movie ({:.0f},{:.0f})\n Do you want to close movie?'.format(
                        tmpconfig.camera_resolution_h, tmpconfig.camera_resolution_v, w, h)):
                    self.cap.release()
                    self.cap = None
                    self.orig_img = None
                    self.buttonpanel.Enable(False)
                    self.slider.Enable(False)
                    update_cameraview = True
                else:
                    dlgShowinfo(self,'Info','Camera configuration was not updated.')
                    return
 
        self.config.load_camera_param(filename)

        self.camera_matrix = self.config.camera_matrix
        self.downscaling_factor = self.config.downscaling_factor

        self.screen = screen()
        self.screen.set_parameters(
            self.config.screen_width/conf.screen_h_res, 
            self.config.screen_rot,
            self.config.screen_offset)

        if update_cameraview:
            im = np.zeros((self.camera_view_height, self.camera_view_width,3),dtype=np.uint8)
            bmp = wx.Bitmap.FromBuffer(im.shape[1], im.shape[0], im)
            self.camera_view.SetBitmap(bmp)


    def open_face_config(self,event):
        filename = dlgAskopenfilename(self, filetypes='Face model (*.cfg)|*.cfg')
        if filename == '':
            return
        
        self.config.load_face_model(filename)
        self.face_model = self.config.face_model
        self.eye_params = self.config.eye_params

    def open_calinfo(self, event):
        if event is not None:
            filename = dlgAskopenfilename(self, filetypes='Calibraion Info (*.csv)|*.csv')
            if filename == '':
                return
        else:
            filename = self.calinfofile

        self.calpoint_listbox.DeleteAllItems()
        with open(filename, 'r', encoding='utf-8', newline='') as fp:
            csvreader = csv.reader(fp)

            idx = 0
            for data in csvreader:
                try:
                    int(data[0])
                except:
                    continue

                self.calpoint_listbox.InsertItem(idx, data[0])
                self.calpoint_listbox.SetItem(idx, 1, data[1])
                self.calpoint_listbox.SetItem(idx, 2, data[2])
                idx += 1
        
        self.calinfofile = filename

    def save_calinfo(self,event):
        if self.calinfofile is None:
            filename = dlgAsksaveasfilename(self, filetypes='Calibraion Info (*.csv)|*.csv')
        else:
            dirname, fname = os.path.split(self.calinfofile)
            filename = dlgAsksaveasfilename(self, filetypes='Calibraion Info (*.csv)|*.csv', initialdir=dirname, initialfile=fname)
        if filename == '':
            return
        
        with open(filename, 'w', encoding='utf-8') as fp:
            fp.write('From,Until,Point\n')
            for idx in range(self.calpoint_listbox.GetItemCount()):
                data = [self.calpoint_listbox.GetItem(idx,col).GetText() for col in range(3)]
                fp.write('{},{},"{}"\n'.format(*data))

    def exit(self,event):
        self.Destroy()

    def jump_base(self,timing):
        if self.cap is None:
            dlgShowerror(self, 'Error', 'Open movie file af first.')
            return

        if timing == 'from':
            col = 0
        elif timing == 'until':
            col = 1
        else:
            dlgShowerror(self, 'Error', 'Timing must be "from" or "until".')
            return

        selected = []
        for idx in range(self.calpoint_listbox.GetItemCount()):
            if self.calpoint_listbox.GetItemState(idx, wx.LIST_STATE_SELECTED) != 0:
                selected.append(idx)
        if len(selected) != 1:
            dlgShowerror(self, 'Error', 'Select a single entry.')
            return
        
        idxstr = self.calpoint_listbox.GetItem(selected[0],col).GetText()
        try:
            idx = int(idxstr)
        except:
            dlgShowerror(self, 'Error', 'Invalid index ({})'.format(idxstr))
            return
        if not (0 < idx < self.movie_frames):
            dlgShowerror(self, 'Error', 'Invalid index ({})'.format(idx))
            return
        
        if self.playing:
            self.playing = False
            self.button_play.SetLabel("Play")
        self.seek_frame = idx

    def jump_from(self,event):
        self.jump_base('from')

    def jump_until(self,event):
        self.jump_base('until')

    def set_base(self,timing):
        if self.cap is None:
            dlgShowerror(self, 'Error', 'Open movie file af first.')
            return

        if timing == 'from':
            col = 0
        elif timing == 'until':
            col = 1
        else:
            dlgShowerror(self, 'Error', 'Timing must be "from" or "until".')
            return

        selected = []
        for idx in range(self.calpoint_listbox.GetItemCount()):
            if self.calpoint_listbox.GetItemState(idx, wx.LIST_STATE_SELECTED) != 0:
                selected.append(idx)
        if len(selected) != 1:
            dlgShowerror(self, 'Error', 'Select a single entry.')
            return
        
        if self.playing:
            self.playing = False
            self.button_play.SetLabel("Play")
        self.calpoint_listbox.SetItem(selected[0], col, str(self.slider.GetValue()))

    def set_from(self,event):
        self.set_base('from')
        
    def set_until(self,event):
        self.set_base('until')
    
    def run_calibration(self,event):
        if self.cap is None:
            dlgShowerror(self, 'Error', 'Open movie file af first.')
            return
        if self.calpoint_listbox.GetItemCount() < 2:
            dlgShowerror(self, 'Error', 'At lease two calibration points are required.')
            return

        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if w != self.config.camera_resolution_h or h != self.config.camera_resolution_v:
            dlgShowerror(self, 'Error',
                'Resolution defined in the cameara parameter ({:.0f},{:.0f}) does not match with that of the current movie ({:.0f},{:.0f})'.format(
                    self.config.camera_resolution_h, self.config.camera_resolution_v, w, h))
            return

        if self.playing:
            self.playing = False
            self.button_play.SetLabel("Play")
        
        self.buttonpanel.Enable(False)
        for id in menu_items_all:
            self.menu_bar.Enable(id, False)

        dlg = calibrationDialog(self)
        dlg.ShowModal()
        if dlg.thread.is_alive():
            dlg.running = False

        self.buttonpanel.Enable(True)
        for id in menu_items_all:
            self.menu_bar.Enable(id, True)
    
    def capture_image(self):
        while self.running:
            if self.cap is None:
                time.sleep(0.1) 
                continue

            if (not self.playing) and (self.seek_frame < 0):
                time.sleep(0.1) 
                continue

            if self.seek_frame >= 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                self.current_frame = self.seek_frame
                self.seek_frame = -1

            ret, im = self.cap.read()
            if not ret:
                self.palying = False
                continue
            
            self.current_frame += 1
            self.slider.SetValue(self.current_frame)
            tdelta = datetime.timedelta(seconds=self.current_frame/self.movie_fps)
            self.slider_label.SetLabelText('Frame {} / {}'.format(self.current_frame, tdelta))

            time.sleep(1/self.movie_fps/2)

            if not self.queue.full():
                self.queue.put(im, False)
                try:
                    wx.PostEvent(self, offline_calibration_app.NewImageEvent())
                except:
                    break
            
    def new_image(self, evt):
        if not self.queue.empty():
            #update image
            self.orig_img = self.queue.get(False)

        if self.orig_img is None:
            return

        im = np.copy(self.orig_img)

        if self.updating_aoi:
            p1 = self.camera_view.ScreenToClient(wx.GetMousePosition())
            left = int(min(self.aoi_p0[0],p1[0]) / self.camera_view_scale)
            right = int(max(self.aoi_p0[0],p1[0]) / self.camera_view_scale)
            top = int(min(self.aoi_p0[1],p1[1]) / self.camera_view_scale)
            bottom = int(max(self.aoi_p0[1],p1[1]) / self.camera_view_scale)
            cv2.rectangle(im, (left, top),(right,bottom),(0,255,255), thickness=1)
        elif self.area_of_interest is not None:
            cv2.rectangle(im, (self.area_of_interest.left(),self.area_of_interest.top()),
                                (self.area_of_interest.right(),self.area_of_interest.bottom()),
                                (0,255,255), thickness=2)

        if self.cb_detect_face.GetValue():
            img = cv2.cvtColor(self.orig_img, cv2.COLOR_BGR2GRAY)
            if self.downscaling_factor == 1.0: # original size
                dets, _ = get_face_boxes(img, engine='dlib_hog')
            else: # downscale camera image
                dets, _ = get_face_boxes(cv2.resize(img, None, fx=self.downscaling_factor, fy=self.downscaling_factor), engine='dlib_hog') # detections, scores, weight_indices
                inv = 1.0/self.downscaling_factor
                # recover rectangle size
                for i in range(len(dets)):
                    dets[i] = dlib.rectangle(int(dets[i].left()*inv), int(dets[i].top()*inv),
                                            int(dets[i].right()*inv), int(dets[i].bottom()*inv))

            for fidx in range(len(dets)):
                if self.area_of_interest is None or self.area_of_interest.contains(dets[fidx]):
                    cv2.rectangle(im, (dets[fidx].left(),dets[fidx].top()), (dets[fidx].right(),dets[fidx].bottom()),(0,255,0),thickness=3)
                else:
                    cv2.rectangle(im, (dets[fidx].left(),dets[fidx].top()), (dets[fidx].right(),dets[fidx].bottom()),(0,0,255),thickness=3)

        im = cv2.resize(im, (int(im.shape[1]*self.camera_view_scale),int(im.shape[0]*self.camera_view_scale)))

        bmp = wx.Bitmap.FromBuffer(im.shape[1], im.shape[0], cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        self.camera_view.SetBitmap(bmp)


if __name__ == '__main__':

    camera_param_file = None
    face_model_file = None

    conf = configuration()

    conf = configuration()
    arg_parser = argparse.ArgumentParser(description='rsgt offline tracker')
    arg_parser.add_argument('--camera_param', type=str, help='camera parameters file')
    arg_parser.add_argument('--face_model', type=str, help='face model file')
    arg_parser.add_argument('--iris_detector', type=str, help='iris detector (ert, peak, enet or path to detector)')
    arg_parser.add_argument('--movie', type=str, help='movie file (required for batch execution)')
    arg_parser.add_argument('--cal_info', type=str, help='calibration information file')
    args = arg_parser.parse_args()

    homedir = Path.home()/'.rsgt'
    if not homedir.exists():
        Path.mkdir(homedir)
        print('info: {} is created.'.format(homedir))
    
    defaultconfig = homedir/'rsgt.cfg'
    if not defaultconfig.exists():
        shutil.copy(module_dir/'resources'/'rsgt.cfg',defaultconfig)
        print('info: default config file is created in {}.'.format(homedir))
    conf.load_application_param(defaultconfig)

    if args.camera_param is None:
        #read default file
        cfgfile = homedir/'camera_param.cfg'
        if not cfgfile.exists():
            shutil.copy(module_dir/'resources'/'camera_param.cfg',cfgfile)
            print('info: default camera parameter file is created in {}.'.format(homedir))
        conf.load_camera_param(str(cfgfile))
        camera_param_file = str(cfgfile)
    else:
        conf.load_camera_param(args.camera_param)

    if args.face_model is None:
        cfgfile = homedir/'face_model.cfg'
        if not cfgfile.exists():
            shutil.copy(module_dir/'resources'/'face_model.cfg',cfgfile)
            print('info: default face model file is created in {}.'.format(homedir))
        conf.load_face_model(str(cfgfile))
        face_model_file = str(cfgfile)
    else:
        conf.load_face_model(face_model_file)

    if args.iris_detector is None:
        iris_detector = get_iris_detector(conf.iris_detector)
    else:
        iris_detector = get_iris_detector(args.iris_detector)
    if iris_detector is None:
        sys.exit()

    app = wx.App(False)
    offline_calibration_app(conf, movie=args.movie, calinfo=args.cal_info, iris_detector=iris_detector)
    app.MainLoop()