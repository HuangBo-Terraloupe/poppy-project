# -*- coding: utf-8 -*-
"""
Created on Fri May 19 00:27:05 2017

@author: Fares
"""

import numpy as np
import time
import pyglet
import threading
import Queue as queue
import datetime
from math import radians,cos,sin
import locale 
#from pyglet.gl import *

OPENGL_VERSION = "2.0"

class Canvas:
    def __init__(self):
        pass
    
    def flush_events(self):
        pass
    
def makeCircle(X,Y,r=6,numPoints=20):
    verts = []
    for i in range(numPoints):
        angle = radians(float(i)/numPoints * 360.0)
        x = r*cos(angle) + X
        y = r*sin(angle) + Y
        #verts.append([x,y])
        verts.append(x)
        verts.append(y)
    #print numPoints, verts
    
    if ((len(verts) <= 0) or (len(verts) != numPoints*2)):
        print "Something went wrong in makeCircle at X: ",X,", Y: ",Y, ", len:", len(verts), ", num:",numPoints
                                                                                    
    return len(verts)//2, ('v2f', tuple(verts))
    

                                                                                    

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
   
class Window:
    def __init__(self,x_plots, y_plots):
        self.canvas = Canvas()
        self.display = pyglet.canvas.get_display()
        self.screen = self.display.get_default_screen()
        
        #use opengl 2.1 as the use of higher versions may lead to bugs
        self.gl_template = pyglet.gl.Config(major_version=int(OPENGL_VERSION[0]),
                                             minor_version=int(OPENGL_VERSION[2]))
        try:
            self.config =self.screen.get_best_config(self.gl_template)
        except pyglet.window.NoSuchConfigException:
            self.gl_template = pyglet.gl.Config()
            self.config = self.screen.get_best_config(self.gl_template)
        
        print "Using OpenGL ", self.gl_template.major_version,".", self.gl_template.minor_version
        
        self.subs = []
        #x_plots = 1
        #y_plots = 0
        x = 20
        y = 40  

        DEFAULT_SUB_WIDTH = 640
        DEFAULT_SUB_HEIGHT = 480
        
        sub_width = DEFAULT_SUB_WIDTH
        sub_height = DEFAULT_SUB_HEIGHT
        
        if (x + (sub_width*x_plots)) >  self.screen.width:
            sub_width = int(float((self.screen.width - x ))//float(x_plots))
        
        if (x + (sub_height*y_plots)) >  self.screen.height:
            sub_height = int(float((self.screen.height - y ))//float(y_plots))
        
        for i in range(x_plots * y_plots):
            
            sub = SubWindow(window_ID = i, w = sub_width, h=sub_height, cfg=self.config)
            if (x_plots * y_plots) > 12:
                sub.enable_bot_msg = False
                sub.enable_fps_msg = False
                sub.enable_top_msg = False
            #sub.set_size(800,230)
            #if (x + (sub.width*x_plots)) <  self.screen.width:
                #sub.set_size(800,230)
                #pass
                #sub.set_size(int(float((self.screen.width - x ))//float(x_plots)), sub.height)
                
            #print "> ", i, ": " ,self.screen.width, " ", x, " ", x_plots, " ", sub.width
            
            #set correct location for subwindows
            #print (i/x_plots)," ", (i%x_plots)
            s_y = (i/x_plots)
            s_x = (i%x_plots)
            if (x_plots * y_plots) == 1:
                x,y = sub.get_location()
            sub.set_location(x + sub.width*s_x,y + sub.height*s_y)
            print "Created subWindow ", i, " on display ", self.display.name
            self.subs.append(sub)
            
        self.ax = np.asarray([np.asarray(self.subs)])     
        
        #exit()
        #print self.ax 
        
    def update(self,dt):
        for sub in self.subs:
            sub.update()
            
    def global_print(self,message):
        for sub in self.subs:
            sub.set_bot_msg(message)
    
    def schedule_updates(self):
        for sub in self.subs:
            pyglet.clock.schedule(sub.update)
            
    def tick(self):
        time.sleep(0.1)
        
MAX_QUEUE_SIZE = 60
class SubWindow(pyglet.window.Window):
    def __init__(self, window_ID = -1, w = 640, h = 480, cfg=None):
        self.window_ID = window_ID
        
        while (True):
            try:
                super(SubWindow,self).__init__(vsync = False, resizable = True,
                                             caption = "Window " + window_ID.__str__(),
                                             width = w, height = h, config=cfg)
                break
            except pyglet.gl.ContextException:
                print "Cannot use OpenGL config ", OPENGL_VERSION, ". Reverting back to original version"
                cfg = None
            
        self.EMPTY_MSG = ""
        
        self.q = queue.Queue(MAX_QUEUE_SIZE)
        self.q_scene = queue.Queue(MAX_QUEUE_SIZE)
        self.q_bot_msg = queue.Queue(MAX_QUEUE_SIZE)
        
        self.last_scene = None
        self.last_bot_msg = self.EMPTY_MSG #"No msg"
        
        self.min_val = +float("inf")
        self.max_val = -float("inf")
        
        self.xlim = []
        self.ylim = []
        
        self.fps = pyglet.clock.ClockDisplay()

        self.label = pyglet.text.Label('Ready', font_name='Times New Roman', 
            font_size=16, x=self.width//2, y=abs(self.height/(1.1)),
            anchor_x='center', anchor_y='center')
        
        self.bot_label = pyglet.text.Label(self.last_bot_msg, font_name='Times New Roman', 
            font_size=16, x=self.width//2, y=abs(self.height/8),
            anchor_x='center', anchor_y='center', multiline=True, width=self.width)
        
        self.enable_bot_msg = True
        self.enable_top_msg = True
        self.enable_fps_msg = True
        #print "Sub"
    
    def super_clear(self):
        super(SubWindow,self).clear()
        
    def clear(self):
        scene = []
        while self.q.not_empty:
            try:
                data = self.q.get(False)
                scene.append(data)
            except queue.Empty:
                break
        
        #print "SCENE: ", scene
        self.q_scene.put(scene)
        pass
        #print "clear> ", datetime.datetime.now()
    
    def set_xlim(self,xlim):
        self.xlim = xlim
        #[-35, 35]
        #print xlim
        
    def set_ylim(self,ylim):
        self.ylim = ylim
        #[-35, 35]
        #print ylim
    
    '''def plot(self,xdata, ydata, linewidth=None, linestyle=None, color=None, 
             marker=None, markersize=None, markeredgewidth=None, 
             markeredgecolor=None, markerfacecolor=None, 
             markerfacecoloralt='none', fillstyle=None, 
             antialiased=None, dash_capstyle=None, 
             solid_capstyle=None, dash_joinstyle=None, solid_joinstyle=None,
             pickradius=5, drawstyle=None, markevery=None, **kwargs):'''
    
    def set_bot_msg(self,message):
        if (not self.enable_bot_msg):
            return False
        
        try:
            self.q_bot_msg.put(message, False)
            self.last_bot_msg = message
        except queue.Full:
            print "Message ",message," not delivered"
            pass
        
    def msg(self,message):
        self.label.text = message
        
    def on_close(self):
        super(SubWindow,self).on_close()
        if (self.window_ID == 0):
            exit()
        else:
            print "Closing Window ", self.window_ID
        
    def on_resize(self,width,height):        
        super(SubWindow,self).on_resize(width,height)
        self.fps.label.y = self.height - 40
        self.label.x = self.width//2
        self.label.y =  max([self.height - 30,abs(self.height/(1.1))])
        self.bot_label.x = self.width//2
        self.bot_label.y = 50#abs(self.height/8)
        
    def update(self, dt):        
        self.set_caption("[" + str(self.window_ID) + "] a " + str(pyglet.clock.get_fps()))
        if (not self.enable_bot_msg):
            return
        message = ""
        try:
            message = self.q_bot_msg.get( False)
        except queue.Empty:
            message = self.last_bot_msg
        pass
        self.bot_label.text = message

        #print "Update ", datetime.datetime.now()
            
    def plot(self,xdata, ydata, line='', **kwargs):
        #self.lock.acquire()
        #print kwargs
        #print "PLOTTING: ", xdata, " ",ydata
        if (type(xdata) != list):
            xdata = [xdata]
        if (type(ydata) != list):
            ydata = [ydata]

        ydata = [-y for y in ydata] # invert y-axis
        xdata, ydata = ydata, xdata # swap axis
        
        data = xdata + ydata
        
        #print "DATA: ", data
        max_data = max(data)
        if (max_data > self.max_val): self.max_val = max_data 
           
        min_data = min(data)
        if (min_data < self.min_val): self.min_val = min_data 
 
               
        self.q.put(data)
        
        #self.set_caption(pyglet.clock.get_fps().__str__())
        #self.lock.release()
    
    def to_world(self, data):
        new_data = []
            #new_val = translate(value, -50.,50.,0.,400.)
        for value in data[0:len(data)//2]:
                new_val = translate(value, float(self.xlim[0]), float(self.xlim[1]),0,self.width)
                new_data.append(new_val)
                
        for value in data[len(data)//2: len(data)]:
                new_val = translate(value, float(self.ylim[0]), float(self.ylim[1]),0,self.height)
                new_data.append(new_val)
            
        return new_data
    
    def on_draw(self):
        #self.clear()
        pyglet.clock.tick(False)
        
        self.super_clear()
        scene = None
        try:
            scene = self.q_scene.get(False)
            self.last_scene = scene
            self.msg(self.EMPTY_MSG)
            #self.msg(locale.format("%.2f", pyglet.clock.get_fps()) + " fps")
        except queue.Empty:
            self.msg("No data in scene queue. Redrawing last scene")
            #print "No data in scene queue. Redrawing last scene"
            if self.last_scene != None:
                scene = self.last_scene
            else:
                #self.msg("Last scene not found!")
                self.msg(self.EMPTY_MSG)
                #print "No last scene found!"
                return
        
        if self.enable_fps_msg: self.fps.draw()
        if self.enable_top_msg: self.label.draw()
        if self.enable_bot_msg: self.bot_label.draw()
        
        for q in scene:
            #print "MAX,MIN: ", self.max_val, " ; ", self.min_val
            #print q
            #print type(q)
            #print len(q)
            
            q = self.to_world(q)
            #print "NEW: ", q
            if (len(q) == 4):
                #print "Line"
                x1 = q[0]
                x2 = q[1]
                y1 = q[2]
                y2 = q[3]
                pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f',(x1,y1,x2,y2)))
            elif (len(q) == 2):
                x = q[0]
                y = q[1]
                pyglet.graphics.draw(1,pyglet.gl.GL_POINTS, ('v2f', (x,y)))
                
                #create circle around points
                numPoints,  vertexData = makeCircle(x,y)
                pyglet.graphics.draw(numPoints, pyglet.gl.GL_POINTS, vertexData)
            else:
                print "Error: Cowerdly refusing to print ",len(q)," points"
                raise Exception
                    
        '''for i in range(3): 
           pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', 
               (50*i, 100*i, 
                300, 100)))'''
       #self.draw_all()
       
    
'''       
    def step(self):
        self.set_caption(pyglet.clock.get_fps().__str__())
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()

        pyglet.clock.tick()'''
    

class Plot:
    def __init__(self):
        #self.clock = pyglet.clock.Clock(200)
        #pyglet.clock.set_default(self.clock)        
        pass
    
    def subplots(self,x_plots, y_plots):         
        self.window = Window(x_plots, y_plots)       
        return self.window, self.window.ax
    
    def run_loop(self):
        pyglet.clock.set_fps_limit(None)
        #pyglet.clock.schedule_interval(self.window.update, 0.0001)
        #pyglet.clock.schedule(self.window.update)
        self.window.schedule_updates()
        pyglet.app.EventLoop().run()
        print "Prgoram finished"
    
    def show(self):
        pass
    
    def ion(self):
        pass
    
    
#fig,ax = subplots(2,2)      
#print type(ax)
#print ax    