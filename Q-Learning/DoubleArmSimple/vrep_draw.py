# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:12:35 2017

"""

#from poppy.creatures import *
from poppy.creatures import PoppyHumanoid

class Plot:
    #poppy instance as paramter from poppy = PoppyTorso(simulator='vrep')
    def __init__(self, poppy = None, simulator = "vrep"):
        if poppy == None:
            print "No poppy instance is given. Creating a new one"
            poppy = PoppyHumanoid(simulator=simulator)
            poppy.reset_simulation()
        
        self.poppy = poppy
        self.io = poppy._controllers[0].io
                                    
        self.objects = []

    def update_object_pos(self, pos, name="goal"):
        if self.objects.count(name) == 0:
            #position = [0.2, 0, 0.8] # X, Y, Z
            #sizes = [0.1, 0.1, 0.1] # in meters
            #mass = 0 # in kg   
            #self.io.add_cube(name, position, sizes, mass)
            self.objects.append(name)
            
        self.io.set_object_position(name, pos)