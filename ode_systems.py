#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:20:54 2017

@author: jason
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin

makeD0          = lambda POINTS: np.asmatrix(np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)) / 2
makeD1          = lambda POINTS: np.asmatrix(-1*np.eye(POINTS-1,POINTS) + np.roll(np.eye(POINTS-1,POINTS),1,1)) / INTERVAL_LENGTH
A               = D1.T @ D1 + D0.T @ D0
proj            = np.ones(POINTS)