import os
import json
import webbrowser
import sys
import ctypes

import numpy as np
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, Qt, QEvent
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import caffe
from caffe import layers as L 
from caffe import params as P 

import io