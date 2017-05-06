# Subarna Tripathi (http://acsweb.ucsd.edu/~stripath/research)
# License: MIT
# 2017-04-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import flags

from model import DCGAN

import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json


flags = tf.app.flags

#flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
#flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_string("checkpoint_dir", "checkpoint_new2", "Directory name to save the checkpoints [checkpoint]")
FLAGS = flags.FLAGS

from PIL import Image
from tensorflow.contrib.framework.python.framework import checkpoint_utils

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, 
                  batch_size=1024, sample_size=1024,
                  num_iters = 80000,
                  LEARNING_RATE=1.)       
    dcgan.reverse_GAN_batch_all_prec(FLAGS)
