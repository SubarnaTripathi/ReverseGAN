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
# flags.DEFINE_integer("epoch", 25, "Epoch" to train [25]")
#flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
#flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
FLAGS = flags.FLAGS


copy_num = 1
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, 
                  batch_size=1024, sample_size=1024,
                  untrained_net = True,
                  external_image=False,
                  num_iters=100000,
                  LEARNING_RATE=1.,
                  #clipping=False,
                  stochastic_clipping=True
                  )       
    dcgan.reverse_GAN_batch_all_prec(FLAGS)   

