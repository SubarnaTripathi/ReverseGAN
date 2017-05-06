# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT
# [2017-04-04] Modifications for latent vector recovery: Subarna Tripathi (http://acsweb.ucsd.edu/~stripath/research)
#   + License: MIT

from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
from six.moves import xrange

from ops import *
from utils import *
import scipy.fftpack as scifft
from PIL import Image
import pickle
from os import listdir

class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1,
                 external_image=False,
                 untrained_net=False,
                 robustness_expt_num = 500,
                 num_iters = 80000,
                 recon_thresh = 0.005,
                 LEARNING_RATE = 1.,
                 clipping = True,
                 stochastic_clipping = False,
                 recover_training_images = False,
		 save_visualizations=True
                ):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, 3]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()

        self.model_name = "DCGAN.model"
        self.from_external_image = external_image
        self.robustness_expt_num = robustness_expt_num

        self.num_iters = num_iters
        self.recon_thresh = recon_thresh
        self.LEARNING_RATE = LEARNING_RATE
        self.stochastic_clipping = stochastic_clipping
        self.clipping = clipping 
        self.save_visualizations = save_visualizations
        viz_rows = int(np.sqrt(self.sample_size*1.))
        for counter in range(viz_rows, 1, -1):
            #print(counter, np.remainder(self.sample_size, counter))
            if (self.sample_size % counter) == 0:
                self.viz_rows = counter
                self.viz_cols = int(self.sample_size/self.viz_rows)
                break

        self.untrained_net = untrained_net
        self.recover_training_images = recover_training_images

    def build_model(self):
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.sample_images= tf.placeholder(
            tf.float32, [None] + self.image_shape, name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)
        
        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits,
                                                    tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_,
                                                    tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        self.mask = tf.placeholder(tf.float32, [None] + self.image_shape, name='mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.mul(self.mask, self.G) - tf.mul(self.mask, self.images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

        # Reverse.
        self.reverse_z = tf.placeholder(tf.float32, [None, self.z_dim], name='reverse_z')
        self.G_hat = self.generator_imposter(self.reverse_z)
        self.reverse_loss_sum = tf.reduce_sum(tf.pow(self.G - self.G_hat, 2), 0)
        self.reverse_loss = tf.reduce_mean(self.reverse_loss_sum)
        self.reverse_grads = tf.gradients(self.reverse_loss, self.reverse_z)
        # reverse
        self.restorer = tf.train.Saver()


    def train(self, config):
        data = glob(os.path.join(config.dataset, "*.png"))
        #np.random.shuffle(data)
        assert(len(data) > 0)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's Brandon Amos'
trained model for faces that's used in the post.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

""")
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

""")

        for epoch in xrange(config.epoch):
            data = glob(os.path.join(config.dataset, "*.png"))
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.images: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    

    def ReverseBatchwithNoise(self, config):
        z_recon_thresh = self.recon_thresh
        iterations = self.num_iters #100000
        LEARNING_RATE = self.LEARNING_RATE 

        tf.initialize_all_variables().run()

        isLoaded = self.load_new(self.checkpoint_dir)
        assert(isLoaded)
        #visualize(self.sess, self, config, 0)

        configurations = ["no_clipping", "standard_clipping", "stochastic_clipping"]

        noise_levels = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e+0]  
        mean = 0        
        
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim)) ## input  
        s_zh = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim)) ## imposter 
        
        orig_image = self.sess.run(self.G, feed_dict={self.z: sample_z}) 
        imposter_image_base = self.sess.run(self.G_hat, feed_dict={self.reverse_z: s_zh}) #sample_zh
      
        rand_shape = (self.sample_size, 64, 64, 3) 
        save_visualizations = True
              
        for config_id in range(len(configurations)):
            ## copy orig and imposter images every time in the loop
            imposter_image = np.copy(imposter_image_base)
            sample_zh = np.copy(s_zh) 

            config_name = configurations[config_id]
            base_result_dir = "result_visualizations/noisy_batch/" + str(config_name) + "/"                    
            if not os.path.exists(base_result_dir):
                os.makedirs(base_result_dir)

            imgName = base_result_dir + "visualization_orig.jpg" 
            save_images(orig_image, [self.viz_rows, self.viz_cols], imgName)

            imgName = base_result_dir + "visualization_imposter_init.jpg" 
            save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName)

            for level in np.arange(len(noise_levels)):
                sample_zh = np.copy(s_zh) 
                learning_rate = LEARNING_RATE 
                sigma = noise_levels[level]**.5
                gnoise = np.random.normal(mean, sigma, rand_shape)
                gnoise = gnoise.reshape(rand_shape)

                noisy_image = np.copy(orig_image)
                noisy_image = noisy_image + gnoise
                noisy_image = np.clip(noisy_image, -1, 1)           

                z_loss = []
                initial_z_loss = np.sum(np.power(sample_z - sample_zh, 2), 0)
                intial_z_loss = np.mean(initial_z_loss)
                z_loss.append(initial_z_loss)

                phi_loss = []
                initial_phi_loss = np.sum(np.power(noisy_image - imposter_image,2), 0)
                initial_phi_loss = np.mean(initial_phi_loss)                      
                phi_loss.append(initial_phi_loss)

                pixel_loss = []
                initial_pixel_loss = np.sum(np.power(orig_image - imposter_image,2), 0)
                initial_pixel_loss = np.mean(initial_pixel_loss)                      
                pixel_loss.append(initial_pixel_loss)

                result_dir = base_result_dir + str(level) + "/"
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                imgName = result_dir + "visualization_noisy_orig.jpg"  
                save_images(noisy_image, [self.viz_rows, self.viz_cols], imgName)

                z_primes_iters = []
                for n_iter in np.arange(iterations):
                    if n_iter > 0:
                        curr_z_loss = np.mean(np.power(sample_zh - sample_z, 2), 1) 
                        curr_z_loss = np.sum(curr_z_loss)
                        phi_loss.append(my_loss)                   
                        z_loss.append(curr_z_loss) 

                        #z_primes_iters.append(sample_zh)

                        ## pixel reconstrcution error
                        curr_pixel_loss = np.mean(np.power(orig_image - imposter_image, 2), 1) 
                        curr_pixel_loss = np.sum(curr_pixel_loss)
                        pixel_loss.append(curr_pixel_loss)

                        if np.abs(curr_z_loss) < self.recon_thresh * self.sample_size:                      
                            break

                    fd = {self.G: noisy_image, ## this is phi(z)
                          self.reverse_z: sample_zh
                          }
                    run = [self.reverse_loss, self.reverse_grads, self.G_hat]
                    [my_loss, my_grads, imposter_image] = self.sess.run(run, feed_dict=fd)

                    my_grads = np.asarray(my_grads[0])
                    #print(sample_zh.shape)
                    #print (n_iter+1, my_loss)  
                
                    ## save after 100-th iterations
                    if n_iter == 99:
                        imgName = result_dir + "visualization_imposter" + str(n_iter+1) + ".png" 
                        save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName) 
                    ######## 
  

                    if (n_iter+1) % 300 == 0:
                        print (n_iter, my_loss, np.mean(np.power(sample_zh - sample_z, 2)))                        
                
                        if n_iter != 0 and save_visualizations:                                          
                            imgName = result_dir + "visualization_imposter" + str(n_iter+1) + ".png"  
                            save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName) 
                         

                    sample_zh = sample_zh - learning_rate * my_grads
                    
                    if config_name != "no_clipping": # do nothing for NO clipping case                    
                        if config_name == "stochastic_clipping" :  ## stochastic clipping
                            for j in range(self.sample_size):  
                                edge1 = np.where(sample_zh[j] >= 1.)[0] #1
                                edge2 = np.where(sample_zh[j] <= -1)[0] #1

                                if edge1.shape[0] > 0:
                                    rand_el1 = np.random.uniform(-1, 1, size=(1, edge1.shape[0])) 
                                    sample_zh[j,edge1] = rand_el1
                                if edge2.shape[0] > 0:
                                    rand_el2 = np.random.uniform(-1, 1, size=(1, edge2.shape[0]))                            
                                    sample_zh[j,edge2] = rand_el2

                                #if edge1.shape[0] > 0 or edge2.shape[0] > 0:
                                    #print (edge1.shape[0], edge2.shape[0])
                        else: ## standard clipping
                            sample_zh = np.clip(sample_zh, -1, 1)

                ## always save the recovered image
                imgName = result_dir + "visualization_recovered.png"  
                save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName) 
                ######## 

                informtion = {'phi_loss': phi_loss, 
                              'z_loss':z_loss,
                              'pixel_loss': pixel_loss,
                              'z': sample_z,
                              'z_prime': sample_zh,
                              'z_primes_iters': z_primes_iters
                             }

                file_name = result_dir + "GAN_loss_batch_noisy.pickle"
                with open(file_name, 'wb') as handle:  
                    pickle.dump(informtion, handle, protocol=pickle.HIGHEST_PROTOCOL)

                ### write complements
                comple_z1 = sample_z - sample_zh  #### G(z - recovered_z)
                comple_z2 = np.ones((self.sample_size, self.z_dim)) - sample_zh  #### G(1 - recovered_z)
                comple_z3 = -sample_zh
                comple_z4 = -sample_z
                
                if config_name != "no_clipping":
                    comple_z1 = np.clip(comple_z1, -1., 1.)
                    comple_z2 = np.clip(comple_z2, -1., 1.)
                                   
                comple_image1 = self.sess.run(self.G, feed_dict={self.z: comple_z1}) 
                comple_image2 = self.sess.run(self.G, feed_dict={self.z: comple_z2}) 
                comple_image3 = self.sess.run(self.G, feed_dict={self.z: comple_z3})                
                comple_image4 = self.sess.run(self.G, feed_dict={self.z: comple_z4})

                imgName1 = result_dir + "z_minus_zh.png" 
                imgName2 = result_dir + "one_minus_zp.png" 
                imgName3 = result_dir + "minus_zh.png"
                imgName4 = result_dir + "minus_z.png"

                save_images(comple_image1, [self.viz_rows, self.viz_cols], imgName1) 
                save_images(comple_image2, [self.viz_rows, self.viz_cols], imgName2) 
                save_images(comple_image3, [self.viz_rows, self.viz_cols], imgName3)
                save_images(comple_image4, [self.viz_rows, self.viz_cols], imgName4)
                #####################
                
        
        #noise_recon_stats = {'stats': stats}
        #file_name = base_result_dir + "noisy_batch_recon_stats.pickle"
        #with open(file_name, 'wb') as handle:  
            #pickle.dump(informtion, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print ('GIR batch with noise complete')

    

    def reverse_GAN_batch_all_prec(self, config):
        tf.initialize_all_variables().run()
       
        base_result_dir = "result_visualizations/"       

        if self.untrained_net:
            base_result_dir = "untrained_net/" + base_result_dir
        else:
            isLoaded = self.load_new(self.checkpoint_dir)
            assert(isLoaded)
        if not os.path.exists(base_result_dir):
            os.makedirs(base_result_dir)

        save_visualizations = self.save_visualizations

        precision_levels = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
       
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim)) ## input  
        s_zh = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim)) ## imposter 
        sample_zh = np.copy(s_zh) 
        configurations = ["no_clipping", "standard_clipping", "stochastic_clipping"]
       
        #tf.get_variable_scope().reuse_variables()
        if not self.from_external_image:                
            orig_image = self.sess.run(self.G, feed_dict={self.z: sample_z})
        else :
            orig_image = np.zeros((self.sample_size, 64, 64, 3))
            img_name = 'unseen_face6.png'  #'unseen_face6.png'   #'test_img.jpg'
            for j in np.arange(self.sample_size):
                orig_image[j] = get_image(img_name, 64)
         
        imposter_image_base = self.sess.run(self.G_hat, feed_dict={self.reverse_z: sample_zh}) #sample_zh
        #print(orig_image.shape, imposter_image.shape)

        for config_id in range(len(configurations)):
            imposter_image = np.copy(imposter_image_base)
            sample_zh = np.copy(s_zh) 

            config_name = configurations[config_id]
            comment=config_name + "Experiment Started"
            print(comment)

            reverse_accuracy_matrix = np.zeros((self.sample_size, len(precision_levels)))
            base_result_dir = "result_visualizations/accuracy/" + str(config_name) + "/"                    
            if not os.path.exists(base_result_dir):
                os.makedirs(base_result_dir)

            if save_visualizations:
                imgName = base_result_dir + "visualization_orig.jpg" 
                save_images(orig_image, [self.viz_rows, self.viz_cols], imgName)
                imgName = base_result_dir + "visualization_imposter_init.jpg" 
                save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName)
       
            #initial_phi_loss = np.mean(np.power(orig_image_pixel - imposter_image_pixel,2))
            if self.from_external_image == 0:
                z_loss = []
                initial_z_loss = np.sum(np.power(sample_z - sample_zh, 2), 0)
                initial_z_loss = np.mean(initial_z_loss)
                z_loss.append(initial_z_loss)
            else:
                recon_zs = []
 
            phi_loss = []
            initial_phi_loss = np.sum(np.power(orig_image - imposter_image,2), 0)  # sum over batch
            initial_phi_loss = np.mean(initial_phi_loss)
            #print(str(0), initial_phi_loss, initial_z_loss)               
            phi_loss.append(initial_phi_loss)

            learning_rate = self.LEARNING_RATE #1. 
            iterations = self.num_iters 
            for n_iter in np.arange(iterations):
                if n_iter % 50000 == 0 and n_iter > 0:  #2000
                    learning_rate /= 2.

                if n_iter > 0:
                    phi_loss.append(my_loss)

                    if self.from_external_image == 0:
                        curr_z_loss = np.power(sample_zh - sample_z, 2)
                        curr_z_loss = np.mean(curr_z_loss, 1)
                        #curr_z_loss = np.mean((curr_z_loss**.5), 1)                          
                        z_loss.append(np.mean(curr_z_loss)) 
                        for p in np.arange(len(precision_levels)):
                            curr_z_loss_p = curr_z_loss   #[:,p]
                            prec_level = precision_levels[p]
                            d = np.where(curr_z_loss_p < prec_level)
                            reverse_accuracy_matrix[d, p] = 1

                        if np.sum(reverse_accuracy_matrix) == self.sample_size * len(precision_levels):
                            break                       

                fd = {self.G: orig_image, ## this is phi(z)
                      self.reverse_z: sample_zh
                     }
                run = [self.reverse_loss, self.reverse_grads, self.G_hat]
                [my_loss, my_grads, imposter_image] = self.sess.run(run, feed_dict=fd)

                my_grads = np.asarray(my_grads[0])  

                if n_iter % 200 == 0:
                    if self.from_external_image == 0:
                        print (n_iter, my_loss, np.mean(np.power(sample_zh - sample_z, 2)))
                    else:
                        print (n_iter, my_loss)
                    if save_visualizations == 1 and n_iter != 0:                                  
                        imgName = base_result_dir + "visualization_imposter" + str(n_iter) + ".jpg"  
                        save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName)

                sample_zh = sample_zh - learning_rate * my_grads            

                if config_name != "no_clipping": # do nothing for NO clipping case                    
                            if config_name == "stochastic_clipping" :  ## stochastic clipping
                                for j in range(self.sample_size):  
                                    edge1 = np.where(sample_zh[j] >= 1.)[0] #1
                                    edge2 = np.where(sample_zh[j] <= -1)[0] #1

                                    if edge1.shape[0] > 0:
                                        rand_el1 = np.random.uniform(-1, 1, size=(1, edge1.shape[0])) 
                                        sample_zh[j,edge1] = rand_el1
                                    if edge2.shape[0] > 0:
                                        rand_el2 = np.random.uniform(-1, 1, size=(1, edge2.shape[0]))                            
                                        sample_zh[j,edge2] = rand_el2

                                    #if edge1.shape[0] > 0 or edge2.shape[0] > 0:
                                        #print (edge1.shape[0], edge2.shape[0])
                            else: ## standard clipping
                                sample_zh = np.clip(sample_zh, -1, 1)


            if self.from_external_image:
                for j in np.arange(self.sample_size):
                    recon_zs.append(sample_zh[j])
                recon_z_info = {'recon_zs':recon_zs}
                file_name = base_result_dir + "recon_zs.pickle"
                with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
                    pickle.dump(recon_z_info, handle, protocol=pickle.HIGHEST_PROTOCOL)               

            if self.from_external_image == 0:
                informtion = {'phi_loss': phi_loss, 
                              'z_loss':z_loss,
                      	      'sample_z': sample_z,
                               'sample_zh': sample_zh}
            else:
                informtion = {'phi_loss': phi_loss,
                              'sample_zh': sample_zh}

            file_name = base_result_dir + "GAN_loss.pickle"
            with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
                pickle.dump(informtion, handle, protocol=pickle.HIGHEST_PROTOCOL)

            accuracy_informtion = {'accuracy_stats': reverse_accuracy_matrix}
            file_name = base_result_dir + "accuracy_stats.pickle"
            with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
                pickle.dump(accuracy_informtion, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print ('GIR accuracy experiment complete')               



    def reverse_GAN_batch_all_prec_posterior(self, config):
        tf.initialize_all_variables().run()
        self.from_external_image = True       
        get_training_images = self.recover_training_images
               
        if get_training_images:
            if self.clipping:
                base_result_dir = "training_images/st_clip_" + str(self.stochastic_clipping) + str("/")
            else:
                base_result_dir = "training_images/Noclip/" 
        else:
            if self.clipping:
                base_result_dir = "unseen_images/st_clip_" + str(self.stochastic_clipping) + str("/")
            else:
                base_result_dir = "unseen_images/Noclip/" 

        if self.untrained_net:
            base_result_dir = "untrained_net/" + base_result_dir
        else:
            isLoaded = self.load_new(self.checkpoint_dir)
            assert(isLoaded)
        if not os.path.exists(base_result_dir):
            os.makedirs(base_result_dir)

        save_visualizations = 1

        precision_levels = [1e-5] #[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        reverse_accuracy_matrix = np.zeros((self.sample_size, len(precision_levels)))

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim)) ## input  
        s_zh = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim)) ## imposter 
        sample_zh = np.copy(s_zh) 
       
        #tf.get_variable_scope().reuse_variables()
        if self.from_external_image == 0:                
            orig_image = self.sess.run(self.G, feed_dict={self.z: sample_z})
        else :
            orig_image = np.zeros((self.sample_size, 64, 64, 3))            
            if get_training_images:
                start_num = np.random.randint(100, 5000) #(0, 10100)
                end_num = start_num + 9000
                img_idx=[np.random.randint(start_num, end_num) for p in range(0, self.sample_size)] 
                print(img_idx)  
                png_files = [f for f in listdir("all_images/gan_train_img_align_celeba/img_align_celeba/")]
                for j in np.arange(self.sample_size):
                    q = img_idx[j]
                    image_name = "all_images/gan_train_img_align_celeba/img_align_celeba/" + png_files[q]
                    orig_image[j] = get_image(img_name, 64)
                #print(orig_image.shape)
            else: 
                start_num = np.random.randint(0, 200) 
                end_num = start_num + 3000
                img_idx=[np.random.randint(start_num, end_num) for p in range(0, self.sample_size)] 
                png_files = [f for f in listdir("all_images/lfw/aligned")]
                print(img_idx)  
                for j in np.arange(self.sample_size):
                    q = img_idx[j]
                    img_name = "all_images/lfw/aligned/" + png_files[q]
                    orig_image[j] = get_image(img_name, 64)
                #print(orig_image.shape)

                #img_name = 'unseen_face6.png'  #'unseen_face7.png'    
                #for j in np.arange(self.sample_size):
                    #orig_image[j] = get_image(img_name, 64)
         
        imposter_image = self.sess.run(self.G_hat, feed_dict={self.reverse_z: sample_zh}) #sample_zh
        #print(orig_image.shape, imposter_image.shape)
        
        if save_visualizations == 1:
            result_dir = base_result_dir 
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            imgName = result_dir + "visualization_orig.jpg"
            save_images(orig_image, [self.viz_rows, self.viz_cols], imgName)   

            imgName = result_dir + "visualization_imposter_init.jpg" # "crazy_imgs_4/visualization_imposter_init.jpg"
            save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName)

        
        #initial_phi_loss = np.mean(np.power(orig_image_pixel - imposter_image_pixel,2))
        if self.from_external_image == 0:
            z_loss = []
            initial_z_loss = np.sum(np.power(sample_z - sample_zh, 2), 0)
            initial_z_loss = np.mean(initial_z_loss)
            z_loss.append(initial_z_loss)
        else:
            recon_zs = []
 
        phi_loss = []
        initial_phi_loss = np.sum(np.power(orig_image - imposter_image,2), 0)  # sum over batch
        initial_phi_loss = np.mean(initial_phi_loss)            
        phi_loss.append(initial_phi_loss)
                 
        learning_rate = self.LEARNING_RATE #1. 
        iterations = self.num_iters 
        for n_iter in np.arange(iterations):
            if n_iter % 50000 == 0 and n_iter > 0:  #2000
                learning_rate /= 2.

            if n_iter > 0:
                phi_loss.append(my_loss)

                if self.from_external_image == 0:
                    curr_z_loss = np.power(sample_zh - sample_z, 2)
                    curr_z_loss = np.mean(curr_z_loss, 1)
                    #curr_z_loss = np.mean((curr_z_loss**.5), 1)       
                    
                    z_loss.append(np.mean(curr_z_loss)) 

                    for p in np.arange(len(precision_levels)):
                        current_z_loss_p = curr_z_loss[:,p]
                        prec_level = precision_levels[p]
                        d = np.where(curr_z_loss_p > p)
                        reverse_accuracy_matrix[d, p] = 1
                    
                    if np.sum(reverse_accuracy_matrix) == self.sample_size * len(precision_levels):
                        break
                        
            
            fd = {self.G: orig_image, ## this is phi(z)
                  self.reverse_z: sample_zh
                 }
            run = [self.reverse_loss, self.reverse_grads, self.G_hat]
            [my_loss, my_grads, imposter_image] = self.sess.run(run, feed_dict=fd)
        
            my_grads = np.asarray(my_grads[0])  
                
            if n_iter % 200 == 0:
                if self.from_external_image == 0:
                    print (n_iter, my_loss, np.mean(np.power(sample_zh - sample_z, 2)))
                else:
                    print (n_iter, my_loss)
                if save_visualizations == 1 and n_iter != 0:                                  
                    imgName = result_dir + "visualization_imposter" + str(n_iter) + ".jpg"  
                    save_images(imposter_image, [self.viz_rows, self.viz_cols], imgName)
                                           
            sample_zh = sample_zh - learning_rate * my_grads            

            if self.clipping:
                if self.stochastic_clipping:  
                    for j in range(self.sample_size):  
                        edge1 = np.where(sample_zh[j] >= 1.)[0] #1
                        edge2 = np.where(sample_zh[j] <= -1)[0] #1

                        if edge1.shape[0] > 0:
                            rand_el1 = np.random.uniform(-1, 1, size=(1, edge1.shape[0])) 
                            sample_zh[j,edge1] = rand_el1
                        if edge2.shape[0] > 0:
                            rand_el2 = np.random.uniform(-1, 1, size=(1, edge2.shape[0]))                            
                            sample_zh[j,edge2] = rand_el2

                        #if edge1.shape[0] > 0 or edge2.shape[0] > 0:
                            #print (edge1.shape[0], edge2.shape[0])
                else:
                    sample_zh = np.clip(sample_zh, -1, 1)

        if self.from_external_image:
            ##### apply discriminator
            fd_orig = {self.images: orig_image}
            fd_imposter = {self.images: imposter_image}
            run = [self.D, self.D_logits]
            [D_score, D_logits] = self.sess.run(run, feed_dict=fd_orig)            
            [D_score_imposter, D_logits_imposter] = self.sess.run(run, feed_dict=fd_imposter)

            print(D_score, D_logits)
            print(D_score_imposter, D_logits_imposter)

            for j in np.arange(self.sample_size):
                recon_zs.append(sample_zh[j])
            recon_z_info = {'recon_zs':recon_zs,
                            'D_scores': D_score,
                            'D_logits' : D_logits,
                            'D_scores_imposter': D_score_imposter,
                            'D_logits_imposter': D_logits_imposter}
            file_name = base_result_dir + "recon_zs.pickle"
            with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
                pickle.dump(recon_z_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

                
        
        if self.from_external_image == 0:
            informtion = {'phi_loss': phi_loss, 
                          'z_loss':z_loss}
        else:
            informtion = {'phi_loss': phi_loss}

        file_name = base_result_dir + "GAN_loss.pickle"
        with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
            pickle.dump(informtion, handle, protocol=pickle.HIGHEST_PROTOCOL)

        accuracy_informtion = {'accuracy_stats': reverse_accuracy_matrix}
        file_name = base_result_dir + "accuracy_stats.pickle"
        with open(file_name, 'wb') as handle:  #'GAN_loss_4.pickle'
            pickle.dump(accuracy_informtion, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print ('Latent space recovery from training or unseen images complete')  


    def complete(self, config):
        os.makedirs(os.path.join(config.outDir, 'hats_imgs'), exist_ok=True)
        os.makedirs(os.path.join(config.outDir, 'completed'), exist_ok=True)

        tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        # data = glob(os.path.join(config.dataset, "*.png"))
        nImgs = len(config.imgs)

        batch_idxs = int(np.ceil(nImgs/self.batch_size))
        if config.maskType == 'random':
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            scale = 0.25
            assert(scale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*scale)
            u = int(self.image_size*(1.0-scale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':
            mask = np.ones(self.image_shape)
        else:
            assert(False)

        for idx in xrange(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            batch_mask = np.resize(mask, [self.batch_size] + self.image_shape)
            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = 8
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, batch_mask)
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],
                        os.path.join(config.outDir, 'masked.png'))

            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: batch_mask,
                    self.images: batch_images,
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                v_prev = np.copy(v)
                v = config.momentum*v - config.lr*g[0]
                zhats += -config.momentum * v_prev + (1+config.momentum)*v
                zhats = np.clip(zhats, -1, 1)

                if i % 50 == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows = np.ceil(batchSz/8)
                    nCols = 8
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-batch_mask)
                    completeed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completeed[:batchSz,:,:,:], [nRows,nCols], imgName)


    def generator(self, z):
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
            [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)


    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')

        return tf.nn.sigmoid(h4), h4
    
    
    def generator_imposter(self, reverse_z):
        tf.get_variable_scope().reuse_variables()
        
        self.reverse_z_, self.h0_w, self.h0_b = linear(reverse_z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.reverse_z_, [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = conv2d_transpose(h0,
            [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = conv2d_transpose(h1,
            [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = conv2d_transpose(h2,
            [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = conv2d_transpose(h3,
            [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)


    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        h0 = tf.reshape(linear(z, self.gf_dim*8*4*4, 'g_h0_lin'),
                        [-1, 4, 4, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = conv2d_transpose(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = conv2d_transpose(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = conv2d_transpose(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = conv2d_transpose(h3, [self.batch_size, 64, 64, 3], name='g_h4')

        return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        max_to_keep=100,
                        global_step=step)

    def load_new(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        checkpoint_include_scopes=["g_h0_lin", "g_h1", "g_h2", "g_h3"]    
        inclusions = [scope.strip() for scope in checkpoint_include_scopes]
        variables_to_restore = []
        for var in tf.all_variables(): 
            for inclusion in inclusions:
                if var.op.name.startswith(inclusion):
                    variables_to_restore.append(var)

        self.restorer = tf.train.Saver(variables_to_restore)

        #ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #if ckpt and ckpt.model_checkpoint_path:
            #self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            #self.restorer.restore(self.sess, ckpt.model_checkpoint_path)
            #return True
        #else:
            #return False


        #checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        checkpoint_path = "checkpoint_new2/DCGAN.model-29002"  #7002
        self.restorer.restore(self.sess, checkpoint_path)
        self.saver.restore(self.sess, checkpoint_path)
        return True



    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            #self.restorer.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
