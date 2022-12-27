import tensorflow as tf
import numpy as np

class RetiFluidNet:

    def __init__(self, num_class=4, input_shape=(256, 256, 1)):
        self.num_class   = num_class
        self.input_shape = input_shape

    def SDA(self, tensor, p_scale=4, SDAblock_nb=0): #Self-adaptive Dual Attention Module
      
      input_tensor = tensor
      _,  H, W, C  = tensor.shape 
        
      tensor = tf.keras.layers.MaxPooling2D(p_scale)(tensor)
      _, Hp, Wp, C = tensor.shape
        
      # =============================================================================
      #     Pixel wise attention    
      # =============================================================================
      """---Pixel wise attention---"""
      ratio1 = np.sqrt((Hp * Wp))
      reshaped_tensor = tf.keras.layers.Reshape(((Hp * Wp), C))(tensor) # Shape ===> (Hp * Wp) * C
      # print(reshaped_tensor.shape)
      transposed_tensor = tf.keras.layers.Permute((2, 1))(reshaped_tensor) # Shape ===> C * (Hp * Wp)
      # print(transposed_tensor.shape)
      x = tf.keras.layers.Dot(axes = (2, 1))([reshaped_tensor, transposed_tensor]) / ratio1  # Shape ===> ((Hp * Wp) * (Hp * Wp)) ### divide by 1 / (sqrt(Hp * Wp))
      # print(x.shape)
      x = tf.keras.layers.Activation('softmax')(x) # Shape ===> ((Hp * Wp) * (Hp * Wp))
      # print(x.shape)
      x = tf.keras.layers.Dot(axes = (2, 1))([x, reshaped_tensor]) # Shape ===> (Hp * Wp) * C
      # print(x.shape)
      
      """layer name='alpha'""" 
      x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), kernel_initializer='ones', name = 'alpha'+str(SDAblock_nb),
                                  padding='same', use_bias=False, activation='relu',
                                  )(tf.expand_dims(x, axis = -1))
                                  # kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1.0, max_value=10.0, rate=0.5, axis=0))(tf.expand_dims(x, axis = -1)),
                                  # kernel_constraint=tf.keras.constraints.NonNeg())(tf.expand_dims(x, axis = -1))
	  
      # Shape ===> (Hp * Wp) * C * 1
      
      add_1 = tf.keras.layers.Reshape((Hp, Wp, C))(x) # Shape ===> Hp * Wp * C
      # print(add_1.shape)
      add_1 = tf.keras.layers.experimental.preprocessing.Resizing(H, W, interpolation='nearest')(add_1) # Shape ===> H * W * C
      # print(add_1.shape)
	  
      # =============================================================================
      #     Channel-wise Attention  
      # =============================================================================
      """---Channel-wise Attention---"""
      ratio2 = np.sqrt((C * C))
      reshaped_tensor = tf.keras.layers.Reshape(((Hp * Wp), C))(tensor) # Shape ===> (Hp * Wp) * C
      # print(reshaped_tensor.shape)
      transposed_tensor = tf.keras.layers.Permute((2, 1))(reshaped_tensor) # Shape ===> C * (Hp * Wp)
      # print(transposed_tensor.shape)
      x = tf.keras.layers.Dot(axes = (2, 1))([transposed_tensor, reshaped_tensor])/ ratio2 # Shape ===> (C * C) ### divide by 1 / (sqrt(C * C))
      # print(x.shape)
      x = tf.keras.layers.Activation('softmax')(x) # Shape ===> (C * C)
      # print(x.shape)
      x = tf.keras.layers.Dot(axes = (2, 1))([reshaped_tensor, x]) # Shape ===> (Hp * Wp) * C
      # print(x.shape)
      
      """layer name='beta'""" 
      x = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1), kernel_initializer='ones', name = 'beta'+str(SDAblock_nb),
                                      padding = 'same', use_bias=False, activation='relu',
                                      )(tf.expand_dims(x, axis = -1))
                                      #kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=1.0, max_value=10.0, rate=0.5, axis=0))(tf.expand_dims(x, axis = -1))
                                      #kernel_constraint=tf.keras.constraints.NonNeg())(tf.expand_dims(x, axis = -1))
      
      # Shape ===> (Hp * Wp) * C * 1
      # print(x.shape)
      add_2 = tf.keras.layers.Reshape((Hp, Wp, C))(x) # Shape ===> Hp * Wp * C
      # print(add_2.shape)
      add_2 = tf.keras.layers.experimental.preprocessing.Resizing(H, W, interpolation='nearest')(add_2) # Shape ===> H * W * C
      # print(add_2.shape)
      
      # =============================================================================
      #     Fusion    
      # =============================================================================
      # print("add_1 : ", add_1.shape)
      # print("add_2 : ", add_2.shape)
      mean = tf.multiply(0.5, tf.keras.layers.Add()([add_1, add_2])) # Shape ===> H * W * C (same as the input shape)
      sda_output = tf.keras.layers.Add()([input_tensor, mean])
      # print(sda_output.shape)
      return sda_output   

    def rmp_block(self, tensor):
    
      _, W, H, C = tensor.shape
    
      pool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides = 2)(tensor)
      pool_2 = tf.keras.layers.MaxPooling2D((3, 3), strides = 3)(tensor)
      pool_3 = tf.keras.layers.MaxPooling2D((5, 5), strides = 5)(tensor)
      pool_4 = tf.keras.layers.MaxPooling2D((6, 6), strides = 6)(tensor)
      
      x1 = tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_1)
      x2 = tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_2)
      x3 = tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_3)
      x4 = tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_4)
      
      conv1 = tf.keras.layers.experimental.preprocessing.Resizing(W, H, interpolation='nearest')(x1)
      conv2 = tf.keras.layers.experimental.preprocessing.Resizing(W, H, interpolation='nearest')(x2)
      conv3 = tf.keras.layers.experimental.preprocessing.Resizing(W, H, interpolation='nearest')(x3)
      conv4 = tf.keras.layers.experimental.preprocessing.Resizing(W, H, interpolation='nearest')(x4)
      
      main_output = tf.keras.layers.concatenate([tensor, conv1, conv2, conv3, conv4]) # Shape  ===> 256*256*36
      return main_output
    
    def SDA_block(self, tensor, p_scale, SDAblock_nb):
        
      main_output = self.SDA(tensor, p_scale, SDAblock_nb)
      return main_output   
    
    def conv_block(self, input_tensor, num_filters):

      x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)  
      x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(x)
      x = tf.keras.layers.BatchNormalization()(x)
      x = tf.keras.layers.Activation('relu')(x)
      return x

    def encoder_block(self, input_tensor, num_filters):

      encoder = self.conv_block(input_tensor, num_filters)
      encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)     
      return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters, p_scale, SDAblock_nb):

      decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)   
      sda = self.SDA_block(concat_tensor, p_scale, SDAblock_nb)
      print('SASC output shape  =', tf.keras.layers.concatenate([sda, concat_tensor], axis=-1).shape)
      decoder = tf.keras.layers.concatenate([sda, concat_tensor, decoder], axis=-1)
      print('decoder input shape=', decoder.shape)
      decoder = tf.keras.layers.BatchNormalization()(decoder)
      decoder = tf.keras.layers.Activation('relu')(decoder)
      decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
      decoder = tf.keras.layers.BatchNormalization()(decoder)
      decoder = tf.keras.layers.Activation('relu')(decoder)  
      decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
      decoder = tf.keras.layers.BatchNormalization()(decoder)
      decoder = tf.keras.layers.Activation('relu')(decoder)
      return decoder
  
    def convert_to_8_channels(self, input_tensor):
        
        out8_1 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(input_tensor[: ,:, :, 0], axis = -1))
        out8_2 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(input_tensor[: ,:, :, 1], axis = -1))
        out8_3 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(input_tensor[: ,:, :, 2], axis = -1))
        out8_4 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(input_tensor[: ,:, :, 3], axis = -1))  
         
        outputs = tf.keras.layers.concatenate([out8_1, out8_2, out8_3, out8_4])
        return outputs
        

    def RetiFluidNet_model(self):
        
        nb_filters = 32
        
        # Input
        inputs = tf.keras.layers.Input(shape = self.input_shape)
        print ('+'*50)
        print ('input shape        =',inputs.shape)
        # image size:256*256    
        
        """Ecoder Block #0"""
        encoder0_pool, encoder0 = self.encoder_block(inputs, nb_filters)
        print ('+'*50)
        print ('encoder0 shape     =',encoder0.shape)
        print ('encoder0_pool shape=',encoder0_pool.shape)
        # image size:128*128
        
        """Ecoder Block #1"""
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, nb_filters*2)
        print ('-'*50)
        print ('encoder1 shape     =',encoder1.shape)
        print ('encoder1_pool shape=',encoder1_pool.shape)
        # image size:64*64
        
        """Ecoder Block #2"""
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, nb_filters*4)
        print ('-'*50)
        print ('encoder2 shape     =',encoder2.shape)
        print ('encoder2_pool shape=',encoder2_pool.shape)
        # image size:32*32
        
        """Ecoder Block #3"""
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, nb_filters*8)
        print ('-'*50)
        print ('encoder3 shape     =',encoder3.shape)
        print ('encoder3_pool shape=',encoder3_pool.shape)
        # image size:16*16
        
        """Ecoder Block #4"""
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, nb_filters*16)
        print ('-'*50)
        print ('encoder4 shape     =',encoder4.shape)
        print ('encoder4_pool shape=',encoder4_pool.shape)
        # image size:8*8
        
        """Center Block"""
        # Latent Space: Center Block Process
        sda = self.SDA_block(encoder4_pool, 4, SDAblock_nb=5)
        center = self.conv_block(sda, nb_filters*32)
        # center = tf.keras.layers.SpatialDropout2D(0.25)(center)
        rmp = self.rmp_block(center)
        print ('-'*50)
        print ('center sda shape   =',sda.shape)
        print ('center_conv shape  =',center.shape)
        print ('center_rmp shape   =',rmp.shape)
        # center output
        # image size:8*8
        
        print('+'*50)
		
        """Decoder Block #4"""
        decoder4 = self.decoder_block(rmp, encoder4, nb_filters*16, 4, SDAblock_nb=4)
        print ('decoder4 shape     =',decoder4.shape)
        # output 4
        output4 = tf.keras.layers.UpSampling2D(size=(16, 16))(decoder4)
        output4 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output4)
        output4 = tf.keras.layers.Activation("softmax", name="output4")(output4)
        bicon_output4 = self.convert_to_8_channels(output4)    
        print ('bicon_output4 shape=',bicon_output4.shape)
        print ('output4 shape      =',output4.shape)
        # output image size:16*16
        
        """Decoder Block #3"""
        print ('-'*50)
        decoder3 = self.decoder_block(decoder4, encoder3, nb_filters*8, 4, SDAblock_nb=3)
        print ('decoder3 shape     =',decoder3.shape)
		# output 3
        output3 = tf.keras.layers.UpSampling2D(size=(8, 8))(decoder3)
        output3 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output3)
        output3 = tf.keras.layers.Activation("softmax", name = "output3")(output3)
        bicon_output3 = self.convert_to_8_channels(output3)    
        print ('bicon_output3 shape=',bicon_output3.shape)
        print ('output3 shape      =',output3.shape)
        # output image size:32*32
        
        """Decoder Block #2"""
        print ('-'*50)
        decoder2 = self.decoder_block(decoder3, encoder2, nb_filters*4, 4, SDAblock_nb=2)
        print ('decoder2 shape     =',decoder2.shape)
        # output 2
        output2 = tf.keras.layers.UpSampling2D(size=(4, 4))(decoder2)
        output2 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output2)
        output2 = tf.keras.layers.Activation("softmax", name = "output2")(output2)
        bicon_output2 = self.convert_to_8_channels(output2)    
        print ('bicon_output2 shape=',bicon_output2.shape)
        print ('output2 shape      =',output2.shape)
        # output image size:64*64
        
        """Decoder Block #1"""
        print ('-'*50)
        decoder1 = self.decoder_block(decoder2, encoder1, nb_filters*2, 4, SDAblock_nb=1)
        print ('decoder1 shape     =',decoder1.shape)
        # output 1
        output1 = tf.keras.layers.UpSampling2D(size=(2, 2))(decoder1)
        output1 = tf.keras.layers.Conv2D(self.num_class, (1, 1))(output1)  
        output1 = tf.keras.layers.Activation("softmax", name = "output1")(output1) 
        bicon_output1 = self.convert_to_8_channels(output1)    
        print ('bicon_output1 shape=',bicon_output1.shape)
        print ('output1 shape      =',output1.shape)
        # output image size:128*128
        
        """Decoder Block #0"""
        print ('-'*50)
        decoder0 = self.decoder_block(decoder1, encoder0, nb_filters, 4, SDAblock_nb=0)
        print ('decoder0 shape     =',decoder0.shape)
        # main output
        outputs = tf.keras.layers.Conv2D(self.num_class, (1, 1))(decoder0)
        Main_output = tf.keras.layers.Activation("softmax",  name='main_output')(outputs)
        
        bicon_output0 = self.convert_to_8_channels(outputs) 
        
        print ('bicon_output0 shape=',bicon_output0.shape)
        print ('output0 shape      =',outputs.shape)
        # output image size:256*256
        
        Bicon_outputs = tf.keras.layers.concatenate([bicon_output0, bicon_output1, bicon_output2, bicon_output3, bicon_output4])
        
        outputs_to_return = tf.keras.layers.concatenate([Bicon_outputs, Main_output, output4, output3, output2, output1])
        
        print ('-'*50)
        print ('Model output shape =',outputs_to_return.shape)
        print ('+'*50)
        
        model = tf.keras.models.Model(inputs=[inputs], outputs = outputs_to_return)  
        
        return model
    
    def __call__(self):
        model =  self.RetiFluidNet_model()
        return model
    
    
model = RetiFluidNet()()
model.summary() 
