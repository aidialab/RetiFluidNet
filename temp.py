import tensorflow as tf
import numpy as np

class Unet:

    def __init__(self, num_class, input_shape = (256,256,1)):
        self.num_class = num_class
        self.input_shape = input_shape
        
    def attention_block_1(self, tensor):  ####################SA model#####################
    
        _, W, H, C = tensor.shape
      
        ###########################################
        x_res = tf.keras.layers.Conv2D(filters = C, kernel_size = (3, 3), padding = 'same')(tensor)
        x_res = tf.keras.layers.BatchNormalization()(x_res)
        x_res = tf.keras.layers.Activation('relu')(x_res)  
      
        x_res = tf.keras.layers.Conv2D(filters = C, kernel_size = (3, 3), padding = 'same')(x_res)
        x_res = tf.keras.layers.BatchNormalization()(x_res)
        x_res = tf.keras.layers.Activation('relu')(x_res)    
        ############################################

        x = tf.keras.layers.MaxPooling2D((2, 2))(tensor)
      
        x = tf.keras.layers.Conv2D(filters = C,kernel_size = (3, 3) ,padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
      
        x = tf.keras.layers.Conv2D(filters = C,kernel_size = (3, 3) ,padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
      
        x = tf.keras.layers.Conv2DTranspose(x.shape[3], (3, 3), strides = (2, 2))(x)
        x = tf.keras.layers.Conv2DTranspose(x.shape[3], (3, 3), strides = (2, 2))(x)
        
        x_att = tf.keras.layers.experimental.preprocessing.Resizing(W, H, interpolation='nearest')(x)
      
        x_output = tf.keras.layers.Multiply()([x_att, x_res])
        x_output = tf.keras.layers.Add()([x_output, x_res])  # Output shape is the same as input shape  
        return x_output

    def attention_block_2(self, tensor): 
      input_tensor = tensor
      _, H1, W1, C1 = tensor.shape
      # reshaped_tensor = tf.reshape(tensor, shape = [(H * W), C])  
      # transposed_tensor = tf.transpose(reshaped_tensor) # Shape ========> C * (H * W)  
      # x = tf.matmul(transposed_tensor, reshaped_tensor) # Shape ========> C * C
      # x = tf.matmul(reshaped_tensor, x)    
      
      # Channel wise attention 
      tensor = tf.keras.layers.MaxPooling2D(4, 4)(tensor)
    
      _, H, W, C = tensor.shape
      
      ratio = np.sqrt((H * W))
      reshaped_tensor = tf.keras.layers.Reshape(((H * W), C))(tensor)# Shape ======> (H * W) * C
      transposed_tensor = tf.keras.layers.Permute((2, 1))(reshaped_tensor)# Shape ========> C * (H * W)
      x = tf.keras.layers.Dot(axes = (2, 1))([transposed_tensor, reshaped_tensor])  ##############divide by 1 / (sqrt(H * W))
      
      x = tf.keras.layers.Activation('softmax')(x)  # Shape ===========> (C * C)
      x = tf.keras.layers.Dot(axes = (2, 1))([reshaped_tensor, x]) / ratio
    
      # x = tf.keras.layers.Multiply()([x, 1 / (H * W)])
      
      add_1= tf.keras.layers.Reshape((H, W, C))(x)  
      add_1 = tf.keras.layers.experimental.preprocessing.Resizing(H1, W1, interpolation='nearest')(add_1)
      

      # Spatial attention
      # reshaped_tensor = tf.reshape(tensor, shape = [(H * W), C])
      # transposed_tensor = tf.transpose(reshaped_tensor) # Shape ========> C * (H * W)  
      # x = tf.matmul(reshaped_tensor, transposed_tensor) # Shape ==========> (H * W) * (H * W)
      # x = tf.matmul(x, reshaped_tensor) # Shape ==========> (H * W) * C

      reshaped_tensor = tf.keras.layers.Reshape(((H * W), C))(tensor)
      transposed_tensor = tf.keras.layers.Permute((2, 1))(reshaped_tensor)  # Shape ======> (H * W) * C
      x = tf.keras.layers.Dot(axes = (2, 1))([reshaped_tensor, transposed_tensor]) 
      x = tf.keras.layers.Activation('softmax')(x)
      x = tf.keras.layers.Dot(axes = (2, 1))([x, reshaped_tensor]) / ratio ##############divide by 1 / (sqrt(H * W))
      # x = tf.keras.layers.Multiply()([x, 1 / (H * W)])
      # add_1 = tf.keras.layers.experimental.preprocessing.Resizing((W*H), C, interpolation='nearest')(x)     
      add_2 = tf.keras.layers.Reshape((H, W, C))(x)
      add_2 = tf.keras.layers.experimental.preprocessing.Resizing(H1, W1, interpolation='nearest')(add_2)
      
      #add_2 = tf.keras.layers.Add()([tensor, x]) 
#      print("add_1 : ", add_1.shape)
#      print("add_2 : ", add_2.shape)#
    
      mean = tf.multiply(0.5, tf.keras.layers.Add()([add_1, add_2]))  #Shape =========> H * W * C (same as the input shape)
      X_attention2 = tf.keras.layers.Add()([input_tensor, mean])
      return X_attention2
  
    def DAC_block(self, tensor):
      _, W, H, C = tensor.shape
      output1 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 1)(tensor)
      output1 = tf.keras.layers.BatchNormalization()(output1)
      output1 = tf.keras.layers.Activation('relu')(output1)
    
      output2 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 3)(tensor)
      output2 = tf.keras.layers.BatchNormalization()(output2)
      output2 = tf.keras.layers.Activation('relu')(output2)  
      output2 = tf.keras.layers.Conv2D(C, (1, 1), padding = 'same', dilation_rate = 1)(output2)
      output2 = tf.keras.layers.BatchNormalization()(output2)
      output2 = tf.keras.layers.Activation('relu')(output2)   
    
      output3 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 1)(tensor)
      output3 = tf.keras.layers.BatchNormalization()(output3)
      output3 = tf.keras.layers.Activation('relu')(output3)   
      output3 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 3)(output3)
      output3 = tf.keras.layers.BatchNormalization()(output3)
      output3 = tf.keras.layers.Activation('relu')(output3)     
      output3 = tf.keras.layers.Conv2D(C, (1, 1), padding = 'same', dilation_rate = 1)(output3)  
      output3 = tf.keras.layers.BatchNormalization()(output3)
      output3 = tf.keras.layers.Activation('relu')(output3)     
    
      output4 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 1)(tensor)
      output4 = tf.keras.layers.BatchNormalization()(output4)
      output4 = tf.keras.layers.Activation('relu')(output4)    
      output4 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 3)(output4)
      output4 = tf.keras.layers.BatchNormalization()(output4)
      output4 = tf.keras.layers.Activation('relu')(output4)   
      output4 = tf.keras.layers.Conv2D(C, (3, 3), padding = 'same', dilation_rate = 5)(output4)  
      output4 = tf.keras.layers.BatchNormalization()(output4)
      output4 = tf.keras.layers.Activation('relu')(output4)  
      output4 = tf.keras.layers.Conv2D(C, (1, 1), padding = 'same', dilation_rate = 1)(output4)  
      output4 = tf.keras.layers.BatchNormalization()(output4)
      output4 = tf.keras.layers.Activation('relu')(output4)   
    
      main_output = tf.keras.layers.Add()([output1, output2, output3, output4])
      
      return main_output    

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
      # conv1 = tf.image.resize(tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_1), size = (W, H), method = 'bilinear')
      # conv2 = tf.image.resize(tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_2), size = (W, H), method = 'bilinear')
      # conv3 = tf.image.resize(tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_3), size = (W, H), method = 'bilinear')
      # conv4 = tf.image.resize(tf.keras.layers.Conv2D(1, (1, 1), padding = 'same', activation = 'relu')(pool_4), size = (W, H), method = 'bilinear')
      main_output = tf.keras.layers.concatenate([tensor, conv1, conv2, conv3, conv4])
      ############################################# IMPORTANT NOTE #############################################
      # main_output = tf.keras.layers.Conv2D(C, (1, 1), activation = 'relu')(main_output)  ## 256*256*32 ========> 256*256*36
      return main_output
    
    def sa_plus_plus(self, tensor):        
      # att_1 = self.attention_block_1(tensor)  
      main_output = self.attention_block_2(tensor)
      # main_output = tf.keras.layers.concatenate([att_1, att_2], axis = -1)         
      return main_output    
    

    def conv_block(self, input_tensor, num_filters):
      encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
      encoder = tf.keras.layers.BatchNormalization()(encoder)
      encoder = tf.keras.layers.Activation('relu')(encoder)  
      encoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
      encoder = tf.keras.layers.BatchNormalization()(encoder)
      encoder = tf.keras.layers.Activation('relu')(encoder)
      return encoder

    def encoder_block(self, input_tensor, num_filters):
      encoder = self.conv_block(input_tensor, num_filters)
      encoder_pool = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)     
      return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters, name):
      decoder = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)   
      sa = self.sa_plus_plus(concat_tensor)
      decoder = tf.keras.layers.concatenate([sa, concat_tensor, decoder], axis=-1)
      decoder = tf.keras.layers.BatchNormalization()(decoder)
      decoder = tf.keras.layers.Activation('relu')(decoder)
      decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
      decoder = tf.keras.layers.BatchNormalization()(decoder)
      decoder = tf.keras.layers.Activation('relu')(decoder)  
      decoder = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
      decoder = tf.keras.layers.BatchNormalization()(decoder)
      decoder = tf.keras.layers.Activation('relu', name = name)(decoder)
      return decoder
  

    def multi_unet_model(self):
        # UNET Model
        inputs = tf.keras.layers.Input(shape = self.input_shape)
        # 256    
        encoder0_pool, encoder0 = self.encoder_block(inputs, 32)
        # 128
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 64)
        # 64
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 128)
        # 32
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 256)
        # 16
        encoder4_pool, encoder4 = self.encoder_block(encoder3_pool, 512)
        # 8
        sa = self.sa_plus_plus(encoder4_pool)
        center = self.conv_block(sa, 1024)
        # center = tf.keras.layers.SpatialDropout2D(0.25)(center)
        rmp = self.rmp_block(center)
        # center
        decoder4 = self.decoder_block(rmp, encoder4, 512, "decoder_4")
        # output 4
        output4 = tf.keras.layers.UpSampling2D(size = (16, 16))(decoder4)
        output4 = tf.keras.layers.Conv2D(self.num_class, (1, 1), activation = 'softmax', name = 'output4')(output4)
        # 16
        decoder3 = self.decoder_block(decoder4, encoder3, 256, "decoder_3")
        output3 = tf.keras.layers.UpSampling2D(size = (8, 8))(decoder3)
        output3 = tf.keras.layers.Conv2D(self.num_class, (1, 1), activation = 'softmax', name = 'output3')(output3)
        # 32
        decoder2 = self.decoder_block(decoder3, encoder2, 128, "decoder_2")
        #output 2
        output2 = tf.keras.layers.UpSampling2D(size = (4, 4))(decoder2)
        output2 = tf.keras.layers.Conv2D(self.num_class, (1, 1), activation = 'softmax', name = 'output2')(output2)
        # 64
        decoder1 = self.decoder_block(decoder2, encoder1, 64, "decoder_1")
        #output 1
        output1 = tf.keras.layers.UpSampling2D(size = (2, 2))(decoder1)
        output1 = tf.keras.layers.Conv2D(self.num_class, (1, 1), activation = 'softmax', name = 'output1')(output1)       
        # 128
        decoder0 = self.decoder_block(decoder1, encoder0, 32, "decoder_0")
        # 256
        outputs = tf.keras.layers.Conv2D(self.num_class, (1, 1), name = 'main_output__')(decoder0)
        outputs = tf.keras.layers.Activation("softmax")(outputs)
        model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs, output4, output3, output2, output1])  
        return model
    
    def __call__(self):        
        model =  self.multi_unet_model()
        return model



def create_model():
  o_model = Unet(4)()  
 

  num_class = 4

  for layer in o_model.layers:
    layer.trainable = 0

  output1  = o_model.get_layer('output1').output
  output2  = o_model.get_layer('output2').output 
  output3  = o_model.get_layer('output3').output
  output4  = o_model.get_layer('output4').output
  decoder0 = o_model.get_layer('decoder_0').output
  outputs = tf.keras.layers.Conv2D(num_class, (1, 1), name = 'main_output')(decoder0)

  out8_1 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(outputs[: ,:, :, 0], axis = -1))
  out8_2 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(outputs[: ,:, :, 1], axis = -1))
  out8_3 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(outputs[: ,:, :, 2], axis = -1))
  out8_4 = tf.keras.layers.Conv2D(8, (1, 1), activation = "sigmoid")(tf.expand_dims(outputs[: ,:, :, 3], axis = -1))        

  main_outputs_ =  tf.keras.layers.concatenate([out8_1, out8_2, out8_3, out8_4], name = 'bicon_outputs')    
  outputs = tf.keras.layers.Activation("softmax",  name = 'main_output_')(outputs)
  outputs_to_return = tf.keras.layers.concatenate([main_outputs_, outputs, output4, output3, output2, output1])
  model = tf.keras.models.Model(inputs=[o_model.input], outputs = outputs_to_return)  
  return model

model = create_model()
#model.summary()


unet = Unet(4)()
# unet.load_weights("/model_3_epoch30.hdf5")

#unet.summary()



def copyModel2Model(model_source,model_target,certain_layer=""):        
    for l_tg,l_sr in zip(model_target.layers,model_source.layers):
        wk0=l_sr.get_weights()
        l_tg.set_weights(wk0)
        if l_tg.name==certain_layer:
            break
    print("model source was copied into model target") 
copyModel2Model(unet,model,"main_output")



print(model.get_layer("conv2d").get_weights())
print("************************************************************/n")
print(unet.get_layer("conv2d_30").get_weights())

print("weigths loaded")


















