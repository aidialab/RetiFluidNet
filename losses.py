import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import time

class Losses:
    
    def __init__(self):
        self.loss =  tf.keras.losses.BinaryCrossentropy()
    
    @tf.function   
    def dice_coeff(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = 4), [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    @tf.function
    def dice_coeff_bicon(self, y_true, y_pred):
        smooth = 1.
        # Flatten
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        return score
    
    @tf.function 
    def iou_loss(self,y_true, y_pred):
        y_true_f=tf.reshape(tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = 4), [-1])
        y_pred_f=tf.reshape(y_pred, [-1])
        inter=tf.reduce_sum(tf.multiply(y_pred_f,y_true_f))
        union=tf.reduce_sum(tf.subtract (tf.add(y_pred_f,y_true_f),tf.multiply(y_pred_f,y_true_f)))
        loss=tf.subtract(tf.constant(1.0, dtype=tf.float32),tf.divide(inter,union))
        return loss
    
    @tf.function    
    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return loss
    
    @tf.function    
    def dice_loss_bicon(self, y_true, y_pred):
        loss = 1 - self.dice_coeff_bicon(y_true, y_pred)
        return loss
    
    @tf.function
    def dice_loss_scale4(self, y_true, y_pred):
        alpha = 1/8.0
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return alpha*loss
    
    @tf.function
    def dice_loss_scale3(self, y_true, y_pred):
        alpha = 1/4.0
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return alpha*loss
    
    @tf.function
    def dice_loss_scale2(self, y_true, y_pred):
        alpha = 1/2.0
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return alpha*loss
    
    @tf.function
    def dice_loss_scale1(self, y_true, y_pred):
        alpha = 1/1.0
        loss = 1 - self.dice_coeff(y_true, y_pred)
        return alpha*loss
    
    @tf.function
    def bce_dice_loss(self,y_true, y_pred):
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + self.dice_loss(y_true, y_pred) + self.iou_loss(y_true, y_pred)
        return loss
    
    def tversky(self, y_true, y_pred):
        smooth=1
        alpha=0.99
        y_true_pos = tf.reshape(tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = 4), [-1])
        y_pred_pos = tf.reshape(y_pred, [-1])      
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    def tversky_loss(self, y_true, y_pred):
        return 1 - self.tversky(y_true, y_pred)

    def focal_tversky_loss(self, y_true,y_pred):
        pt_1 = self.tversky(y_true, y_pred)       
        gamma = 0.75
        focal_weight = K.clip((1-pt_1), K.epsilon(), 1.0)
        return K.pow(focal_weight, gamma)

    def Focaltver_CEsparse_loss(self, y_true, y_pred):
        loss = 1 * (tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + 0.5 * self.focal_tversky_loss(y_true, y_pred))
        return loss

    def gen_dice(self, y_true, y_pred, eps=1e-6):
        """both tensors are [b, h, w, classes] and y_pred is in logit form"""

        # [b, h, w, classes]
        y_true = tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = 4)
        y_true_shape = tf.shape(y_true)

        # [b, h*w, classes]
        y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
        y_pred = tf.reshape(y_pred, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

        # [b, classes]
        # count how many of each class are present in 
        # each image, if there are zero, then assign
        # them a fixed weight of eps
        counts = tf.reduce_sum(y_true, axis=1)
        weights = 1. / (counts ** 4)
        weights = tf.where(tf.math.is_finite(weights), weights, eps)

        multed = tf.reduce_sum(y_true * y_pred, axis=1)
        summed = tf.reduce_sum(y_true + y_pred, axis=1)

        # [b]
        numerators = tf.reduce_sum(weights*multed, axis=-1)
        denom = tf.reduce_sum(weights*summed, axis=-1)
        dices = 1. - 2. * numerators / denom
        dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
        return tf.reduce_mean(dices)
    
    @tf.function
    def edge_loss(self, glo_map,vote_out,edge,target):
      #  start = time.time()
        pred_mask_min  = tf.math.reduce_min(vote_out , axis = -1)
        # pred_mask_max  = tf.math.reduce_max(vote_out , axis = -1)
        pred_mask_min  = 1 - pred_mask_min
        pred_mask_min  = pred_mask_min * edge
        decouple_map   = glo_map*(1-edge)+pred_mask_min
       # stop = time.time()
        #print("Esp Time For edge_loss : ", stop - start)
        minloss = self.loss(target, decouple_map)
        return minloss
               
    @tf.function          
    def Bilater_voting(self, c_map,hori_translation,verti_translation):
        
        #start = time.time()
        _, row, column, channel = c_map.shape
        #vote_out = tf.zeros(shape = [row, column, channel])
        
        right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
        left         = tf.matmul (c_map[:, :, :, 3],tf.transpose(hori_translation, perm = [1, 0]))
        left_bottom  = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:, :,:, 5]) 
        left_bottom  = tf.matmul(left_bottom,tf.transpose(hori_translation, perm = [1, 0]))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,tf.transpose(hori_translation, perm = [1, 0]))
        bottom       = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 7])
        right_bottom = tf.matmul(right_bottom,hori_translation)
        
        a1 = tf.multiply(c_map[:,:,:, 3], right)       
        a2 = tf.multiply(c_map[:,:,:, 4], left)
        a3 = tf.multiply(c_map[:,:,:, 1], bottom)
        a4 = tf.multiply(c_map[:,:,:, 6], up)
        a5 = tf.multiply(c_map[:,:,:, 2], left_bottom)
        a6 = tf.multiply(c_map[:,:,:, 5], right_above)
        a7 = tf.multiply(c_map[:,:,:, 0], right_bottom)
        a8 = tf.multiply(c_map[:,:,:, 7], left_above)
        
        vote_out = tf.stack([a7, a3, a5, a1, a2, a6, a4, a8], axis = -1)
       # stop = time.time()
        #print("Esp Time For Bilater_voting : ", stop - start)
        return vote_out  
    
    def sal2conn(self, mask):
        ## converte the saliency mask into a connectivity mask
        ## mask shape: H*W, output connectivity shape: H*W*8
        mask        = np.squeeze(np.array(mask))        
        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, 0)
            
        batch, rows, cols  = mask.shape
        conn        = np.zeros(shape = (batch, rows, cols, 8))
        up          = np.zeros(shape = (batch, rows, cols))#move the orignal mask to up
        down        = np.zeros(shape = (batch, rows, cols))
        left        = np.zeros(shape = (batch, rows, cols))
        right       = np.zeros(shape = (batch, rows, cols))
        up_left     = np.zeros(shape = (batch, rows, cols))
        up_right    = np.zeros(shape = (batch, rows, cols))
        down_left   = np.zeros(shape = (batch, rows, cols))
        down_right  = np.zeros(shape = (batch, rows, cols))
    
        up[:,:rows-1, :]             = mask[:,1:rows,:]
        down[:,1:rows,:]             = mask[:,0:rows-1,:]
        left[:,:,:cols-1]            = mask[:,:,1:cols]
        right[:,:,1:cols]            = mask[:,:,:cols-1]
        up_left[:,0:rows-1,0:cols-1] = mask[:,1:rows,1:cols]
        up_right[:,0:rows-1,1:cols]  = mask[:,1:rows,0:cols-1]
        down_left[:,1:rows,0:cols-1] = mask[:,0:rows-1,1:cols]
        down_right[:,1:rows,1:cols]  = mask[:,0:rows-1,0:cols-1]  
        
        conn[:,:,:,0] = mask*down_right
        conn[:,:,:,1] = mask*down
        conn[:,:,:,2] = mask*down_left
        conn[:,:,:,3] = mask*right
        conn[:,:,:,4] = mask*left
        conn[:,:,:,5] = mask*up_right
        conn[:,:,:,6] = mask*up
        conn[:,:,:,7] = mask*up_left
        conn = conn.astype(np.float32)
        
        return conn
    
    @tf.function
    def tf_sal2conn(self, mask):
        #start  = time.time()
        output = tf.numpy_function(self.sal2conn, [mask], tf.float32) 
        #stop   = time.time()
        #print("Esp Time For tf_sal2conn : ", stop - start)
        return output 

    def numpy_full_like(self, x, y):
        return np.full_like(x, y)
    
    @tf.function
    def tf_full_like(self, x, y):
        return tf.numpy_function(self.numpy_full_like, [x, y], tf.float32)          

                
    @tf.function
    def bicon_loss_new(self, y_ture, y_pred):
        #y_ture shape : batchsize * 256 * 256
        # start = time.time()
        
        # start0 = time.time()
        y_ture8 = self.tf_sal2conn(y_ture) #Output Shape : batch_size * 256 * 256 * 8 
        # stop0 = time.time()
        #print("tf_sal2conn time : ",  stop0 - start0)   
        #find edge GT
       # 
        # start1 = time.time()
        sum_conn = tf.math.reduce_sum(y_pred, axis = -1)
        edge  = tf.where(sum_conn < 8, self.tf_full_like(sum_conn, 1), self.tf_full_like(sum_conn, 0))
        edge1 = tf.where(sum_conn > 0, self.tf_full_like(sum_conn, 1), self.tf_full_like(sum_conn, 0))           
        edge  = tf.multiply(edge, edge1)
        # stop1   = time.time()
        #print("find edge gt time : ",  stop1 - start1)  

        # construct the translation matrix
        # start2 = time.time()
        hori_translation = np.zeros(shape  = (y_pred.shape[2],y_pred.shape[2]), dtype = np.float32)
        for i in range(y_pred.shape[2]-1):
            hori_translation[i,i+1] = 1

        verti_translation = np.zeros(shape = (y_pred.shape[1],y_pred.shape[1]), dtype = np.float32)
        for j in range(y_pred.shape[1]-1):
            verti_translation[j,j+1] = 1
        # stop2   = time.time()
       # print("construct the translation matrix time : ",  stop2 - start2) 

        # start3 = time.time()
        vote_out = self.Bilater_voting(y_pred,hori_translation,verti_translation) # output Shape batchsize * 256 * 256 * 8
        # stop3   = time.time()
       # print("Bilater_voting time : ",  stop3 - start3) 
     
        # start4 = time.time()
        glo_map = tf.math.reduce_max(vote_out, axis = -1) # reduce mean changed to reduce max(based on the paper), output Shape : batchsize * 256 * 256
        # stop4 = time.time()
        #print("glo_map time : ",  stop4 - start4) 


       # start5 = time.time()
        de_loss  = self.edge_loss(glo_map,vote_out,edge,y_ture)   
        # stop5 = time.time()
        #print("de_loss time : ",  stop5 - start5)   

        # start6 = time.time()
        # if include_dice == True:
        #     dice_loss = self.dice_loss_bicon(y_ture, glo_map)
        dice_loss = self.dice_loss_bicon(y_ture, glo_map)
        # stop6 = time.time()
       # print("dice_loss_bicon time : ",  stop6 - start6) 

        # start7 = time.time()  
        loss_con_const = self.loss(y_ture8, y_pred)
        # stop7 = time.time()
        #print("loss_con_const time : ",  stop7 - start7)  
        # if include_dice == True:
        #     loss = de_loss + loss_con_const + dice_loss
        # else:    
        #     loss = de_loss + loss_con_const 
        # loss = de_loss + loss_con_const
        loss = 0.5 * (de_loss) + loss_con_const + dice_loss
        # loss = dice_loss
        # stop   = time.time()
       # print("Esp Time To Calculate one step bicon loss : ", stop - start)    
        return loss

    @tf.function
    def training_loss(self, y_true, y_pred):
       
        start = time.time()
        y_true_bicon = tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = 4)

        y_true_bicon = tf.squeeze(y_true_bicon) # batchsize * 256 * 256 * 4
        
        # bicon_output_0 = y_pred[:, :, :, 0:32]
        # bicon_output_1 = y_pred[:, :, :, 32:64]
        # bicon_output_2 = y_pred[:, :, :, 64:96]
        # bicon_output_3 = y_pred[:, :, :, 96:128]
        # bicon_output_4 = y_pred[:, :, :, 128:160]
        
        outputs = y_pred[:, :, :, 160:164] #Side output of the main output
        output4 = y_pred[:, :, :, 164:168] #Side output4
        output3 = y_pred[:, :, :, 168:172] #Side output3
        output2 = y_pred[:, :, :, 172:176] #Side output2
        output1 = y_pred[:, :, :, 176:180] #Side output1     

        start = time.time()
        bicon_loss = {}
        for i in range(5):
            bicon_output   = y_pred[:, :, :, i * 32 : (i + 1) * 32]
            
            output_bicon_1 = bicon_output[:, :, :, :8]    #Background, shape : batchsize * 256 * 256 * 8
            output_bicon_2 = bicon_output[:, :, :, 8:16]  #IRF
            output_bicon_3 = bicon_output[:, :, :, 16:24] #SRF
            output_bicon_4 = bicon_output[:, :, :, 24:32] #PED
            loss_bicon_1 = self.bicon_loss_new(y_true_bicon[:, :, :, 0], output_bicon_1) #Background with Layer 0 of ground truth, shape of y_true_bicon[:, :, :, 0] : batchsize * 256 * 256
            loss_bicon_2 = self.bicon_loss_new(y_true_bicon[:, :, :, 1], output_bicon_2) #IRF with Layer 1 of ground truth, shape of y_true_bicon[:, :, :, 1]        : batchsize * 256 * 256
            loss_bicon_3 = self.bicon_loss_new(y_true_bicon[:, :, :, 2], output_bicon_3) #SRF with Layer 2 of ground truth, shape of y_true_bicon[:, :, :, 2]        : batchsize * 256 * 256
            loss_bicon_4 = self.bicon_loss_new(y_true_bicon[:, :, :, 3], output_bicon_4) #PED with Layer 3 of ground truth, shape of y_true_bicon[:, :, :, 3]        : batchsize * 256 * 256
            bicon_loss["decoder_" + str(i)] = 1.   * (loss_bicon_1 + loss_bicon_2 + loss_bicon_3 + loss_bicon_4)
         
        stop = time.time() 
        print("Esp Time To Calculate bicon Loss is {}".format(stop - start))
        Multi_scale_bicon_loss = (1/8.0) * bicon_loss["decoder_4"] + (1/4.0) * bicon_loss["decoder_3"] + (1/2.0) * bicon_loss["decoder_2"] + bicon_loss["decoder_1"] + bicon_loss["decoder_0"]
        
        loss_side_main = self.dice_loss(y_true, outputs) #Main output
        loss_side_1    = self.dice_loss(y_true, output1) #output of the encoder 1
        loss_side_2    = self.dice_loss(y_true, output2) #output of the encoder 2
        loss_side_3    = self.dice_loss(y_true, output3) #output of the encoder 3
        loss_side_4    = self.dice_loss(y_true, output4) #output of the encoder 4
        
        Multi_scale_dice_loss  = 1 * loss_side_main + 1 * loss_side_1 + (1/2.0) * loss_side_2 + (1/4.0) * loss_side_3 + (1/8.0) * loss_side_4
        joint_loss = 0.05 * (Multi_scale_bicon_loss) + 1.0 * Multi_scale_dice_loss
        # joint_loss = Multi_scale_bicon_loss
        stop = time.time()
        print("Esp Time To Calculate Loss is {}".format(stop - start))
        return joint_loss


    @tf.function    
    def dice(self, y_true, y_pred):

        y_pred = y_pred[:, :, :, 0:32]
        output_bicon_1 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, :8]))      # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 1
        output_bicon_2 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, 8:16]))    # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 2
        output_bicon_3 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, 16:24]))   # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 3
        output_bicon_4 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, 24:32]))   # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 4
        
        pred_data = tf.concat([output_bicon_1, output_bicon_2,output_bicon_3, output_bicon_4], axis = -1) # Concat to form new  prediction tensor, output shape  : (256,256,4)
              
        dice = self.dice_coeff(y_true, pred_data) # Calculate Dice coeff
        return dice
    
    @tf.function    
    def dice2(self, y_true, y_pred):

        y_true_bicon = tf.keras.backend.one_hot(tf.cast(y_true, 'int32'),num_classes = 4)
        y_true_bicon = tf.squeeze(y_true_bicon) # batchsize * 256 * 256 * 4
        y_true_bicon = tf.expand_dims(y_true_bicon, axis = 0)
        y_pred = y_pred[:, :, :, 0:32]
        output_bicon_1 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, :8]))      # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 1
        output_bicon_2 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, 8:16]))    # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 2
        output_bicon_3 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, 16:24]))   # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 3
        output_bicon_4 = tf.convert_to_tensor(self.bv_test_new(y_pred[:, :, :, 24:32]))   # Input shape : (256,256,8) -> output shape : (256,256,1), Layer 4

        dice_coeff_layer1 = self.dice_coeff_bicon(y_true_bicon[:, :, :, 0], output_bicon_1)
        dice_coeff_layer2 = self.dice_coeff_bicon(y_true_bicon[:, :, :, 1], output_bicon_2)
        dice_coeff_layer3 = self.dice_coeff_bicon(y_true_bicon[:, :, :, 2], output_bicon_3)
        dice_coeff_layer4 = self.dice_coeff_bicon(y_true_bicon[:, :, :, 3], output_bicon_4)

        print("Background Dice : ", dice_coeff_layer1)
        print("IRF Dice        : ", dice_coeff_layer2)
        print("SRF Dice        : ", dice_coeff_layer3)
        print("PED Dice        : ", dice_coeff_layer4)
              
        dice = dice_coeff_layer1 + dice_coeff_layer2 + dice_coeff_layer3 + dice_coeff_layer4
        return dice
    

    @tf.function           
    def ConMap2Mask_prob(self, c_map,hori_translation,verti_translation):
        '''
        continuous bilateral voting
        '''

        _, row, column, channel = c_map.shape
        vote_out = tf.zeros(shape = [row, column, channel])
    
        right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
        left         = tf.matmul (c_map[:, :, :, 3],tf.transpose(hori_translation, perm = [1, 0]))
        left_bottom  = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:, :,:, 5]) 
        left_bottom  = tf.matmul(left_bottom,tf.transpose(hori_translation, perm = [1, 0]))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,tf.transpose(hori_translation, perm = [1, 0]))
        bottom       = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 7])
        right_bottom = tf.matmul(right_bottom,hori_translation)
        
     
        a1 = (c_map[:,:,:, 3]) * (right)        
        a2 = (c_map[:,:,:, 4]) * (left)
        a3 = (c_map[:,:,:, 1]) * (bottom)
        a4 = (c_map[:,:,:, 6]) * (up)
        a5 = (c_map[:,:,:, 2]) * (left_bottom)
        a6 = (c_map[:,:,:, 5]) * (right_above)
        a7 = (c_map[:,:,:, 0]) * (right_bottom)
        a8 = (c_map[:,:,:, 7]) * (left_above)
        
        vote_out = tf.stack([a7, a3, a5, a1, a2, a6, a4, a8], axis = -1)
        
        pred_mask = tf.math.reduce_mean(vote_out, axis=-1)
        pred_mask = tf.squeeze(pred_mask)
        return pred_mask        
            
    @tf.function 
    def ConMap2Mask_prob_new(self, c_map,hori_translation,verti_translation):
        '''
        continuous bilateral voting
        '''
        # if len(c_map.shape) == 3:
        #     c_map = tf.expand_dims(c_map, 0)   
        
        # print("cmap : ", c_map.shape)
        _, row, column, channel = c_map.shape
        # c_map = tf.squeeze(c_map)
        vote_out = tf.zeros(shape = [row, column, channel])
    
        # print(c_map[1,4].shape)
        right        = tf.matmul(c_map[ :, :, :, 4],hori_translation)
        left         = tf.matmul (c_map[:, :, :, 3],tf.transpose(hori_translation, perm = [1, 0]))
        left_bottom  = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:, :,:, 5]) 
        left_bottom  = tf.matmul(left_bottom,tf.transpose(hori_translation, perm = [1, 0]))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,tf.transpose(hori_translation, perm = [1, 0]))
        bottom       = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(tf.transpose(verti_translation, perm = [1, 0]), c_map[:,:,:, 7])
        right_bottom = tf.matmul(right_bottom,hori_translation)
        
        
        # print(a1[0][0][100])
        a1 = (c_map[:,:,:, 3]) * (right)        
        a2 = (c_map[:,:,:, 4]) * (left)
        a3 = (c_map[:,:,:, 1]) * (bottom)
        a4 = (c_map[:,:,:, 6]) * (up)
        a5 = (c_map[:,:,:, 2]) * (left_bottom)
        a6 = (c_map[:,:,:, 5]) * (right_above)
        a7 = (c_map[:,:,:, 0]) * (right_bottom)
        a8 = (c_map[:,:,:, 7]) * (left_above)
        
        vote_out = tf.stack([a7, a3, a5, a1, a2, a6, a4, a8], axis = -1)
        
        pred_mask = tf.math.reduce_max(vote_out, axis=-1) #tf.math.reduce_mean changed to tf.math.reduce_max
        pred_mask = tf.squeeze(pred_mask)
        return pred_mask        
            
    @tf.function         
    def bv_test(self, y_pred):
        '''
        generate the continous global map from output connectivity map as final saliency output 
        via bilateral voting
        '''        
        #construct the translation matrix
        hori_translation = np.zeros(shape  = (y_pred.shape[2],y_pred.shape[2]), dtype = np.float32)
        for i in range(y_pred.shape[2]-1):
            hori_translation[i,i+1] = 1
        verti_translation = np.zeros(shape = (y_pred.shape[1],y_pred.shape[1]), dtype = np.float32)
        for j in range(y_pred.shape[1]-1):
            verti_translation[j,j+1] = 1    
    
        pred = self.ConMap2Mask_prob(y_pred,hori_translation,verti_translation)
        return pred            
    
    @tf.function             
    def bv_test_new(self, y_pred):
        '''
        generate the continous global map from output connectivity map as final saliency output 
        via bilateral voting
        '''        
        #construct the translation matrix
        hori_translation = np.zeros(shape  = (y_pred.shape[2],y_pred.shape[2]), dtype = np.float32)
        for i in range(y_pred.shape[2]-1):
            hori_translation[i,i+1] = 1
        verti_translation = np.zeros(shape = (y_pred.shape[1],y_pred.shape[1]), dtype = np.float32)
        for j in range(y_pred.shape[1]-1):
            verti_translation[j,j+1] = 1    
    
        pred = self.ConMap2Mask_prob_new(y_pred,hori_translation,verti_translation)
        return pred    
                    
from tensorflow.keras.callbacks import  Callback
# from tqdm import tqdm
losses = Losses()
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=()):

        super(Callback, self).__init__()

        self.X_val, self.y_val = [], []
        for image, mask in validation_data:
            self.X_val.append(image)
            self.y_val.append(mask) 
        print("Data Is Ready!")

    def on_epoch_end(self, epoch, logs={}):
        
        dices = []
        for i in range(len(self.X_val)):
            y_pred = self.model.predict(self.X_val[i])
            score = losses.dice(self.y_val[i], y_pred)
            dices.append(score)
        print(" - val_Dice: {:.6f} ".format(np.mean(dices)))
