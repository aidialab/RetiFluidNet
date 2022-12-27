import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from sklearn import  metrics
from tqdm import tqdm


class Results:
    
    def __init__(self):
        self.layers_names = ['background', 'IRF', 'SRF', 'PED']
    
    
    def recall(self, predictions, labels):  
        #print("recall")      
        recall_ = {}            
        for i in range(labels.shape[-1]):            
            label_1D = K.flatten(labels[:, :, i])
            pred_1D  = K.flatten(predictions[:, :, i])
            recall_[self.layers_names[i]] = metrics.recall_score(label_1D,pred_1D, labels=np.unique(pred_1D))                
            
        return recall_
    
    
    
    def precision(self, predictions, labels):     
        #print("precision")   
        precision_ = {}        
        for i in range(labels.shape[-1]):            
            label_1D = K.flatten(labels[:, :, i])
            pred_1D  = K.flatten(predictions[:, :, i])
            precision_[self.layers_names[i]] = metrics.precision_score(label_1D,pred_1D, labels=np.unique(pred_1D))                
            
        return precision_
       
    
    
    def accuracy(self, predictions, labels):
        #print("Accuracy")        
        accuracy_ = {}        
        for i in range(labels.shape[-1]):            
            label_1D = K.flatten(labels[:, :, i])
            pred_1D  = K.flatten(predictions[:, :, i])
            accuracy_[self.layers_names[i]] = metrics.accuracy_score(label_1D,pred_1D)                
            
        return accuracy_
       
     
    
    def f1_score(self, predictions, labels):   
        #print("F1_Score")     
        f1_score_ = {}        
        for i in range(labels.shape[-1]):            
            label_1D = K.flatten(labels[:, :, i])
            pred_1D  = K.flatten(predictions[:, :, i])
            f1_score_[self.layers_names[i]] = metrics.f1_score(label_1D,pred_1D, labels=np.unique(pred_1D))         
            
        return f1_score_
    
    
    def balanced_acc(self, predictions, labels):  
        #print("BAcc")      
        balanced_accuracy_ = {}        
        for i in range(labels.shape[-1]):            
            label_1D = K.flatten(labels[:, :, i])
            pred_1D  = K.flatten(predictions[:, :, i])
            balanced_accuracy_[self.layers_names[i]] = metrics.balanced_accuracy_score(label_1D,pred_1D)                
            
        return balanced_accuracy_
       
    
    def dice_coef(self, predictions, labels, smooth=1):
        #print("dice")
        #labels = labels[0]
        #labels = tf.keras.utils.to_categorical(labels, num_classes = 4) #496,64,1 ====> 496,64,4
        dice = {}
        for i in range(labels.shape[-1]):
            label_1D = K.flatten(labels[:, :, i])
            pred_1D  = K.flatten(predictions[:, :, i])
            intersection = K.sum(label_1D * pred_1D)            
            try:
                dice_coff = (2. * intersection + smooth) / (K.sum(label_1D) + K.sum(pred_1D) + smooth)
                dice[self.layers_names[i]] = float(dice_coff)
            except:
                dice_coff = 0
                dice[self.layers_names[i]] = dice_coff
        return dice
    
    def iou(self, predictions, labels, smooth=1):
        iou = {}
        for i in range(labels.shape[-1]):
          label_1D = K.flatten(labels[:, :, i])
          pred_1D  = K.flatten(predictions[:, :, i])
          intersection = K.sum(label_1D * pred_1D)
          union = K.sum(label_1D + pred_1D - (label_1D * pred_1D))
                
          try:
              iou_coff = (intersection + smooth) / (union + smooth)
              iou[self.layers_names[i]] = float(iou_coff)
          except:
              iou_coff = 0
              iou[self.layers_names[i]] = iou_coff
        return iou 
           
    
    def results_per_layer(self, predictions, labels):
        
        predictions = np.array(predictions)
        print(predictions.shape)
        predictions = np.squeeze(predictions)
        num_samples, W, H, C = predictions.shape
        pred_data = np.zeros(shape = (num_samples, 256, 256, 4))
        for i in range(num_samples):
            sample = predictions[i]
            sample = np.expand_dims(sample, axis = 0)
            output_bicon_1 = self.bv_test(sample[:, :, :, :8])    #Layer 1
            output_bicon_2 = self.bv_test(sample[:, :, :, 8:16])  #Layer 2
            output_bicon_3 = self.bv_test(sample[:, :, :, 16:24]) #Layer 3
            output_bicon_4 = self.bv_test(sample[:, :, :, 24:32]) #Layer 4
            pred_data[i, :, :, 0] = np.expand_dims(output_bicon_1, axis = 0)
            pred_data[i, :, :, 1] = np.expand_dims(output_bicon_2, axis = 0)
            pred_data[i, :, :, 2] = np.expand_dims(output_bicon_3, axis = 0)
            pred_data[i, :, :, 3] = np.expand_dims(output_bicon_4, axis = 0)
            
        predictions = pred_data
        predictions = np.squeeze(predictions)        
        predictions = np.argmax(predictions, axis = -1)
        predictions = np.array(tf.keras.utils.to_categorical(predictions , 4))
         

        acc_value, dice_value, f1_score_value, bacc_value, precision_value, recall_value, iou_value  = [], [], [], [], [], [], []
        acc_dict, dice_dict, f1_score_dict, bacc_dict, precision_dict, recall_dict, iou_dict         = {}, {}, {}, {}, {}, {}, {}
        acc_mean, dice_mean, f1_score_mean, bacc_mean, precision_mean, recall_mean, iou_mean         = {}, {}, {}, {}, {}, {}, {}
        for i in range(4):            
            acc_dict[self.layers_names[i]]          = 0            
            dice_dict[self.layers_names[i]]         = 0
            bacc_dict[self.layers_names[i]]         = 0
            f1_score_dict[self.layers_names[i]]     = 0
            precision_dict[self.layers_names[i]]    = 0
            recall_dict[self.layers_names[i]]       = 0
            iou_dict[self.layers_names[i]]          = 0
            
        
        labels_ = []
        for image, label in labels:    
            label = tf.squeeze(label)
            labels_.append(tf.keras.backend.one_hot(tf.cast(label, 'int32'),num_classes = 4)) 
        print("\n")    
        for predicted_mask in tqdm(range(predictions.shape[0])):
            acc_value.append(self.accuracy(predictions[predicted_mask], labels_[predicted_mask]))
            f1_score_value.append(self.f1_score(predictions[predicted_mask], labels_[predicted_mask]))
            precision_value.append(self.precision(predictions[predicted_mask], labels_[predicted_mask]))
            bacc_value.append(self.balanced_acc(predictions[predicted_mask], labels_[predicted_mask]))
            # specificity_value.append(self.speceficity(predictions[predicted_mask], labels_[predicted_mask]))
            recall_value.append(self.recall(predictions[predicted_mask], labels_[predicted_mask]))            
            dice_value.append(self.dice_coef(predictions[predicted_mask], labels_[predicted_mask], 1))
            iou_value.append(self.iou(predictions[predicted_mask], labels_[predicted_mask]))
        number_of_samples = len(acc_value)
        for i in range(number_of_samples):
                
            acc_dict[self.layers_names[0]] += acc_value[i][self.layers_names[0]]
            acc_dict[self.layers_names[1]] += acc_value[i][self.layers_names[1]]
            acc_dict[self.layers_names[2]] += acc_value[i][self.layers_names[2]]
            acc_dict[self.layers_names[3]] += acc_value[i][self.layers_names[3]]
            
            f1_score_dict[self.layers_names[0]] += f1_score_value[i][self.layers_names[0]]
            f1_score_dict[self.layers_names[1]] += f1_score_value[i][self.layers_names[1]]
            f1_score_dict[self.layers_names[2]] += f1_score_value[i][self.layers_names[2]]
            f1_score_dict[self.layers_names[3]] += f1_score_value[i][self.layers_names[3]]

            precision_dict[self.layers_names[0]] += precision_value[i][self.layers_names[0]]
            precision_dict[self.layers_names[1]] += precision_value[i][self.layers_names[1]]
            precision_dict[self.layers_names[2]] += precision_value[i][self.layers_names[2]]
            precision_dict[self.layers_names[3]] += precision_value[i][self.layers_names[3]]

            bacc_dict[self.layers_names[0]] += bacc_value[i][self.layers_names[0]]
            bacc_dict[self.layers_names[1]] += bacc_value[i][self.layers_names[1]]
            bacc_dict[self.layers_names[2]] += bacc_value[i][self.layers_names[2]]
            bacc_dict[self.layers_names[3]] += bacc_value[i][self.layers_names[3]]            
            
            
            dice_dict[self.layers_names[0]] += dice_value[i][self.layers_names[0]]
            dice_dict[self.layers_names[1]] += dice_value[i][self.layers_names[1]]
            dice_dict[self.layers_names[2]] += dice_value[i][self.layers_names[2]]
            dice_dict[self.layers_names[3]] += dice_value[i][self.layers_names[3]]  
              
            
            recall_dict[self.layers_names[0]] += recall_value[i][self.layers_names[0]]
            recall_dict[self.layers_names[1]] += recall_value[i][self.layers_names[1]]
            recall_dict[self.layers_names[2]] += recall_value[i][self.layers_names[2]]
            recall_dict[self.layers_names[3]] += recall_value[i][self.layers_names[3]]      
            
            iou_dict[self.layers_names[0]] += iou_value[i][self.layers_names[0]]
            iou_dict[self.layers_names[1]] += iou_value[i][self.layers_names[1]]
            iou_dict[self.layers_names[2]] += iou_value[i][self.layers_names[2]]
            iou_dict[self.layers_names[3]] += iou_value[i][self.layers_names[3]]                  
            
            
        for layer_name in self.layers_names:
            
            print("----------------- " + layer_name + " -----------------")
            # print("Recall            : ", recall_dict[layer_name]  / number_of_samples)
            print("Accuracy          : ", acc_dict[layer_name]  / number_of_samples)
            print("Balaned Accuracy  : ", bacc_dict[layer_name]  / number_of_samples)
            # print("Speceficity       : ", specificity_dict[layer_name] / number_of_samples)
            # print("Precision         : ", precision_dict[layer_name] / number_of_samples)
            # print("F1 score          : ", f1_score_dict[layer_name] / number_of_samples)
            print("Dice              : ", dice_dict[layer_name] / number_of_samples)
            print("Iou               : ", iou_dict[layer_name] / number_of_samples)
            
            
             
            recall_mean[layer_name]  = recall_dict[layer_name]  / number_of_samples
            acc_mean[layer_name]  = acc_dict[layer_name]  / number_of_samples
            bacc_mean[layer_name]  = bacc_dict[layer_name]  / number_of_samples
            #specificity_mean[layer_name]  = specificity_dict[layer_name]  / number_of_samples
            precision_mean[layer_name]  = precision_dict[layer_name]  / number_of_samples
            f1_score_mean[layer_name]  = f1_score_dict[layer_name]  / number_of_samples 
            dice_mean[layer_name]  = dice_dict[layer_name]  / number_of_samples 
            iou_mean[layer_name]  = iou_dict[layer_name]  / number_of_samples 
            
        return acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean
        
    def print_overall_results(self, overall_results, dataset_name="DATABASE"):
        acc_mean, dice_mean, f1_score_mean, precision_mean, bacc_mean, recall_mean, iou_mean   = overall_results[0]
        acc_mean1, dice_mean1, f1_score_mean1, precision_mean1, bacc_mean1, recall_mean1, iou_mean1  = overall_results[1]
        acc_mean2, dice_mean2, f1_score_mean2, precision_mean2, bacc_mean2, recall_mean2, iou_mean2  = overall_results[2]
        result_acc_mean, result_acc_std = {}, {}
        result_bacc_mean, result_bacc_std = {}, {}
        result_recall_mean, result_recall_std = {}, {}
        result_precision_mean, result_precision_std = {}, {}
        result_dice_mean, result_dice_std = {}, {}
        result_f1_mean, result_f1_std = {}, {}
        result_iou_mean, result_iou_std = {}, {}
        # result_specificity_mean, result_specificity_std = {}, {}
        
        for i in range(4):

          result_acc_mean[i] = np.round(np.mean([acc_mean[self.layers_names[i]] * 100.0 , acc_mean1[self.layers_names[i]]* 100.0, acc_mean2[self.layers_names[i]]* 100.0]), 2)
          result_acc_std[i]  = np.round(np.std([acc_mean[self.layers_names[i]] * 100.0 , acc_mean1[self.layers_names[i]] * 100.0, acc_mean2[self.layers_names[i]]* 100.0]), 2)

          result_bacc_mean[i] = np.round(np.mean([bacc_mean[self.layers_names[i]] * 100.0, bacc_mean1[self.layers_names[i]]* 100.0 , bacc_mean2[self.layers_names[i]]* 100.0]), 2)
          result_bacc_std[i]  = np.round(np.std([bacc_mean[self.layers_names[i]] * 100.0, bacc_mean1[self.layers_names[i]]* 100.0 , bacc_mean2[self.layers_names[i]]* 100.0]), 2)

          result_recall_mean[i] = np.round(np.mean([recall_mean[self.layers_names[i]]* 100.0 , recall_mean1[self.layers_names[i]]* 100.0 , recall_mean2[self.layers_names[i]]* 100.0]), 2)
          result_recall_std[i]  = np.round(np.std([recall_mean[self.layers_names[i]] * 100.0, recall_mean1[self.layers_names[i]]* 100.0 , recall_mean2[self.layers_names[i]]* 100.0]), 2)          
          result_precision_mean[i] = np.round(np.mean([precision_mean[self.layers_names[i]]* 100.0 , precision_mean1[self.layers_names[i]] * 100.0, precision_mean2[self.layers_names[i]]* 100.0]), 2)
          result_precision_std[i]  = np.round(np.std([precision_mean[self.layers_names[i]]* 100.0 , precision_mean1[self.layers_names[i]] * 100.0, precision_mean2[self.layers_names[i]]* 100.0]), 2)

          result_dice_mean[i] = np.round(np.mean([dice_mean[self.layers_names[i]]* 100.0 , dice_mean1[self.layers_names[i]] * 100.0, dice_mean2[self.layers_names[i]]* 100.0]), 2)
          result_dice_std[i]  = np.round(np.std([dice_mean[self.layers_names[i]]* 100.0 , dice_mean1[self.layers_names[i]]* 100.0 , dice_mean2[self.layers_names[i]]* 100.0]), 2)

          result_f1_mean[i] = np.round(np.mean([f1_score_mean[self.layers_names[i]]* 100.0 , f1_score_mean1[self.layers_names[i]]* 100.0 , f1_score_mean2[self.layers_names[i]]* 100.0]), 2)
          result_f1_std[i] = np.round(np.std([f1_score_mean[self.layers_names[i]]* 100.0 , f1_score_mean1[self.layers_names[i]] * 100.0, f1_score_mean2[self.layers_names[i]]* 100.0]), 2)                        


          result_iou_mean[i] = np.round(np.mean([iou_mean[self.layers_names[i]] * 100.0, iou_mean1[self.layers_names[i]] * 100.0, iou_mean2[self.layers_names[i]]* 100.0]), 2)
          result_iou_std[i]  = np.round(np.std([iou_mean[self.layers_names[i]] * 100.0, iou_mean1[self.layers_names[i]]* 100.0 , iou_mean2[self.layers_names[i]]* 100.0]), 2)

          # result_specificity_mean[i] = np.round(np.mean([specificity_mean[self.layers_names[i]]* 100.0 , specificity_mean1[self.layers_names[i]] * 100.0, specificity_mean2[self.layers_names[i]]* 100.0]), 2)
          # result_specificity_std[i]  = np.round(np.std([specificity_mean[self.layers_names[i]] * 100.0, specificity_mean1[self.layers_names[i]] * 100.0, specificity_mean2[self.layers_names[i]]* 100.0]), 2)
          
              # print("----------------- " + "OVERALL RESULTS" + " -----------------")
              # print("----------------- " + self.layers_names[i] + " -----------------")
              # print("Accuracy           : " ,  ()
              # print("Balanced Accuracy  : " ,  (bacc_mean[self.layers_names[i]] + bacc_mean1[self.layers_names[i]] + bacc_mean2[self.layers_names[i]]) / 3)
              # print("Recall             : " ,  (recall_mean[self.layers_names[i]] + recall_mean1[self.layers_names[i]] + recall_mean2[self.layers_names[i]]) / 3)
              # print("Precision          : " ,  (precision_mean[self.layers_names[i]] + precision_mean1[self.layers_names[i]] + precision_mean2[self.layers_names[i]]) / 3)
              # print("Specificity        : " ,  (specificity_mean[self.layers_names[i]] + specificity_mean1[self.layers_names[i]] + specificity_mean2[self.layers_names[i]]) / 3)
              # print("Dice               : " ,  (dice_mean[self.layers_names[i]] + dice_mean1[self.layers_names[i]] + dice_mean2[self.layers_names[i]]) / 3)
              # print("F1 score           : " ,  (f1_score_mean[self.layers_names[i]] + f1_score_mean1[self.layers_names[i]] + f1_score_mean2[self.layers_names[i]]) / 3)
              # print("Iou                : " ,  (iou_mean[self.layers_names[i]] + iou_mean1[self.layers_names[i]] + iou_mean2[self.layers_names[i]]) / 3)
          
            


        def res(mean, std):
            return "{}\u00B1{}".format(mean, std)

        info = {'Accuracy'           : [res(result_acc_mean[0],result_acc_std[0]), res(result_acc_mean[1],result_acc_std[1]), res(result_acc_mean[2],result_acc_std[2]), res(result_acc_mean[3],result_acc_std[3])],
                'Balanced Accuracy'  : [res(result_bacc_mean[0],result_bacc_std[0]), res(result_bacc_mean[1],result_bacc_std[1]), res(result_bacc_mean[2],result_bacc_std[2]), res(result_bacc_mean[3],result_bacc_std[3])],
                # 'Precision'          : [res(result_precision_mean[0],result_precision_std[0]), res(result_precision_mean[1],result_precision_std[1]), res(result_precision_mean[2],result_precision_std[2]), res(result_precision_mean[3],result_precision_std[3])],
                # 'Recall'             : [res(result_recall_mean[0],result_recall_std[0]), res(result_recall_mean[1],result_recall_std[1]), res(result_recall_mean[2],result_recall_std[2]), res(result_recall_mean[3],result_recall_std[3])],
                # 'Specificity'        : [res(result_specificity_mean[0],result_specificity_std[0]), res(result_specificity_mean[1],result_specificity_std[1]), res(result_specificity_mean[2],result_specificity_std[2]), res(result_specificity_mean[3],result_specificity_std[3])],
                # 'F1_score'           : [res(result_f1_mean[0],result_f1_std[0]), res(result_f1_mean[1],result_f1_std[1]), res(result_f1_mean[2],result_f1_std[2]), res(result_f1_mean[3],result_f1_std[3])],
                'Dice'               : [res(result_dice_mean[0],result_dice_std[0]), res(result_dice_mean[1],result_dice_std[1]), res(result_dice_mean[2],result_dice_std[2]), res(result_dice_mean[3],result_dice_std[3])],
                'IoU'                : [res(result_iou_mean[0],result_iou_std[0]), res(result_iou_mean[1],result_iou_std[1]), res(result_iou_mean[2],result_iou_std[2]), res(result_iou_mean[3],result_iou_std[3])]}

        print('\n Overall Results on %s'%dataset_name)
        print(tabulate(info, headers='keys', tablefmt='fancy_grid',
                      showindex=['Background', 'IRF','SRF', 'PED'],
                      missingval='N/A'))   
                
    def ConMap2Mask_prob(self, c_map,hori_translation,verti_translation):
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
        left         = tf.matmul (c_map[:, :, :, 3],hori_translation.transpose(1,0))
        left_bottom  = tf.matmul(verti_translation.transpose(1,0), c_map[:, :,:, 5])
        left_bottom  = tf.matmul(left_bottom,hori_translation.transpose(1,0))
        right_above  = tf.matmul(verti_translation, c_map[:,:,:, 2])
        right_above  = tf.matmul(right_above,hori_translation)
        left_above   = tf.matmul(verti_translation, c_map[:,:,:, 0])
        left_above   = tf.matmul(left_above,hori_translation.transpose(1,0))
        bottom       = tf.matmul(verti_translation.transpose(1,0), c_map[:,:,:, 6])
        up           = tf.matmul(verti_translation, c_map[:,:,:, 1])
        right_bottom = tf.matmul(verti_translation.transpose(1,0), c_map[:,:,:, 7])
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
            
                    
                                                                                                                                                                                                                                                                                                           
