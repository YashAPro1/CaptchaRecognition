import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pdpip 
from dataclasses import dataclass
import keras
import tensorflow as tf
from keras import layers
import numpy as np


@dataclass
class ModelIngestionConf:
    train_data_path = os.path.join('artifacts','12')
    
model = tf.keras.models.load_model(os.path.join('artifacts','12'))

def temporal_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + keras.backend.epsilon())
    input_length = tf.cast(input_length, tf.int32)

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

prediction_model = keras.models.Model(
    model.input[0], model.get_layer(name="dense2").output
)
characters = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

        # Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = temporal_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :
    ]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

class DataIngestion:
    def __init__(self,batchimage):
        self.model_conf = ModelIngestionConf()
        self.batchimage = batchimage
    
    def initiate_data(self):
        logging.info("We have started the Model ingestion part now...")
        # Mapping characters to integers
        
        try:
            preds = prediction_model.predict(self.batchimage)
            pred_texts = decode_batch_predictions(preds)
            print("Hi there")
            print(pred_texts[0][:5])
        except Exception as e:
            raise CustomException(e,sys)