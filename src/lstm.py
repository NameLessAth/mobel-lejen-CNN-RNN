import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from activations import Activations

class LSTM_Network:
    def __init__(self, keras_model):
        self.model = keras_model
        self.extract_weights()
    
    def extract_weights(self):
        self.weights = {}
        self.layer_info = []
        
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, layers.Embedding): # layer embedding
                self.weights[f'embedding_{i}'] = layer.get_weights()[0]
                self.layer_info.append(('embedding', i))
            elif isinstance(layer, layers.LSTM): # layer lstm
                lstm_weights = layer.get_weights()
                W = lstm_weights[0] 
                U = lstm_weights[1]
                bias = lstm_weights[2]  
                units = layer.units
    
                self.weights[f'lstm_{i}'] = {
                    'W_i': W[:, :units],         
                    'W_f': W[:, units:2*units],    
                    'W_c': W[:, 2*units:3*units],
                    'W_o': W[:, 3*units:],       
                    'U_i': U[:, :units],     
                    'U_f': U[:, units:2*units],   
                    'U_c': U[:, 2*units:3*units],
                    'U_o': U[:, 3*units:],
                    'b_i': bias[:units],
                    'b_f': bias[units:2*units],
                    'b_c': bias[2*units:3*units],
                    'b_o': bias[3*units:],
                    'units': units,
                    'return_sequences': layer.return_sequences
                }
                self.layer_info.append(('lstm', i))
                
            elif isinstance(layer, layers.Bidirectional): # layer bidirectional lstm
                forward_weights = layer.forward_layer.get_weights()
                backward_weights = layer.backward_layer.get_weights()
                
                units = layer.forward_layer.units
                
                self.weights[f'bilstm_{i}'] = {
                    'forward': self.extract_weights_lstm(forward_weights, units),
                    'backward': self.extract_weights_lstm(backward_weights, units),
                    'units': units,
                    'return_sequences': layer.return_sequences
                }
                self.layer_info.append(('bidirectional', i))
                
            elif isinstance(layer, layers.Dense): # layer dense
                dense_weights = layer.get_weights()
                self.weights[f'dense_{i}'] = {
                    'W': dense_weights[0],
                    'b': dense_weights[1] if len(dense_weights) > 1 else None,
                    'activation': layer.activation.__name__
                }
                self.layer_info.append(('dense', i))
                
            elif isinstance(layer, layers.Dropout): # layer dropout
                self.layer_info.append(('dropout', i))
    
    def extract_weights_lstm(self, lstm_weights, units):
        W = lstm_weights[0]
        U = lstm_weights[1] 
        bias = lstm_weights[2]
        return {
            'W_i': W[:, :units],
            'W_f': W[:, units:2*units], 
            'W_c': W[:, 2*units:3*units],
            'W_o': W[:, 3*units:],
            'U_i': U[:, :units],
            'U_f': U[:, units:2*units],
            'U_c': U[:, 2*units:3*units],
            'U_o': U[:, 3*units:],
            'b_i': bias[:units],
            'b_f': bias[units:2*units],
            'b_c': bias[2*units:3*units], 
            'b_o': bias[3*units:],
            'units': units 
        }
    
    def lstm_forward(self, x_t, h_pre, c_pre, weights):
        # Input gate
        i_t = Activations.sigmoid(np.dot(x_t, weights['W_i']) + np.dot(h_pre, weights['U_i']) + weights['b_i'])
        # Forget gate  
        f_t = Activations.sigmoid(np.dot(x_t, weights['W_f']) + np.dot(h_pre, weights['U_f']) + weights['b_f'])
        # Candidate values
        C_t = np.tanh(np.dot(x_t, weights['W_c']) + np.dot(h_pre, weights['U_c']) + weights['b_c'])
        # Cell state
        c_post = f_t * c_pre + i_t * C_t
        # Output gate
        o_t = Activations.sigmoid(np.dot(x_t, weights['W_o']) + np.dot(h_pre, weights['U_o']) + weights['b_o'])
        # Hidden state
        h_post = o_t * np.tanh(c_post)
        
        return h_post, c_post
    
    def lstm_forward_sequence(self, inputnya, weights, return_sequences=False):
        batch_size, seq_len, input_dim = inputnya.shape
        units = weights['units']
        # sementara initiate pake 0
        h_t = np.zeros((batch_size, units))
        c_t = np.zeros((batch_size, units))
        
        if return_sequences:
            outputs = np.zeros((batch_size, seq_len, units))
        
        for t in range(seq_len):
            h_t, c_t = self.lstm_forward(inputnya[:, t, :], h_t, c_t, weights)
            if return_sequences:
                outputs[:, t, :] = h_t
        
        if return_sequences:
            return outputs
        else:
            return h_t
    
    def bidirectional_lstm_forward(self, inputnya, weights, return_sequences=False):
        forward_output = self.lstm_forward_sequence(inputnya, weights['forward'], return_sequences)

        X_reversed = inputnya[:, ::-1, :] 
        backward_output = self.lstm_forward_sequence(X_reversed, weights['backward'], return_sequences)
        
        if return_sequences:
            backward_output = backward_output[:, ::-1, :]
            output = np.concatenate([forward_output, backward_output], axis=-1)
        else:
            output = np.concatenate([forward_output, backward_output], axis=-1)
            
        return output
    
    def forward_propagation(self, inputnya):
        current_input = inputnya.copy()
        
        for layer_type, layer_idx in self.layer_info:
            
            if layer_type == 'embedding':
                embedding_weights = self.weights[f'embedding_{layer_idx}']
                current_input = embedding_weights[current_input.astype(int)]
                
            elif layer_type == 'lstm':
                lstm_weights = self.weights[f'lstm_{layer_idx}']
                current_input = self.lstm_forward_sequence(
                    current_input, lstm_weights, lstm_weights['return_sequences']
                )
                
            elif layer_type == 'bidirectional':
                bilstm_weights = self.weights[f'bilstm_{layer_idx}']
                current_input = self.bidirectional_lstm_forward(
                    current_input, bilstm_weights, bilstm_weights['return_sequences']
                )
                
            elif layer_type == 'dense':
                dense_weights = self.weights[f'dense_{layer_idx}']
                current_input = np.dot(current_input, dense_weights['W'])
                if dense_weights['b'] is not None:
                    current_input += dense_weights['b']
                
                if dense_weights['activation'] == 'softmax':
                    current_input = Activations.softmax(current_input)
                elif dense_weights['activation'] == 'relu':
                    current_input = np.maximum(0, current_input)
                elif dense_weights['activation'] == 'sigmoid':
                    current_input = Activations.sigmoid(current_input)
                elif dense_weights['activation'] == 'tanh':
                    current_input = np.tanh(current_input)
                    
            # elif layer_type == 'dropout':
        
        return current_input
    
    def predict(self, inputnya):
        return self.forward_propagation(inputnya)
    
    def cmp_keras(self, X):
        keras_pred = self.model.predict(X, verbose=0)
        diy_pred = self.predict(X)
        
        max_diff = np.max(np.abs(keras_pred - diy_pred))
        mean_diff = np.mean(np.abs(keras_pred - diy_pred))
        
        return max_diff, mean_diff