import tensorflow as tf
import numpy as np
from keras.layers.core import Dense
from keras.layers import Conv1D,Flatten, BatchNormalization, Input
from keras import backend as K
from keras.models import Model
import copy
from keras.callbacks import ModelCheckpoint

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

k = 50   #dataword length                 
N = k*2  #codeword length, rate 1/2 encoder is used  

#Setting SNR range
train_SNR_Eb = [-1,0,1,2,3,4,5,6,7]    
       
#Calculate PSF from Eb/N0
def get_sigma(train_SNR_Eb):
    
    train_SNR_Es = train_SNR_Eb + 10*np.log10(1.0*k/N)
    train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))
    return(train_sigma)


#Setting model parameters         
optimizer = 'adam'           
loss = 'mse'                
batch_size=1024
train_batch=1024
test_batch=1024


#BER calculation
def ber(y_true, y_pred):
    return  K.mean(K.cast(K.not_equal(y_true, K.round(y_pred)),dtype='float32'))


#CNN network design
input_batch=Input(shape=(N,1))
conv1 = Conv1D(10, 24, activation='relu', input_shape=(N,1),padding='same')(input_batch)
batch_norm1=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv1)
conv2 = Conv1D(50, 24, activation='relu',padding='same')(batch_norm1)
batch_norm2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv2)
conv3 = Conv1D(50, 24, activation='relu',padding='same')(batch_norm2)
batch_norm3=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(conv3)
flatten = Flatten()(batch_norm3)
msg_out = Dense(k, activation='sigmoid')(flatten)

model=Model(inputs=input_batch,outputs=msg_out)
model.compile(optimizer=optimizer, loss=loss, metrics=[ber])

#Convolution encoder
def convenc(data,g1,g2):
    z1=0
    z2=0
    x1=(np.convolve(data,g1)%2)[0:k]
    x2=(np.convolve(data,g2)%2)[0:k]
    enc_msg=np.zeros([2*k])
    for p in range(2*k):
        if p%2==0:
            enc_msg[p]=x1[z1]
            z1=z1+1
        else:
            enc_msg[p]=x2[z2]
            z2=z2+1
    return(enc_msg)    



#Loading weights
model.load_weights(str(k)+'_conv.hdf5')


#Setting SNR range
SNR_dB_start_Eb = -1
SNR_dB_stop_Eb = 7
SNR_points = 9


SNR_dB_start_Es = SNR_dB_start_Eb + (10*np.log10(1.0*k/N))
SNR_dB_stop_Es = SNR_dB_stop_Eb + (10*np.log10(1.0*k/N))

SNR_range=np.linspace(SNR_dB_start_Es, SNR_dB_stop_Es, SNR_points)
sigmas=np.sqrt(1/(2*10**(SNR_range/10)))

n_samples=10 #Number of samples to test on for each SNR. Currently set as 10 for demo purpose. 
#Testing
print('Testing')
for i in range(0,len(sigmas)):
    
    for ii in range(0,n_samples):
        #Generating dataword and codeword
        uncoded = np.zeros((train_batch,k)) 
        x=np.zeros((train_batch,N))
        #(5,7), memory 2 convolutional encoder used
        g1=[1,1,1] #7
        g2=[1,0,1] #5
        for zz in range(0,train_batch):
            uncoded[zz,:] = np.random.randint(0,2,size=(k)) 
            encoded = convenc(uncoded[zz,:],g1,g2)
            x[zz,:]=copy.deepcopy(encoded)
            
        #Modulate
      	if(k==100):
        	s_test = -2*x + 1
        else:
        	s_test = 2*x - 1

        y_test=np.zeros(np.shape(s_test))
       	#Adding noise
        for kk in range(0,len(s_test)):
            y_test[kk] = s_test[kk] + np.random.normal(0, sigmas[i], np.shape(s_test[kk]))

        #Decoding noisy message
        y_test=np.expand_dims(y_test,axis=2)
        msg_rec=model.predict(y_test)
        #Calculating metrics
        err_block=0
        for qq in range(0,len(msg_rec)):
            if list(np.round(msg_rec[qq,:]))!=list(uncoded[qq,:]):
                err_block=err_block+1
        bler=err_block*1.0/len(msg_rec)
        print(round(SNR_range[i]),ii,bler, model.evaluate(y_test, uncoded, batch_size=test_batch, verbose=2)[1])
 
