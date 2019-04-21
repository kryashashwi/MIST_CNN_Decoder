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


k = 500 #dataword length                    
N = k*2 #codeword length (rate 1/2 encoder is used)
#Loading encoding matrices  
Linv=np.load('Linv'+str(k)+'.npy')
Uinv=np.load('Uinv'+str(k)+'.npy')
H=np.load('H'+str(k)+'.npy')
#Setting SNR range
train_SNR_Eb = [-1,0,1,2,3,4,5,6,7]    

#Get PSD using Eb/N0        
def get_sigma(train_SNR_Eb):
    
    train_SNR_Es = train_SNR_Eb + 10*np.log10(1.0*k/N)
    train_sigma = np.sqrt(1/(2*10**(train_SNR_Es/10)))
    return(train_sigma)


#Model parameters
optimizer = 'adam'           
loss = 'mse'               
train_batch=1024
test_batch=1024


#BER function
def ber(y_true, y_pred):
    return  K.mean(K.cast(K.not_equal(y_true, K.round(y_pred)),dtype='float32'))


#CNN Network design
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

#LDPC Encoder
def LDPC_encoder(s):
    s=np.reshape(s,(k,1))
    N=np.shape(H)[1]
    M=np.shape(H)[0]
    z=np.mod(np.matmul(H[:,N-M:],s),2)

    c=np.mod(np.matmul(Uinv,np.matmul(Linv,z)),2)
    enc_msg=np.reshape(np.concatenate((c,s),axis=0),(1,2*k))
    return(enc_msg)   


def encoder():
    uncoded = np.zeros((train_batch,k)) 
    x=np.zeros((train_batch,N))
    for i in range(0,train_batch):
        uncoded[i,:] = np.random.randint(0,2,size=(k)) 
        encoded = LDPC_encoder(uncoded[i,:])
        x[i,:]=copy.deepcopy(encoded[0])
   
    #Modulate
    x=2*x -1
    return x,uncoded

#Load model weights
model.load_weights('LDPC'+str(k)+'.hdf5')

#Testing
test_batch=1024
SNR_dB_start_Eb = -1
SNR_dB_stop_Eb = 7
SNR_points = 9

#Setting SNR range
SNR_dB_start_Es = SNR_dB_start_Eb + 10*np.log10(1.0*k/N)
SNR_dB_stop_Es = SNR_dB_stop_Eb + 10*np.log10(1.0*k/N)

SNR_range=np.linspace(SNR_dB_start_Es, SNR_dB_stop_Es, SNR_points)
sigmas=np.sqrt(1/(2*10**(SNR_range/10)))


n_samples=10 #Number of samples to test on for each SNR. Currently set as 10 for demo purpose. 
print('Testing')
for i in range(0,len(sigmas)):
    
    for ii in range(0,n_samples):
        
        #Get dataword and codeword
        x1,uncoded1=encoder()
        #Add noise
        for iii in range(0,len(x1)):
            w=np.random.normal(0, sigmas[i], np.shape(x1[iii]))
            x1[iii,:]=x1[iii,:]+w


        #Decoding noisy message
        x1=np.expand_dims(x1,axis=2)
        msg_rec=model.predict(x1)
        #Calculating metrics
        err_block=0
        for qq in range(0,len(msg_rec)):
           
            if list(np.round(msg_rec[qq,:]))!=list(uncoded1[qq,:]):
                err_block=err_block+1
        bler=err_block*1.0/len(msg_rec)
        print(round(SNR_range[i]),ii,bler, model.evaluate(x1, uncoded1, batch_size=test_batch, verbose=2)[1])
