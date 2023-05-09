import tensorflow as tf
from keras import backend as K
from keras import backend as k
from keras import layers
from keras.layers.core import Lambda
import numpy as np
from numpy import array 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dropout,Bidirectional,Input, Embedding, Dense,Concatenate,Flatten, Multiply,Average,Subtract,Reshape,GRU
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras import optimizers,regularizers
import statistics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import multilabel_confusion_matrix
import ast
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline

def expandDim(x):
    x=k.expand_dims(x, 1)
    return x


def linearConv(var):
    ten1,ten2=var[0],var[1]
    ten1=expandDim(ten1)
    print("ten1************",ten1)
    ten2=expandDim(ten2)
    print("ten2************",ten2)
    arr1=tf.nn.conv1d(ten1, ten2, padding='SAME', stride=1)
    arr1=tf.squeeze(arr1,axis=1)
    print("arr1####:::::",arr1)
    return arr1

def attentionScores(var):
    Q_t,K_t,V_t=var[0],var[1],var[2]
    scores = tf.matmul(Q_t, K_t, transpose_b=True)
    print("first scores shape:::::",scores.shape)
    distribution = tf.nn.softmax(scores)
    scores=tf.matmul(distribution, V_t)
    print("scores shape:::::",scores.shape)
    return scores

def create_resample(train_sequence,train_sequence_affect,train_stance_enc,train_emo,train_int):
    df = pd.DataFrame(list(zip(train_sequence,train_sequence_affect,train_stance_enc,train_emo,train_int)), columns =['text','affect','stance','emo','int'],index=None)
    ambi=(df[df['stance'] == 0])
    print("len ambi",len(ambi))
    blv=(df[df['stance'] == 1])
    print("len blv",len(blv))
    deny=(df[df['stance'] == 2])
    print("len deny",len(deny))

    upsampled1 = resample(ambi,replace=True, # sample with replacement
                          n_samples=len(blv), # match number in majority class
                          random_state=27)
    upsampled2 = resample(deny,replace=True, # sample with replacement
                          n_samples=len(blv), # match number in majority class
                          random_state=27)

    upsampled = pd.concat([blv,upsampled1,upsampled2])
    upsampled=upsampled.sample(frac=1)
    print("After oversample train data : ",len(upsampled))
    print("After oversampling, instances of tweet act classes in oversampled data :: ",upsampled.stance.value_counts())

    train_data=upsampled
    train_sequence=[]
    train_stance_enc=[]
    train_emo=[]
    train_int=[]
    train_sequence_affect=[]

   
    for i in range(len(train_data)):
        train_sequence.append(train_data.text.values[i])
        train_stance_enc.append(train_data.stance.values[i])
        train_emo.append(train_data.emo.values[i])
        train_int.append(train_data.int.values[i])
        train_sequence_affect.append(train_data.affect.values[i])

    print("len of all::::",len(train_sequence),len(train_stance_enc),len(train_emo),len(train_int),len(train_sequence_affect))


    return train_sequence,train_sequence_affect,train_stance_enc,train_emo,train_int



data=pd.read_csv("../final_data.csv", delimiter=";", na_filter= False) 
print("data :: ",len(data))


########### creating multiple lists as per the daataframe

li_text=[]
li_stance=[]
li_emo1=[]
li_id=[]
li_emo=[]
li_int1,li_int=[],[]
li_affect=[]

for i in range(len(data)):
    li_id.append(data.tweetid.values[i])
    li_stance.append(data.stance.values[i])
    li_text.append((data.text.values[i]))
    li_affect.append((data.affect_words.values[i]))
    li_emo1.append((data.emotions.values[i]))
    li_int1.append((data.emotion_intensity.values[i]))


for i in range(len(li_emo1)):
    x=li_emo1[i]
    x=ast.literal_eval(x)
    y=[]
    for j in x:
        j=int(j)
        y.append(j)
    li_emo.append(y)
    x=li_int1[i]
    x=ast.literal_eval(x)
    y=[]
    for j in x:
        j=int(j)
        y.append(j)
    li_int.append(y)

print("li_stance np unique:::",np.unique(li_stance,return_counts=True))
print("li_emo np unique:::",len(li_emo))
print("li_int np unique:::",len(li_int))

########### converting labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=li_stance
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)

label_enc=total_integer_encoded

label_cat=to_categorical(total_integer_encoded)

label_emo=np.array(li_emo)
label_int=np.array(li_int)

########## converting text modality into sequence of vectors ############
total_text = [x.lower() for x in li_text] 

tokenizer_btw = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)
model_btw = AutoModel.from_pretrained("vinai/bertweet-base")
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model_btw.resize_token_embeddings(len(tokenizer_btw))
pipe = pipeline('feature-extraction', model=model_btw, tokenizer=tokenizer_btw)
pipe_data = pipe(total_text)

input_sequence=[]
set_token=set()
for i in range(len(total_text)):
    sent=total_text[i]
    embed=pipe_data[i][0]
    li_token=pipe.tokenizer.encode(sent)
    input_sequence.append(li_token)

MAX_SEQ=50
padded_docs = pad_sequences(input_sequence, maxlen=MAX_SEQ, padding='post')

total_sequence=padded_docs#text
vocab_size = 63212 + 1
print("vocab_size:::",vocab_size)
print("downloading ###")
embedding_matrix= np.load("../embedding_matrix_btweet/embed_matrix_bertweet.npy")
print("embedding matrix ****************",embedding_matrix.shape)


#############affect words embedding vector ###########
total_affect= li_affect
total_affect = [x.lower() for x in total_affect] 

pipe_data = pipe(total_affect)

input_sequence_affect=[]
set_token=set()
for i in range(len(total_affect)):
    sent=total_affect[i]
    embed=pipe_data[i][0]
    li_token=pipe.tokenizer.encode(sent)
    input_sequence_affect.append(li_token)

MAX_SEQ=30
padded_docs = pad_sequences(input_sequence_affect, maxlen=MAX_SEQ, padding='post')

total_sequence_affect=padded_docs#text
vocab_size_affect = 60219 + 1
print("vocab_size_affect:::",vocab_size_affect)
print("downloading ###")
embedding_matrix_affect= np.load("../embedding_matrix_btweet/embed_matrix_bertweet_affect.npy")
print("embedding_matrix_affect ****************",embedding_matrix_affect.shape)

MAX_LENGTH=50
MAX_LENGTH_AFFECT=30

#######data for K-fold #########

list_acc_stance,list_f1_stance,list_prec_stance,list_rec_stance=[],[],[],[]
list_acc_emo,list_f1_emo,list_prec_emo,list_rec_emo=[],[],[],[]
list_acc_tox,list_f1_tox,list_prec_tox,list_rec_tox=[],[],[],[]


kf=StratifiedKFold(n_splits=5, random_state=None,shuffle=False)
fold=0
results=[]
for train_index,test_index in kf.split(total_sequence,label_enc):
    print("K FOLD ::::::",fold)
    fold=fold+1

    ############## Shared input #############

    input_shared = Input (shape = (MAX_LENGTH, ))
    input_text_shared = Embedding(vocab_size, 768, weights=[embedding_matrix], input_length=50, name='text_embed_share')(input_shared)
    input_text_flatten=input_text_shared
    input_text_flatten=Flatten()(input_text_shared)
    input_text_flatten= Dense(128, activation="relu")(input_text_flatten)

    input_affect = Input (shape = (MAX_LENGTH_AFFECT, ))
    input_affect_shared = Embedding(vocab_size_affect, 768, weights=[embedding_matrix_affect], input_length=30, name='affect_embed_share')(input_affect)
    input_affect_flatten=input_affect_shared
    input_affect_flatten=Flatten()(input_affect_shared)
    input_affect_flatten= Dense(128, activation="relu")(input_affect_flatten)

    #####convolution module ######
    EAEM_output=Lambda(linearConv)([input_text_flatten,input_affect_flatten])
    print("EAEM_output:::",EAEM_output)
    # EAEM_output= Reshape((-1,1))(EAEM_output)

    EAEM_output_final=Average()([EAEM_output,input_text_flatten])
    EAEM_output_final= Reshape((-1,1))(EAEM_output_final)
    print("EAEM_output_final:::",EAEM_output_final)

    #### stance #######
    lstm_stance = Bidirectional(GRU(128, name='lstm_stance', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(EAEM_output_final)
    Q_s= Dense(128, activation="relu")(lstm_stance)
    K_s= Dense(128, activation="relu")(lstm_stance)
    V_s= Dense(128, activation="relu")(lstm_stance)
    IA_stance=Lambda(attentionScores)([Q_s,K_s,V_s])
    
    #### emotion #######
    lstm_emo = Bidirectional(GRU(128, name='lstm_emo', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(EAEM_output_final)
    Q_e= Dense(128, activation="relu")(lstm_emo)
    K_e= Dense(128, activation="relu")(lstm_emo)
    V_e= Dense(128, activation="relu")(lstm_emo)
    IA_emo=Lambda(attentionScores)([Q_e,K_e,V_e])

    #####intensity ######
    lstm_int = Bidirectional(GRU(128, name='lstm_int', activation='tanh',dropout=.2,kernel_regularizer=regularizers.l2(0.07)))(EAEM_output_final)
    Q_t= Dense(128, activation="relu")(lstm_int)
    K_t= Dense(128, activation="relu")(lstm_int)
    V_t= Dense(128, activation="relu")(lstm_int)
    IA_int=Lambda(attentionScores)([Q_t,K_t,V_t])

    #### shared output ##############
    shared_input=Average()([IA_stance,IA_emo,IA_int])
    shared_output= Dense(128, activation="relu")(shared_input)
    
    int_diff=layers.subtract([IA_stance,shared_output])
    intg_mul=Multiply()([IA_stance,shared_output])
    IM_output1=Concatenate()([IA_stance,shared_output,int_diff,intg_mul])
    stance_output=Dense(3, activation="softmax", name="task_stance")(IM_output1)
    print("stance_output:::::",stance_output)

    int_diff=layers.subtract([IA_emo,shared_output])
    intg_mul=Multiply()([IA_emo,shared_output])
    IM_output2=Concatenate()([IA_emo,shared_output,int_diff,intg_mul])
    emo_output=Dense(8, activation="sigmoid", name="task_emo")(IM_output2)
    print("emo_output:::::",emo_output)

    int_diff=layers.subtract([IA_int,shared_output])
    intg_mul=Multiply()([IA_int,shared_output])
    IM_output3=Concatenate()([IA_int,shared_output,int_diff,intg_mul])
    int_output=Dense(8, activation="relu", name="task_int")(IM_output3)
    print("int_output:::::",int_output)

  
    model=Model([input_shared,input_affect],[stance_output,emo_output,int_output])


    ##Compile
    model.compile(optimizer=Adam(0.001),loss={'task_stance':'categorical_crossentropy','task_emo':'binary_crossentropy','task_int':'mse'},
    loss_weights={'task_stance':1.0,'task_emo':0.5,'task_int':0.3},metrics=['accuracy'])    
    print(model.summary())

    #### model fit ############
    test_sequence,train_sequence=total_sequence[test_index],total_sequence[train_index]
    test_sequence_affect,train_sequence_affect=total_sequence_affect[test_index],total_sequence_affect[train_index]
    
    test_stance_enc,train_stance_enc=label_enc[test_index],label_enc[train_index]
    test_emo,train_emo=label_emo[test_index],label_emo[train_index]
    test_int,train_int=label_int[test_index],label_int[train_index]
    
    train_sequence,train_sequence_affect,train_stance_enc,train_emo,train_int=create_resample(train_sequence,train_sequence_affect,train_stance_enc,train_emo,train_int)

    train_sequence=np.array(train_sequence)
    train_sequence_affect=np.array(train_sequence_affect)
    train_emo=np.array(train_emo)
    test_emo=np.array(test_emo)
    train_int=np.array(train_int)
    test_int=np.array(test_int)
    train_stance=to_categorical(train_stance_enc)

    model.fit([train_sequence,train_sequence_affect],[train_stance,train_emo,train_int],shuffle=True,validation_split=0.2,epochs=20,verbose=2)
    predicted = model.predict([test_sequence,test_sequence_affect])
    print(predicted)

    stance_specific=predicted[0]
    result_=stance_specific
    p_1 = np.argmax(result_, axis=1)
    test_accuracy=accuracy_score(test_stance_enc, p_1)
    list_acc_stance.append(test_accuracy)
    print("test accuracy::::",test_accuracy)
    target_names = ['ambiguous','believe','deny']
    class_rep=classification_report(test_stance_enc, p_1)
    print("specific confusion matrix",confusion_matrix(test_stance_enc, p_1))
    print(class_rep)
    class_rep=classification_report(test_stance_enc, p_1, target_names=target_names,output_dict=True)
    macro_avg=class_rep['macro avg']['f1-score']
    macro_prec=class_rep['macro avg']['precision']
    macro_rec=class_rep['macro avg']['recall']
    print("macro f1 score",macro_avg)
    list_f1_stance.append(macro_avg)
    list_prec_stance.append(macro_prec)
    list_rec_stance.append(macro_rec)
    
############# stance 
print("Stance ::::::::::::::::::::::")
print("ACCURACY :::::::::::: #############")
print("Accuracy  ::: ",list_acc_stance)
print("Mean, STD DEV", statistics.mean(list_acc_stance),statistics.stdev(list_acc_stance))

print("F1  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("F1 ::: ",list_f1_stance)
print("MTL Mean, STD DEV", statistics.mean(list_f1_stance),statistics.stdev(list_f1_stance))


print("Precision  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Precision ::: ",list_prec_stance)
print("MTL Mean, STD DEV", statistics.mean(list_prec_stance),statistics.stdev(list_prec_stance))

print("Recall  $$$$$$$$$$$$$$$$$ ::::::::::::")
print("Recall ::: ",list_rec_stance)
print("MTL Mean, STD DEV", statistics.mean(list_rec_stance),statistics.stdev(list_rec_stance))