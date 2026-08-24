# -*- coding: utf-8 -*-
"""
@author: cquesada

Model architecture 
"""

from keras.layers import Concatenate, Input, LSTM, Dense
from keras.models import Model

concatenator_all = Concatenate(axis=-1)
def model(time_steps, n_pca, n_snp, n_soil):

    weather_input = Input(shape = (time_steps, 3), name = "Weather_Input")   
    pca_input = Input(shape = (n_pca, ), name ="PCA_Input")  
    snp_input = Input(shape = (n_snp, ), name = "SNP_Input")
    soil_input = Input(shape = (n_soil,), name = "Soil_Input")
    
    # Encoder LSTM for weather
    lstm, state_h, state_c = LSTM(128, return_state=True, return_sequences=True)(weather_input)
    lstm, state_h, state_c = LSTM(128, return_state=True, return_sequences=True)(lstm)
    lstm, state_h, state_c = LSTM(128, return_state=True, return_sequences=False)(lstm)
    lstm = Dense(128, activation = "relu") (lstm)
    lstm = Dense(128, activation = "relu") (lstm)
    lstm = Dense(64, activation = "relu")(lstm)
    lstm = Dense(32, activation = "relu")(lstm)
    lstm = Dense(16, activation = "relu")(lstm)
    lstm = Dense(8, activation = "relu")(lstm)

    # input PCA
    
    l_pca = Dense(128, activation = "relu")(pca_input)
    l_pca = Dense(128, activation = "relu")(l_pca)
    l_pca = Dense(64, activation = "relu")(l_pca)
    l_pca = Dense(32, activation = "relu")(l_pca)  
    l_pca = Dense(16, activation = "relu")(l_pca)   
    l_pca = Dense(8, activation = "relu")(l_pca)
    
    # process the snp data

    l_snp = Dense(1024, activation = "relu")(snp_input)
    l_snp = Dense(1024, activation = "relu")(l_snp)
    l_snp = Dense(512, activation = "relu")(l_snp)
    l_snp = Dense(256, activation = "relu")(l_snp)
    l_snp = Dense(128, activation = "relu")(l_snp)
    l_snp = Dense(64, activation = "relu")(l_snp)
    l_snp = Dense(32, activation = "relu")(l_snp)      
    l_snp = Dense(16, activation = "relu")(l_snp)
    l_snp = Dense(8, activation = "relu")(l_snp)    

    # Input soil and management
    l_soilman = Dense(64, activation = "relu")(soil_input)
    l_soilman = Dense(64, activation = "relu")(l_soilman)
    l_soilman = Dense(32, activation = "relu")(l_soilman)
    l_soilman = Dense(16, activation = "relu")(l_soilman)
    l_soilman = Dense(8, activation = "relu")(l_soilman)

    # Concatenate all
    l_all = concatenator_all([lstm,l_snp, l_pca, l_soilman])   

    # Output regression layer
    yhat = Dense (1, activation = "sigmoid")(l_all)   
        
    pred_model = Model([weather_input, pca_input, snp_input, soil_input], yhat)   # Prediction Model
        
    return pred_model

