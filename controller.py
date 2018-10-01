import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, Dropout, SimpleRNN, CuDNNGRU, CuDNNLSTM
from tensorflow.python.keras.optimizers import RMSprop, SGD
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import copy

RUDDER = 0
PROPELLER = 1

load_columns = ['zz', 'vx', 'vy', 'vzz', 'rudder_demanded', 'propeller_demanded', 'distance_midline', 'distance_target']
inp_rudder = ['zz', 'vx', 'vy', 'vzz', 'distance_midline']
inp_prop = ['zz', 'vx', 'vy', 'vzz', 'distance_midline', 'distance_target']
target_rudder = 'rudder_demanded'
target_prop = 'propeller_demanded'

class Train_Log:
    def __init__(self, type_model):
        self.type_model = type_model
        self.train_time = None
        self.loss = None
        self.flag_error = False
        self.log_error = None

    def write_log(self, train_code):
        file = open('./keras_logs/' + train_code + '/train_log_' + train_code + '.txt', 'a+')
        if self.type_model == RUDDER:
            file.write('\n\n*** Treinamento RUDDER ***')
        else:
            file.write('\n\n*** Treinamento PROPELLER ***')

        if self.flag_error:
            file.write('\n\tResultado: ERRO')
            file.write('\n\tLog de erro:')
            file.write('\n' + str(self.log_error))
        else:
            file.write('\n\tResultado: CONCLUÍDO SEM ERRO')
            file.write('\n\tTempo de treinamento: %s' % str(self.train_time))
            file.write('\n\tLoss: %s' % self.loss)
        file.close()

class Controller:
    def __init__(self):
        self.model_rudder = None
        self.model_prop = None
        self.num_model = 0
        self.train_code = None
        self.loaded_data = False
        self.has_rudder_scaler = False
        self.has_prop_scaler = False
        self.x_rudder_scaler = None
        self.y_rudder_scaler = None
        self.x_prop_scaler = None
        self.y_prop_scaler = None
        self.rudder_logger = Train_Log(RUDDER)
        self.prop_logger = Train_Log(PROPELLER)

    def load_model(self, train_code):
        self.train_code = train_code
        self.model_rudder = load_model('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder.h5')
        self.model_prop = load_model('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller.h5')
        self.x_rudder_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder_x_scaler.save')
        self.y_rudder_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder_y_scaler.save')
        self.x_prop_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller_x_scaler.save')
        self.y_prop_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller_y_scaler.save')
        self.has_prop_scaler = True
        self.has_rudder_scaler = True
        self.loaded_data = False
        self.rudder_logger = Train_Log(RUDDER)
        self.prop_logger = Train_Log(PROPELLER)

    def select_rudder_action(self, input):
        print('Entrada rudder: ' + str(input))
        input = self.x_rudder_scaler.transform(np.expand_dims(input, 0))
        input = np.expand_dims(input, 0)
        output = self.model_rudder.predict(input)
        output = self.y_rudder_scaler.inverse_transform(output[0])
        return output[0][0]

    def select_prop_action(self, input):
        print('Entrada propeller: ' + str(input))
        input = self.x_prop_scaler.transform(np.expand_dims(input, 0))
        input = np.expand_dims(input, 0)
        output = self.model_prop.predict(input)
        output = self.y_prop_scaler.inverse_transform(output[0])
        output = np.asarray(pd.cut(output[:, 0], [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], right=False, labels=[-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        return output[0]

    def train(self, files_path, num_model=1):
        batch_size = 256
        sequence_length = 1000

        self.set_num_model(num_model)
        self.set_train_code()

        if not self.loaded_data:
            print('****** CARREGANDO DADOS ******')
            df = self.load_data(files_path)
            self.loaded_data = True
            print('DF shape:' + str(df.shape))

        print('\n****** TREINAMENTO MODELO ' + self.train_code + ' ******')
        try:
            self.run_train(df, batch_size, sequence_length, RUDDER)
        except Exception as e:
            self.rudder_logger.flag_error = True
            self.rudder_logger.log_error = e
        self.rudder_logger.write_log(self.train_code)

        try:
            self.run_train(df, batch_size, sequence_length, PROPELLER)
        except Exception as e:
            self.prop_logger.flag_error = True
            self.prop_logger.log_error = e
        self.prop_logger.write_log(self.train_code)

    def set_num_model(self, num_model):
        self.num_model = num_model

    def set_train_code(self):
        year = str(datetime.datetime.now().year)
        month = str(datetime.datetime.now().month)
        day = str(datetime.datetime.now().day)
        hour = str(datetime.datetime.now().hour)
        minute = str(datetime.datetime.now().minute)
        self.train_code = 'model_' + year + '-' + month + '-' + day + '_' + hour + '-' + minute

    def model_rudder_0(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_0(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_1(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_1(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_2(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_2(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_3(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_3(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_4(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_4(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_5(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_5(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_6(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_6(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_7(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_7(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_8(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_8(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_9(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_9(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_10(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_10(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_11(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_11(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_12(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_12(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=256,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_13(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_13(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_14(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_14(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_15(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_15(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_16(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_16(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                      return_sequences=True,
                      input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_17(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_17(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_18(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_18(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_19(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_19(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_20(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_20(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_21(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_21(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=64,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=64,
                     return_sequences=True,
                     input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softsign'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def load_data(self, loc):
        file = os.listdir(loc)
        df = pd.DataFrame()
        for f in file:
            path = loc + f
            dt = pd.read_csv(path, sep=" ", skiprows=11)
            df = df.append(dt, ignore_index=True)
        print('Dados carregados: ' + str(df.shape[0]) + ', ' + str(df.shape[1]))
        return df[load_columns]

    def run_train(self, df, batch_size, sequence_length, type):
        x_train, y_train, x_test, y_test, validation_data = self.prepare_data(df, type)
        generator = self.batch_generator(batch_size, sequence_length, x_train, y_train)

        num_x_signals = x_train.shape[1]
        num_y_signals = y_train.shape[1]

        if type == RUDDER:
            model = getattr(self, 'model_rudder_' + str(self.num_model))(num_x_signals, num_y_signals)
            print('\t>>> RUDDER <<<')
        else:
            model = getattr(self, 'model_prop_' + str(self.num_model))(num_x_signals, num_y_signals)
            print('\t>>> PROPELLER <<<')

        print(model.summary())
        callbacks = self.define_callback(type)
        init_time = datetime.datetime.now()
        model.fit_generator(generator=generator,
                            epochs=100,
                            steps_per_epoch=100,
                            validation_data=validation_data,
                            callbacks=callbacks)
        train_time = datetime.datetime.now() - init_time
        loss = self.run_test(model, x_test, y_test, type)
        print("Loss: " + str(loss))

        strout = 'rudder' if type == RUDDER else 'propeller'
        model.save('./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strout + '.h5')

        # scaler save
        if type == RUDDER:
            x_scaler_filename = './keras_logs/' + self.train_code + '/' + self.train_code + '_' + strout + '_x_scaler.save'
            joblib.dump(self.x_rudder_scaler, x_scaler_filename)
            y_scaler_filename = './keras_logs/' + self.train_code + '/' + self.train_code + '_' + strout + '_y_scaler.save'
            joblib.dump(self.y_rudder_scaler, y_scaler_filename)
        else:
            x_scaler_filename = './keras_logs/' + self.train_code + '/' + self.train_code + '_' + strout + '_x_scaler.save'
            joblib.dump(self.x_prop_scaler, x_scaler_filename)
            y_scaler_filename = './keras_logs/' + self.train_code + '/' + self.train_code + '_' + strout + '_y_scaler.save'
            joblib.dump(self.y_prop_scaler, y_scaler_filename)

        if type == RUDDER:
            self.model_rudder = model
            self.rudder_logger.train_time = train_time
            self.rudder_logger.loss = loss
        else:
            self.model_prop = model
            self.prop_logger.train_time = train_time
            self.prop_logger.loss = loss

    def run_test(self, model, x_test, y_test, type):
        if type == RUDDER:
            strtype = "rudder"
        else:
            strtype = "propeller"

        try:
            model.load_weights('./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strtype + '.keras')
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

        return model.evaluate(x=np.expand_dims(x_test, axis=0),
                              y=np.expand_dims(y_test, axis=0))

    def prepare_data(self, df, type):
        # TODO: Tirar esse hardcode
        #Caso 05 usado como validacao
        #df.loc[66955:84409]

        if type == RUDDER:
            input_columns = inp_rudder
            target_column = target_rudder
        else:
            input_columns = inp_prop
            target_column = target_prop

        x_data = df[input_columns].values
        y_data = df[target_column].values
        y_data = np.reshape(y_data, (len(y_data), 1))

        x_train = np.delete(x_data, np.s_[66955:84410], axis=0)
        y_train = np.delete(y_data, np.s_[66955:84410], axis=0)
        x_test = x_data[66955:84410]
        y_test = y_data[66955:84410]

        if (type == RUDDER and not self.has_rudder_scaler) or (type == PROPELLER and not self.has_prop_scaler):
            self.generate_scaler(df, type)

        if type == RUDDER:
            x_train_scaled = self.x_rudder_scaler.transform(x_train)
            x_test_scaled = self.x_rudder_scaler.transform(x_test)
            y_train_scaled = self.y_rudder_scaler.transform(y_train)
            y_test_scaled = self.y_rudder_scaler.transform(y_test)
        else:
            x_train_scaled = self.x_prop_scaler.transform(x_train)
            x_test_scaled = self.x_prop_scaler.transform(x_test)
            y_train_scaled = self.y_prop_scaler.transform(y_train)
            y_test_scaled = self.y_prop_scaler.transform(y_test)

        validation_data = (np.expand_dims(x_test_scaled, axis=0),
                           np.expand_dims(y_test_scaled, axis=0))

        return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, validation_data

    def generate_scaler(self, df, type):
        if type == RUDDER:
            input_columns = inp_rudder
            target_column = target_rudder
        else:
            input_columns = inp_prop
            target_column = target_prop

        x_data = df[input_columns].values
        y_data = df[target_column].values
        y_data = np.reshape(y_data, (len(y_data), 1))

        x_scaler = MinMaxScaler()
        x_scaler.fit(x_data)
        y_scaler = MinMaxScaler()
        y_scaler.fit(y_data)

        if type == RUDDER:
            self.x_rudder_scaler = x_scaler
            self.y_rudder_scaler = y_scaler
            self.has_rudder_scaler = True
        else:
            self.x_prop_scaler = x_scaler
            self.y_prop_scaler = y_scaler
            self.has_prop_scaler = True

    def batch_generator(self, batch_size, sequence_length, x_data, y_data):
        while True:
            x_shape = (batch_size, sequence_length, x_data.shape[1])
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            y_shape = (batch_size, sequence_length, y_data.shape[1])
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            for i in range(batch_size):
                # TODO: Arrumar para nao considerar quebra entre os arquivos
                idx = np.random.randint(x_data.shape[0] - sequence_length)
                x_batch[i] = x_data[idx:idx + sequence_length]
                y_batch[i] = y_data[idx:idx + sequence_length]

            yield (x_batch, y_batch)

    def define_callback(self, type):
        if type == RUDDER:
            strtype = "rudder"
        else:
            strtype = "propeller"
        callback_checkpoint = ModelCheckpoint(filepath='./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strtype + '.keras',
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)
        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=5, verbose=1)
        callback_tensorboard = TensorBoard(log_dir='./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strtype + '_log/',
                                           histogram_freq=0,
                                           write_graph=False)
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                               factor=0.1,
                                               min_lr=1e-4,
                                               patience=0,
                                               verbose=1)
        callbacks = [callback_early_stopping,
                     callback_checkpoint,
                     callback_tensorboard,
                     callback_reduce_lr]

        return callbacks

    def test_case(self, num_case):
        strcase = '0' + str(num_case) if num_case < 10 else str(num_case)

        df2 = pd.read_csv('C:\\Users\\AlphaCrucis_Control1\\PycharmProjects\\Data_Filter\\Output\\Compilado\\Caso' + strcase + '.txt', sep=" ", skiprows=11)
        df2 = df2[load_columns]

        x_rudder = df2[inp_rudder].values
        y_rudder = df2[target_rudder].values
        y_rudder = np.reshape(y_rudder, (len(y_rudder), 1))

        x_prop = df2[inp_prop].values
        y_prop = df2[target_prop].values
        y_prop = np.reshape(y_prop, (len(y_prop), 1))

        if not self.has_rudder_scaler:
            self.x_rudder_scaler = joblib.load('./keras_logs/rudder_x_scaler.save')
            self.y_rudder_scaler = joblib.load('./keras_logs/rudder_y_scaler.save')

        if not self.has_prop_scaler:
            self.x_prop_scaler = joblib.load('./keras_logs/propeller_x_scaler.save')
            self.y_prop_scaler = joblib.load('./keras_logs/propeller_y_scaler.save')

        x_rudder = self.x_rudder_scaler.transform(x_rudder)
        x_rudder = np.expand_dims(x_rudder, axis=0)
        y_rudder_pred = self.model_rudder.predict(x_rudder)
        y_rudder_pred = self.y_rudder_scaler.inverse_transform(y_rudder_pred[0])

        x_prop = self.x_prop_scaler.transform(x_prop)
        x_prop = np.expand_dims(x_prop, axis=0)
        y_prop_pred = self.model_prop.predict(x_prop)
        y_prop_pred = self.y_prop_scaler.inverse_transform(y_prop_pred[0])

        time = np.arange(0, len(y_rudder_pred)) / 10

        y_prop_pred = np.asarray(pd.cut(y_prop_pred[:, 0], [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], right=False, labels=[-4, -3, -2, -1, 0, 1, 2, 3, 4]))

        plt.figure(figsize=(15, 5))
        plt.plot(time, y_prop, label='true')
        plt.plot(time, y_prop_pred, label='pred')
        plt.legend()
        plt.savefig('./keras_logs/' + self.train_code + '/' + self.train_code + '_prop_caso_' + strcase + '.png')
        #plt.show()

        plt.figure(figsize=(15, 5))
        plt.plot(time, y_rudder, label='true')
        plt.plot(time, y_rudder_pred, label='pred')
        plt.legend()
        plt.savefig('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder_caso_' + strcase + '.png')
        #plt.show()

if __name__ == '__main__':

    #print(model_rudder.select_action([10, 15, 20]))

    #model generator and scaler generator
    # for i in range(5):
    year = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month)
    day = str(datetime.datetime.now().day)
    file = open('./keras_logs/train_log_' + year + '-' + month + '-' + day + '.txt', 'a+')
    for i in range(22):
        try:
            model = Controller()
            model.train('C:\\Users\\AlphaCrucis_Control1\\PycharmProjects\\Data_Filter\\Output\\Compilado\\', i)
            file.write('\n*** Treinamento ' + model.train_code + ': OK')
            model.test_case(10)
        except Exception as e:
            file.write('\n*** Treinamento ' + model.train_code + ': NOK')
            file.write('\n\t' + str(e))
    file.close()

    #model test
    #model = Controller()
    #model.load_model('model_2018-9-28_17-20')
    #model.test_case(10)