'''
[AKUM-14/12/2018]
Arquivo principal do controlador para fazer o treinamento e execucao
'''

'''
Importacao das bibliotecas a serem utilizadas
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, Dropout, SimpleRNN, CuDNNGRU, CuDNNLSTM
from tensorflow.python.keras.optimizers import RMSprop, SGD, Adagrad, Adadelta
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

'''
Parametro globais auxiliares
'''
# Constantes de identificacao de leme e maquina
RUDDER = 0
PROPELLER = 1

# Lista de parametros a serem carregados dos arquivos das simulacoes
load_columns = ['zz', 'vx', 'vy', 'vzz', 'rudder_demanded', 'propeller_demanded', 'distance_midline', 'distance_target']

# Lista de parametros utilizados como entradas na rede neural do leme e da maquina
inp_rudder = ['zz', 'vx', 'vy', 'vzz', 'distance_midline']
inp_prop = ['zz', 'vx', 'vy', 'vzz', 'distance_midline', 'distance_target']

# Lista de saida da rede neural do leme e da maquina
target_rudder = 'rudder_demanded'
target_prop = 'propeller_demanded'

'''
Classe para auxiliar na geracao do log de treinamento
'''
class Train_Log:
    # Construtor da classe
    # Entrada:
    #   type_model: RUDDER ou PROPELLER
    # Saida:
    #   None
    def __init__(self, type_model):
        self.type_model = type_model
        self.train_time = None
        self.loss = None
        self.flag_error = False
        self.log_error = None

    # Escrita padrao do log de treinamento salvo na raiz da pasta do modelo "train_log_<train_code>.txt" - erros, tempo de treinamento e loss
    # Entrada:
    #   train_code: codigo unico de treinamento "model_<data>_<horario>_<tipo_modelo>"
    # Saida:
    #   None
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

'''
Classe para definicao do controlador supervisionado
'''
class Controller:
    # Construtor da classe
    # Entrada:
    #   None
    # Saida:
    #   None
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
        self.first_run_rudder_flag = True
        self.first_run_prop_flag = True

    # Inicializacao do modelo para fazer uma simulacao a partir de um controlador treinado
    # Entrada:
    #   train_code: codigo unico de treinamento "model_<data>_<horario>_<tipo_modelo>"
    #   num_model: codigo do modelo dentro deste codigo fonte - model_prop_<num_model> ou model_rudder_<num_model>
    #   flag_rnn: flag para indicar se o modelo a ser carregado e' do tipo recorrente para fazer a transicao stateful e stateless
    # Saida:
    #   None
    def load_model(self, train_code, num_model, flag_rnn):
        # Salva o codigo unico de treinamento no objeto
        self.train_code = train_code

        if flag_rnn:  # Precisa fazer transicao stateless para stateful
            # Reinstancia o modelo do leme de forma stateful
            self.model_rudder = getattr(self, 'model_rudder_' + str(num_model))(len(inp_rudder), 1, True)
            # Carrega o modelo treinado e transfere os pesos para o novo modelo stateful
            aux_rudder = load_model('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder.h5')
            for i in range(len(self.model_rudder.layers)):
                self.model_rudder.layers[i].set_weights(aux_rudder.layers[i].get_weights())

            # Reinstancia o modelo do propulsor de forma stateful
            self.model_prop = getattr(self, 'model_prop_' + str(num_model))(len(inp_prop), 1, True)
            # Carrega o modelo treinado e transfere os pesos para o novo modelo stateful
            aux_prop = load_model('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller.h5')
            for i in range(len(self.model_prop.layers)):
                self.model_prop.layers[i].set_weights(aux_prop.layers[i].get_weights())
        else:  # Nao precisa fazer a transicao, basta carregar o modelo treinado
            self.model_rudder = load_model('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder.h5')
            self.model_prop = load_model('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller.h5')

        # Carrega os scalers usados nos dados de treinamento
        self.x_rudder_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder_x_scaler.save')
        self.y_rudder_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder_y_scaler.save')
        self.x_prop_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller_x_scaler.save')
        self.y_prop_scaler = joblib.load('./keras_logs/' + self.train_code + '/' + self.train_code + '_propeller_y_scaler.save')

        # Inicializa as flags adequadamente
        self.has_prop_scaler = True
        self.has_rudder_scaler = True
        self.loaded_data = False
        self.rudder_logger = Train_Log(RUDDER)
        self.prop_logger = Train_Log(PROPELLER)
        self.first_run_rudder_flag = True
        self.first_run_prop_flag = True

    # Define o comando de leme adequado
    # Entrada:
    #   input: vetor com o estado da embarcacao no mesmo formato que inp_rudder
    # Saida:
    #   None
    def select_rudder_action(self, input):
        print('Entrada rudder: ' + str(input))
        # Reescala o estado baseado no treinamento
        input = self.x_rudder_scaler.transform(np.expand_dims(input, 0))
        # Adequa o tamanho do vetor
        input = np.expand_dims(input, 0)
        # Requisita o comando de leme previsto para a rede
        output = self.model_rudder.predict(input)
        # Reescala inversamente baseado no treinamento
        output = self.y_rudder_scaler.inverse_transform(output[0])
        print('Saida rudder: ' + str(output))
        return output[0][0]

    # Define o comando de leme adequado
    # Entrada:
    #   input: vetor com o estado da embarcacao no mesmo formato que inp_prop
    # Saida:
    #   None
    def select_prop_action(self, input):
        print('Entrada propeller: ' + str(input))
        # Reescala o estado baseado no treinamento
        input = self.x_prop_scaler.transform(np.expand_dims(input, 0))
        # Adequa o tamanho do vetor
        input = np.expand_dims(input, 0)
        # Requisita o comando de leme previsto para a rede
        output = self.model_prop.predict(input)
        # Reescala inversamente baseado no treinamento
        output = self.y_prop_scaler.inverse_transform(output[0])
        # Discretiza o comando de maquina
        output = np.asarray(pd.cut(output[:, 0], [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5], right=False, labels=[-4, -3, -2, -1, 0, 1, 2, 3, 4]))
        print('Saida propeller: ' + str(output))
        return output[0]

    # Realiza o treinamento de uma nova rede neural
    # Entrada:
    #   files_path: local dos dados de treinamento
    #   num_model: codigo do modelo dentro deste codigo fonte - model_prop_<num_model> ou model_rudder_<num_model>
    # Saida:
    #   None
    def train(self, files_path, num_model=1):
        # Quantidade de batches treinados em paralelo
        batch_size = 256

        # Tamanho de cada batch - quantidade de estados em cada batch
        sequence_length = 1000

        # Define parametros de identificacao do objeto
        self.set_num_model(num_model)
        self.set_train_code()

        # Faz o carregamento dos dados de treinamento
        if not self.loaded_data:
            print('****** CARREGANDO DADOS ******')
            df, limits = self.load_data(files_path)
            self.loaded_data = True
            print('DF shape:' + str(df.shape))

        print('\n****** TREINAMENTO MODELO ' + self.train_code + ' ******')
        # Faz o treinamento do leme
        try:
            self.run_train(df, limits, batch_size, sequence_length, RUDDER)
        except Exception as e:
            self.rudder_logger.flag_error = True
            self.rudder_logger.log_error = e
        self.rudder_logger.write_log(self.train_code)

        # Faz o treinamento do propulsor
        try:
            self.run_train(df, limits, batch_size, sequence_length, PROPELLER)
        except Exception as e:
            self.prop_logger.flag_error = True
            self.prop_logger.log_error = e
        self.prop_logger.write_log(self.train_code)

    # Setter do numero do modelo
    # Entrada:
    #   num_model: codigo do modelo dentro deste codigo fonte - model_prop_<num_model> ou model_rudder_<num_model>
    # Saida:
    #   None
    def set_num_model(self, num_model):
        self.num_model = num_model

    # Gerador e setter do codigo de treinamento
    # Entrada:
    #   None
    # Saida:
    #   None
    def set_train_code(self):
        # train_code: codigo unico de treinamento "model_<data>_<horario>_<tipo_modelo>"
        year = str(datetime.datetime.now().year)
        month = str(datetime.datetime.now().month)
        day = str(datetime.datetime.now().day)
        hour = str(datetime.datetime.now().hour)
        minute = str(datetime.datetime.now().minute)
        self.train_code = 'model_' + year + '-' + month + '-' + day + '_' + hour + '-' + minute

    # Carregamento dos dados de treinamento
    # Entrada:
    #   loc: endereco dos arquivos de treinamento
    # Saida:
    #   df: dataframe com os estados das simulacoes para treinamento segundo load_columns
    #   limits: vetor com as posicoes dos ultimos estados de cada treinamento contido em df
    def load_data(self, loc):
        # Caso queira reduzir o numero de pontos utilizados para o treinamento, basta aumentar essa variavel
        # Se alterar, nao esquecer de ir em simulation_settings e mudar a discretizacao do tempo na simulacao
        step_drop = 1

        # Define os limites de estados entre simulacoes para nao criar batches com dados de duas simulacoes diferentes
        limits = list()

        # Carrega os dados
        file = os.listdir(loc)
        df = pd.DataFrame()
        for f in file:
            path = loc + f
            dt = pd.read_csv(path, sep=" ", skiprows=11)
            dt = dt.iloc[::step_drop]
            df = df.append(dt, ignore_index=True)
            limits.append(df.shape[0]-1)
        print('Dados carregados: ' + str(df.shape[0]) + ', ' + str(df.shape[1]))
        return df[load_columns], limits

    # Realiza o treinamento da rede neural
    # Entrada:
    #   df: dataframe com os estados das simulacoes para treinamento
    #   limits: vetor com as posicoes dos ultimos estados de cada treinamento contido em df
    #   batch_size: quantidade de batches em treinamento em paralelo
    #   sequence_length: tamanho de cada - quantidade de estados
    #   type: RUDDER ou PROPELLER
    # Saida:
    #   None
    def run_train(self, df, limits, batch_size, sequence_length, type):
        # Separa as simulacoes de treinamento e de teste e gera os batches de treinamento
        x_train, y_train, x_test, y_test, validation_data = self.prepare_data(df, type)
        generator = self.batch_generator(limits, batch_size, sequence_length, x_train, y_train)

        num_x_signals = x_train.shape[1]
        num_y_signals = y_train.shape[1]

        # Define o modelo de treinamento segundo num_model e type
        if type == RUDDER:
            model = getattr(self, 'model_rudder_' + str(self.num_model))(num_x_signals, num_y_signals)
            print('\t>>> RUDDER <<<')
        else:
            model = getattr(self, 'model_prop_' + str(self.num_model))(num_x_signals, num_y_signals)
            print('\t>>> PROPELLER <<<')
        print(model.summary())

        # Define os callbacks
        callbacks = self.define_callback(type)

        # Salva o tempo para mensurar o tempo de treinamento
        init_time = datetime.datetime.now()

        # Inicia o treinamento
        # Pode-se calibrar os valores de epoch e steps_per_epoch
        model.fit_generator(generator=generator,
                            epochs=100,
                            steps_per_epoch=100,
                            validation_data=validation_data,
                            callbacks=callbacks)

        # Define o tempo de treinamento e o valor da acuracia nos dados de teste
        train_time = datetime.datetime.now() - init_time
        loss = self.run_test(model, x_test, y_test, type)
        print("Loss: " + str(loss))

        # Salva o modelo
        strout = 'rudder' if type == RUDDER else 'propeller'
        model.save('./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strout + '.h5')

        # Salva os scalers utilizados
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

        # Salva os dados para gerar o relatorio de treinamento
        if type == RUDDER:
            self.model_rudder = model
            self.rudder_logger.train_time = train_time
            self.rudder_logger.loss = loss
        else:
            self.model_prop = model
            self.prop_logger.train_time = train_time
            self.prop_logger.loss = loss

    # Realiza o teste para verificar a performance
    # Entrada:
    #   model: modelo a ser testado
    #   x_test: vetor com os estados que servem de teste
    #   y_test: vetor com as saidas esperadas de comparacao
    #   type: RUDDER ou PROPELLER
    # Saida:
    #   Resultado do teste - loss
    def run_test(self, model, x_test, y_test, type):
        if type == RUDDER:
            strtype = "rudder"
        else:
            strtype = "propeller"

        # Carrega os pesos do modelo
        try:
            model.load_weights('./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strtype + '.keras')
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)

        # Retorna o loss
        return model.evaluate(x=np.expand_dims(x_test, axis=0),
                              y=np.expand_dims(y_test, axis=0))

    # Preparo dos dados para realizar o treinamento e o teste
    # Entrada:
    #   df: dataframe com os estados das simulacoes para treinamento
    #   type: RUDDER ou PROPELLER
    # Saida:
    #   x_train_scaled: vetor com os dados dos estados de treino escalados
    #   y_train_scaled: vetor com os dados da saida de treino escalados
    #   x_test_scaled: vetor com os dados dos estados de teste escalados
    #   y_test_scaled: vetor com os dados da saida de teste escalados
    #   validation_data: vetor com os dados de validacao enquanto gera o treinamento
    def prepare_data(self, df, type):
        # Hardcode de alguns casos para validacao
        #Caso 05 usado como validacao
        #validation_range = range(66955, 84410)
        #Caso 01 usado como validacao
        validation_range = range(17208)

        # Flag para forcar um overfit nos dados - treino = teste
        flag_overfit = False

        # Seleciona os parametros da rede
        if type == RUDDER:
            input_columns = inp_rudder
            target_column = target_rudder
        else:
            input_columns = inp_prop
            target_column = target_prop

        # Separa em estados de entrada e comandos de saida
        x_data = df[input_columns].values
        y_data = df[target_column].values
        y_data = np.reshape(y_data, (len(y_data), 1))

        # Overfit faz dados de treino = teste, sem overfit faz separado
        if flag_overfit:
            x_train = x_data
            y_train = y_data
            x_test = x_data
            y_test = y_data
        else:
            x_train = np.delete(x_data, np.s_[validation_range], axis=0)
            y_train = np.delete(y_data, np.s_[validation_range], axis=0)
            x_test = x_data[validation_range]
            y_test = y_data[validation_range]

        # Verifica a necessidade de gerar um scaler dos dados para normalizar - melhora no treinamento
        if (type == RUDDER and not self.has_rudder_scaler) or (type == PROPELLER and not self.has_prop_scaler):
            self.generate_scaler(df, type)

        # Escala os dados de treinamento e teste
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

        # Gera o vetor de validacao
        validation_data = (np.expand_dims(x_test_scaled, axis=0),
                           np.expand_dims(y_test_scaled, axis=0))

        return x_train_scaled, y_train_scaled, x_test_scaled, y_test_scaled, validation_data

    # Gerador do scaler dos dados pelo uso de MinMaxScaler
    # Entrada:
    #   df: dataframe com os estados das simulacoes para treinamento
    #   type: RUDDER ou PROPELLER
    # Saida:
    #   None
    def generate_scaler(self, df, type):
        # Seleciona os parametros da rede
        if type == RUDDER:
            input_columns = inp_rudder
            target_column = target_rudder
        else:
            input_columns = inp_prop
            target_column = target_prop

        # Seleciona o dados
        x_data = df[input_columns].values
        y_data = df[target_column].values
        y_data = np.reshape(y_data, (len(y_data), 1))

        # Cria e define os limites do scaler
        x_scaler = MinMaxScaler()
        x_scaler.fit(x_data)
        y_scaler = MinMaxScaler()
        y_scaler.fit(y_data)

        # Guarda o scaler
        if type == RUDDER:
            self.x_rudder_scaler = x_scaler
            self.y_rudder_scaler = y_scaler
            self.has_rudder_scaler = True
        else:
            self.x_prop_scaler = x_scaler
            self.y_prop_scaler = y_scaler
            self.has_prop_scaler = True

    # Gerador de batches para realizar os treinamentos
    # Baseado em: https://www.youtube.com/watch?v=6f67zrH-_IE
    # Entrada:
    #   limits: vetor com as posicoes dos ultimos estados de cada treinamento contido em df
    #   batch_size: quantidade de batches em treinamento em paralelo
    #   sequence_length: tamanho de cada - quantidade de estados
    #   x_data: vetor com os dados dos estados de treino escalados
    #   y_data: vetor com os dados de saida de treino escalados
    # Saida:
    #   Batches de estado e de saidas esperadas
    def batch_generator(self, limits, batch_size, sequence_length, x_data, y_data):
        while True:
            x_shape = (batch_size, sequence_length, x_data.shape[1])
            x_batch = np.zeros(shape=x_shape, dtype=np.float16)

            y_shape = (batch_size, sequence_length, y_data.shape[1])
            y_batch = np.zeros(shape=y_shape, dtype=np.float16)

            for i in range(batch_size):
                # Gera um valor randomico que se ele + sequence_length ultrapasse algum limite de simulacao, os ultimos sequence_length estados desta simulacao sao usados no batch
                idx = np.random.randint(x_data.shape[0] - sequence_length)
                last_idx = idx + sequence_length
                for j in range(len(limits)):
                    if limits[j] > idx:
                        limit_idx = limits[j]
                        break
                if last_idx > limit_idx:
                    idx = limit_idx - sequence_length
                    last_idx = idx + sequence_length
                x_batch[i] = x_data[idx:last_idx]
                y_batch[i] = y_data[idx:last_idx]

            yield (x_batch, y_batch)

    # Criar os callbacks para avaliar o desempenho do treinamento
    # Baseado em: https://www.youtube.com/watch?v=6f67zrH-_IE
    # Entrada:
    #   type: RUDDER ou PROPELLER
    # Saida:
    #   Callbacks
    def define_callback(self, type):
        if type == RUDDER:
            strtype = "rudder"
        else:
            strtype = "propeller"

        # Define os checkpoints
        callback_checkpoint = ModelCheckpoint(filepath='./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strtype + '.keras',
                                              monitor='val_loss',
                                              verbose=1,
                                              save_weights_only=True,
                                              save_best_only=True)
        # Define uma parada antes do treinamento completo caso loss nao esteja decrescendo
        callback_early_stopping = EarlyStopping(monitor='val_loss',
                                                patience=5, verbose=1)
        # Log para o tensorflow
        callback_tensorboard = TensorBoard(log_dir='./keras_logs/' + self.train_code + '/' + self.train_code + '_' + strtype + '_log/',
                                           histogram_freq=0,
                                           write_graph=False)
        # Reducao gradual da taxa de aprendizagem
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

    # Faz o teste de avaliacao para verificar aderencia aos dados de simulacoes passadas
    # Entrada:
    #   num_case: numero do caso simulado base para teste
    # Saida:
    #   None
    def test_case(self, num_case):
        # Obtem os dados do caso simulado e faz o processamento de com o scaler e separacao entre entrada e saida
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

        # Faz a previsao das saidas com as redes e processa os dados
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

        # Define uma plotagem comparativa entre o simulado e o gerado pela rede neural
        plt.figure(figsize=(15, 5))
        plt.plot(time, y_prop, label='Original')
        plt.plot(time, y_prop_pred, label='Rede Neural')
        plt.legend()
        plt.title('Comando de máquina')
        plt.ylabel('Comando (-)')
        plt.xlabel('Tempo (s)')
        plt.savefig('./keras_logs/' + self.train_code + '/' + self.train_code + '_prop_caso_' + strcase + '.png')

        plt.figure(figsize=(15, 5))
        plt.plot(time, y_rudder, label='Original')
        plt.plot(time, y_rudder_pred, label='Rede Neural')
        plt.legend()
        plt.title('Comando de leme')
        plt.ylabel('Ângulo (rad)')
        plt.xlabel('Tempo (s)')
        plt.savefig('./keras_logs/' + self.train_code + '/' + self.train_code + '_rudder_caso_' + strcase + '.png')

    """
    ******************************************
    ******************************************
    *********** MODELOS UTILIZADOS ***********
    ******************************************
    ******************************************
    """
    # Cada metodo deve comecar com 'model_' seguido do termo 'rudder' ou 'prop' dependendo de qual a funcao do modelo e terminando uma um numero sequencial a partir de 0 sem repeticao
    # Cada metodo deve ter pelo menos duas entradas: num_x_signals e num_y_signals que sao os numeros de parametros de estados e de saidas
    # Em redes neurais recorrente existe mais um parametro flag_stateful que define se o modelo deve ser criado de maneira stateful (simulacao) ou stateless (treinamento)
    # O retorno de cada metodo deve ser o modelo final projetado
    # Para chamar cada metodo, basta utilizar getattr com o tipo (rudder ou prop) e o seu numero

    def model_rudder_0(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
        else:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        batch_input_shape=(1, None, num_x_signals),
                        stateful=True))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_0(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
        else:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        batch_input_shape=(1, None, num_x_signals),
                        stateful=True))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_1(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_1(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_2(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_2(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_3(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_3(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_4(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adagrad(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_4(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adagrad(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_5(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_5(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_6(self, num_x_signals, num_y_signals, flag_stateful = False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
        else:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        batch_input_shape=(1, None, num_x_signals),
                        stateful=True))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='tanh'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_6(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,)))
        else:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        batch_input_shape=(1, None, num_x_signals),
                        stateful=True))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='tanh'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_7(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softplus'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_7(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softplus'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_8(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(CuDNNGRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(CuDNNLSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_8(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(CuDNNGRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(CuDNNLSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_9(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_9(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_10(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(Dense(units=128,
                          input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        dropout=0.1))
        else:
            mdl.add(Dense(units=128,
                          input_shape=(None, num_x_signals,),
                          batch_input_shape=(1, None, num_x_signals)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        dropout=0.1,
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_10(self, num_x_signals, num_y_signals,flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(Dense(units=128,
                        input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        dropout=0.1))
        else:
            mdl.add(Dense(units=128,
                          input_shape=(None, num_x_signals,),
                          batch_input_shape=(1, None, num_x_signals)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        dropout=0.1,
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_11(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        else:
            mdl.add(Dense(units=128,
                          input_shape=(None, num_x_signals,),
                          batch_input_shape=(1, None, num_x_signals)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        dropout=0.2,
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_11(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
            mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        else:
            mdl.add(Dense(units=128,
                          input_shape=(None, num_x_signals,),
                          batch_input_shape=(1, None, num_x_signals)))
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        dropout=0.2,
                        stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_12(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_12(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_13(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adagrad(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_13(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adagrad(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_14(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_14(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_15(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='tanh'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_15(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='tanh'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_16(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softplus'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_16(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                    input_shape=(None, num_x_signals,)))
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softplus'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_17(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
            mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        else:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        batch_input_shape=(1, None, num_x_signals),
                        stateful=True))
            mdl.add(LSTM(units=128,
                         return_sequences=True,
                         input_shape=(None, num_x_signals,),
                         stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_17(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
            mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        else:
            mdl.add(GRU(units=128,
                        return_sequences=True,
                        input_shape=(None, num_x_signals,),
                        batch_input_shape=(1, None, num_x_signals),
                        stateful=True))
            mdl.add(LSTM(units=128,
                         return_sequences=True,
                         input_shape=(None, num_x_signals,),
                         stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_18(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_18(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_19(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_19(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    dropout=0.2))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_20(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_20(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,),
                    recurrent_dropout=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_21(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adagrad(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_21(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adagrad(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_22(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_22(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = Adadelta(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_23(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='tanh'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_23(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='tanh'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_24(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softplus'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_24(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(GRU(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(LSTM(units=128,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dense(num_y_signals, activation='softplus'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_25(self, num_x_signals, num_y_signals):
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

    def model_prop_25(self, num_x_signals, num_y_signals):
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

    def model_rudder_26(self, num_x_signals, num_y_signals):
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

    def model_prop_26(self, num_x_signals, num_y_signals):
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

    def model_rudder_27(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_27(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_28(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_28(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_29(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_29(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_30(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_30(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_31(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=128,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_31(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=128,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_32(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_32(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_33(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_33(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_34(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=128,
                      activation='sigmoid'))
        mdl.add(Dense(units=128,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_34(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=128,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=128,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_35(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_35(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_36(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_36(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(units=64,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_37(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_37(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_38(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_38(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_39(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_39(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_40(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softmax'))
        mdl.add(Dense(units=32,
                      activation='softmax'))
        mdl.add(Dense(units=32,
                      activation='softmax'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_40(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softmax'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softmax'))
        mdl.add(Dense(units=32,
                      activation='softmax'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_41(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='elu'))
        mdl.add(Dense(units=32,
                      activation='elu'))
        mdl.add(Dense(units=32,
                      activation='elu'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_41(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='elu'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='elu'))
        mdl.add(Dense(units=32,
                      activation='elu'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_42(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='selu'))
        mdl.add(Dense(units=32,
                      activation='selu'))
        mdl.add(Dense(units=32,
                      activation='selu'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_42(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='selu'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='selu'))
        mdl.add(Dense(units=32,
                      activation='selu'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_43(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softplus'))
        mdl.add(Dense(units=32,
                      activation='softplus'))
        mdl.add(Dense(units=32,
                      activation='softplus'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_43(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softplus'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softplus'))
        mdl.add(Dense(units=32,
                      activation='softplus'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_44(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softsign'))
        mdl.add(Dense(units=32,
                      activation='softsign'))
        mdl.add(Dense(units=32,
                      activation='softsign'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_44(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softsign'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='softsign'))
        mdl.add(Dense(units=32,
                      activation='softsign'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_45(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='relu'))
        mdl.add(Dense(units=32,
                      activation='relu'))
        mdl.add(Dense(units=32,
                      activation='relu'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_45(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='relu'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='relu'))
        mdl.add(Dense(units=32,
                      activation='relu'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_46(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='tanh'))
        mdl.add(Dense(units=32,
                      activation='tanh'))
        mdl.add(Dense(units=32,
                      activation='tanh'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_46(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='tanh'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='tanh'))
        mdl.add(Dense(units=32,
                      activation='tanh'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_47(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_47(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='sigmoid'))
        mdl.add(Dense(units=32,
                      activation='sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_48(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='hard_sigmoid'))
        mdl.add(Dense(units=32,
                      activation='hard_sigmoid'))
        mdl.add(Dense(units=32,
                      activation='hard_sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_48(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='hard_sigmoid'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='hard_sigmoid'))
        mdl.add(Dense(units=32,
                      activation='hard_sigmoid'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_49(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='exponential'))
        mdl.add(Dense(units=32,
                      activation='exponential'))
        mdl.add(Dense(units=32,
                      activation='exponential'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_49(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='exponential'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='exponential'))
        mdl.add(Dense(units=32,
                      activation='exponential'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_50(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=32,
                      activation='linear'))
        mdl.add(Dense(units=32,
                      activation='linear'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_50(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=32,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=32,
                      activation='linear'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_51(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=256,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=128,
                      activation='linear'))
        mdl.add(Dense(units=32,
                      activation='linear'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_51(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=256,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=128,
                      activation='linear'))
        mdl.add(Dense(units=32,
                      activation='linear'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_52(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(CuDNNGRU(units=256,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=128,
                    return_sequences=True))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=32,
                    return_sequences=True))
        else:
            mdl.add(CuDNNGRU(units=256,
                             return_sequences=True,
                             input_shape=(None, num_x_signals,),
                             batch_input_shape=(1, None, num_x_signals),
                             stateful=True))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=128,
                             return_sequences=True,
                             stateful=True))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=32,
                             return_sequences=True,
                             stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_52(self, num_x_signals, num_y_signals, flag_stateful=False):
        mdl = Sequential()
        if not flag_stateful:
            mdl.add(CuDNNGRU(units=256,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=128,
                    return_sequences=True))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=32,
                    return_sequences=True))
        else:
            mdl.add(CuDNNGRU(units=256,
                             return_sequences=True,
                             input_shape=(None, num_x_signals,),
                             batch_input_shape=(1, None, num_x_signals),
                             stateful=True))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=128,
                             return_sequences=True,
                             stateful=True))
            mdl.add(Dropout(rate=0.1))
            mdl.add(CuDNNGRU(units=32,
                             return_sequences=True,
                             stateful=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_53(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=512,
                      input_shape=(None, num_x_signals,),
                      activation='tanh'))
        mdl.add(Dense(units=256,
                      activation='tanh'))
        mdl.add(Dense(units=128,
                      activation='tanh'))
        mdl.add(Dense(units=64,
                      activation='tanh'))
        mdl.add(Dense(units=32,
                      activation='tanh'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_53(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=512,
                      input_shape=(None, num_x_signals,),
                      activation='tanh'))
        mdl.add(Dense(units=256,
                      activation='tanh'))
        mdl.add(Dense(units=128,
                      activation='tanh'))
        mdl.add(Dense(units=64,
                      activation='tanh'))
        mdl.add(Dense(units=32,
                      activation='tanh'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_54(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(CuDNNGRU(units=256,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=128,
                    return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=64,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=32,
                    return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=16,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_54(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(CuDNNGRU(units=256,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=128,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=64,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=32,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=16,
                         return_sequences=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_55(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(CuDNNGRU(units=256,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=128,
                    return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=64,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=32,
                    return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=16,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=8,
                         return_sequences=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_55(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(CuDNNGRU(units=256,
                    return_sequences=True,
                    input_shape=(None, num_x_signals,)))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=128,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=64,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=32,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=16,
                         return_sequences=True))
        mdl.add(Dropout(rate=0.1))
        mdl.add(CuDNNGRU(units=8,
                         return_sequences=True))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_rudder_56(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=15,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=10,
                      activation='linear'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

    def model_prop_56(self, num_x_signals, num_y_signals):
        mdl = Sequential()
        mdl.add(Dense(units=10,
                      input_shape=(None, num_x_signals,),
                      activation='linear'))
        mdl.add(Dense(units=5,
                      activation='linear'))
        mdl.add(Dense(num_y_signals, activation='sigmoid'))
        optimizer = RMSprop(lr=0.001)
        mdl.compile(loss='mean_squared_error', optimizer=optimizer)
        return mdl

if __name__ == '__main__':
    model = Controller()
    model.load_model('model_2018-10-19_22-32')
    model.test_case(2)