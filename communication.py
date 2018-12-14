'''
[AKUM-14/12/2018]
Arquivo de interface entre environment e controller
'''

'''
Importacao das bibliotecas a serem utilizadas
'''
import controller
import environment
from simulation_settings import *
from viewer import Viewer
from state_helper import State_Helper
import os
import time
import pandas as pd
import openpyxl
import json

# Treina o modelo, gera o teste com simulacoes passadas e valida no Dyna
# Entrada:
#   num_model: vetor com os codigos dos modelos de controller.py - model_prop_<num_model> ou model_rudder_<num_model>
#   num_test: vetor com os numeros dos casos simulados base para teste
#   init_state_list: vetor com os estados das condicoes iniciais de simulacao no formato [posicao_x, posicao_y, angulo_aproamento leste antihorario, velocidade_avanco, velocidade_deriva, velocidade_guinada]
#   flag_rnn: define se o modelo e' ou nao recorrente
# Saida:
#   None
def train_evaluate_mode(num_model, num_test, init_state_list, flag_rnn):
    # Duas etapas: treinamento e teste nos casos simulados para depois fazer a simulacao de todos os modelos no Dyna
    model_list = list()

    # Treinamento e teste
    for i in num_model:
        # Instancia o controlador
        control = controller.Controller()
        try:
            # Define os dados da simulacao a serem guardados para posterior analise mais criteriosa ou plotagem
            dfLogger = pd.DataFrame(columns=['NumModel', 'TrainCode', 'Rudder Loss', 'Propeller Loss', 'Rudder Train Time', 'Propeller Train Time', 'Rudder Model', 'Propeller Model'])
            # Treino da rede e guarda o modelo treinado
            control.train('C:\\Users\\AlphaCrucis_Control1\\PycharmProjects\\Data_Filter\\Output\\CompiladoSelecionados\\', i)
            model_list.append(control.train_code)

            # Gera o log de treinamento com resumo em planilha do excel
            str_rudder = []
            str_prop = []
            control.model_rudder.summary(print_fn=lambda x: str_rudder.append(x + '\n'))
            control.model_prop.summary(print_fn=lambda x: str_prop.append(x + '\n'))
            str_rudder = ''.join(str_rudder)
            str_prop = ''.join(str_prop)
            dfLogger.loc[len(dfLogger)] = [i, control.train_code, control.rudder_logger.loss, control.prop_logger.loss, str(control.rudder_logger.train_time), str(control.prop_logger.train_time), str_rudder, str_prop]

            # Gera a verificacao do controlador com os dados das simulacoes
            for j in num_test:
                control.test_case(j)
        except Exception as e:
            try:
                dfLogger.append([i, control.train_code, 'Error', 'Error', 'Error', 'Error'])
            except Exception as e:
                dfLogger.append([i, 'Error', 'Error', 'Error', 'Error', 'Error'])
                write_log(control.train_code, flag_error=True, error_log=e, flag_excel=True, dfLogger=dfLogger)
        # Escreve o log do treinamento e teste de comparacao com as simulacoes
        write_log(control.train_code, flag_excel=True, dfLogger=dfLogger)

        # Deleta o controlador
        del control

    # Validacao com o Dyna
    for train_code in model_list:
        # No primeiro de modelo gera um cabecalho no log
        flag_header = True

        # Realiza a validacao em varios casos iniciais
        for init_state in init_state_list:
            try:
                # Avalia com o Dyna e gera o log de simulacao
                final_flag, step, dfStates, viewer = evaluate_model(train_code, init_state, num_model if flag_rnn else None)
                write_log(train_code, init_state=init_state, flag_header=flag_header, final_flag=final_flag, num_step=step, dfStates=dfStates, viewer=viewer)
                flag_header = False
            except Exception as e:
                write_log(train_code, flag_error=True, error_log=e)

            # Importante ter o sleep para ter tempo de fechar por completo a simulacao antes de comecar outro
            time.sleep(30)

# Valida modelo no Dyna
# Entrada:
#   train_code: codigo unico de treinamento "model_<data>_<horario>_<tipo_modelo>"
#   start_pos: estado da condicao inicial de simulacao no formato [posicao_x, posicao_y, angulo_aproamento leste antihorario, velocidade_avanco, velocidade_deriva, velocidade_guinada]
#   num_model: codigo do modelo de controller.py - model_prop_<num_model> ou model_rudder_<num_model>
# Saida:
#   final_flag: 1 se chegou ate o final do canal e -1 se ocorreu colisao
#   step: quantidade de steps ate a simulacao parar
#   dfStates: conjunto de estados da simulacao
#   viewer: objeto do gerador do turtle de plotagem
def evaluate_model(train_code, start_pos, num_model=None):
    print('Ângulo inicial: ' + str(start_pos[2]))

    # Adaptacao do angulo de aproamento para norte horario
    start_pos[2] = -270 - start_pos[2]

    # Se nao tiver num_model definido e' porque nao precisa do modelo original para ser convertido em stateful
    if num_model is None:
        flag_rnn = False
    else:
        flag_rnn = True

    # Carrega o controlador
    control = controller.Controller()
    control.load_model(train_code, num_model, flag_rnn)

    # Inicializa o ambiente de teste
    env = environment.Environment()
    env.set_up()

    # Inicializa o visualizador da simulacao
    viewer = Viewer()
    viewer.plot_boundary(buoys)
    viewer.plot_goal(goal, 100)

    # Inicia a embarcacao na posicao
    env.set_single_start_pos_mode(start_pos)
    env.move_to_next_start()

    # Inicia a ferramenta para facilitar a tarefa de gerar os estados
    with open('./suape-local.json') as f:
        json_file = json.load(f)
    p3dfile_json = json_file['scenarios'][0]['p3d']
    st_help = State_Helper(p3dfile_json, list_buoys, target)

    # Armazenador dos estados
    dfStates = pd.DataFrame(columns=['time', 'x', 'y', 'zz', 'vx', 'vy', 'vzz', 'rudder_demanded', 'propeller_demanded', 'distance_midline', 'distance_target', 'distance_port', 'distance_starboard', 'cog', 'sog', 'local_cog'])

    # Inicia a simulacao
    for step in range(evaluation_steps):
        # Adequa o estado para a versao local
        state = st_help.convert_cartesian_to_local(env.get_state())

        # Plota no turtle
        viewer.plot_position(state[ST_POSX], state[ST_POSY], state[ST_POSZZ])

        # Gera a entrada no formato dos controladores
        input_rudder_state = st_help.get_inputtable_rudder_state(state)
        input_prop_state = st_help.get_inputtable_prop_state(state)

        # Gera os estados auxiliares para fazer a analise posterior a simulacao
        aux_state = st_help.get_aux_states(state)

        # Obtem os comandos de leme e de maquina previsto pelo controlador
        rudder_level = control.select_rudder_action(input_rudder_state)
        prop_level = control.select_prop_action(input_prop_state)

        # Armazena o estado para avaliacao posterior a simulacao
        dfStates.loc[len(dfStates)] = [env.simulation.get_current_time(), state[0], state[1], -270 - state[2], state[3], state[4], state[5], rudder_level, prop_level, input_prop_state[4], input_prop_state[5],
                                       aux_state[0], aux_state[1], aux_state[2], aux_state[3], aux_state[4]]

        # Converte o comando de leme e maquina para o formato do Dyna
        rudder_level = st_help.get_converted_rudder(rudder_level)
        prop_level = st_help.get_converted_propeller(prop_level)

        # Faz um avanco na simulacao
        env.step(rudder_level, prop_level)

        # Atualiza o mapeamento do ambiente
        st_help.set_position(state)

        # Verifica se a simulacao deve ser finalizada
        final_flag = env.is_final()
        print('Time: ' + str(env.simulation.get_current_time()))
        print("***Evaluation step " + str(step + 1) + " Completed***\n")
        if final_flag != 0:
            break

    # Finaliza o ambiente
    env.finish()
    del env

    return final_flag, step, dfStates, viewer

# Escreve o log da simulacao
# Entrada:
#   train_code: codigo unico de treinamento "model_<data>_<horario>_<tipo_modelo>"
#   init_state: estado da condicao inicial de simulacao no formato [posicao_x, posicao_y, angulo_aproamento leste antihorario, velocidade_avanco, velocidade_deriva, velocidade_guinada]
#   final_flag: 1 se chegou ate o final do canal e -1 se ocorreu colisao
#   num_step: quantidade de steps ate a simulacao parar
#   flag_header: flag de impressao do cabecalho
#   flag_error: flag indicativa de que ocorreu um erro
#   error_log: descricao do erro que ocorreu
#   flag_excel: gerar log resumido na tabela excel
#   dfLogger: resumo a ser escrito no excel
#   dfState: conjunto de estados da simulacao
#   viewer: plotagem do turtle a ser salvo
# Saida:
#   None
def write_log(train_code, init_state=None, final_flag=0, num_step=0, flag_header=False, flag_error=False, error_log=None, flag_excel=False, dfLogger=None, dfStates=None, viewer=None):
    # Dados para o codigo do log <data>_<horario>
    year = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month)
    day = str(datetime.datetime.now().day)
    hour = str(datetime.datetime.now().hour)
    minute = str(datetime.datetime.now().minute)

    if not flag_excel:
        # Precisa redirecionar a pasta raiz pois o environment faz uma troca
        os.chdir('..')
        file = open('./keras_logs/simulation_log_' + year + '-' + month + '-' + day + '.txt', 'a+')

        # Impressao do cabecalho
        if flag_header:
            file.write('\n\n*** Treinamento ' + train_code + ':')

        # Impressao do erro na execucao do teste ou do resultado da manobra de validacao
        if flag_error:
            file.write('\n\tCondição inicial: ' + str(init_state) + ': Erro na execução do teste:')
            file.write('\n\t' + str(error_log))
        else:
            if final_flag == 1:
                file.write('\n\tCondição inicial: ' + str(init_state) + ': Manobra realizada com sucesso - Final flag: ' + str(final_flag))
            else:
                file.write('\n\tCondição inicial: ' + str(init_state) + ': Ocorreu colisão - Final flag: ' + str(final_flag))
            file.write('\n\tQuantidade de steps: ' + str(num_step))
        file.close()

        # Obtem o codigo do caso utilizado na simulacao - condicoes ambientais constantes no p3d
        with open('./dyna/suape-local.json') as f:
            json_file = json.load(f)
        case_code = json_file['scenarios'][0]['p3d'].split('-')[0]

        # Salva o conjunto de estados da simulacao do Dyna
        dfStates.to_csv('./keras_logs/' + train_code + '/simulation_log_' + year + '-' + month + '-' + day + '_' + hour + '-' + minute + '_res' + str(final_flag) + '_' + case_code + '_' + str(init_state) + '.txt', sep=' ', index=False)

        # Salva a plotagem dos estados no turtle
        viewer.save_plot(train_code, year + '-' + month + '-' + day + '_' + hour + '-' + minute + '_res' + str(final_flag) + '_' + case_code + '_' + str(init_state))

    else:
        # Salva o resumo do modelo no excel para facilitar encontrar os dados do modelo
        workbook = openpyxl.load_workbook('./keras_logs/treinamentos.xlsx')
        writer = pd.ExcelWriter('./keras_logs/treinamentos.xlsx', engine='openpyxl')
        writer.book = workbook
        writer.sheets = {ws.title: ws for ws in workbook.worksheets}
        write_header = False
        try:
            startrow = writer.sheets['log_' + year + '-' + month + '-' + day].max_row
        except:
            startrow = 0
            write_header = True
        dfLogger.to_excel(writer, sheet_name='log_' + year + '-' + month + '-' + day, header=write_header, index=False, startrow=startrow)
        writer.save()


if __name__ == '__main__':
    # Para fazer a avaliacao em varias simulacoes com o Dyna
    init_state_list = [
            [11000, 5240, -165.64, 3, 0, 0],
            [11000, 5300, -165.64, 3, 0, 0],
            [11000, 5280, -165.64, 3, 0, 0],
            [11000, 5320, -165.64, 3, 0, 0],
            [11000, 5320, -165.64, 3, 0, 0]]
    train_code_list = [
            'model_2018-10-9_12-37']
    for train_code in train_code_list:
        for init_state in init_state_list:
            final_flag, step, dfStates = evaluate_model(train_code, init_state)
            write_log(train_code, init_state=init_state, final_flag=final_flag, num_step=step, dfStates=dfStates)
            time.sleep(30)


    # Para fazer o treinamento de varios modelos
    #num_models = [6, 11, 17, 52]#range(12, 56)
    #test_list = [5, 10, 12, 20, 29]
    #test_list = [1, 2, 8, 16, 19, 20, 21, 24, 25, 28, 31, 33]
    #test_list = [1, 8, 20]
    #train_evaluate_mode(num_models, test_list, init_state_list, True)
    #final_flag, step, dfStates = evaluate_model('model_2018-10-11_10-43', [11631.89, 5447.37, -162.75, 3.95, -0.51, 0.11])
    #write_log('model_2018-10-11_10-43', init_state=[11631.89, 5447.37, -162.75, 3.95, -0.51, 0.11], flag_header=True, final_flag=final_flag, num_step=step, dfStates=dfStates)
