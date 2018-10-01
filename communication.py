import controller
import environment
from simulation_settings import *
from viewer import Viewer
from state_helper import State_Helper
import os
import time

def train_evaluate_mode(num_model, num_test, init_state_list):
    # Define o controlador
    model_list = list()
    for i in range(num_model):
        control = controller.Controller()
        try:
            control.train('C:\\Users\\AlphaCrucis_Control1\\PycharmProjects\\Data_Filter\\Output\\Compilado\\', i)
            model_list.append(control.train_code)
            for j in num_test:
                control.test_case(j)
        except Exception as e:
            write_log(control.train_code, flag_error=True, error_log=e)
        del control

    for train_code in model_list:
        flag_header = True
        for init_state in init_state_list:
            try:
                final_flag, step = evaluate_model(train_code, init_state)
                write_log(train_code, init_state, flag_header=flag_header, final_flag=final_flag, num_step=step)
                flag_header = False
            except Exception as e:
                write_log(train_code, flag_error=True, error_log=e)
            time.sleep(30)

def evaluate_model(train_code, start_pos):
    # Posicao inicial da embarcacao [x, y, zz, vx, vy, vzz] = [ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ]
    # start_pos = [11000, 5240, -103.5, 3, 0, 0]
    start_pos[2] = -270 - start_pos[2] # CUIDADO COM ESSE ANGULO

    # Carrega o controlador
    control = controller.Controller()
    control.load_model(train_code)

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
    st_help = State_Helper(p3dfile, list_buoys, target)

    # Inicia a simulacao
    for step in range(evaluation_steps):
        state = env.get_state()
        viewer.plot_position(state[ST_POSX], state[ST_POSY], state[ST_POSZZ])
        input_rudder_state = st_help.get_inputtable_rudder_state(state)
        input_prop_state = st_help.get_inputtable_prop_state(state)

        rudder_level = st_help.get_converted_rudder(control.select_rudder_action(input_rudder_state))
        prop_level = st_help.get_converted_propeller(control.select_prop_action(input_prop_state))
        #env.step(0, 0)
        env.step(rudder_level, prop_level)
        st_help.set_position(state)
        final_flag = env.is_final()
        print("***Evaluation step " + str(step + 1) + " Completed***\n")
        if final_flag != 0:
            break

    # Finaliza o ambiente
    env.finish()

    return final_flag, step

def write_log(train_code, init_state, final_flag=0, num_step=0, flag_header=False, flag_error=False, error_log=None):
    os.chdir('..')
    year = str(datetime.datetime.now().year)
    month = str(datetime.datetime.now().month)
    day = str(datetime.datetime.now().day)
    file = open('./keras_logs/simulation_log_' + year + '-' + month + '-' + day + '.txt', 'a+')

    if flag_header:
        file.write('\n\n*** Treinamento ' + train_code + ':')

    if flag_error:
        file.write('\n\tCondição inicial: ' + str(init_state) + ': Erro na execução do teste:')
        file.write('\n\t' + str(error_log))
    else:
        if final_flag == 0:
            file.write('\n\tCondição inicial: ' + str(init_state) + ': Manobra realizada com sucesso - Final flag: ' + str(final_flag))
        else:
            file.write('\n\tCondição inicial: ' + str(init_state) + ': Ocorreu colisão - Final flag: ' + str(final_flag))
        file.write('\n\tQuantidade de steps: ' + str(num_step))


    file.close()


if __name__ == '__main__':
    #evaluate_model('model/model_rudder_2018-9-12-16-49.h5')
    #evaluate_model('model_2018-10-1_12-31')
    num_models = 22
    init_state_list = [[11632.01, 5501.04, -165.64, 5.06, -0.35, -0.03]]
    test_list = [10, 20]
    train_evaluate_mode(num_models, test_list, init_state_list)
