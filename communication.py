import controller
import environment
from simulation_settings import *
from viewer import Viewer
from state_helper import State_Helper

def evaluate_model(path=None):
    # Posicao inicial da embarcacao [x, y, zz, vx, vy, vzz] = [ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ]
    # start_pos = [11000, 5240, -103.5, 3, 0, 0]
    start_pos = [11000, 5240, -103.5, 3, 0, 0]

    # Define o controlador
    control = controller.Controller()
    # TODO: Melhorar isso
    control.train('C:\\Users\\AlphaCrucis_Control1\\PycharmProjects\\Data_Filter\\Output\\Compilado\\', 0)
    control.test_case(10)

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
        env.step(rudder_level, prop_level)
        st_help.set_position(state)
        final_flag = env.is_final()
        print("\n***Evaluation step " + str(step + 1) + " Completed***")
        if final_flag != 0:
            break


if __name__ == '__main__':
    #evaluate_model('model/model_rudder_2018-9-12-16-49.h5')
    evaluate_model('model/model_rudder_2018-9-12-16-49.h5')