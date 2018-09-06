import controller
import environment
from simulation_settings import *
from viewer import Viewer
from state_helper import State_Helper

def evaluate_model(path):
    # Posicao inicial da embarcacao [x, y, zz, vx, vy, vzz] = [ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ]
    start_pos = [11000, 5240, -103.5, 3, 0, 0]

    # Define o controlador
    control = controller.Controller(path)

    # Inicializa o ambiente de teste
    env = environment.Environment()
    env.set_up()

    # Iniciailiza o visualizador da simulacao
    viewer = Viewer()
    viewer.plot_boundary(buoys)
    viewer.plot_goal(goal, 100)

    # Inicia a embarcacao na posicao
    env.set_single_start_pos_mode(start_pos)
    env.move_to_next_start()

    # Inicia a ferramenta para facilitar a tarefa de gerar os estados
    st_help = State_Helper(p3dfile, list_buoys, target)

    # Parametros de teste
    # TODO: Calcular direito os parametros de entrada da rede
    IN_VX = 4.18
    IN_VY = -0.18
    IN_TG = 6024
    DESC_FACTOR = 0.5

    # Inicia a simulacao
    for step in range(evaluation_steps):
        state = env.get_state()
        viewer.plot_position(state[ST_POSX], state[ST_POSY], state[ST_POSZZ])
        # TODO: Substituir pelo novo estado
        # input_state = st_help.get_inputtable_state(state)
        # action = control.select_action(input_state)
        action = control.select_action([IN_VX, IN_VY, IN_TG])
        # TODO: Pegar o comando de leme da rede neural
        env.step(1, action)
        IN_TG = IN_TG - DESC_FACTOR


if __name__ == '__main__':
    evaluate_model('model/model.h5')