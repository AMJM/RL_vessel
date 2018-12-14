'''
[AKUM-12/12/2018]
Codigo fonte original de https://github.com/zeamendola89/RL_vessel/tree/refac com novos parametros para simulacao de rede neural supervisionada (alteracoes indicadas)

Muito importante pois contem os parametros de controle da simulacao
'''

import datetime
from geometry_helper import GeometryHelper
import actions
import numpy as np
from state_helper import Point

timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def channel_angle_e_ccw(point_a, point_b):
    line = np.array(point_b) - np.array(point_a)
    support = np.array((point_a[0]+10,point_a[1])) - np.array(point_a)
    c = np.dot(line, support) / np.linalg.norm(line) / np.linalg.norm(support)  # -> cosine of the angle
    angle = np.arccos(np.clip(c, -1, 1))
    return 360 - np.rad2deg(angle)

variables_file = "experiment_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
learner_file = "agent" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
sample_file = "samples/samples" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
q_file = "q_table" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
main_loop_iterations = 10
max_fit_iterations = 2000
max_tuples_per_batch = 20000000
maximum_training_steps = 20000000
max_episodes = 5000000
steps_between_actions = 100
funnel_start = (14000, 7000)
N01 = (11724.8015, 5582.9127)
N03 = (9191.6506, 4967.8532)
N05 = (6897.7712, 4417.3295)
N07 = (5539.2417, 4089.4744)
Final = (5812.6136, 3768.5637)
N06 = (6955.9288, 4227.1846)
N04 = (9235.8653, 4772.7884)
N02 = (11770.3259, 5378.4429)
funnel_end = (14000, 4000)
plot = False
goal_heading_e_ccw = channel_angle_e_ccw(N03, N05)

'''
[AKUM-12/12/2018] Inicio das alteracoes
'''
# Variavel para controle do tipo de simulacao
SIM_RL = 0      # Simulacao de reinforcement learning
SIM_SUP = 1     # Simulacao supervisionada
simulation_type = SIM_SUP

# Constantes para ordenacao no vetor de estado
ST_POSX = 0
ST_POSY = 1
ST_POSZZ = 2
ST_VELX = 3
ST_VELY = 4
ST_VELZZ = 5
ST_TARGET = 1
ST_MID = 0
ST_DPB = 0
ST_DSB = 1
ST_COG = 0
ST_SOG = 1
ST_LOCAL_COG = 2

# Vetor do posicionamento das boias
list_buoys = [Point(11724.8015, 5582.9127), Point(11770.3259, 5378.4429), Point(9191.6506, 4967.8532), Point(9235.8653, 4772.7884),
              Point(6897.7712, 4417.3295), Point(6955.9288, 4227.1846), Point(5539.2417, 4089.4744), Point(5812.6136, 3768.5637)]

# Ponto de define o ponto final da embarcacao
target = Point(5790.0505, 3944.9947)

# Vetores com as velocidades da embarcacao discretizadas, depende muito de qual embarcacao esta sendo simulada
discrete_velocity = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
#discrete_velocity = [-0.8, -0.6, -0.4, -0.35, 0.0, 0.35, 0.4, 0.7, 0.8]
#discrete_velocity = [-0.8, -0.7, -0.5, -0.4, 0.0, 0.4, 0.5, 0.7, 0.8]

# Constantes de referencia para maximo leme e numero de ordem de maquinas possiveis
MAX_RUDDER_RAD = 0.61
NUM_PROP_VEL = 4

# Parametros de controle da simulacao
# Quantidade maxima de passos simulados
evaluation_steps = 100000
# Quantidade de passos entre verificacao de estados e envio de comandos
steps = 1 #20
# Tempo de simulacao entre passos - muito importante para redes neurais recorrentes, precisam ser compativeis
step_increment = 0.1 #0.5
'''
[AKUM-12/12/2018] Fim das alteracoes
'''

goal_vel_lon = 3
buoys = (funnel_start, N01, N03, N05, N07, Final, N06, N04, N02, funnel_end)
vessel_id = '36'
rudder_id = '0'
thruster_id = '0'
scenario = 'default'
goal = ((N07[0]+Final[0])/2, (N07[1]+Final[1])/2)
goal_factor = 100
upper_shore = [funnel_start, N01, N03, N05, N07]
lower_shore = [N06, N04, N02, funnel_end]

geom_helper = GeometryHelper()
geom_helper.set_ship_geometry(((-5, 0), (0, 10), (5, 0)))
geom_helper.set_boundary_points(buoys)
geom_helper.set_goal_rec(goal[0], goal[1])
geom_helper.set_shore_lines(upper_shore, lower_shore)
geom_helper.set_guidance_line()

action_space = actions.BaseAction('rudder_complete')


# viewer = Viewer()
# viewer.plot_boundary(buoys)
# viewer.plot_goal(goal, 100)
