'''
[AKUM-18/12/2018]
Arquivo de gerenciamento de environment adaptado do original para considerar um aprendizado supervisionado
'''

'''
Importacao das bibliotecas a serem utilizadas
'''
# -*- coding: utf-8 -*-
import threading
import buzz_python
import itertools
import utils
import subprocess
import os
from geometry_helper import is_inbound_coordinate
from simulation_settings import *
import pickle
import random

'''
Classe para gerenciar o environment de simulacao
'''
class Environment(buzz_python.session_subscriber):
    # Construtor da classe
    # Entrada:
    #   rw_mapper: obj RewardMapper para aprendizado por reforço - None para aprendizado supervisionado
    # Saida:
    #   None
    def __init__(self, rw_mapper=None):
        super(Environment, self).__init__()
        self.simulation_id = 'sim'
        self.control_id = '555'
        self.chat_address = '127.0.0.1'
        self.simulation = []
        self.dyna_ctrl_id = '407'
        self.allow_advance_ev = threading.Event()
        self.dyna_ready_ev = threading.Event()
        self.own = []
        self.vessel = []
        self.rudder = []
        self.thruster = []
        self.max_angle = 0
        self.max_rot = 0
        if rw_mapper is None and simulation_type == SIM_RL:
            print("ERRO: Reinforcement learning sem reward map")
        self.reward_mapper = rw_mapper
        self.init_state = list()
        self._final_flag = False
        self.initial_states_sequence = None
        os.chdir('./dyna') # Cuidado se quiser salvar os dados, pode comecar a dar problema pela mudanca da raiz da pasta
        self.dyna_proc = None
        self.accumulated_starts = list()

    # Verifica se a simulacao pode ser considerada finalizada
    # Entrada:
    #   None
    # Saida:
    #   ret: 1 chegou ao final do canal, -1 se colidiu na margem, 0 simulacao nao finalizada
    def is_final(self):
        ret = 0
        if geom_helper.reached_goal():
            ret = 1
        elif geom_helper.ship_collided():
            ret = -1
        print("Final step:", ret)
        return ret

    def on_state_changed(self, state):
        if state == buzz_python.STANDBY:
            print("Dyna ready")
            self.dyna_ready_ev.set()

    def on_time_advanced(self, time):
        step = self.simulation.get_current_time_step()

    def on_time_advance_requested(self, process):
        if process.get_id() == self.dyna_ctrl_id:
            self.allow_advance_ev.set()

    # Inicializacao do ambiente de simulacao
    # Entrada:
    #   None
    # Saida:
    #   None
    def set_up(self):
        self.dyna_proc = subprocess.Popen(
            ['Dyna_reset_prop.exe ', '--pid', '407', '-f', 'suape-local.json', '-c', '127.0.0.1'])
        # ds = buzz_python.create_bson_data_source(self.mongo_addr, self.dbname)
        ds = buzz_python.create_bson_data_source('suape-local.json')
        ser = buzz_python.create_bson_serializer(ds)
        self.simulation = buzz_python.create_simco_simulation(self.simulation_id, self.control_id, ser)
        self.simulation.connect(self.chat_address)
        self.init(self.simulation)
        scn = ser.deserialize_scenario(scenario)
        self.simulation.add(scn)
        factory = self.simulation.get_element_factory()
        r_c = factory.build_runtime_component(self.dyna_ctrl_id)
        self.simulation.add(r_c)
        self.own = self.simulation.get_runtime_component(self.control_id)

        self.simulation.build()
        self.simulation.set_current_time_increment(step_increment)
        self.start()
        self.vessel = self.simulation.get_vessel(vessel_id)

        # self.reset_state(self.init_state[0], self.init_state[1], self.init_state[2],
        #                      self.init_state[3], self.init_state[4], self.init_state[5])
        # self.reward_mapper.update_ship(-200, -200, 10,, 0, 0
        self.simulation.advance_time()
        self.advance()
        # self.initial_states_sequence = itertools.cycle(self.get_initial_states())

        # self.init_state = next(self.initial_states_sequence)
        self.advance()
        self.get_propulsion()

    # Inicializacao da simulacao
    # Entrada:
    #   None
    # Saida:
    #   None
    def start(self):
        self.dyna_ready_ev.wait()
        self.simulation.start()
        self.dyna_ready_ev.clear()

    # [AKUM] Nao tenho certeza, mas acredito que com o id do navio definido seja possivel obter informacoes para o comandos que sao definidos neste metodo
    # Entrada:
    #   None
    # Saida:
    #   None
    def get_propulsion(self):
        thr_tag = buzz_python.thruster_tag()
        self.simulation.is_publisher(thr_tag, thr_tag.SMH_DEMANDED_ROTATION)

        rdr_tag = buzz_python.rudder_tag()
        self.simulation.is_publisher(rdr_tag, rdr_tag.SMH_DEMANDED_ANGLE)

        thr_ev_taken_tag = buzz_python.thruster_control_taken_event_tag()
        self.simulation.is_publisher(thr_ev_taken_tag, True)

        rdr_ev_taken_tag = buzz_python.rudder_control_taken_event_tag()
        self.simulation.is_publisher(rdr_ev_taken_tag, True)

        self.thruster = self.vessel.get_thruster(thruster_id)
        ctrl_ev = buzz_python.create_thruster_control_taken_event(self.thruster)
        ctrl_ev.set_controlled_fields(thr_tag.SMH_DEMANDED_ROTATION)
        ctrl_ev.set_controller(self.own)
        self.simulation.publish_event(ctrl_ev)

        self.rudder = self.vessel.get_rudder(rudder_id)
        ctrl_ev_r = buzz_python.create_rudder_control_taken_event(self.rudder)
        ctrl_ev_r.set_controlled_fields(rdr_tag.SMH_DEMANDED_ANGLE)
        ctrl_ev_r.set_controller(self.own)
        self.simulation.publish_event(ctrl_ev_r)
        self.simulation.update(self.thruster)
        self.max_rot = self.thruster.get_maximum_rotation()
        self.simulation.update(self.rudder)
        self.max_angle = self.rudder.get_maximum_angle()

    # Obtem o estado da embarcacao
    # Entrada:
    #   None
    # Saida:
    #   Estado: vetor com o estado [posicao_x, posicao_y, angulo_aproamento norte horario, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    def get_state(self):
        self.simulation.sync(self.vessel)
        lin_pos_vec = self.vessel.get_linear_position()
        ang_pos_vec = self.vessel.get_angular_position()
        lin_vel_vec = self.vessel.get_linear_velocity()
        ang_vel_vec = self.vessel.get_angular_velocity()
        x = lin_pos_vec[0]
        y = lin_pos_vec[1]
        theta = ang_pos_vec[2]
        xp = lin_vel_vec[0]
        yp = lin_vel_vec[1]
        theta_p = ang_vel_vec[2]
        return x, y, theta, xp, yp, theta_p

    # Avanca a simulacao para o proximo passo de tempo
    # Entrada:
    #   None
    # Saida:
    #   None
    def advance(self):
        self.allow_advance_ev.wait()
        self.simulation.advance_time()
        self.allow_advance_ev.clear()

    # Envia os comandos de maquina e leme para prosseguir a simulacao
    # Entrada:
    #   angle_level: porcentagem do leme maximo
    #   rot_level: porcentagem da rotacao maxima
    # Saida:
    #   Estado: vetor com o estado [posicao_x, posicao_y, angulo_aproamento norte horario, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    def step(self, angle_level, rot_level):
        """**Implement Here***
            The state transition. The agent executed the action in the parameter
            :param rot_level:
            :param angle_level:
        """

        print('Rotation level: ', rot_level)
        print('Angle level: ', angle_level)
        self.thruster.set_demanded_rotation(rot_level * self.max_rot)
        self.simulation.update(self.thruster)
        self.rudder.set_demanded_angle(angle_level * self.max_angle)
        self.simulation.update(self.rudder)

        cycle = 0
        for cycle in range(steps):
            self.advance()
        statePrime = self.get_state()  # Get next State
        print('statePrime: ', statePrime)

        if simulation_type == SIM_RL:
            self.reward_mapper.update_ship(statePrime[0], statePrime[1], statePrime[2], statePrime[3], statePrime[4],
                                           statePrime[5], angle_level, rot_level)
            rw = self.reward_mapper.get_reward()
            return statePrime, rw
        else:
            return statePrime

    # Define o estado inicial da proxima simulacao
    # Entrada:
    #   None
    # Saida:
    #   None
    def move_to_next_start(self):
        self.init_state = next(self.initial_states_sequence)

    # Define simulacao unica com apenas um estado inicial
    # Entrada:
    #   init_state: vetor com o estado [posicao_x, posicao_y, angulo_aproamento norte horario, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    # Saida:
    #   None
    def set_single_start_pos_mode(self, init_state=None):
        if not init_state:
            org_state = self.get_state()
            vel_l, vel_drift, theta = utils.global_to_local(org_state[3], org_state[4], org_state[2])
            self.init_state = (org_state[0], org_state[1], org_state[2], vel_l, vel_drift, 0)
        else:
            self.init_state = (init_state[0], init_state[1], init_state[2], init_state[3], init_state[4], init_state[5])
        self.reset_state_localcoord(self.init_state[0], self.init_state[1], self.init_state[2], self.init_state[3],
                                    self.init_state[4], self.init_state[5])
        dummy_list = list()
        dummy_list.append(self.init_state)
        self.initial_states_sequence = itertools.cycle(dummy_list)

    # Define o estado na embarcacao do Dyna
    # Entrada:
    #   x: posicao cartesiana em x (m)
    #   y: posicao cartesiana em y (m)
    #   theta: angulo de aproamento em graus norte horario (º)
    #   vel_lon: velocidade de avanco (m/s)
    #   vel_drift: velocidade de deriva (m/s)
    #   vel_theta: velocidade de guinada (º/s)
    # Saida:
    #   None
    def reset_state_localcoord(self, x, y, theta, vel_lon, vel_drift, vel_theta):
        # Apparently Dyna ADV is using theta n_cw
        self.vessel.set_linear_position([x, y, 0.00])
        self.vessel.set_linear_velocity([vel_lon, vel_drift, 0.00])
        self.vessel.set_angular_position([0.00, 0.00, theta])
        self.vessel.set_angular_velocity([0.00, 0.00, vel_theta])
        if simulation_type == SIM_RL:
            self.reward_mapper.initialize_ship(x, y, theta, vel_lon, vel_drift, vel_theta)
        self.simulation.sync(self.vessel)
        self.advance()
        self.advance()
        self.advance()
        self.simulation.sync(self.vessel)

    # Finaliza a simulacao
    # Entrada:
    #   None
    # Saida:
    #   None
    def finish(self):
        if self.simulation:
            self.simulation.stop()
            self.dyna_proc.terminate()

    # Metodo destrutor
    # Entrada:
    #   None
    # Saida:
    #   None
    def __del__(self):
        if self.simulation:
            self.simulation.stop()
            self.simulation.disconnect()


if __name__ == "__main__":
    test = Environment()
    test.set_up()
    for i in range(100):
        ret = test.step(0, 0)
        print(ret)
