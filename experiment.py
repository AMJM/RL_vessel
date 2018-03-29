import argparse
import sys
import scipy.io as io
import blabla
import os
import qlearning
import environment
import datetime
import actions
from viewer import Viewer
import pickle
import learner
import utils

variables_file = "experiment_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
learner_file = "agent" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
q_file = "q_table" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
main_loop_iterations = 10
max_fit_iterations = 100
max_steps_per_batch = 2000
maximum_training_steps = 20000000
evaluation_steps = 1000
max_episodes = 400

steps_between_actions = 20
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
goal_heading_e_ccw = utils.channel_angle_e_ccw(N03, N05)
goal_vel_lon = 1.5
buoys = (funnel_start, N01, N03, N05, N07, Final, N06, N04, N02, funnel_end)
vessel_id = '36'
rudder_id = '0'
thruster_id = '0'
scenario = 'default'
goal = ((N07[0]+Final[0])/2, (N07[1]+Final[1])/2)
goal_factor = 100


def load_pickle_file(file_to_load):
    with open(file_to_load, 'rb') as infile:
        var_list = pickle.load(infile)
        episodes_list = list()
        while True:
            try:
                ep = pickle.load(infile)
                episodes_list.append(ep)
            except EOFError as e:
                break
    return var_list, episodes_list

def load_agent(file_to_load):
    agent_obj = None
    with open(file_to_load, 'rb') as infile:
        agent_obj = pickle.load(infile)
    return agent_obj

def replay_trajectory(episodes):
    view = Viewer()
    view.plot_boundary(buoys)
    view.plot_goal(goal, goal_factor)
    for episode in episodes:
        transitions_list = episode['transitions_list']
        for transition in transitions_list:
            state = transition[0]
            view.plot_position(state[0], state[1], state[2])
    view.freeze_screen()


def train_from_batch(episodes):
    batch_learner = learner.Learner(learner_file)
    batch_size = 0
    for episode in episodes:
        remaining = max_steps_per_batch - len(episode['transitions_list']) - batch_size
        if remaining >= 0:
            batch_learner.add_to_batch(episode['transitions_list'], episode['final_flag'])
            batch_size += len(episode['transitions_list'])
        else:
            batch_learner.add_to_batch(episode['transitions_list'][0:remaining], 0)
    batch_learner.fit_batch(max_fit_iterations)


def main():
    action_space_name = 'simple_action_space'
    action_space = actions.BaseAction(action_space_name)
    agent = qlearning.QLearning(q_file, action_space=action_space)
    env = environment.Environment(buoys, steps_between_actions, vessel_id,
                                  rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw, goal_vel_lon, plot)
    with open(variables_file, 'wb') as outfile:
        pickle_vars = dict()
        pickle_vars['action_space'] = action_space_name
        env.set_up()
        env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
        agent.exploring = True
        pickle.dump(pickle_vars, outfile)
        for episode in range(max_episodes):
            episode_dict = dict()
            episode_transitions_list = list()
            final_flag = 0
            env.new_episode()
            for step in range(maximum_training_steps):
                state = env.get_state()
                print('Yaw:', state[2])
                rot, angle = agent.select_action(state)
                state_rime, reward = env.step(rot, angle)
                # state_rime, reward = env.step(0, 0)
                print('Reward:', reward)
                transition = (state, (rot, angle), state_rime, reward)
                final_flag = env.is_final()
                agent.observe_reward(state, rot, angle, state_rime, reward, final_flag)
                print("***Training step "+str(step+1)+" Completed")
                episode_transitions_list.append(transition)
                if final_flag != 0:
                    env.step(0, 0)
                    env.step(0, 0)
                    break
            episode_dict['episode_number'] = episode
            episode_dict['transitions_list'] = episode_transitions_list
            episode_dict['final_flag'] = final_flag
            pickle_vars['ep#'+str(episode)] = episode_dict
            pickle.dump(episode_dict, outfile)
        #Now that the training has finished, the agent can use his policy without updating it
    with open(learner_file, 'wb') as outfile:
        pickle.dump(agent, outfile)

def evaluate_agent(ag_obj):
    env = environment.Environment(buoys, steps_between_actions, vessel_id,
                                  rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw, goal_vel_lon, plot)
    env.set_up()
    null_action_number = int(len(actions.action_combinations) / 2)
    ag_obj.exploring = False
    # env.start_original_episode()
    env.step(0, 0)
    init_state = [8000, 4600, -103, 2, 0, 0]
    env.set_single_start_pos_mode(init_state)
    for step in range(evaluation_steps):
        # Mostly the same as training, but without observing the rewards
        # The first step is to define the current state
        state = env.get_state()
        # The agent selects the action according to the state
        action = ag_obj.select_action(state)
        # The state transition is processed
        env.step(0, 0)
        print("***Evaluation step " + str(step + 1) + " Completed")

    
    

if __name__ == '__main__':
    # main()
    loaded_vars, ep_list = load_pickle_file('experiment_20180328215022')
    # ag = load_agent('agent20180326132514')
    # evaluate_agent(ag)
    # replay_trajectory(ep_list)
    train_from_batch(ep_list)

    # agent = load_agent('agent_20180308083538')
