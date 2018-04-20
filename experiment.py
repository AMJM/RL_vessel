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
import reward
import json
import argparse

variables_file = "experiment_" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
learner_file = "agent" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
sample_file = "samples/samples" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
q_file = "q_table" + datetime.datetime.now().strftime('%Y%m%d%H%M%S')
main_loop_iterations = 10
max_fit_iterations = 2000
max_tuples_per_batch = 20000000
maximum_training_steps = 20000000
evaluation_steps = 1000
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

def train_from_samples(sample_files):
    replace_reward = reward.RewardMapper(plot_flag=False, r_mode_='exp_border_target')
    replace_reward.set_boundary_points(buoys)
    replace_reward.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    batch_learner = learner.Learner(r_m_=replace_reward)
    for file in sample_files:
        batch_learner.load_sample_file(file)
    batch_learner.set_up_agent()
    batch_learner.fqi_step(50)

def train_from_batch(episodes, pickle_vars):
    replace_reward = reward.RewardMapper(plot_flag=False, r_mode_='exp_border_target_rot_angle')
    replace_reward.set_boundary_points(buoys)
    replace_reward.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    batch_learner = learner.Learner(file_to_save=learner_file, action_space_name=pickle_vars['action_space'],
                                    r_m_=replace_reward)
    batch_size = 0
    for episode in episodes:
        remaining = max_tuples_per_batch - len(episode['transitions_list']) - batch_size
        if remaining >= 0:
            batch_learner.add_to_batch(episode['transitions_list'], episode['final_flag'])
            batch_size += len(episode['transitions_list'])
        else:
            batch_learner.add_to_batch(episode['transitions_list'][0:abs(remaining)], 0)
            break
    batch_learner.set_up_agent()
    batch_learner.fqi_step(max_fit_iterations)

def train_from_single_episode(episodes, pickle_vars, ep_number):
    env = environment.Environment(buoys, steps_between_actions, vessel_id,
                                  rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw, goal_vel_lon,
                                  False)

    replace_reward = reward.RewardMapper(plot_flag=False, r_mode_='exp_border_target_rot_angle')
    replace_reward.set_boundary_points(buoys)
    replace_reward.set_goal(goal, goal_heading_e_ccw, goal_vel_lon)
    batch_learner = learner.Learner(file_to_save=learner_file, action_space_name=pickle_vars['action_space'],
                                    r_m_=replace_reward)
    episode = episodes[ep_number]
    with open('debug_ep.txt', 'w') as outfile:
        for transition in episode['transitions_list']:
            print(transition[0], file=outfile)
            print(list(transition[1]), file=outfile)
            print(transition[2], file=outfile)
            print(transition[3], file=outfile)
            print('\n', file=outfile)
    batch_learner.add_to_batch(episode['transitions_list'], episode['final_flag'])
    batch_learner.set_up_agent()
    for it in range(max_fit_iterations):
        if it % 10 == 0:
            batch_learner.fqi_step(1, debug=True)
        else:
            batch_learner.fqi_step(1, debug=False)
        # if it % 10 == 0:
        #     env.set_up()
        #     env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
        #     env.new_episode()
        #     final_flag = 0
        #     total_reward = 0
        #     for step in range(evaluation_steps):
        #         state = env.get_state()
        #         action = batch_learner.select_action(state)
        #         nxt, rw = env.step(action[0], action[1])
        #         total_reward += rw
        #         final_flag = env.is_final()
        #         print("***Evaluation step " + str(step + 1) + " Completed")
        #         if final_flag != 0:
        #             break
        #     env.finish()
        #     print('For FQI iteration: ',it,' Total reward: ', total_reward, ' and result: ', final_flag)


def sample_transitions(start_state=0, end_state=-1):
    #TODO Implement
    action_space_name = 'simple_action_space'
    action_space = actions.BaseAction(action_space_name)
    env = environment.Environment(buoys, steps_between_actions, vessel_id,
                                  rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw, goal_vel_lon, False)
    env.set_up()
    env.set_sampling_mode(start_state, end_state)
    transitions_list = list()
    for episode in range(max_episodes):
        print('### NEW STARTING STATE ###',episode)
        #TODO fix onconsistency in order of methods from Environment
        env.move_to_next_start()
        final_flag = 0
        for action in action_space.action_combinations:
            # env.set_up()
            env.reset_to_start()
            for i in range(2):
                state = env.get_state()
                angle = action[0]
                rot = action[1]
                state_prime, rw = env.step(angle, rot)
                print('Reward:', rw)
                final_flag = env.is_final()
                # New format
                transition = (state, (angle, rot), state_prime, rw, final_flag)
                transitions_list.append(transition)
        if episode % 100 == 0 and episode != 0:
            with open(sample_file+'_s_'+str(start)+'_'+str(episode), 'wb') as outfile:
                pickle.dump(transitions_list, outfile)
                transitions_list = list()





def main():
    action_space_name = 'simple_action_space'
    action_space = actions.BaseAction(action_space_name)
    agent = qlearning.QLearning(q_file, epsilon=0.1, action_space=action_space, gamma=0.9)
    env = environment.Environment(buoys, steps_between_actions, vessel_id,
                                  rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw, goal_vel_lon, False)
    # with open(variables_file, 'wb') as outfile:
    pickle_vars = dict()
    pickle_vars['action_space'] = action_space_name
    env.set_up()
    env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
    # env.set_single_start_pos_mode([6000, 4000, -103.5, 3, 0, 0])
    agent.exploring = True
    # pickle.dump(pickle_vars, outfile)
    for episode in range(500):
        print('###STARTING EPISODE ', episode)
        episode_dict = dict()
        episode_transitions_list = list()
        final_flag = 0
        env.move_to_next_start()
        env.reset_to_start()
        for step in range(500):
            state = env.get_state()
            print('Yaw:', state[2])
            angle, rot = agent.select_action(state)
            state_prime, reward = env.step(angle, rot)
            # state_rime, reward = env.step(0, 0)
            print('Reward:', reward)
            transition = (state, (angle, rot), state_prime, reward)
            final_flag = env.is_final()
            agent.observe_reward(state, angle, rot, state_prime, reward, final_flag)
            print("***Training step "+str(step+1)+" Completed")
            episode_transitions_list.append(transition)
            if final_flag != 0:
                break
                # state_rime, reward = env.step(0, 0)
                # state_rime, reward = env.step(0, 0)
            # episode_dict['episode_number'] = episode
            # episode_dict['transitions_list'] = episode_transitions_list
            # episode_dict['final_flag'] = final_flag
            # pickle_vars['ep#'+str(episode)] = episode_dict
            # pickle.dump(episode_dict, outfile)
            # env.finish()
        #Now that the training has finished, the agent can use his policy without updating it
    with open(learner_file, 'wb') as outfile:
        pickle.dump(agent, outfile)


def evaluate_agent(ag_obj):
    env = environment.Environment(buoys, steps_between_actions, vessel_id,
                                  rudder_id, thruster_id, scenario, goal, goal_heading_e_ccw, goal_vel_lon, True)
    env.set_up()
    agent = learner.Learner(load_saved_regression=ag_obj, action_space_name='simple_action_space', nn_=True)
    env.set_single_start_pos_mode([8000, 4600, -103.5, 3, 0, 0])
    # env.set_single_start_pos_mode([6600, 4200, -102, 3, 0, 0])
    env.move_to_next_start()
    final_flag = 0
    with open('debug.txt', 'w') as outfile:
        for step in range(evaluation_steps):
            state = env.get_state()
            print(state, file=outfile)
            action = agent.select_action(state)
            print(action, file=outfile)
            state_prime, reward = env.step(action[0], action[1])
            print(state_prime, file=outfile)
            print(reward, file=outfile)
            print('\n', file=outfile)
            final_flag = env.is_final()
            print("***Evaluation step " + str(step + 1) + " Completed")
            if final_flag != 0:
                break



    
    

if __name__ == '__main__':
    # main()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('s', type=str, help='Start')
    parser.add_argument('e', type=str, help='End')

    args = parser.parse_args()
    start = 0
    end = -1
    if args.s:
        start = args.s
    if args.e:
        end = args.e
    # sample_transitions(start, end)
    # ag = load_agent('default_agentSequential_r_exp_border_target')
    evaluate_agent('default_agentSequential_r_exp_border_target.h5')
    #
    #
    # loaded_vars, ep_list = load_pickle_file('experiment_b__')
    # train_from_single_episode(ep_list, loaded_vars, 1)
    # files_list = ['experiment_a__', 'experiment_b__',  'experiment_c__', 'experiment_d__', 'experiment_e__']
    # files_list = ['experiment_full_action_a', 'experiment_full_action_b', 'experiment_full_action_c']
    # ep = list()
    # for file in files_list:
    #     loaded_vars, ep_list = load_pickle_file(file)
    #     ep = ep + ep_list
    # train_from_batch(ep, loaded_vars)
    # replay_trajectory(ep)
    # train_from_batch(ep_list, loaded_vars)
    # train_from_samples(['samples20180419231035_s_0_100','samples20180419231035_s_0_200','samples20180419231035_s_0_300',
    #                     'samples20180419231035_s_0_400'])

