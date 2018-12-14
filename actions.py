'''
[AKUM-12/12/2018]
Codigo fonte original sem alteracoes de https://github.com/zeamendola89/RL_vessel/tree/refac
'''

from itertools import product

spaces = {'simple_action_space': {
        'rudder_lvls':[-0.5, 0.0, 0.5],
        'thruster_lvls':[-0.5, 0.0, 0.5]
        },
        'cte_rotation': {
        'rudder_lvls':[-0.5, 0.0, 0.5],
        'thruster_lvls':[-0.5]
        },
        'stable': {
            'rudder_lvls': [-0.5, 0.0, 0.5],
            'thruster_lvls': [0.2]
        },
        'rudder_complete':{
            'rudder_lvls':[-0.86, -0.71, -0.57, -0.43, -0.29, -0.14, 0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86],
            'thruster_lvls':[0.6]
        },
        'complete':{
            'rudder_lvls':[-0.86, -0.71, -0.57, -0.43, -0.29, -0.14, 0.0, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86],
            'thruster_lvls':[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6]
        }
}


class BaseAction(object):
    def __init__(self, space_name):
        action_space = spaces[space_name]
        self.rudder_lvls = action_space['rudder_lvls']
        self.thruster_lvls = action_space['thruster_lvls']
        self.action_combinations = list(map(list, product(self.rudder_lvls, self.thruster_lvls)))
        self.possible_actions = range(len(self.action_combinations))

    def map_from_action(self, action_number):
        comb = self.action_combinations[action_number]
        return comb[0], comb[1]

    def all_agent_actions(self):
        return self.possible_actions