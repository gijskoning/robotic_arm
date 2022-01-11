# data logger code

import os
import pandas as pd

class Save_Expert():

    def __init__(self, expert_task, dataset_no):
        # what task does the expert perform
        self.expert_task = expert_task
        # number of the dataset
        self.dataset_no = dataset_no

        self.save_dir = "expert_data_set"
        self.save_path = os.path(self.save_dir + "_" + self.expert_task + "_" + self.dataset_no)

    def saveSet(self, observations, actions):
        # save observations and actions in a pandas df
        save_data = pd.DataFrame(observations, actions)
        save_data.colums = ["Observations", "Actions"]
        save_data.to_csv(self.save_path)


class DataLog():
    # logs observation space and actions

    def __init__(self):
        self.actions = []
        self.observations = []

    def append_to_log(self, observation, action):
        self.actions.append(action)
        self.observations.append(observation)