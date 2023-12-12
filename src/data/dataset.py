import json
from glob import glob
from tqdm import tqdm
import os

def parse_json(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    return data

class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.label = parse_json(data_dir + 'label.json')
        self.states = self.label['states']
        self.actions = self.label['actions']
        self.images = sorted(glob(data_dir + 'images/*.png'))

    # def __init__(self, data_dir1, data_dir2, data_dir3):
    #     data_dirs = [data_dir1, data_dir2, data_dir3]
    #     self.states = []
    #     self.actions = []
    #     self.images = []

    #     for data_dir in data_dirs:
    #         self.label = parse_json(data_dir + 'label.json')
    #         self.states.append(self.label['states'])
    #         self.actions.append(self.label['actions'])
    #         self.images.append(sorted(glob(data_dir + 'images/*.png')))

    def __len__(self):
        assert len(self.states) == len(self.actions) == len(self.images)
        return len(self.states)

    def __getitem__(self, idx):
        return {'state': self.states[idx], 'action': self.actions[idx], 'image': self.images[idx]}

class CartPoleDataset(Dataset):
    def __init__(self):
        data_dir = '../data/CartPole/'
        super().__init__(data_dir)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        cart_pos, cart_vel, pole_angle, pole_angular_vel = item['state']
        item['state'] = {'cart_pos': cart_pos, 'cart_vel': cart_vel, 'pole_angle': pole_angle, 'pole_angular_vel': pole_angular_vel}
        item['action'] = int(item['action'])
        return item

class MountainCarDataset(Dataset):
    def __init__(self):
        data_dir = '../data/MountainCar/'
        super().__init__(data_dir)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        car_pos, car_vel = item['state']
        item['state'] = {'car_pos': car_pos, 'car_vel': car_vel}
        item['action'] = int(item['action'])
        return item

class PendulumDataset(Dataset):
    def __init__(self):
        data_dir = '../data/Pendulum/'
        super().__init__(data_dir)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)

        x, y, angular_vel = item['state'][0]
        item['state'] = {'x': x, 'y': y, 'angular_vel': angular_vel}
        item['action'] = float(item['action'])
        return item
    
class MergedDataset(Dataset):
    def __init__(self):
        data_dir1 = '../data/CartPole/'
        data_dir2 = '../data/MountainCar/'
        data_dir3 = '../data/Pendulum/'
        super().__init__(data_dir1, data_dir2, data_dir3)

if __name__ == '__main__':
    dataset = CartPoleDataset()
    import pdb; pdb.set_trace()