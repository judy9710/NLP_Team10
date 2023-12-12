import os
from tqdm import tqdm
from glob import glob

'''
renaming the images with zfill(5)
'''

def main():
    data_dirs = ['./Pendulum/']
    for data_dir in data_dirs:
        images = sorted(glob(data_dir + 'images/*.png'))
        for i in tqdm(range(len(images))):
            os.rename(images[i], data_dir + 'images/' + str(i).zfill(5) + '.png')

if __name__ == '__main__':
    main()