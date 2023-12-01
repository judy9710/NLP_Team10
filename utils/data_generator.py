import matplotlib.pyplot as plt
import json


def generate_data(env_name, env, model, vec_env=False):

    data_size = 10000

    data = {
        'states': [],
        'actions': []
    }

    i = 0
    base_dir = '../data/' + env_name + '/'
    while i < data_size:

        if vec_env:
            obs = env.reset()
        else:
            obs, info = env.reset()

        terminated = False

        while not terminated:
            action, _ = model.predict(obs)

            screen = env.render()
            filepath = base_dir + 'images/' + str(i).zfill(5) + '.png'
            plt.imsave(filepath, screen)

            data['states'].append(obs.tolist())
            if env_name == 'Pendulum':
                data['actions'].append(float(action))
            else:
                data['actions'].append(int(action))

            if vec_env:
                obs, _, terminated, _ = env.step(action)
            else:
                obs, _, terminated, _, _ = env.step(action)

            i += 1
            if i >= data_size:
                break

    filepath = base_dir + 'label.json'
    with open(filepath, 'w') as fp:
        json.dump(data, fp)

    print(i)

    plt.close()
