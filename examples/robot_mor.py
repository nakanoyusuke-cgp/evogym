import gym
import evogym.envs

import numpy as np


expr_nums = [0, 0, 0, 1, 1, 2, 2, 3]
gens = [60, 0, 30, 20, 60, 60, 60, 60]
ranks = [1, 2, 10, 5, 1, 1, 8, 1]

exprs = [
    "obeserver-vis1",
    "hunting_creeper_ga",
    "huntCreeper_vis1",
    "huntCreeper_vis1-v1",
]

if __name__ == '__main__':
    for i in range(len(expr_nums)):
        expr_name = exprs[expr_nums[i]]
        gen = gens[i]
        rank = ranks[i]


        p1 = "saved_data/" + expr_name + "/generation_" + str(gen) + "/output.txt"
        idx = int(np.loadtxt(p1)[rank - 1][0])
        print("idx:", idx)
        

        path = "saved_data/" + expr_name + "/generation_" + str(gen) + "/structure/" + str(idx) + ".npz"
        print("path:", path)
        b_c = np.load(path)
        body = b_c["arr_0"]
        print("arr:", body)
        connection = b_c["arr_1"]
        
        env = gym.make('Walker-v0', body=body)
        env.reset()
        action = (env.action_space.sample() * 0) + 0
        env.step(action)
        # env.reset()
        
        env.sim.translate_object(0., 5., "robot")


        env.render()
        # while True:
        #     action = env.action_space.sample()-1
        #     ob, reward, done, info = env.step(action)
        #     env.render()

        #     if done:
        #         env.reset()

        _ = input()
        
        env.close()
