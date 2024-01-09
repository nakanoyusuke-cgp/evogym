import numpy as np
import pandas as pd
import base64
import os, sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, "../../"))
load_dir = os.path.abspath(os.path.join(curr_dir, "../../examples/saved_data/"))

sys.path.insert(0, root_dir)
sys.path.insert(1, load_dir)


expr_name = "obeserver-vis1"
env_name = ""

save_dir = ""


def to_genecode(body: np.array) -> str:
    # 7 ^ 25通り
    # bodyを25桁の7進数とみなして, 10進数のintに変換
    # 9byte、bigエンディアン（実はどっちでもいい）に変換
    # base64にエンコード&デコード　文字列を返す
    str_n = ''
    for e in body.astype(np.int64).reshape(-1):
        str_n += str(e)
    return base64.b64encode(int(str_n, 7).to_bytes(9, "big")).decode()

def get_generations(load_dir, exp_name):
    gen_list = os.listdir(os.path.join(load_dir, exp_name))
    gen_count = 0
    while gen_count < len(gen_list):
        try:
            gen_list[gen_count] = int(gen_list[gen_count].split("_")[1])
        except:
            del gen_list[gen_count]
            gen_count -= 1
        gen_count += 1
    return [i for i in range(gen_count)]

def get_exp_gen_data(exp_name, load_dir, gen):
    robot_data = []
    gen_data_path = os.path.join(load_dir, exp_name, f"generation_{gen}", "output.txt")
    f = open(gen_data_path, "r")
    for line in f:
        robot_data.append((int(line.split()[0]), float(line.split()[1])))
    return robot_data

def make_df():
    # column: [generation, genecode, rank, fitness, n_empty_voxel, n_soft_voxels, n_hard_voxels, n_h_act_voxels, n_v_act_voxels, n_pred_voxels, n_vis_voxels]
    generations = []
    genecodes = []
    indices = []
    ranks = []
    fitnesses = []
    bodies = []
    n_empty_voxels = []
    n_soft_voxels = []
    n_hard_voxels = []
    n_h_act_voxels = []
    n_v_act_voxels = []
    n_pred_voxels = []
    n_vis_voxels = []

    # gen_list = os.listdir(os.path.join(load_dir, expr_name))
    gen_list = get_generations(load_dir=load_dir, exp_name=expr_name)
    gen_list = [0]
    
    # print(get_generations(load_dir=load_dir, exp_name=expr_name))
    # print(get_exp_gen_data(exp_name=expr_name, load_dir=load_dir, gen=0))

    for g in gen_list:

        # idx, reward = get_exp_gen_data(exp_name=expr_name, load_dir=load_dir, gen=g)
        for rank, (idx, reward) in enumerate(get_exp_gen_data(exp_name=expr_name, load_dir=load_dir, gen=g)):

            # load body
            body = np.load(os.path.join(load_dir, expr_name, "generation_" + str(g), "structure", str(idx) + ".npz"))['arr_0']

            # generate
            _map = [
                0,   # EMPTY
                1,   # RIGID
                2,   # SOFT
                3,   # H_ACT
                4,   # V_ACT
                -1,  # FIXED
                5,   # PRED
                -1,  # PREY
                6,   # VIS
            ]
            body_for_genecode = body
            # -1 check
            genecode = to_genecode(body=body_for_genecode)
            
            
            # for debug
            print(rank, idx, reward)
            print(body)
            print(genecode)

            generations += [g]
            genecodes += [genecode]
            indices += [idx]
            ranks += [rank]
            fitnesses += [reward]
            bodies += [body]
            n_empty_voxels += []
            n_soft_voxels += []
            n_hard_voxels += []
            n_h_act_voxels += []
            n_v_act_voxels += []
            n_pred_voxels += []
            n_vis_voxels += []

    
    # print(curr_dir)
    # print(os.path.join(curr_dir, ".."))
    # print(os.path.abspath(os.path.join(curr_dir, "../..")))
    # print(root_dir)



    n_generations = None



    # for g in range(n_generations):
    #     pass
    
    # df = pd.DataFrame()


if __name__ == "__main__":
    # for i in range(50):
    #     body =  np.random.randint(0, 7, (5,5))
    #     gc = to_genecode(body)
    #     print(gc)

    make_df()
