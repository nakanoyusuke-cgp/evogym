import numpy as np
import pandas as pd
import base64
import os, sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '../../examples/save_data/')
sys.path.insert(0, root_dir)

expr_name = ""
env_name = ""


def to_genecode(body: np.array) -> str:
    # 7 ^ 25通り
    # bodyを25桁の7進数とみなして, 10進数のintに変換
    # 9byte、bigエンディアン（実はどっちでもいい）に変換
    # base64にエンコード&デコード　文字列を返す
    str_n = ''
    for e in body.astype(np.int64).reshape(-1):
        str_n += str(e)
    return base64.b64encode(int(str_n, 7).to_bytes(9, "big")).decode()

def make_df():
    # column: [generation, genecode, rank, fitness, n_empty_voxel, n_soft_voxels, n_hard_voxels, n_h_act_voxels, n_v_act_voxels, n_pred_voxels, n_vis_voxels]
    generations = []
    genecodes = []
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

    
    n_generations = 
    

    df = pd.DataFrame()


if __name__ == "__main__":
    for i in range(50):
        body =  np.random.randint(0, 7, (5,5))
        gc = to_genecode(body)
        print(gc)
