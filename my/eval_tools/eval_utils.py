import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import base64
import os, sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, "../../"))
load_dir = os.path.abspath(os.path.join(curr_dir, "../../examples/saved_data/"))

sys.path.insert(0, root_dir)
sys.path.insert(1, load_dir)


expr_name = "obeserver-vis1"


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


def get_n_vis_lines(body: np.ndarray):
    result = 0
    padded_body = np.pad(body, 1)
    for i, e in enumerate(body.reshape(-1)):
        if e != 6:
            continue
        x = i % 5 + 1
        y = i // 5 + 1

        if padded_body[y][x+1] == 0:
            result += 1
        if padded_body[y+1][x] == 0:
            result += 1
        if padded_body[y-1][x] == 0:
            result += 1
        if padded_body[y][x-1] == 0:
            result += 1

    return result


def make_df():
    # column: [generation, genecode, rank, fitness, n_empty_voxel, n_soft_voxels, n_hard_voxels, n_h_act_voxels, n_v_act_voxels, n_pred_voxels, n_vis_voxels]
    generations = []
    genecodes = []
    indices = []
    ranks = []
    fitnesses = []
    bodies = []
    n_empty_voxels = []
    n_rigid_voxels = []
    n_soft_voxels = []
    n_h_act_voxels = []
    n_v_act_voxels = []
    n_pred_voxels = []
    n_vis_voxels = []
    n_vis_lines = []

    # gen_list = os.listdir(os.path.join(load_dir, expr_name))
    gen_list = get_generations(load_dir=load_dir, exp_name=expr_name)
    # gen_list = [0]

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
            f = np.frompyfunc(lambda x: _map[np.int64(x)], 1, 1)

            body_for_genecode = f(body)
            # -1 check
            genecode = to_genecode(body=body_for_genecode)
            
            n_voxel_types = [0] * 7
            for e in body_for_genecode.reshape(-1):
                n_voxel_types[e] += 1

            n_vis_line = get_n_vis_lines(body=body_for_genecode)

            # # for debug
            # print(rank, idx, reward)
            # print(body)
            # print(body_for_genecode)
            # print(genecode)
            # print(n_voxel_types)
            # print(n_vis_line)

            generations += [g]
            genecodes += [genecode]
            indices += [idx]
            ranks += [rank]
            fitnesses += [reward]
            bodies += [body]
            n_empty_voxels += [n_voxel_types[0]]
            n_rigid_voxels += [n_voxel_types[1]]
            n_soft_voxels += [n_voxel_types[2]]
            n_h_act_voxels += [n_voxel_types[3]]
            n_v_act_voxels += [n_voxel_types[4]]
            n_pred_voxels += [n_voxel_types[5]]
            n_vis_voxels += [n_voxel_types[6]]
            n_vis_lines += [n_vis_line]
    
    df = pd.DataFrame({
        'generation': generations,
        'genecode': genecodes,
        'index': indices,
        'rank': ranks,
        'fitness': fitnesses,
        'body': bodies,
        'n_empty_voxels': n_empty_voxels,
        'n_rigid_voxels': n_rigid_voxels,
        'n_soft_voxels': n_soft_voxels,
        'n_h_act_voxels': n_h_act_voxels,
        'n_v_act_voxels': n_v_act_voxels,
        'n_pred_voxels': n_pred_voxels,
        'n_vis_voxels': n_vis_voxels,
        'n_vis_lines': n_vis_lines,
    })

    return df


def add_column_survive_terms(df: pd.DataFrame):
    df['survive_terms'] = 0
    for index, items in df.iterrows():
        # df.loc[index, 'survive_terms'] = 1
        # 世代を抽出, 前の世代で同じコードの個体があればsurvivetermを加算
        # 同じ個体が複数いた場合は一旦エラー

        generation = items['generation']
        if generation == 0:
            continue
        
        m = df[(df['generation'] == generation - 1)]
        m = m[m['genecode'] == items['genecode']]
        # m = df[(df['generation'] == generation - 1) & df['genecode'] == items['genecode']]
        if len(m) > 1:
            print("error: utils.py l.184")

        if len(m) == 1:
            df.loc[index, 'survive_terms'] = m['survive_terms'].values[0] + 1


def reward_graph(df, title, save_path):
    y_mean = df.groupby('generation')['fitness'].mean()
    y_max = df.groupby('generation')['fitness'].max()
    y_median = df.groupby('generation')['fitness'].median()
    x = np.arange(len(y_mean))
    
    plt.plot(x, y_max, label="max")
    plt.plot(x, y_median, label="median")
    plt.plot(x, y_mean, label="mean")
    plt.title(title, fontsize=18)
    plt.ylim(0, 1000)
    plt.ylabel("reward", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


def reward_scatter(df, title, save_path):
    x1 = df[df['survive_terms']>0]['generation']
    y1 = df[df['survive_terms']>0]['fitness']
    x2 = df[df['survive_terms']<=0]['generation']
    y2 = df[df['survive_terms']<=0]['fitness']

    plt.scatter(x1, y1, c='blue', linewidths=0.01, label='survivors')
    plt.scatter(x2, y2, c='orange', linewidths=0.01, label='new-born')
    plt.title(title, fontsize=18)
    plt.ylim(0, 1000)
    plt.ylabel("reward", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


def vis_line_graph(df, title, save_path):
    # 視線ボクセルを使ったロボットが成熟することを示したい
    # 折れ線グラフ
    # 視線数の平均をプロット
    # new-bornの平均が
    # survivor, new-born, all
    y_nb = df[df['survive_terms'] <= 0][['generation', 'n_vis_lines']].groupby('generation').mean()['n_vis_lines']
    y_sv = df[df['survive_terms'] > 0][['generation', 'n_vis_lines']].groupby('generation').mean()['n_vis_lines']
    y_all = df[['generation', 'n_vis_lines']].groupby('generation').mean()['n_vis_lines']
    x_nb = np.arange(len(y_nb))
    x_sv = np.arange(len(y_sv))
    x_all = np.arange(len(y_all))
    plt.plot(x_nb, y_nb, label='new-born')
    plt.plot(x_sv, y_sv, label='survivors')
    plt.plot(x_all, y_all, label='all')
    plt.title(title, fontsize=18)
    plt.ylabel("number of vis-lines", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


def survive_hist(df, title, save_path):
    # x:term, y:pop

    # エラー確認
    if df.groupby(['genecode', 'survive_terms']).count()['generation'].max() != 1:
        print("Error")
        exit(1)

    survive_terms = df[['genecode', 'survive_terms']].groupby('genecode').max()

    x = []
    y = []

    # ---

    for g in range(survive_terms.max()[0]+1):
        x += [g]
        y += [len(survive_terms[survive_terms['survive_terms']>=g])]

    x = np.array(x)
    y = np.array(y)

    # ---

    # st = survive_terms.reset_index()
    # st = st.groupby('survive_terms').count()
    # x = st.index
    # y = st

    # ---

    plt.bar(x, y, width=1.0)
    plt.title(title, fontsize=18)
    # plt.ylim(0, 13)
    plt.ylabel("population", fontsize=16)
    plt.xlabel("survive_terms(accumulated)", fontsize=16)
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":

    # N_EXPR = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    N_EXPR = [0]

    EXPR = [
        # [expr_name, env_name, save_name,]
        ["obeserver-vis1",          "Observer-vis"],        # 0
        ["huntCreeperBaseline",     "HuntCreeper"],         # 1
        ["huntCreeperBaselineVis",  "HuntCreeper-vis"],     # 2
        ["huntHopperBaseline",      "HuntHopper"],          # 3
        ["huntHopperBaselineVis",   "HuntHopper-vis"],      # 4
        ["huntFlyerBaseline",       "HuntFlyer"],           # 5
        ["huntFlyerBaselineVis",    "HuntFlyer-vis"],       # 6
        ["huntCreeperBaselineVisRandomPop", "HuntCreeper-vis (Random Spawn)"],  # 7
        ["huntCreeperBaselineVis-ms", "HuntCreeper-vis (More Survive)"],        # 8
    ]

    # ---

    for n_expr in N_EXPR:
        expr_name = EXPR[n_expr][0]
        expr_name_for_report = EXPR[n_expr][1]

        df = make_df()
        add_column_survive_terms(df=df)

        # 報酬グラフ作成
        save_path = 'plots/reward_graph(' + expr_name + ').png'
        reward_graph(df, '報酬値の統計量推移\n' + expr_name_for_report, save_path)

        # 報酬散布図作成
        save_path = 'plots/reward_scatter(' + expr_name + ').png'
        reward_scatter(df, '報酬値散布図\n' + expr_name_for_report, save_path)

        # 視線本数グラフ
        save_path = 'plots/vis_line_graph(' + expr_name + ').png'
        vis_line_graph(df, '平均視線本数の推移\n' + expr_name_for_report, save_path)

        # 生存期間ヒストグラム
        save_path = 'plots/survive_hist(' + expr_name + ').png'
        survive_hist(df, '累積生存期間別の個体数\n' + expr_name_for_report, save_path)


