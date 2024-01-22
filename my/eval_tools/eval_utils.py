import numpy as np
import pandas as pd
import time
import japanize_matplotlib
import matplotlib.pyplot as plt
import base64
import os, sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, "../../"))
load_dir = os.path.abspath(os.path.join(curr_dir, "../../examples/saved_data/"))

sys.path.insert(0, root_dir)
sys.path.insert(1, load_dir)


INTERVAL = 4.0


expr_name = "obeserver-vis1"


def to_genecode(body: np.array) -> str:
    # 7 ^ 25通り
    # bodyを25桁の7進数とみなして, 10進数のintに変換
    # 9byte、bigエンディアン（実はどっちでもいい）に変換
    # base64にエンコード&デコード　文字列を返す
    str_n = ''
    for e in body.astype(np.int64).reshape(-1):
        str_n += str(e)
    return base64.b64encode(int(str_n, 7).to_bytes(9, "big"), altchars=b"+@").decode()


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


def reward_graph(df, title, save_path, ylim):
    y_mean = df.groupby('generation')['fitness'].mean()
    y_max = df.groupby('generation')['fitness'].max()
    y_median = df.groupby('generation')['fitness'].median()
    x = np.arange(len(y_mean))

    plt.plot(x, y_max, label="max")
    plt.plot(x, y_median, label="median")
    plt.plot(x, y_mean, label="mean")
    plt.title(title, fontsize=18)
    plt.ylim(0, ylim)
    plt.ylabel("reward", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


def reward_scatter(df, title, save_path, ylim):
    x1 = df[df['survive_terms']>0]['generation']
    y1 = df[df['survive_terms']>0]['fitness']
    x2 = df[df['survive_terms']<=0]['generation']
    y2 = df[df['survive_terms']<=0]['fitness']

    plt.scatter(x1, y1, c='blue', linewidths=0.01, label='survivors')
    plt.scatter(x2, y2, c='orange', linewidths=0.01, label='new-born')
    plt.title(title, fontsize=18)
    plt.ylim(0, ylim)
    plt.ylabel("reward", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


def vis_line_graph(df, title, save_path, y_lim):
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
    plt.ylim(y_lim)
    plt.legend(fontsize=16)
    plt.savefig(save_path)
    plt.show()


def survive_hist(df, title, save_path, limit_survive_terms):
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
    plt.ylabel("population", fontsize=16)
    plt.xlabel("survive_terms(accumulated)", fontsize=16)
    plt.xlim((0, limit_survive_terms))
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    exec_test = False

    # N_EXPR = [
    #     0,
    #     1, 2, 3, 4, 5, 6, 7,
    #     8, 9, 10, 11, 12, 13, 14, 15
    # ]

    N_EXPR = [
        0,
        1, 2, 3, 4, 6,
        8, 9, 10, 11, 12, 13,
        14, 15
    ]

    limit_survive_terms = 43
    nvl_y_lim = [0, 10]

    EXPR = [
        # [expr_name, save_name, episode_steps, is_vis_task]
        ["obeserver-vis1",          "(a)Observer-vis",         1000,   True,   ],  # 0

        ["hunting_creeper_ga",      "(b)HuntCreeper",          1000,   False,  ],  # 1
        ["huntCreeper_vis1",        "(c)HuntCreeper-vis",      1000,   True,   ],  # 2
        ["huntCreeper_vis1-v1", "(d)HuntCreeper-vis (Distant)",1000,   True,   ],  # 3
        ["ga_hopper",               "(e)HuntHopper",           1000,   False,  ],  # 4
        ["huntHopperVis",           "(f)HuntHopper-vis",       1000,   True,   ],  # 5
        ["ga_flyer",                "(g)HuntFlyer",            1000,   False,  ],  # 6
        ["huntFlyerVis",            "(h)HuntFlyer-vis",        1000,   True,   ],  # 7

        ["huntCreeperBaseline",     "(i)HuntLargeCreeper",      500,    False, ],  # 8
        ["huntCreeperBaselineVis",  "(j)HuntLargeCreeper-vis",  500,    True,  ],  # 9
        ["huntHopperBaseline",      "(k)HuntLargeHopper",       500,    False, ],  # 10
        ["huntHopperBaselineVis",   "(l)HuntLargeHopper-vis",   500,    True,  ],  # 11
        ["huntFlyerBaseline",       "(m)HuntLargeFlyer",        500,    False, ],  # 12
        ["huntFlyerBaselineVis",    "(n)HuntLargeFlyer-vis",    500,    True,  ],  # 13
        ["huntCreeperBaselineVisRandom",
                    "(o)HuntLargeCreeper-vis (Random Spawn)",   500,    True,  ],  # 14
        ["huntCreeperBaselineVis-ms",
                    "(p)HuntLargeCreeper-vis (More Survive)",   500,    True,  ],  # 15
    ]

    # ---

    n_plots = 0
    count_plot = 0
    for n_expr in N_EXPR:
        if EXPR[n_expr][3]:
            n_plots += 4
        else:
            n_plots += 3

    print("plot", n_plots, "graphs")

    for n_expr in N_EXPR:
        expr_name = EXPR[n_expr][0]
        expr_name_for_report = EXPR[n_expr][1]
        episode_steps = EXPR[n_expr][2]
        is_vis_task = EXPR[n_expr][3]

        df = make_df()
        add_column_survive_terms(df=df)

        if exec_test:

            # ---

            print(expr_name, "v_vis_lines:", df['n_vis_lines'].max())
            print(expr_name, "survive_terms:", df['survive_terms'].max())

            # ---

            continue

        # 報酬グラフ作成
        count_plot += 1
        print("plot:", expr_name, "reward_graph", f"({count_plot}/{n_plots})")
        save_path = 'plots/reward_graph(' + expr_name + ').png'
        reward_graph(df, '報酬値の統計量推移\n' + expr_name_for_report, save_path, ylim=episode_steps)

        # 報酬散布図作成
        count_plot += 1
        print("plot:", expr_name, "reward_scatter", f"({count_plot}/{n_plots})")
        save_path = 'plots/reward_scatter(' + expr_name + ').png'
        reward_scatter(df, '報酬値散布図\n' + expr_name_for_report, save_path, ylim=episode_steps)

        if is_vis_task:
            # 視線本数グラフ
            count_plot += 1
            print("plot:", expr_name, "vis_line_graph", f"({count_plot}/{n_plots})")
            save_path = 'plots/vis_line_graph(' + expr_name + ').png'
            vis_line_graph(df, '平均視線本数の推移\n' + expr_name_for_report, save_path, y_lim=nvl_y_lim)

        # 生存期間ヒストグラム
        count_plot += 1
        print("plot:", expr_name, "survive_hist", f"({count_plot}/{n_plots})")
        save_path = 'plots/survive_hist(' + expr_name + ').png'
        survive_hist(df, '誕生後の経過世代数に対する生存個体数\n' + expr_name_for_report, save_path, limit_survive_terms=limit_survive_terms)

        time.sleep(INTERVAL)
