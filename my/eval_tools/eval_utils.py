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


INTERVAL = 5.0


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


def reward_graph(df, title, ylim):
    y_mean = df.groupby('generation')['fitness'].mean()
    y_max = df.groupby('generation')['fitness'].max()
    y_median = df.groupby('generation')['fitness'].median()
    x = np.arange(len(y_mean))

    plt.plot(x, y_max, label="max")
    plt.plot(x, y_median, label="median")
    plt.plot(x, y_mean, label="mean")
    plt.title(title)
    plt.ylim(0, ylim)
    plt.ylabel("reward")
    plt.xlabel("generation")
    plt.legend()


def reward_scatter(df, title, ylim):
    x1 = df[df['survive_terms']>0]['generation']
    y1 = df[df['survive_terms']>0]['fitness']
    x2 = df[df['survive_terms']<=0]['generation']
    y2 = df[df['survive_terms']<=0]['fitness']

    plt.scatter(x1, y1, c='blue', linewidths=0.01, label='survivors')
    plt.scatter(x2, y2, c='orange', linewidths=0.01, label='new-born')
    plt.title(title)
    plt.ylim(0, ylim)
    plt.ylabel("reward")
    plt.xlabel("generation")
    plt.legend()


def vis_line_graph(df, title, y_lim):
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
    plt.title(title)
    plt.ylabel("number of vis-lines")
    plt.xlabel("generation")
    plt.ylim(y_lim)
    plt.legend()


def survive_hist(df, title, limit_survive_terms):
    # エラー確認
    if df.groupby(['genecode', 'survive_terms']).count()['generation'].max() != 1:
        print("Error")
        exit(1)

    survive_terms = df[['genecode', 'survive_terms']].groupby('genecode').max()

    x = []
    y = []

    for g in range(survive_terms.max()[0]+1):
        x += [g]
        y += [len(survive_terms[survive_terms['survive_terms'] >= g])]

    x = np.array(x)
    y = np.array(y)

    one_generation_survivors = y[1]
    max_survivor = survive_terms.max()

    # x_ticks = np.hstack((np.linspace(0, 40, 5), max_survivor))
    # y_ticks = np.hstack((np.linspace(0, len(survive_terms), 6), one_generation_survivors))

    plt.bar(x, y, width=1.0)
    plt.title(title)
    plt.ylabel("population")
    plt.xlabel("generation (accumulated)")
    plt.xlim((0, limit_survive_terms))
    # _, x_labels = plt.xticks(x_ticks)
    # _, y_labels = plt.yticks(y_ticks)
    # x_labels[-1].set_color('blue')
    # y_labels[-1].set_color('blue')

    plt.annotate(f'(1, {one_generation_survivors})', xy=(1, one_generation_survivors),
                 xytext=(10, len(survive_terms) * 2.5 / 5),
                 # xycoords="axes fraction",
                 # textcoords="axes fraction",
                 arrowprops=dict(facecolor='black', shrink=0.10, width=2, headlength=7, headwidth=10),
                 fontsize=plt.rcParams["legend.fontsize"],
                 horizontalalignment='center',
                 verticalalignment='top')

    plt.annotate(f'({max_survivor.values[0]}, {y[-1]})', xy=(max_survivor, y[-1]),
                 xytext=(33, len(survive_terms) / 5),
                 # xycoords="axes fraction",
                 # textcoords="axes fraction",
                 arrowprops=dict(facecolor='black', shrink=0.10, width=2, headlength=7, headwidth=10),
                 fontsize=plt.rcParams["legend.fontsize"],
                 horizontalalignment='center',
                 verticalalignment='top')
    # plt.annotate(f'Surviving\nmore than\none generation:\n({one_generation_survivors})', xy=(1, one_generation_survivors),
    #              xytext=(15, len(survive_terms) * 2.5 / 5),
    #              arrowprops=dict(facecolor='black', shrink=0.10, width=2, headlength=7, headwidth=10),
    #              fontsize=plt.rcParams["legend.fontsize"],
    #              horizontalalignment='center')
    #
    # plt.annotate(f'Longest\nsurviving\ngenerations:\n({max_survivor.values[0]})', xy=(max_survivor, y[-1]),
    #              xytext=(33, len(survive_terms) / 5),
    #              arrowprops=dict(facecolor='black', shrink=0.10, width=2, headlength=7, headwidth=10),
    #              fontsize=plt.rcParams["legend.fontsize"],
    #              horizontalalignment='center')


def change_plt_params(layout_type):
    # plt.rcParams['font.family'] = 'sans-serif'
    if layout_type == 3:
        plt.rcParams["axes.titlesize"] = 26
        plt.rcParams["axes.labelsize"] = 26
        plt.rcParams["xtick.labelsize"] = 19
        plt.rcParams["ytick.labelsize"] = 19
        plt.rcParams['figure.subplot.bottom'] = 0.18
        plt.rcParams['figure.subplot.left'] = 0.18
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.framealpha"] = 0.4
        plt.rcParams["legend.fontsize"] = 20

    elif layout_type == 2:
        plt.rcParams["axes.titlesize"] = 18
        plt.rcParams["axes.labelsize"] = 20
        plt.rcParams["xtick.labelsize"] = 16
        plt.rcParams["ytick.labelsize"] = 16
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.16
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.framealpha"] = 0.4
        plt.rcParams["legend.fontsize"] = 18

    else:
        print("illegal layout_type:", layout_type)


if __name__ == "__main__":
    exec_test = False
    save_graph = True
    plot_reward_graph = True
    plot_reward_scatter = True
    plot_survive_hist = True
    plot_vis_line_graph = True

    # N_EXPR = [
    #     0,
    #     1, 2, 3, 4, 5, 6, 7,
    #     8, 9, 10, 11, 12, 13, 14, 15
    # ]

    N_EXPR = [
        # 0,
        # 1, 2, 3, 4, 5, 6,
        # 7,
        # 8, 9, 10, 11, 12, 13,
        14,
    ]

    limit_survive_terms = 43
    nvl_y_lim = [0, 10]

    EXPR = [
        # [index, expr_name, save_name, episode_steps, is_vis_task, layout_type]
        ["1,1", "obeserver-vis1",          "(a)Observer-vis",         1000,   True,  2],  # 0

        ["2,1", "hunting_creeper_ga",      "(b)HuntCreeper",          1000,   False, 3],  # 1
        ["2,4", "huntCreeper_vis1",        "(c)HuntCreeper-vis",      1000,   True,  3],  # 2
        ["2,2", "ga_hopper",               "(d)HuntHopper",           1000,   False, 3],  # 3
        ["2,5", "huntHopperVis",           "(e)HuntHopper-vis",       1000,   True,  3],  # 4
        ["2,3", "ga_flyer",                "(f)HuntFlyer",            1000,   False, 3],  # 5
        ["2,6", "huntFlyerVis",            "(g)HuntFlyer-vis",        1000,   True,  3],  # 6

        ["3,1", "huntCreeper_vis1-v1", "(h)HuntCreeper-vis (Distant)",1000,   True,  2],  # 7

        ["4,1", "huntCreeperBaseline",     "(i)HuntLargeCreeper",      500,   False, 3],  # 8
        ["4,4", "huntCreeperBaselineVis",  "(j)HuntLargeCreeper-vis",  500,   True,  3],  # 9
        ["4,2", "huntHopperBaseline",      "(k)HuntLargeHopper",       500,   False, 3],  # 10
        ["4,5", "huntHopperBaselineVis",   "(l)HuntLargeHopper-vis",   500,   True,  3],  # 11
        ["4,3", "huntFlyerBaseline",       "(m)HuntLargeFlyer",        500,   False, 3],  # 12
        ["4,6", "huntFlyerBaselineVis",    "(n)HuntLargeFlyer-vis",    500,   True,  3],  # 13

        ["5,1", "huntCreeperBaselineVisRandom",
                    "(o)HuntLargeCreeper-vis \n(Random Spawn)",   500,    True,  3],  # 14
        ["5,2", "huntCreeperBaselineVis-ms",
                    "(p)HuntLargeCreeper-vis (More Survive)",   500,    True,  2],  # 15
    ]

    save_path_option = "tmp"

    # ---

    n_plots = 0
    count_plot = 0
    for n_expr in N_EXPR:
        if EXPR[n_expr][4]:
            n_plots += 4
        else:
            n_plots += 3

    print("plot", n_plots, "graphs")

    for n_expr in N_EXPR:
        idx1, idx2 = EXPR[n_expr][0].split(",")
        expr_name = EXPR[n_expr][1]
        expr_name_for_report = EXPR[n_expr][2]
        episode_steps = EXPR[n_expr][3]
        is_vis_task = EXPR[n_expr][4]
        layout_type = EXPR[n_expr][5]

        df = make_df()
        add_column_survive_terms(df=df)

        if exec_test:

            # ---

            print(expr_name, "v_vis_lines:", df['n_vis_lines'].max())
            print(expr_name, "survive_terms:", df['survive_terms'].max())

            # ---

            continue

        # レイアウトの設定
        change_plt_params(layout_type)

        # 報酬グラフ作成
        if plot_reward_graph:
            count_plot += 1
            print("plot:", expr_name, "reward_graph", f"({count_plot}/{n_plots})")
            save_path = os.path.join('plots', save_path_option, f'({idx1}-a-{idx2}) reward_graph ({expr_name}).png')
            title = ('報酬値の統計量推移\n' if layout_type == 2 else "") + expr_name_for_report
            reward_graph(df, title, ylim=episode_steps)
            if save_graph:
                plt.savefig(save_path)
            plt.show()

        # 報酬散布図作成
        if plot_reward_scatter:
            count_plot += 1
            print("plot:", expr_name, "reward_scatter", f"({count_plot}/{n_plots})")
            save_path = os.path.join('plots', save_path_option, f'({idx1}-b-{idx2}) reward_scatter ({expr_name}).png')
            title = ('報酬値散布図\n' if layout_type == 2 else "") + expr_name_for_report
            reward_scatter(df, title, ylim=episode_steps)
            if save_graph:
                plt.savefig(save_path)
            plt.show()

        # 生存期間ヒストグラム
        if plot_survive_hist:
            count_plot += 1
            print("plot:", expr_name, "survive_hist", f"({count_plot}/{n_plots})")
            save_path = os.path.join('plots', save_path_option, f'({idx1}-c-{idx2}) survive_hist ({expr_name}).png')
            title = ('誕生後の経過世代数に対する生存個体数\n' if layout_type == 2 else "") + expr_name_for_report
            survive_hist(df, title, limit_survive_terms=limit_survive_terms)
            if save_graph:
                plt.savefig(save_path)
            plt.show()

        if is_vis_task:
            # 視線本数グラフ
            if plot_vis_line_graph:
                count_plot += 1
                print("plot:", expr_name, "vis_line_graph", f"({count_plot}/{n_plots})")
                save_path = os.path.join('plots', save_path_option, f'({idx1}-d-{idx2}) vis_line_graph ({expr_name}).png')
                title = ('平均視線本数の推移\n' if layout_type == 2 else "") + expr_name_for_report
                vis_line_graph(df, title, y_lim=nvl_y_lim)
                if save_graph:
                    plt.savefig(save_path)
                plt.show()

        time.sleep(INTERVAL)
