import os, sys

# pycharmの設定でroot\examples\をpathに追加済み

from make_gifs import Job, GIF_RESOLUTION

# GIF_RESOLUTION = (1280/5, 720/5)


# 0(observer)は過去の出力を使う
N_EXPR = [1, 2, 3, 4, 5, 6,]

EXPR = [
    # [expr_name, env_name, save_name,]
    ["obeserver-vis1",          "Observer_vis1-v0",             "observerVis"],             # 0
    ["huntCreeperBaseline",     "HuntCreeperBaseline-v0",       "huntCreeperBaseline"],     # 1
    ["huntCreeperBaselineVis",  "HuntCreeperBaselineVis-v0",    "huntCreeperBaselineVis"],  # 2
    ["huntHopperBaseline",      "HuntHopperBaseline-v0",        "huntHopperBaseline"],      # 3
    ["huntHopperBaselineVis",   "HuntHopperBaselineVis-v0",     "huntHopperBaselineVis"],   # 4
    ["huntFlyerBaseline",       "HuntFlyerBaseline-v0",         "huntFlyerBaseline"],       # 5
    ["huntFlyerBaselineVis",    "HuntFlyerBaselineVis-v0",      "huntFlyerBaselineVis"],    # 6
]


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.abspath(os.path.join(curr_dir, "../../examples/saved_data/"))
    save_dir = os.path.join(load_dir, 'all_media/baselines')

    for expr in N_EXPR:
        my_job = Job(
            name=EXPR[expr][2],
            experiment_names=[EXPR[expr][0]],
            env_names=[EXPR[expr][1]],
            ranks=[i for i in range(10)],
            load_dir=load_dir,
            generations=[i for i in range(31)],
            organize_by_experiment=False,
            organize_by_generation=True,
        )

        my_job.generate(load_dir=load_dir, save_dir=save_dir)
