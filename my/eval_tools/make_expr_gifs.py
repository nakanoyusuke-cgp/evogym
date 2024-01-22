import os, sys

# pycharmの設定でroot\examples\をpathに追加済み

from make_gifs import Job, GIF_RESOLUTION

GIF_RESOLUTION = (1280/5, 720/5)


# 0(observer)は過去の出力を使う
N_EXPR = [
    # 0,
    # 1, 2, 3, 4, 5, 6, 7,
    # 8, 9, 10, 11, 12, 13,
    # 14, 15,
    0, 1, 2, 3, 4, 6
]


RANKS = {

    -1: [i for i in range(10)],
}

GENERATIONS = {

    # -1: [i for i in range(31)],
    -1: [i for i in range(63)],
}


EXPR = [
    # [expr_name, env_name, save_name,]
    ["obeserver-vis1",      "Observer_vis1-v0",     "observerVis"               ],  # 0

    ["hunting_creeper_ga",  "HuntCreeper-v0",       "HuntCreeper"               ],  # 1
    ["huntCreeper_vis1",    "HuntCreeper_vis1-v0",  "HuntCreeper-vis"           ],  # 2
    ["huntCreeper_vis1-v1", "HuntCreeper_vis1-v0",  "HuntCreeper-vis (Distant)" ],  # 3
    ["ga_hopper",           "HuntHopper-v0",        "HuntHopper"                ],  # 4
    ["huntHopperVis",       "HuntHopperVis-v0",     "HuntHopper-vis"            ],  # 5
    ["ga_flyer",            "HuntFlyer-v0",         "HuntFlyer"                 ],  # 6
    ["huntFlyerVis",        "HuntFlyerVis-v0",      "HuntFlyer-vis"             ],  # 7

    ["huntCreeperBaseline",     "HuntCreeperBaseline-v0",   "huntCreeperBaseline"   ],  # 8
    ["huntCreeperBaselineVis",  "HuntCreeperBaselineVis-v0","huntCreeperBaselineVis"],  # 9
    ["huntHopperBaseline",      "HuntHopperBaseline-v0",    "huntHopperBaseline"    ],  # 10
    ["huntHopperBaselineVis",   "HuntHopperBaselineVis-v0", "huntHopperBaselineVis" ],  # 11
    ["huntFlyerBaseline",       "HuntFlyerBaseline-v0",     "huntFlyerBaseline"     ],  # 12
    ["huntFlyerBaselineVis",    "HuntFlyerBaselineVis-v0",  "huntFlyerBaselineVis"  ],  # 13

    ["huntCreeperBaselineVisRandom",    "HuntCreeperBaselineVis-v1",
                                        "HuntLargeCreeper-vis (Random Spawn)"   ],  # 14
    ["huntCreeperBaselineVis-ms",       "HuntCreeperBaselineVis-v0",
                                        "HuntLargeCreeper-vis (More Survive)"   ],  # 15
]


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    load_dir = os.path.abspath(os.path.join(curr_dir, "../../examples/saved_data/"))
    save_dir = os.path.join(load_dir, 'all_media/baselines')

    for expr in N_EXPR:
        if RANKS.keys().__contains__(expr):
            ranks = RANKS[expr]
        else:
            ranks = RANKS[-1]

        if GENERATIONS.keys().__contains__(expr):
            generations = GENERATIONS[expr]
        else:
            generations = GENERATIONS[-1]

        print("make gif:")
        print("\texpr-name:", EXPR[expr][0])
        print("\tenv-name:", EXPR[expr][1])
        print("\tsave-name:", EXPR[expr][2])
        print("\tgenerations:", generations)
        print("\tranks:", ranks)

        # print("Gif resolution", GIF_RESOLUTION)

        my_job = Job(
            name=EXPR[expr][2],
            experiment_names=[EXPR[expr][0]],
            env_names=[EXPR[expr][1]],
            ranks=ranks,
            load_dir=load_dir,
            generations=generations,
            organize_by_experiment=False,
            organize_by_generation=True,
        )

        my_job.generate(load_dir=load_dir, save_dir=save_dir)
