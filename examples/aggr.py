import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data_tmp = []
    max_fitnesses = []
    median_fitnesses = []
    mean_fitnesses = []

    # 0, 3, 4, 5
    expr_num = 5

    # [expr_name, graph_title]
    exprs = [
        ["hunting_creeper_ga", "HuntCreeper"],
        ["ga_hopper", "Hopper"],
        ["ga_flyer", "Flyer"],

        ["obeserver-vis1", "Observer-vis"],
        ["huntCreeper_vis1", "HuntCreeper-vis"],
        ["huntCreeper_vis1-v1", "HuntCreeper-vis(distant)"],
    ]

    expr, title = exprs[expr_num]

    
    for i in range(63):
        path = "saved_data/" + expr + "/generation_" + str(i) + "/output.txt"
        ary = np.loadtxt(path)[:, 1]
        max_fitnesses.append(np.max(ary))
        median_fitnesses.append(np.median(ary))
        mean_fitnesses.append(np.mean(ary))
        data_tmp.append(ary)
    
    y1 = np.vstack(max_fitnesses)
    y2 = np.vstack(median_fitnesses)
    y3 = np.vstack(mean_fitnesses)
    x = np.arange(0, 63)

    plt.plot(x, y1, label="max")
    plt.plot(x, y2, label="median")
    plt.plot(x, y3, label="mean")
    plt.title(title)
    plt.ylim(0, 1000)
    plt.ylabel("reward", fontsize=16)
    plt.xlabel("generation", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
