import os

import numpy as np
from PIL import Image, ImageDraw

import eval_utils
import make_expr_gifs
import make_snapshot

curr_dir = os.path.dirname(os.path.abspath(__file__))
content_root = os.path.abspath(os.path.join(curr_dir, "../../"))

DEFAULT_IDC_F = [1, 100, 200, 300, 400, 498]


def get_gif_save_dir(expr_name):
    expr_dict = {e_name: s_dir for e_name, _, s_dir in make_expr_gifs.EXPR}
    save_dir = expr_dict[expr_name]
    return os.path.join(content_root, "examples/saved_data/all_media/baselines", save_dir)


def create_tile_image_with_contours(array, color_map, cell_size=50):
    height, width = array.shape[0] * cell_size, array.shape[1] * cell_size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            left = x * cell_size
            top = y * cell_size
            right = left + cell_size
            bottom = top + cell_size

            if array[y, x] != 0:
                draw.rectangle((left, top, right-1, bottom-1), fill=color_map.get(array[y, x], (0, 0, 0)))

            if x > 0 and array[y, x] != array[y, x - 1] and (array[y, x - 1] == 0 or array[y, x] == 0):
                draw.line([left, top, left, bottom], fill="black", width=1)
            if y > 0 and array[y, x] != array[y - 1, x] and (array[y, x] == 0 or array[y - 1, x] == 0):
                draw.line([left, top, right, top], fill="black", width=1)
            if x == 0 and array[y, x] != 0:
                draw.line([left, top, left, bottom], fill="black", width=1)
            if y == 0 and array[y, x] != 0:
                draw.line([left, top, right, top], fill="black", width=1)

    for y in range(array.shape[0]):
        if array[y, -1] != 0:
            draw.line([(width - 1, y * cell_size), (width - 1, (y + 1) * cell_size)], fill="black", width=1)
    for x in range(array.shape[1]):
        if array[-1, x] != 0:
            draw.line([(x * cell_size, height - 1), ((x + 1) * cell_size, height - 1)], fill="black", width=1)

    return image


# 色のマッピングを定義
color_map = {
    0: (255, 255, 255),  # EMPTY
    1: (38, 38, 38),      # RIGID
    2: (191, 191, 191),      # SOFT
    3: (253, 142, 62),    # ACT_H
    4: (109, 175, 214),    # ACT_V
    5: (38, 38, 38),      # FIXED
    6: (26, 230, 179),      # PRED
    7: (179, 230, 26),      # PREY
    8: (179, 26, 204),    # VIS
}


if __name__ == "__main__":
    EXPR = [
        # [expr_name, [gen, rank]]
        ["obeserver-vis1",[
            [60, 0], [0, 1], [30, 9]
        ]],  # 0

        ["hunting_creeper_ga",[
            [20, 4], [60, 0]
        ]],  # 1

        ["huntCreeper_vis1",[
            [60, 0], [60, 7]
        ]],  # 2

        ["huntCreeper_vis1-v1",[
            [60, 0]
        ]],  # 3

        ["ga_hopper",[

        ]],  # 4

        ["huntHopperVis",[

        ]],  # 5

        ["ga_flyer",[

        ]],  # 6

        ["huntFlyerVis",[

        ]],  # 7

        ["huntCreeperBaseline",[
            [30, 0], [30, 8]
        ]],  # 8

        ["huntCreeperBaselineVis",[
            [30, 0]
        ]],  # 9

        ["huntHopperBaseline",[
            [30, 0], [8, 2], [17, 4]
        ]],  # 10

        ["huntHopperBaselineVis",[
            [30, 0], [17, 1], [11, 4]
        ]],  # 11

        ["huntFlyerBaseline",[
            [23, 0]
        ]],  # 12

        ["huntFlyerBaselineVis",[
            [29, 6], [28, 9], [22, 4]
        ]],  # 13

        ["huntCreeperBaselineVisRandom",[
            [22, 1],
            [20, 2], [15, 8]
        ]],  # 14

        ["huntCreeperBaselineVis-ms",[

        ]],  # 15
    ]

    for expr in EXPR:
        if len(expr[1]) == 0:
            continue

        expr_name = expr[0]
        eval_utils.expr_name = expr_name
        df = eval_utils.make_df()
        for ind in expr[1]:
            r = ind[0]
            g = ind[1]
            idc_f = DEFAULT_IDC_F if len(ind) == 2 else ind[2]

            # robot designs
            robot = df[(df["generation"] == g) & (df["rank"] == r)]
            if len(robot) != 1:
                print("error")
                exit(1)
            genecode = robot["genecode"].values[0]
            arr = robot["body"].values[0]
            print(f"expr_name:{expr_name}/generation:{g}/rank:{r}/genecode:{genecode}")
            img = create_tile_image_with_contours(arr, color_map)
            img.save(os.path.join(curr_dir, "designs", f"{expr_name}-g{g}r{r}-{genecode}.png"))

            # snapshots
            expr_media_path = get_gif_save_dir(expr_name)
            ss_save_path = os.path.join(curr_dir, "snapshors", f"{expr_name}-g{g}r{r}-{genecode}.png")
            ss = make_snapshot.make_snapshot(expr_media_path, generation=g, rank=r, indices_of_frame=idc_f)
            ss.save(ss_save_path)
