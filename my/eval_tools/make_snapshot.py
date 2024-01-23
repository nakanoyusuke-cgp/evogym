from PIL import Image, ImageSequence, ImageDraw
import os, sys, glob
from operator import itemgetter

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, "../../"))

sys.path.insert(0, root_dir)

media_root = os.path.join(root_dir, r"examples/saved_data/all_media")
output_folder_path = os.path.join(root_dir, r"my/eval_tools/snapshots")

LOG_GIF_PATH_MAP = [
    # [expr_name, save_name,]
    ["obeserver-vis1",      "observerVis"               ],  # 0

    ["hunting_creeper_ga",  "hunting_creeper_ga"               ],  # 1
    ["huntCreeper_vis1",    "huntCreeper_vis1"           ],  # 2
    ["huntCreeper_vis1-v1", "huntCreeper_vis1-v1" ],  # 3
    ["ga_hopper",           "ga_hopper"                ],  # 4
    ["huntHopperVis",       "HuntHopper-vis"            ],  # 5
    ["ga_flyer",            "ga_flyer"                 ],  # 6
    ["huntFlyerVis",        "HuntFlyer-vis"             ],  # 7

    ["huntCreeperBaseline",     "huntCreeperBaseline"   ],  # 8
    ["huntCreeperBaselineVis",  "huntCreeperBaselineVis"],  # 9
    ["huntHopperBaseline",      "huntHopperBaseline"    ],  # 10
    ["huntHopperBaselineVis",   "huntHopperBaselineVis" ],  # 11
    ["huntFlyerBaseline",       "huntFlyerBaseline"     ],  # 12
    ["huntFlyerBaselineVis",    "huntFlyerBaselineVis"  ],  # 13

    ["huntCreeperBaselineVisRandom",    "HuntLargeCreeper-vis (Random Spawn) high res"   ],  # 14
    ["huntCreeperBaselineVis-ms",       "HuntLargeCreeper-vis (More Survive)"   ],  # 15
]


def expr_name_to_gif_path(expr_name):
    expr_dict = {log:gif for log, gif in LOG_GIF_PATH_MAP}
    return expr_dict[expr_name]


def make_snapshot(expr_media_path, generation, rank,  indices_of_frame):
    gen_path = os.path.join(expr_media_path, "generation_" + str(generation))
    matched_files = glob.glob(os.path.join(gen_path, str(rank) + "_(*).gif"))
    if len(matched_files) != 1:
        print("error, matched multiple files or no files")
        print("\tquery:", os.path.join(gen_path, str(rank) + "_(*).gif"))
        print("\tresult:", matched_files)
        exit(1)
    gif_path = matched_files[0]

    n_frames = len(indices_of_frame)
    image = Image.open(gif_path)
    frame_durations = image.info['duration']
    width, height = image.size
    frames_all = ImageSequence.all_frames(image)
    getter = itemgetter(*indices_of_frame)
    frames = [f.convert('RGB') for f in getter(frames_all)]
    snap_shot = Image.new('RGB', (width * n_frames, height))
    draw = ImageDraw.Draw(snap_shot)
    for (i, f) in enumerate(frames):
        snap_shot.paste(f, (i * width, 0))
        if not i == 0:
            draw.line([i * width, 0, i * width, height], fill='gray', width=1)

    return snap_shot


if __name__ == "__main__":
    for expr in EXPRS:
        media_path = expr[0]
        robots = expr[1]
        for g, r, idc_f in robots:
            expr_media_path = os.path.join(media_root, media_path)
            snapshot = make_snapshot(expr_media_path=expr_media_path, generation=g, rank=r, indices_of_frame=idc_f)
            snapshot.save(os.path.join(output_folder_path, f"{media_path}-g{g}r{r}.png"))
