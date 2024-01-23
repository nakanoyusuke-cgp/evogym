from PIL import Image, ImageSequence, ImageDraw
import os, sys, glob
from operator import itemgetter

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, "../../"))

sys.path.insert(0, root_dir)

media_root = os.path.join(root_dir, r"examples/saved_data/all_media")
output_folder_path = os.path.join(root_dir, r"my/eval_tools/snapshots")

EXPRS = [
    # [media_path, [gen, rank, [indices of frames]]
    ["baselines/observerVis",[
        [0, 0, [1, 200, 400, 600, 800, 900]]
        # [60, 0], [0, 1], [30, 9]
    ]],  # 0

    ["hunting_creeper_ga",[
        # [20, 4], [60, 0]
    ]],  # 1

    ["huntCreeper_vis1",[
        # [60, 0], [60, 7]
    ]],  # 2

    ["huntCreeper_vis1-v1",[
        # [60, 0]
    ]],  # 3

    ["ga_hopper",[

    ]],  # 4

    ["huntHopperVis",[

    ]],  # 5

    ["ga_flyer",[

    ]],  # 6

    ["huntFlyerVis",[

    ]],  # 7

    [r"baselines\huntCreeperBaseline",[
        # [30, 0], [30, 8]
    ]],  # 8

    [r"baselines\huntCreeperBaselineVis",[
        # [30, 0]
    ]],  # 9

    [r"baselines\huntHopperBaseline",[
        # [30, 0], [8, 2], [17, 4]
    ]],  # 10

    [r"baselines\huntHopperBaselineVis",[
        # [30, 0], [17, 1], [11, 4]
    ]],  # 11

    [r"baselines\huntFlyerBaseline",[
        # [23, 0]
    ]],  # 12

    [r"baselines\huntFlyerBaselineVis",[
        # [29, 6], [28, 9], [22, 4]
    ]],  # 13

    [r"baselines\HuntLargeCreeper-vis (Random Spawn)",[
        # [22, 1],
        # [20, 2], [15, 8]
    ]],  # 14

    [r"baselines\HuntLargeCreeper-vis (More Survive)",[

    ]],  # 15
]


def make_snapshot(expr_media_path, generation, rank,  indices_of_frame):
    gen_path = os.path.join(expr_media_path, "generation_" + str(generation))
    matched_files = glob.glob(os.path.join(gen_path, str(rank) + "_(*).gif"))
    if len(matched_files) != 1:
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


for expr in EXPRS:
    media_path = expr[0]
    robots = expr[1]
    for g, r, idc_f in robots:
        expr_media_path = os.path.join(media_root, media_path)
        snapshot = make_snapshot(expr_media_path=expr_media_path, generation=g, rank=r, indices_of_frame=idc_f)
        snapshot.save(os.path.join(output_folder_path, f"{media_path}-g{g}r{r}.png"))
