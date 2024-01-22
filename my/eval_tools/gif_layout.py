from PIL import Image, ImageSequence
import os, sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(curr_dir, "../../"))

sys.path.insert(0, root_dir)

media_root = os.path.join(root_dir, r"examples\saved_data\all_media")

expr_media_paths = [
    # [r"baselines\huntCreeperBaseline",      [i for i in range(31)],],
    # [r"baselines\huntCreeperBaseline",      [i for i in range(31)],],
    # [r"baselines\huntCreeperBaselineVis",  [i for i in range(31)],],
    # [r"baselines\huntHopperBaseline",       [i for i in range(31)],],
    # [r"baselines\huntHopperBaselineVis",   [i for i in range(31)],],
    # [r"baselines\huntFlyerBaseline",        [i for i in range(31)],],
    # [r"baselines\huntFlyerBaselineVis",    [i for i in range(31)],],

    # [r"baselines\huntCreeperBaseline",      [i for i in range(31)],],
    # [r"baselines\huntCreeperBaselineVis",  [i for i in range(31)],],
    # [r"baselines\huntHopperBaseline",       [i for i in range(31)],],
    # [r"baselines\huntHopperBaselineVis",   [i for i in range(31)],],
    # [r"baselines\huntFlyerBaseline",        [i for i in range(31)],],
    # [r"baselines\huntFlyerBaselineVis",    [i for i in range(31)],],

    [r"baselines\HuntLargeCreeper-vis (Random Spawn) high res",    [i for i in range(31)],],
    # [r"baselines\HuntLargeCreeper-vis (More Survive)",    [i for i in range(31)],],
]

# 結合したGIF画像を保存するフォルダのパスを指定します
output_folder_path = os.path.join(root_dir, r"my\eval_tools\media")


# 2×5のレイアウトで結合する関数
def create_montage(gen_dir, output_filename):
    gif_files_path = [f for f in os.listdir(gen_dir) if f.endswith('.gif')]
    gif_files_path.sort()
    images = [Image.open(os.path.join(gen_dir, p)) for p in gif_files_path]

    # フレームのディスプレイ時間を設定
    frame_durations = images[0].info['duration']

    # 2×5のレイアウトに結合
    width, height = images[0].size
    frames = []

    for elements in zip(*[ImageSequence.Iterator(i) for i in images]):
        elements_rgba = [e.convert("RGB") for e in elements]
        frame = Image.new('RGB', (width * 2, height * 5))

        for i in range(10):
            x = i % 2
            y = i // 2
            frame.paste(elements_rgba[i], (x * width, y * height))

        frames.append(frame)

    frames[0].save(
        output_filename,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations,
        loop=0,
    )


def create_montage_multiple_generations(load_path, save_dir, generations):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("directory was made:", save_dir)
    for g in generations:
        gen_dir = os.path.join(load_path, "generation_" + str(g))
        create_montage(gen_dir=gen_dir, output_filename=os.path.join(save_dir, f"gen_{g}.gif"))


for expr in expr_media_paths:
    load_path = os.path.join(media_root, expr[0])
    save_path = os.path.join(output_folder_path, expr[0])
    create_montage_multiple_generations(load_path=load_path, save_dir=save_path, generations=expr[1])
