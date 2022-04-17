import argparse
import gzip
import os
import pickle
import sys
from io import BytesIO

import cv2
import numpy as np
import PIL.Image
import scipy as sp
from tqdm import tqdm

import dnnlib
import dnnlib.tflib as tflib
from mtd_video import MTDVideo


def get_image_diff_array(image_1, image_2):

    image_1 = np.asarray(PIL.Image.open(image_1), dtype=np.uint8)
    image_2 = np.asarray(PIL.Image.open(image_2), dtype=np.uint8)
    return np.clip(
        (image_1.astype(np.int16) - image_2.astype(np.int16)) / 2 + 128, 0, 255
    )


def generate_mtd_video(
    network_pkl,
    truncation_psi,
    outdir,
    seed,
    num_waypoints,
    num_frames_between_waypoints,
    quality,
    key_frame_only,
    format="jpeg",
    smooth=False,
    max_img_size=512,

    num_upper_waypoints=1,
):
    tflib.init_tf()
    os.makedirs(outdir, exist_ok=True)
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
    )
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi
    rnd = np.random.RandomState(seed)
    z_size = Gs.input_shape[1]
    max_origin_img_dim = max(Gs.output_shape[2:])
    max_img_dim = min(max_origin_img_dim, max_img_size)
    img_ratio = max_img_dim / max_origin_img_dim

    img_shape = (
        np.round(np.array(Gs.output_shape[2:]) * img_ratio).astype(int).tolist()
        + Gs.output_shape[1:2]
    )
    num_points_dim_0 = num_frames_between_waypoints
    num_points_dim_1 = num_frames_between_waypoints

    if num_upper_waypoints > 1:
        dim_1_gap = 0
    else:
        dim_1_gap = num_points_dim_1 // 4
    assert num_waypoints > num_upper_waypoints

    z_upper = []
    for i in range(num_upper_waypoints):
        z_upper.append(rnd.randn(1, z_size))
    start_z_upper = z_upper[0]

    z_bottom = []
    for i in range(num_waypoints):
        z_bottom.append(rnd.randn(1, z_size))
    start_z = z_bottom[0]

    images_array = [
        [BytesIO() for k in range(num_points_dim_1)]
        for ij in range(num_waypoints * num_points_dim_0)
    ]
    key_frames = {}
    print("generate images")
    count = 0
    for i in range(num_waypoints):
        print(f"waypoint {i}")
        if i == num_waypoints - 1:
            target_z = z_bottom[0]
            target_z_upper = z_upper[0]
        else:
            target_z = z_bottom[i + 1]
            target_z_upper = z_upper[(i + 1) % num_upper_waypoints]
        z_diff_dim_0 = (target_z - start_z) / num_points_dim_0

        z_diff_dim_0_upper = (target_z_upper - start_z_upper) / num_points_dim_0

        for j in tqdm(range(num_points_dim_0)):
            z_dim_0 = start_z + z_diff_dim_0 * j
            z_dim_1_target = start_z_upper + z_diff_dim_0_upper * j
            z_diff_dim_1 = (z_dim_1_target - z_dim_0) / (num_points_dim_1 + dim_1_gap)
            for k in range(num_points_dim_1):
                z = z_dim_0 + z_diff_dim_1 * k
                image_list = Gs.run(z, None, **Gs_kwargs)

                image = cv2.resize(image_list[0], img_shape[0:2])
                if smooth:
                    image = sp.ndimage.filters.gaussian_filter(
                        image, [1, 1, 0], mode="constant"
                    )
                image = (image // 2) * 2
                PIL.Image.fromarray(image.astype(np.uint8), "RGB").save(
                    images_array[count][k], format=format, quality=quality
                )
                # every 3 frame make it a keyframe
                if j % 3 == 0:
                    key_frames[(i * num_points_dim_0 + j, k)] = images_array[count][k]
                # save certain images for visualization
                if j == 0 and (k == 0 or k == num_points_dim_1 - 1):
                    PIL.Image.fromarray(image, "RGB").save(
                        f"{outdir}/frame_i{i}_k{k}.png", quality=quality
                    )
            count += 1
        start_z = target_z
        start_z_upper = target_z_upper

    print("calculate diff array")
    diff_array = [
        [[None for dir in range(2)] for k in range(num_points_dim_1)]
        for ij in range(num_waypoints * num_points_dim_0)
    ]

    for ij in tqdm(range(num_waypoints * num_points_dim_0)):

        for k in range(num_points_dim_1):
            if (ij, k) in key_frames:
                pass

            if not key_frame_only:
                image_diff_dim_left = get_image_diff_array(
                    images_array[ij][k], images_array[ij - 1][k]
                )
                buffer_left_diff = BytesIO()
                PIL.Image.fromarray(image_diff_dim_left.astype(np.uint8), "RGB").save(
                    buffer_left_diff, format=format, quality=quality
                )
                size_left = len(buffer_left_diff.getvalue())
                size_down = 0
                size_up = 0
                if k > 0:
                    image_diff_dim_down = image_diff_dim_up
                    buffer_down_diff = BytesIO()
                    PIL.Image.fromarray(
                        image_diff_dim_down.astype(np.uint8), "RGB"
                    ).save(buffer_down_diff, format=format, quality=quality)
                    if diff_array[ij][k - 1][1] is None:
                        size_down = len(buffer_down_diff.getvalue())
                    else:
                        size_down = 0
                if k != num_points_dim_1 - 1:
                    image_diff_dim_up = get_image_diff_array(
                        images_array[ij][k + 1], images_array[ij][k]
                    )
                    buffer_up_diff = BytesIO()
                    PIL.Image.fromarray(image_diff_dim_up.astype(np.uint8), "RGB").save(
                        buffer_up_diff, format=format, quality=quality
                    )
                    size_up = len(buffer_up_diff.getvalue())

                size_key = len(images_array[ij][k].getvalue())

                if size_key <= size_left + size_up + size_down:
                    key_frames[(ij, k)] = images_array[ij][k]

                else:
                    diff_array[ij - 1][k][0] = buffer_left_diff
                    if k != 0:
                        if diff_array[ij][k - 1][1] is None:
                            diff_array[ij][k - 1][1] = buffer_down_diff
                    if k != num_points_dim_1 - 1:
                        diff_array[ij][k][1] = buffer_up_diff
            else:
                key_frames[(ij, k)] = images_array[ij][k]

    print(f"saving file, number of keyframes {len(key_frames)}")

    mtd = MTDVideo(
        diff_array=diff_array,
        diff_array_shape=(num_waypoints * num_points_dim_0, num_points_dim_1, 2),
        key_frames=key_frames,
    )
    fp = gzip.open(f"{outdir}/video.mtd", "wb")
    pickle.dump(mtd, fp)
    fp.close()


# ----------------------------------------------------------------------------

_examples = """examples:

  # Generate mtd video
  python %(prog)s generate-mtd-video --network=../trained_networks/{your network}.pkl --truncation-psi=0.5 --seed=0 --outdir="results/my_mtd_video"

"""
# ----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="""StyleGAN2 mtd video generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.""",
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(help="Sub-commands", dest="command")

    parser_generate_mtd_video = subparsers.add_parser(
        "generate-mtd-video", help="Generate MTD video"
    )
    parser_generate_mtd_video.add_argument(
        "--network", help="Network pickle filename", dest="network_pkl", required=True
    )
    parser_generate_mtd_video.add_argument(
        "--truncation-psi",
        type=float,
        help="Truncation psi (default: %(default)s)",
        default=0.5,
    )
    parser_generate_mtd_video.add_argument(
        "--outdir", help="Where to save the output", required=True
    )
    parser_generate_mtd_video.add_argument(
        "--seed", type=int, help="List of random seeds", default=0
    )
    parser_generate_mtd_video.add_argument(
        "--num-waypoints", type=int, help="Number of waypoints", default=6
    )
    parser_generate_mtd_video.add_argument(
        "--num-frames-between-waypoints", type=int, help="Number of frames between waypoints", default=30
    )
    parser_generate_mtd_video.add_argument(
        "--quality", type=int, help="The quality of video 0 to 95", default=90
    )
    parser_generate_mtd_video.add_argument(
        "--key-frame-only", type=bool, help="Store every frame as keyframe", default=False
    )

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop("command")

    if subcmd is None:
        print("Error: missing subcommand.  Re-run with --help for usage.")
        sys.exit(1)

    generate_mtd_video(**kwargs)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
