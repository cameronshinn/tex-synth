import argparse
from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from tqdm import tqdm

WINDOW_SIZE = 7
SEED_PATCH_SIZE = 3
EPS = 0.0005
GAUSSIAN_STD = 0.3

parser = argparse.ArgumentParser(
    prog='python tex_synth.py',
    description='Performs texture synthesis by non-parametric sampling. Generates a new image from a source image by replicating the observed patterns.',
)
parser.add_argument('in_image', help='path to source image')
parser.add_argument('out_image', help='path to save synthesized output image to')
parser.add_argument('-r', '--resolution', nargs=2, type=int, required=True, metavar=('height', 'width'), help='resolution of output image')
parser.add_argument('-w', '--window-size', type=int, default=WINDOW_SIZE, metavar='size', help='size of square sampling window taken over input image')
parser.add_argument('-s', '--seed-patch-size', type=int, default=SEED_PATCH_SIZE, metavar='size', help='size of starting point patch, sampled from input image')
parser.add_argument('-e', '--epsilon', type=float, default=EPS, metavar='eps', help='distance threshold from closest window to sample other windows from')
parser.add_argument('-d', '--gaussian-std', type=float, default=GAUSSIAN_STD, metavar='std', help='gaussian standard deviation for centered window distance weighting')
parser.add_argument('-g', '--use-gpu', action='store_true', help='enable GPU usage by PyTorch')


def gaussian_kernel_2d(size, std=1):
    std = size * std  # Make width/height of kernel be 1 standard deviation
    g1 = torch.signal.windows.gaussian(size, std=std).reshape(-1, 1)
    g2 = torch.signal.windows.gaussian(size, std=std).reshape(1, -1)
    gauss_2d = g1 @ g2
    return gauss_2d


def dist_func(p, samples, gaussian_std=1, mask=None):
    assert len(p.shape) == 3
    assert len(samples.shape) == 4
    assert p.shape[0] % 2 == 1
    assert p.shape[1] % 2 == 1
    if mask == None:
        mask = torch.ones(p.shape).bool()  # Full mask by default
    assert mask.shape == p.shape
    sq_dists = (samples - p)**2
    sq_dists = sq_dists.nan_to_num(nan=0)  # Synth window will have many NANs in it
    gaussian_2d = gaussian_kernel_2d(sq_dists.shape[-2], std=GAUSSIAN_STD)
    gaussian_weighted = sq_dists * gaussian_2d.view(1, *gaussian_2d.shape, 1)  # Broadcast over first and last dimensions
    masked = gaussian_weighted * mask / mask.sum()  # Scale distance by the number of feature dimensions used
    return masked.sum((-3, -2, -1))  # Sum over spatial and channel dimensions, should be a 1d vector


def sample_with_p(x, p):
    assert p.sum().isclose(torch.ones(1))
    assert x.shape == p.shape
    assert len(x.shape) == 1 and len(p.shape) == 1
    cdf = p.cumsum(0)
    idx = torch.searchsorted(cdf, torch.rand(1))
    return x[idx], idx


def square_spiral_iterator(center_box_len, boundary_shape):
    assert center_box_len % 2 == 1
    assert boundary_shape[0] > center_box_len and boundary_shape[1] > center_box_len

    # Assume middle of center box is at (0, 0)
    half_box_len = center_box_len // 2
    top_left = (-half_box_len, -half_box_len)

    # Start above top left corner
    y, x = top_left
    y -= 1

    # Initialization is based on the fact that we start at the top left
    dy = -1
    dx = 0

    spaces_needed = boundary_shape[0] * boundary_shape[1] - center_box_len**2
    counted_spaces = 0

    while counted_spaces < spaces_needed:
        boundary_box_coords = (boundary_shape[0] // 2 + y, boundary_shape[1] // 2 + x)

        if boundary_box_coords[0] >= boundary_shape[0] or boundary_box_coords[1] >= boundary_shape[1] or boundary_box_coords[0] < 0 or boundary_box_coords[1] < 0:
            if y == -x or (y > 0 and y == x) or (y < 0 and y == x-1):
                dy, dx = dx, -dy

            y += dy
            x += dx
            continue

        counted_spaces += 1
        yield y, x  # Matrix dimension order

        if y == -x or (y > 0 and y == x) or (y < 0 and y == x-1):
            dy, dx = dx, -dy

        y += dy
        x += dx


def get_window(image, center_point, window_shape):
    assert len(image.shape) == 3
    assert window_shape[0] % 2 == 1 and window_shape[1] % 2 == 1

    padding_left = window_shape[1] // 2
    padding_right = window_shape[1] // 2
    padding_top = window_shape[0] // 2
    padding_bottom = window_shape[0] // 2

    pad_image = F.pad(image, (0, 0, padding_left, padding_right, padding_top, padding_bottom), value=float('nan'))

    return pad_image[
        center_point[0]:center_point[0]+window_shape[0],
        center_point[1]:center_point[1]+window_shape[1]
    ]


def main():
    args = parser.parse_args()
    input_file = args.in_image
    output_file = args.out_image
    synth_img_res = args.resolution
    window_size = args.window_size
    seed_patch_size = args.seed_patch_size
    eps = args.epsilon
    gaussian_std = args.gaussian_std
    use_gpu = args.use_gpu

    if use_gpu:
        torch.set_default_device('cuda')

    # Read and preprocess image
    img = Image.open(input_file)
    img = torch.from_numpy(np.array(img)) / 255  # Change to torch tensor and scale 0-1
    if use_gpu:
        img = img.cuda()
    img = img[:, :, :3]  # Remove alpha channel

    # Get windows of source image
    img_4d = img.permute(2, 0, 1).unsqueeze(0)  # Unfold requires a batch dimension
    img_windows = F.unfold(img_4d, (window_size, window_size)).squeeze()  # Remove batch dimension after unfold
    img_windows = img_windows.reshape(3, window_size, window_size, -1).permute(3, 1, 2, 0)  # Reassemble into L, H, W, C patches

    # Generate seed center patch for image synthesis
    patch_top_left = (
        torch.randint(img.shape[0] - seed_patch_size, (1,)).item(),
        torch.randint(img.shape[1] - seed_patch_size, (1,)).item()
    )

    y0, x0 = patch_top_left
    y1 = y0 + seed_patch_size
    x1 = x0 + seed_patch_size
    seed_patch = img[y0:y1, x0:x1]

    # Create synthetic image tensor and fill in seed patch
    synth_img = torch.empty(synth_img_res + [3]).fill_(float('nan'))  # Empty RGB image
    seed_patch_top = synth_img_res[0] // 2 - seed_patch_size // 2
    seed_patch_left = synth_img_res[1] // 2 - seed_patch_size // 2
    synth_img[seed_patch_top:seed_patch_top+seed_patch_size,seed_patch_left:seed_patch_left+seed_patch_size, :] = seed_patch

    total = synth_img_res[0] * synth_img_res[1] - seed_patch_size**2
    for y, x in tqdm(square_spiral_iterator(seed_patch_size, synth_img_res), total=total):
        y += synth_img_res[0] // 2
        x += synth_img_res[1] // 2

        synth_window = get_window(synth_img, (y, x), (window_size, window_size))
        mask = synth_window.isnan().logical_not()
        mask[window_size//2, window_size//2] = False  # Mask out center pixel since that's what we are predicting (it should be anyways)

        # Find closest window
        distances = dist_func(synth_window, img_windows, gaussian_std=gaussian_std, mask=mask)
        _, sorted_indices = torch.sort(distances)
        closest_window = img_windows[sorted_indices[0]]

        # Find close windows to closest window
        distances = dist_func(closest_window, img_windows, gaussian_std=gaussian_std)
        is_close = distances < eps
        close_windows = img_windows[is_close, :, :, :]

        prob = torch.ones(is_close.sum()) / is_close.sum()  # equal likelihood
        sampled_window_idx, _ = sample_with_p(torch.arange(is_close.sum()), prob)
        sampled_window = close_windows[sampled_window_idx, :, :, :].squeeze()  # Remove sampled dimension

        synth_img[y, x, :] = sampled_window[int(window_size//2), int(window_size//2), :]  # Use center pixel from selected window

    synth_img = (synth_img * 255).to('cpu', dtype=torch.uint8)
    img_out = Image.fromarray(synth_img.detach().numpy())
    img_out.save(output_file)


if __name__ == '__main__':
    main()
