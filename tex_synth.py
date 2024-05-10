from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import matplotlib.pyplot as plt
import sys
import torch
from torch.nn import functional as F
from tqdm import tqdm

W = 15
GAUSSIAN_STD = 0.3
EPS = 0.001
SEED_PATCH_SIZE = 7
SYNTH_IMG_RES = [400, 400]
# SYNTH_IMG_RES = [50, 50]

def gaussian_kernel_2d(size, std=1):
    # std = size * 0.2
    std = size * std  # Make width/height of kernel be 1 standard deviation
    g1 = torch.signal.windows.gaussian(size, std=std).reshape(-1, 1)
    g2 = torch.signal.windows.gaussian(size, std=std).reshape(1, -1)
    gauss_2d = g1 @ g2
    return gauss_2d


# def dist_func(p, samples, gaussian_std=1, mask=None):
#     assert len(p.shape) == 2
#     assert len(samples.shape) == 3
#     assert p.shape[0] % 2 == 1
#     assert p.shape[1] % 2 == 1
#     if mask == None:
#         mask = torch.ones(p.shape).bool()  # Full mask by default
#     assert mask.shape == p.shape
#     sq_dists = (samples - p)**2
#     sq_dists = sq_dists.nan_to_num(nan=0)  # Synth window will have many NANs in it
#     gaussian_weighted = sq_dists * gaussian_kernel_2d(sq_dists.shape[-1], std=gaussian_std)
#     masked = gaussian_weighted * mask / mask.sum()  # Scale distance by the number of dimensions used
#     return masked.sum((-2, -1))


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


# def get_window(image, center_point, window_shape):
#     assert window_shape[0] % 2 == 1 and window_shape[1] % 2 == 1

#     padding_left = window_shape[1] // 2
#     padding_right = window_shape[1] // 2
#     padding_top = window_shape[0] // 2
#     padding_bottom = window_shape[0] // 2

#     pad_image = F.pad(image, (padding_left, padding_right, padding_top, padding_bottom), value=float('nan'))

#     return pad_image[
#         center_point[0]:center_point[0]+window_shape[0],
#         center_point[1]:center_point[1]+window_shape[1]
#     ]

def get_window(image, center_point, window_shape):
    assert len(image.shape) == 3
    assert window_shape[0] % 2 == 1 and window_shape[1] % 2 == 1

    padding_left = window_shape[1] // 2
    padding_right = window_shape[1] // 2
    padding_top = window_shape[0] // 2
    padding_bottom = window_shape[0] // 2

    pad_image = F.pad(image, (0, 0, padding_left, padding_right, padding_top, padding_bottom), value=float('nan'))

    # # TODO: Adjust coordinates to work on padded image
    # top = center_point[0] - window_shape[0] // 2
    # bottom = center_point[0] + window_shape[0] // 2
    # left = center_point[1] - window_shape[1] // 2
    # right = center_point[1] + window_shape[1] // 2

    return pad_image[
        center_point[0]:center_point[0]+window_shape[0],
        center_point[1]:center_point[1]+window_shape[1]
    ]


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    torch.set_default_device('cuda')

    # Read and preprocess image
    img = Image.open(input_file)
    img = torch.from_numpy(np.array(img)).cuda() / 255  # Change to torch tensor and scale 0-1
    img = img[:, :, :3]  # Remove alpha channel

    # Get windows of source image
    img_4d = img.permute(2, 0, 1).unsqueeze(0)  # Unfold requires a batch dimension
    img_windows = F.unfold(img_4d, (W, W)).squeeze()  # Remove batch dimension after unfold
    img_windows = img_windows.reshape(3, W, W, -1).permute(3, 1, 2, 0)  # Reassemble into L, H, W, C patches

    # Generate seed center patch for image synthesis
    patch_top_left = (
        torch.randint(img.shape[0] - SEED_PATCH_SIZE, (1,)).item(),
        torch.randint(img.shape[1] - SEED_PATCH_SIZE, (1,)).item()
    )

    y0, x0 = patch_top_left
    y1 = y0 + SEED_PATCH_SIZE
    x1 = x0 + SEED_PATCH_SIZE
    seed_patch = img[y0:y1, x0:x1]

    # Create synthetic image tensor and fill in seed patch
    synth_img = torch.empty(SYNTH_IMG_RES + [3]).fill_(float('nan'))  # Empty RGB image
    seed_patch_top = SYNTH_IMG_RES[0] // 2 - SEED_PATCH_SIZE // 2
    seed_patch_left = SYNTH_IMG_RES[1] // 2 - SEED_PATCH_SIZE // 2
    synth_img[seed_patch_top:seed_patch_top+SEED_PATCH_SIZE,seed_patch_left:seed_patch_left+SEED_PATCH_SIZE, :] = seed_patch

    num_close = []
    total = SYNTH_IMG_RES[0] * SYNTH_IMG_RES[1] - SEED_PATCH_SIZE**2

    for y, x in tqdm(square_spiral_iterator(SEED_PATCH_SIZE, SYNTH_IMG_RES), total=total):
        y += SYNTH_IMG_RES[0] // 2
        x += SYNTH_IMG_RES[1] // 2

        synth_window = get_window(synth_img, (y, x), (W, W))
        mask = synth_window.isnan().logical_not()
        mask[W//2, W//2] = False  # Mask out center pixel since that's what we are predicting (it should be anyways)

        # Find closest window
        distances = dist_func(synth_window, img_windows, gaussian_std=GAUSSIAN_STD, mask=mask)
        _, sorted_indices = torch.sort(distances)
        closest_window = img_windows[sorted_indices[0]]

        # Find close windows to closest window
        distances = dist_func(closest_window, img_windows, gaussian_std=GAUSSIAN_STD)
        is_close = distances < EPS
        close_windows = img_windows[is_close, :, :, :]

        num_close.append(is_close.sum().item())

        prob = torch.ones(is_close.sum()) / is_close.sum()  # equal likelihood
        sampled_window_idx, _ = sample_with_p(torch.arange(is_close.sum()), prob)
        sampled_window = close_windows[sampled_window_idx, :, :, :].squeeze()  # Remove sampled dimension

        synth_img[y, x, :] = sampled_window[int(W//2), int(W//2), :]  # Use center pixel from selected window

    synth_img = (synth_img * 255).to('cpu', dtype=torch.uint8)
    img_out = Image.fromarray(synth_img.detach().numpy())
    # img_out.convert('L')
    img_out.save(output_file)



# def main():
#     input_file = sys.argv[1]
#     output_file = sys.argv[2]

#     torch.set_default_device('cuda')

#     # Read and preprocess image
#     img = Image.open(input_file)
#     img = img.convert('L')  # Convert to grayscale for now
#     img = torch.from_numpy(np.array(img)).cuda() / 255  # Change to torch tensor and scale 0-1

#     # Get windows of source image
#     img_4d = img[None, None, :, :]  # Unfold only accepts 4D tensors
#     img_windows = F.unfold(img_4d, (W, W))[0,:,:]
#     img_windows = img_windows.transpose(1, 0).reshape(-1, W, W)

#     # Generate seed center patch for image synthesis
#     patch_top_left = (
#         torch.randint(img.shape[0] - SEED_PATCH_SIZE, (1,)).item(),
#         torch.randint(img.shape[1] - SEED_PATCH_SIZE, (1,)).item()
#     )

#     y0, x0 = patch_top_left
#     y1 = y0 + SEED_PATCH_SIZE
#     x1 = x0 + SEED_PATCH_SIZE
#     seed_patch = img[y0:y1, x0:x1]

#     # Create synthetic image tensor and fill in seed patch
#     synth_img = torch.empty(SYNTH_IMG_RES).fill_(float('nan'))
#     seed_patch_top = SYNTH_IMG_RES[0] // 2 - SEED_PATCH_SIZE // 2
#     seed_patch_left = SYNTH_IMG_RES[1] // 2 - SEED_PATCH_SIZE // 2
#     synth_img[seed_patch_top:seed_patch_top+SEED_PATCH_SIZE,seed_patch_left:seed_patch_left+SEED_PATCH_SIZE] = seed_patch
#     idx = (synth_img.shape[0] // 2 - SEED_PATCH_SIZE // 2, synth_img.shape[1] // 2 - SEED_PATCH_SIZE // 2)

#     num_close = []
#     total = SYNTH_IMG_RES[0] * SYNTH_IMG_RES[1] - SEED_PATCH_SIZE**2

#     for y, x in tqdm(square_spiral_iterator(SEED_PATCH_SIZE, SYNTH_IMG_RES), total=total):
#         y += SYNTH_IMG_RES[0] // 2
#         x += SYNTH_IMG_RES[1] // 2

#         synth_window = get_window(synth_img, (y, x), (W, W))
#         mask = synth_window.isnan().logical_not()
#         mask[W//2, W//2] = False  # Mask out center pixel since that's what we are predicting (it should be anyways)

#         # Find closest window
#         distances = dist_func(synth_window, img_windows, gaussian_std=GAUSSIAN_STD, mask=mask)
#         _, sorted_indices = torch.sort(distances)
#         closest_window = img_windows[sorted_indices[0]]

#         # Find close windows to closest window
#         distances = dist_func(closest_window, img_windows, gaussian_std=GAUSSIAN_STD)
#         is_close = distances < EPS
#         close_windows = img_windows[is_close, :, :]

#         num_close.append(is_close.sum().item())

#         prob = torch.ones(is_close.sum()) / is_close.sum()  # equal likelihood
#         sampled_window_idx, _ = sample_with_p(torch.arange(is_close.sum()), prob)
#         sampled_window = close_windows[sampled_window_idx, :, :].squeeze()  # Remove sampled dimension

#         synth_img[y, x] = sampled_window[int(W//2), int(W//2)]  # Use center pixel from selected window

#     synth_img = (synth_img * 255).to('cpu', dtype=torch.uint8)
#     img_out = Image.fromarray(synth_img.detach().numpy())
#     # img_out.convert('L')
#     img_out.save(output_file)


if __name__ == '__main__':
    main()
