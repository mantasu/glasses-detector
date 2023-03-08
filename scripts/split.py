import os
import re
import tqdm
import torch
import random
import argparse
import numpy as np
import torchsr.models

from PIL import Image
from typing import Iterable
from scipy.io import loadmat
from torchvision.transforms.functional import to_pil_image, to_tensor

def parse_args() -> dict[str, int | float | str | list[str | int] | None]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True,
        help="Path to the dataset directory")
    parser.add_argument("--criteria", type=str, required=True,
        help=f"Criteria based on which each file will be distinguished "
             f"(sunglasses or no sunglasses). If it is a string of the form "
             f"[dir|file]/keyword (e.g., 'dir/sunglasses'), it will be "
             f"checked if it is contained in the filename or directory the "
             f"file is in. If it is a csv/txt/mat/etc file, the line which "
             f"contains image name will be queried for a label as the loop "
             f"goes through image files at the image name entry")
    parser.add_argument("--filter", nargs='+', type=str, default=[],
        help=f"Ignore filenames which contain provided strings (use --filter "
             f"'substr1' 'substr2' ...). Defaults to [].")
    parser.add_argument("--val-size", type=float, default=None,
        help="Validation size (if needed). Defaults to None.")
    parser.add_argument("--test-size", type=float, default=None,
        help="Test size (if needed). Defaults to None.")
    parser.add_argument("--sr-scale", type=int, default=1, choices=[1, 2, 3, 4, 8],
        help=f"By how much to increase the image quality (super resolution "
             f"fraction). Defaults to 1.")
    parser.add_argument("--resize", nargs=2, type=int, default=[None, None],
        help=f"The width and the height to resize the image to. Defaults to "
             f"[None, None].")
    parser.add_argument("--dim-mismatch", type=str, default="crop", choices=["crop", "ignore"],
        help=f"How to deal with image dimension mismatch. If 'crop', then the "
             f"larger dimension will be cropped from both sides (e.g., if "
             f"width is 6 and height is 4, width will be cropped to 4). "
             f"Defaults to 'crop'.")
    parser.add_argument("--seed", type=int, default=None,
        help=f"The seed for the randomizer to shuffle the lists (determines "
             f"the order of train/val/test split). Defaults to None (random "
             f"every time).")
    parser.add_argument("--sr-model", type=str, default="ninasr_b2",
        help=f"The super resolution model to use. For available options, see "
             f"https://pypi.org/project/torchsr/. Defaults to 'ninasr_b0'.")

    return vars(parser.parse_args())

def read_sof_mat(path: str) -> tuple[callable, callable]:    
    def get_item(filename, samples):
        for key in samples.keys():
            if re.match(key, filename):
                return samples[key]

    metadata, samples = loadmat(path)["metadata"][0], {}

    for sample in metadata:
        name = f"^{sample[-1][0][0][0][:-1]}.*"
        is_sunglasses = sample[10][0][0] in [2, 3]
        landmarks = np.array(sample[12][0]).reshape(-1, 2)
        samples[name] = {"is_sunglasses": is_sunglasses, "landmarks": landmarks}
    
    criteria_fn = lambda _, x: get_item(x, samples)["is_sunglasses"]
    landmark_fn = lambda _, x: get_item(x, samples)["landmarks"]
    
    return criteria_fn, landmark_fn


def align_and_crop_face(image: Image.Image, face_landmarks: np.ndarray, left_eye_index: int = 3, right_eye_index: int = 0, face_factor: float = 0.25) -> Image.Image:
    """
    Aligns and center-crops a face from a PIL image using the provided face alignment points and eye indices.

    Args:
        image: A PIL image.
        face_landmarks: A numpy array of shape (N, 2) representing the face alignment points.
        left_eye_index: The index of the left eye landmark in the face_landmarks array.
        right_eye_index: The index of the right eye landmark in the face_landmarks array.
        face_factor: The factor of the face area relative to the output image.

    Returns:
        A PIL image of the aligned and cropped face.
    """

    # Calculate the center point between the left and right eyes
    left_eye = face_landmarks[left_eye_index]
    right_eye = face_landmarks[right_eye_index]
    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Calculate the angle between the line connecting the eyes and the horizontal axis
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Rotate the image around the center point
    rotated_image = image.rotate(-angle, center=center)

    # Calculate the dimensions of the output image based on the face_factor
    face_width = np.linalg.norm(right_eye - left_eye)
    output_width = int(face_width / face_factor)
    output_height = output_width

    # Calculate the coordinates of the top-left corner of the output image
    x = center[0] - output_width // 2
    y = center[1] - output_height // 2

    # Crop the rotated image to the desired size and return it
    output_image = rotated_image.crop((x, y, x + output_width, y + output_height))

    return output_image

def on_file(root: str,
            name: str,
            criteria_fn: callable,
            landmark_fn: callable = None,
            sr_model: torch.nn.Module | None = None,
            width: int | None = None,
            height: int | None = None,
            dim_mismatch: str = "crop") -> tuple[Image.Image, bool]:
    image = Image.open(os.path.join(root, name))
    is_sunglasses = criteria_fn(root, name)
    mode = image.mode

    if landmark_fn is not None:
        landmarks = landmark_fn(root, name)
        image = align_and_crop_face(image, landmarks)
    
    if sr_model is not None:
        image = to_tensor(image).to(next(sr_model.parameters()).device)
        image = sr_model(image.unsqueeze(0)).clip(0, 1)
        image = to_pil_image(image.squeeze(0)).convert(mode)
    
    if dim_mismatch == "ignore":
        pass
    elif image.size[0] != image.size[1] and dim_mismatch == "crop":
        new_size = min(width, height)
        left, right = (width - new_size) // 2, (width + new_size) // 2
        top, bottom = (height - new_size) // 2, (height + new_size) // 2
        image = image.crop((left, top, right, bottom))
    
    if width is not None or height is not None:
        new_width = image.size[0] if width is None else width
        new_height = image.size[1] if height is None else height
        image = image.resize((new_width, new_height))
    
    return image, is_sunglasses

def save_images_to_dir(name_image_pairs: list[tuple[str, Image.Image]], dir_path: str):
    if len(name_image_pairs) == 0:
        return
    
    os.makedirs(dir_path, exist_ok=True)
    
    for name, image in name_image_pairs:
        name = os.path.splitext(name)[0] + ".jpg"
        save_path = os.path.join(dir_path, name)
        image.save(save_path)

def process_dir(data_dir: str, on_file: callable, filter: Iterable = [],
                val_size: float | None = None, test_size: float | None = None,
                seed: int = None):
    sunglasses, no_sunglasses = [], []

    # Count the total number of files in the directory tree, init tqdm
    total = sum(len(files) for _, _, files in os.walk(data_dir)) + 3
    pbar = tqdm.tqdm(desc="Processing data", total=total)

    for root, _, files in os.walk(data_dir):
        for name in files:
            if any(x in name for x in filter):
                pbar.update(1)
                continue
            
            image, is_sunglasses = on_file(root, name)

            if is_sunglasses:
                sunglasses.append((name, image))
            else:
                no_sunglasses.append((name, image))
            
            pbar.update(1)
    
    pbar.update(1)
    pbar.set_description("Shuffling and splitting data")

    sunglasses = sorted(sunglasses, key=lambda x: x[0])
    no_sunglasses = sorted(no_sunglasses, key=lambda x: x[0])
    
    if seed is not None:
        random.seed(seed)
    
    random.shuffle(sunglasses)
    random.shuffle(no_sunglasses)

    num_train_sunglasses = len(sunglasses)
    num_train_no_sunglasses = len(no_sunglasses)
    num_val_sunglasses, num_val_no_sunglasses = 0, 0
    num_test_sunglasses, num_test_no_sunglasses = 0, 0

    if val_size is not None:
        assert 0 < val_size < 1, "Provide valid val fraction"
        num_val_sunglasses = round(len(sunglasses) * val_size)
        num_val_no_sunglasses = round(len(no_sunglasses) * val_size)
        num_train_sunglasses -= num_val_sunglasses
        num_train_no_sunglasses -= num_val_no_sunglasses
    
    if test_size is not None:
        assert 0 < test_size < 1, "Provide valid test fraction"
        num_test_sunglasses = round(len(sunglasses) * test_size)
        num_test_no_sunglasses = round(len(no_sunglasses) * test_size)
        num_train_sunglasses -= num_test_sunglasses
        num_train_no_sunglasses -= num_test_no_sunglasses
    
    def split_list(lst: list, index: int):
        return lst[:index], lst[index:]
    
    sunglasses_train, remainder = split_list(sunglasses, num_train_sunglasses)
    sunglasses_val, sunglasses_test = split_list(remainder, num_val_sunglasses)
    
    no_sunglasses_train, remainder = split_list(no_sunglasses, num_train_no_sunglasses)
    no_sunglasses_val, no_sunglasses_test = split_list(remainder, num_val_no_sunglasses)

    sunglasses_train_dir = os.path.join(data_dir, "train/sunglasses")
    no_sunglasses_train_dir = os.path.join(data_dir, "train/no_sunglasses")
    sunglasses_val_dir = os.path.join(data_dir, "val/sunglasses")
    no_sunglasses_val_dir = os.path.join(data_dir, "val/no_sunglasses")
    sunglasses_test_dir = os.path.join(data_dir, "test/sunglasses")
    no_sunglasses_test_dir = os.path.join(data_dir, "test/no_sunglasses")

    pbar.update(1)
    pbar.set_description("Saving results")

    save_images_to_dir(sunglasses_train, sunglasses_train_dir)
    save_images_to_dir(sunglasses_val, sunglasses_val_dir)
    save_images_to_dir(sunglasses_test, sunglasses_test_dir)
    save_images_to_dir(no_sunglasses_train, no_sunglasses_train_dir)
    save_images_to_dir(no_sunglasses_val, no_sunglasses_val_dir)
    save_images_to_dir(no_sunglasses_test, no_sunglasses_test_dir)

    pbar.update(1)
    pbar.set_description("Done")
    pbar.close()
    

def main():
    kwargs = parse_args()
    [width, height] = kwargs.pop("resize")
    dim_mismatch = kwargs.pop("dim_mismatch")
    landmark_fn, sr_model = None, None

    if (sr_scale := kwargs.pop("sr_scale")) > 1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_fn = getattr(torchsr.models, kwargs.pop("sr_model"))
        sr_model = model_fn(scale=sr_scale, pretrained=True).to(device).eval()

        for param in sr_model.parameters():
            param.requires_grad = False
    
    if "dir/" in (criteria := kwargs.pop("criteria")):
        criteria_fn = lambda x, _: criteria[4:] in os.path.basename(x)
    elif "file/" in criteria:
        criteria_fn = lambda _, x: criteria[5:] in x
    elif ".mat" in criteria:
        criteria_fn, landmark_fn = read_sof_mat(criteria)
    
    on_file_fn = lambda x, y: on_file(x, y, criteria_fn, landmark_fn, sr_model,
                                      width, height, dim_mismatch)
    
    process_dir(kwargs.pop("data_dir"), on_file_fn, **kwargs)


if __name__ == "__main__":
    main()
