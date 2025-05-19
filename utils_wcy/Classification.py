import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import copy

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
transforms_image_vgg = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor()
])
transforms_image_tensorflow = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor (this gives shape (C, H, W))
        transforms.Resize((224, 224)),  # Resize to the input size of the model
])

denormalize = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std])
])

to_Tensor = transforms.ToTensor()
to_Normalize = transforms.Normalize(mean=mean, std=std)


def denormalize_tf(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = tf.constant(mean, dtype=tensor.dtype)
    std = tf.constant(std, dtype=tensor.dtype)
    return tensor * std + mean

# Define the denormalization function for PyTorch tensors
def denormalize_torch(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    denormalize = torch.nn.Sequential(
        torch.nn.Normalize(
            mean=[-m / s for m, s in zip(mean, std)],
            std=[1 / s for s in std]
        )
    )
    return denormalize(tensor)


# Universal function to handle both types of tensors
def denormalize_and_convert_to_numpy(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Check if the input is a TensorFlow tensor
    if isinstance(tensor, tf.Tensor):
        denormalized_tensor = denormalize_tf(tensor, mean, std)
        return denormalized_tensor.numpy().squeeze()  # Convert to NumPy

    # Check if the input is a PyTorch tensor
    elif isinstance(tensor, torch.Tensor):
        denormalized_tensor = denormalize_torch(tensor, mean, std)
        return denormalized_tensor.cpu().numpy().suqeeze()  # Convert to NumPy, ensuring it's on CPU
    else:
        raise TypeError("Input must be a TensorFlow EagerTensor or a PyTorch Tensor.")

def tf_normalize(tensor, mean, std):
    """
    Normalize a TensorFlow tensor with the specified mean and standard deviation.
    Equivalent to torch's transforms.Normalize.
    """
    mean = tf.constant(mean)
    std = tf.constant(std)

    mean = tf.reshape(mean, [1, 1, 3])
    std = tf.reshape(std, [1, 1, 3])

    return (tensor - mean) / std


def Classify(file, model, tpdv):
    model.eval()

    if isinstance(file, str):
        file = Image.open(file)
        tensor_file = transforms_image(file)
    elif isinstance(file, Image.Image):
        tensor_file = transforms_image(file)
    elif isinstance(file, torch.Tensor):
        tensor_file = file
    elif isinstance(file, np.ndarray):
        tensor_file = torch.tensor(file).to(**tpdv)
    else:
        raise TypeError(f"The input shape of the file {type(file)} is invalid")
    tensor_file = tensor_file[None, ...]
    # tensor_file = tensor_file.reshape((-1, tensor_file.shape[0], tensor_file.shape[1], tensor_file.shape[2]))
    tensor_file = tensor_file.to(**tpdv)
    pred = model(tensor_file).detach().cpu().numpy()
    # feature = feature_model(tensor_file).reshape(1, -1).detach().cpu().numpy()
    if isinstance(file, Image.Image):
        file.close()
    del tensor_file
    torch.cuda.empty_cache()
    return pred


# def Feature_Extract(file, feature_model, tpdv, model_arch='resnet', adv=False, tensor_feature=False, tensor_file_require=False, perturbation=None, is_tensorflow=False, flag=False):
#
#     if isinstance(file, str):
#         file = Image.open(file)
#         if adv:
#             image_np = np.array(file)
#             smoothed_img_np = cv2.GaussianBlur(image_np, (5, 5), 1.5)
#             file = Image.fromarray(smoothed_img_np)
#     if not is_tensorflow:
#         feature_model.eval()
#
#     if isinstance(file, np.ndarray) and not flag:
#         if file.shape[0] == 3:  # Check if the first dimension is the channel dimension
#             file = np.transpose(file, (1, 2, 0))
#         file = Image.fromarray((file * 255).astype(np.uint8))
#     elif isinstance(file, np.ndarray) and flag:
#         if file.shape[0] == 3:  # Check if the first dimension is the channel dimension
#             file = np.transpose(file, (1, 2, 0))
#     if not flag:
#         if perturbation is not None:
#             if 'resnet' in model_arch:
#                 tensor_file = to_Tensor(file)
#                 tensor_file += perturbation
#                 tensor_file = torch.clamp(tensor_file, 0, 1)
#                 tensor_file = to_Normalize(tensor_file)
#             elif 'vgg' in model_arch:
#                 tensor_file = transforms_image_vgg(file)
#                 tensor_file += perturbation
#                 tensor_file = torch.clamp(tensor_file, 0, 1)
#                 tensor_file = to_Normalize(tensor_file)
#             elif 'tensorflow' in model_arch:
#                 tensor_file = transforms_image_tensorflow(file)
#                 tensor_file += perturbation
#                 tensor_file = tf.clip_by_value(tensor_file, 0, 1)
#                 tensor_file = tf_normalize(tensor_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             else:
#                 raise ValueError(f"No model arch name {model_arch}")
#         else:
#             if 'resnet' in model_arch:
#                 if isinstance(file, np.ndarray):
#                     tensor_file = to_Tensor(file)
#                     if tensor_file.shape[0] != 3 and tensor_file.shape[-1] != 3:
#                         tensor_file = tensor_file.permute(1, 2, 0)
#                     tensor_file = to_Normalize(tensor_file)
#                 elif isinstance(file, torch.Tensor):
#                     if file.ndim == 4:  # If batch dimension exists, take the first element
#                         file = file[0]
#                     if file.shape[0] != 3 and file.shape[-1] != 3:
#                         file = file.permute(1, 2, 0)  # Swap dimensions if needed
#                     tensor_file = to_Normalize(file)
#                 else:
#                     tensor_file = transforms_image(file)
#             elif 'vgg' in model_arch:
#                 tensor_file = transforms_image_vgg(file)
#                 tensor_file = to_Normalize(tensor_file)
#             elif 'tensorflow' in model_arch:
#                 tensor_file = transforms_image_tensorflow(file)
#                 tensor_file = tf_normalize(tensor_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             else:
#                 raise ValueError(f"No model arch name {model_arch}")
#     else:
#         if 'resnet' in model_arch:
#             if isinstance(file, np.ndarray):
#                 tensor_file = to_Tensor(file)
#                 if tensor_file.shape[0] != 3 and tensor_file.shape[-1] != 3:
#                     tensor_file = tensor_file.permute(1, 2, 0)
#                 tensor_file = to_Normalize(tensor_file)
#             elif isinstance(file, torch.Tensor):
#                 if file.ndim == 4:  # If batch dimension exists, take the first element
#                     file = file[0]
#                 if file.shape[0] != 3 and file.shape[-1] != 3:
#                     file = file.permute(1, 2, 0)  # Swap dimensions if needed
#                 tensor_file = to_Normalize(file)
#             else:
#                 tensor_file = transforms_image(file)
#         elif 'vgg' in model_arch:
#             tensor_file = transforms_image_vgg(file)
#             tensor_file = to_Normalize(tensor_file)
#         elif 'tensorflow' in model_arch:
#             if isinstance(file, np.ndarray):
#                 tensor_file = tf.convert_to_tensor(file, dtype=tf.float32)
#                 tensor_file = tf.transpose(tensor_file, perm=[2, 0, 1])
#                 tensor_file = tf_normalize(tensor_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         else:
#             raise ValueError(f"No model arch name {model_arch}")
#     tensor_file = tensor_file[None, ...]
#     # tensor_file = tensor_file.reshape((-1, tensor_file.shape[0], tensor_file.shape[1], tensor_file.shape[2]))
#     if 'tensorflow' in model_arch:
#         tensor_file = tf.transpose(tensor_file, perm=[0, 2, 3, 1])
#         feature = feature_model(tensor_file)
#     else:
#         tensor_file = tensor_file.to(**tpdv)
#         if not tensor_feature:
#             feature = feature_model(tensor_file).reshape(1, -1).detach().cpu().numpy()
#         else:
#             feature = feature_model(tensor_file).reshape(1, -1)
#     # feature = feature_model(tensor_file).reshape(1, -1).detach().cpu().numpy()
#     if isinstance(file, Image.Image):
#         file.close()
#     # torch.cuda.empty_cache()
#     if tensor_file_require:
#         return denormalize_and_convert_to_numpy(tensor_file), feature
#     else:
#         del tensor_file
#         return feature


def Feature_Extract(file, feature_model, tpdv, model_arch='resnet', adv=False, tensor_feature=False, tensor_file_require=False, perturbation=None, is_tensorflow=False, flag=False):
    if not is_tensorflow:
        feature_model.eval()

    if isinstance(file, str):
        file = Image.open(file)
        if adv:
            image_np = np.array(file)
            file = cv2.GaussianBlur(image_np, (5, 5), 1.5)
        else:
            file = np.array(file)

    if isinstance(file, np.ndarray):
        if file.shape[0] == 3:
            file = np.transpose(file, (1, 2, 0))
    if perturbation is not None:
        if is_tensorflow:
            tensor_file = transforms_image_tensorflow(file)
            tensor_file += perturbation
            tensor_file = tf.transpose(tensor_file, perm=[1, 2, 0])
            tensor_file = tf.clip_by_value(tensor_file, clip_value_min=0, clip_value_max=1)
            tensor_file = tf_normalize(tensor_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            tensor_file = to_Tensor(file)
            tensor_file += perturbation
            tensor_file = torch.clamp(tensor_file, 0, 1)
            tensor_file = to_Normalize(tensor_file)
    else:
        if is_tensorflow:
            tensor_file = transforms_image_tensorflow(file)
            tensor_file = tensor_file.permute(1, 2, 0)
            tensor_file = tensor_file.numpy()
            tensor_file = tf.convert_to_tensor(tensor_file)
            tensor_file = tf_normalize(tensor_file, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            tensor_file = to_Tensor(file)
            tensor_file = to_Normalize(tensor_file)
    tensor_file = tensor_file[None, ...]
    if is_tensorflow:
        feature = feature_model(tensor_file)
    else:
        feature = feature_model(tensor_file.to(**tpdv))
        if len(feature.shape) == 4 and feature.shape[-1] == 1 and feature.shape[-2] == 1:
            feature = feature.view(feature.shape[0], -1)
    if tensor_file_require:
        return denormalize_and_convert_to_numpy(tensor_file), feature
    else:
        del tensor_file
        return feature


def Comput_SSIM(original_image, perturbed_image, model='VBPR'):
    ori_file = cv2.imread(original_image)
    if isinstance(perturbed_image, str):
        adv_image = cv2.imread(perturbed_image)
    else:
        adv_image = copy.deepcopy(perturbed_image) * 255
        if ori_file.shape[-1] == 3 and adv_image.shape[-1] != 3:
            adv_image = np.transpose(adv_image, (1, 2, 0))

    if model != 'VBPR' and model != 'AMR':
        ori_file = cv2.resize(ori_file, (adv_image.shape[1], adv_image.shape[0]))  # Resize to match shape


    # Compute SSIM using skimage
    ssim_value_skimage, _ = ssim(ori_file, adv_image, full=True, channel_axis=2, data_range=255)
    cv2.destroyAllWindows()
    del adv_image
    return ssim_value_skimage
