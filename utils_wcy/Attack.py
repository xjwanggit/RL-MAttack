import copy
import os.path
import sys
sys.path.append(r'/home/lxh/wcy/Reinforcement-Learning-Against-RecSys/utils_wcy')
import numpy as np
import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from dataset.datasets import *
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50
from Configuration import args_parser
import tensorflow as tf

to_tensor = transforms.ToTensor()
transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tf_transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize for TensorFlow models
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Denormalization for later use
denormalize = transforms.Normalize(
    mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std=[1 / s for s in [0.229, 0.224, 0.225]]
)

def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename[:-4], npy)


def save_image(image, filename, mode='lossless'):
    """
    Store an image to hard disk
    Args:
        image (pytorch tensor): image to save
        filename (str): filename
        mode (str): either lossless or lossy
    """

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    if mode == 'lossy':
        torchvision.utils.save_image(image, filename)
    elif mode == 'lossless':
        save_np(npy=image.cpu().numpy(), filename=filename)
    else:
        torchvision.utils.save_image(image, filename)
        save_np(npy=image.cpu().numpy(), filename=filename)

def modify_numpy_array(arr):
    # 检查数组的维度
    if len(arr.shape) == 3:
        if arr.shape[0] == 3:  # 如果是 (c, h, w)
            return np.transpose(arr, (1, 2, 0))  # 转换为 (h, w, c)
        elif arr.shape[-1] == 3:  # 如果是 (h, w, c)，保持原样
            return arr
    # 如果不符合要求，返回 None 或者抛出异常，取决于你的需求
    return None


@tf.function
def compute_gradients(model, input_image, target_label, epsilon):
    """
    Compute the gradient of the loss with respect to the input image.
    Args:
        model: The classification model (with CNN-F backbone).
        input_image: The original input image (shape: [1, height, width, 3]).
        target_label: The target label as a one-hot vector.
        epsilon: Perturbation size.
    Returns:
        Perturbed image.
    """
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image, training=False)
        loss = tf.keras.losses.categorical_crossentropy(target_label, prediction)

    # Compute the gradient of the loss with respect to the input image
    gradient = tape.gradient(loss, input_image)

    # Normalize the gradient to unit norm (optional)
    signed_grad = tf.sign(gradient)

    # Create a perturbed image by adding the signed gradient
    perturbed_image = input_image + epsilon * signed_grad

    # Clip the image to ensure pixel values remain valid (0 to 1)
    perturbed_image = tf.clip_by_value(perturbed_image, 0.0, 1.0)

    return perturbed_image, signed_grad


def ensemble_perturb(models, image, perturbation_size, target, model):
    # If input is a file path, open the image using PIL
    if isinstance(image, str):
        file = Image.open(image).convert('RGB')  # Ensure 3 channels
        x = tf_transform_image(file) if "tensorflow" in model else transform_image(file)

    # If input is a numpy array, convert it to a PIL image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] == 3:
            # Convert from (C, H, W) to (H, W, C) for PIL compatibility
            image = np.transpose(image, (1, 2, 0))
        # Apply transformations based on model type
        x = tf_transform_image(image) if "tensorflow" in model else transform_image(image)

    # # If input is already a tensor
    # elif isinstance(image, torch.Tensor):
    #     # If it's a TensorFlow model, ensure the tensor is resized and normalized
    #     if "tensorflow" in model:
    #         # Resize if the tensor is not already [3, 224, 224]
    #         if image.shape[-2:] != (224, 224):
    #             image = transforms.Resize((224, 224))(image)
    #         x = transform_image(image)  # Normalize the resized tensor
    #     else:
    #         # For non-TensorFlow models, apply normalization only
    #         x = transform_image(image)
    else:
        raise ValueError("Unsupported input type. Provide a string path, numpy array, or tensor.")
    # Prepare the image tensor and ensure it is on the correct device
    image = x.unsqueeze(0).to(f"cuda:{param.GPU}")

    perturbed_image = image.clone().detach().requires_grad_(True)
    total_grad = 0.0
    for model in models:
        if isinstance(model, tf.keras.Model):
            perturbed_image_tf = tf.convert_to_tensor(
                perturbed_image.permute(0, 2, 3, 1).detach().cpu().numpy(), dtype=tf.float32
            )  # (B, C, H, W) -> (B, H, W, C)
            # TensorFlow Model Gradient Computation
            with tf.GradientTape() as tape:
                tape.watch(perturbed_image_tf)
                target_numpy = target.cpu().numpy()
                target_tf = tf.constant([target_numpy], dtype=tf.int64)
                output = model(perturbed_image_tf)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=output, labels=target_tf
                )
            grad = -tape.gradient(loss, perturbed_image_tf)
            total_grad += torch.tensor(grad.numpy()).permute(0, 3, 1, 2).to(f"cuda:{param.GPU}")
        else:
            model.eval()
            output = model(perturbed_image)
            loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target]).to(image.device))
            loss = -loss
            model.zero_grad()
            loss.backward(retain_graph=True)
            total_grad += perturbed_image.grad  # Accumulate gradients

        # Update the perturbation using the accumulated gradients
    perturbed_image = perturbed_image + perturbation_size * total_grad.sign()
    perturbed_image = denormalize(perturbed_image)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure valid pixel range
    return perturbed_image.squeeze().detach().cpu().numpy()


def add_intermediate_perturb(model, image, perturbation_size, target, alpha):
    model.eval()  # Set model to evaluation mode

    # Image transformation: Convert image to tensor and normalize
    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    denormalize = transforms.Normalize(
        mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )

    # Load or convert the input image
    if isinstance(image, str):
        file = Image.open(image)
        x = transform_image(file)
    else:
        x = transforms.ToTensor()(image).permute(1, 2, 0)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

    # Prepare the image tensor and ensure it is on the correct device
    image = x.unsqueeze(0).to(f"cuda:{param.GPU}")

    # Select appropriate intermediate layers for different architectures
    model_layers = {
        'vgg16': 'features.28',  # Last conv layer in VGG16
        'resnet18': 'layer4.1.conv2',  # Use a conv layer instead of ReLU
        'resnet50': 'layer4.2.conv3',
    }

    # Determine the model type and select the appropriate layer
    model_name = param.arch.lower()
    if model_name not in model_layers:
        raise ValueError(f"Unsupported model type: {model_name}")

    layer_name = model_layers[model_name]  # Select the correct intermediate layer

    # Store the intermediate activations
    intermediate_activations = None

    # Hook function to capture activations
    def forward_hook(module, input, output):
        nonlocal intermediate_activations
        intermediate_activations = output
        output.retain_grad()  # Retain gradient for the output

    # Register the hook on the specified layer
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(forward_hook)  # Capture activations

    # Multi-step PGD attack using intermediate gradients
    perturbed_image = image.clone().detach().requires_grad_(True)

    # Forward pass to get predictions and intermediate activations
    output = model(perturbed_image)

    # Compute the loss with respect to the target
    loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target]).to(image.device))

    # Zero gradients and perform the backward pass
    model.zero_grad()
    loss = -loss
    loss.backward(retain_graph=True)

    # Check if the intermediate gradient was captured
    if intermediate_activations.grad is None:
        raise RuntimeError(f"Intermediate gradients not captured for layer {layer_name}.")

    # Use the captured intermediate gradients to compute input-level gradients
    input_grad = torch.autograd.grad(
        outputs=intermediate_activations,
        inputs=perturbed_image,
        grad_outputs=intermediate_activations.grad,
        retain_graph=True
    )[0]

    # Perform the PGD update step
    perturbed_image = perturbed_image + alpha * input_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, image - perturbation_size, image + perturbation_size)
    perturbed_image = denormalize(perturbed_image)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure valid pixel range

    # Clear gradients for the next iteration
    if intermediate_activations.grad is not None:
        intermediate_activations.grad.zero_()
    if perturbed_image.grad is not None:
        perturbed_image.grad.zero_()
    # Remove the hook after use
    handle.remove()

    return perturbed_image.squeeze().detach().cpu().numpy()


def Add_perturb(model, x, eps, attack_method, iteration_number=7, eps_iter=0.01, target=False, tensor_file=False):
    model.eval()
    transform_image = transforms.Compose([transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray (H x W x C) in the range[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    if isinstance(x, str):
        file = Image.open(x)
        x = transform_image(file)
    else:
        file = x
        x = transforms.ToTensor()(x).permute(1, 2, 0)
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(x)

    x = x[None, ...].to(f"cuda:{param.GPU}")

    if attack_method == 'FGSM':
        if target:
            target = target[None, ...]
            x_adv = fast_gradient_method(model, x, eps, np.inf, y=target, targeted=True)
        else:
            x_adv = fast_gradient_method(model, x, eps, np.inf)
    elif attack_method == 'PGD':
        if target:
            target = target[None, ...]
            x_adv = projected_gradient_descent(model, x, eps, eps_iter, iteration_number, np.inf, y=target, targeted=True)
        else:
            x_adv = projected_gradient_descent(model, x, eps, eps_iter, iteration_number, np.inf)
    else:
        raise ValueError(f"No attack method named {attack_method}")

    denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                           std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    adv_perturbed_out = denormalize(x_adv[0])

    # Clip before saving image to memory
    adv_perturbed_out[adv_perturbed_out < 0.0] = 0.0
    adv_perturbed_out[adv_perturbed_out > 1.0] = 1.0
    if not tensor_file:
        adv_image = adv_perturbed_out.detach().cpu().numpy()
        # adv_image = modify_numpy_array(adv_image)
    else:
        adv_image = adv_perturbed_out
    # pil_img = transforms.ToPILImage()(adv_perturbed_out)
    if isinstance(file, Image.Image):
        file.close()
    del x_adv, x, adv_perturbed_out
    torch.cuda.empty_cache()
    return adv_image

def Save_img(image, user_id, item_index, mode='lossy'):
    save_file_path = f'./results/{param.model}/{param.dataset}/{param.attack}/Subs_{param.arch}-Rec_{param.rec_arch}/user_id{user_id}/'
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    save_file_name = os.path.join(save_file_path, f'attack_img_{item_index}.jpg')

    # Save image
    if not torch.is_tensor(image):
        if image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        image = to_tensor(image)
    # save_image(image=image, filename=os.path.join(param.save_path, save_file_name), mode='lossy')
    save_image(image=image, filename=save_file_name, mode=mode)

def Save_img_discount_Exp(image, file_path, eps):

    save_file_path = f'./results/{param.dataset}/{param.attack}/discount_reason/Substi_{param.arch}-Rec_{param.rec_arch}/eps_{eps}'
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    save_file_name = os.path.join(save_file_path, file_path.split('.')[0] + '_adv.jpg')

    # Save image
    if not torch.is_tensor(image):
        image = transforms.ToTensor()(image)
    # save_image(image=image, filename=os.path.join(param.save_path, save_file_name), mode='lossy')
    save_image(image=image, filename=save_file_name, mode='None')

param = args_parser()

# param = args_parser()
# model = resnet50(pretrained=True)
# file_path = r'D:\Recommender System Project\TAaMR-master\data\amazon_men\pgd_806_770_eps16_eps_it0.01045751633986928_nb_it10_linf_images\images\177.jpg'
# file_name = file_path.split('\\')[-1]
# save_file_name = file_name[:-4] + '_adv.jpg'
#
# image_file = Image.open(file_path)
# image_file.show(title='original')
#
# adv_perturbed_out = Add_perturb(model, file_path, 16/255, 'FGSM')
#
# denormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
#
# adv_perturbed_out = denormalize(adv_perturbed_out[0])
#
# # Clip before saving image to memory
# adv_perturbed_out[adv_perturbed_out < 0.0] = 0.0
# adv_perturbed_out[adv_perturbed_out > 1.0] = 1.0
#
# pil_img = transforms.ToPILImage()(adv_perturbed_out)
# pil_img.show()
#
# # Save image
# save_image(image=adv_perturbed_out, filename=os.path.join(param.save_path, save_file_name), mode='lossy')
