import torch
import torchvision.transforms as transforms
from PIL import Image
from utils_wcy.Configuration import args_parser

class WeightedEnsembleAttack:
    def __init__(self, models):
        self.param = args_parser()
        self.models = models
        self.transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.denormalize = transforms.Normalize(
            mean=[-m / s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
            std=[1 / s for s in [0.229, 0.224, 0.225]]
        )
        self.device = f"cuda:{self.param.GPU}"
        self.reset()

    def reset(self):
        """Reset the attack state: success/failure counts and perturbations."""
        self.success_counts = [1] * len(self.models)  # Initialize to avoid div by zero
        self.failure_counts = [1] * len(self.models)
        self.perturbations = []  # Clear any stored perturbations

    def compute_weights(self):
        """Compute normalized weights based on success probabilities."""
        probabilities = [
            self.success_counts[i] / (self.success_counts[i] + self.failure_counts[i])
            for i in range(len(self.models))
        ]
        total_prob = sum(probabilities)
        weights = [p / total_prob for p in probabilities]
        return weights

    def update_counts(self, model_index, success):
        """Update success or failure counts for a specific model."""
        if success:
            self.success_counts[model_index] += 1
        else:
            self.failure_counts[model_index] += 1

    def ensemble_perturb(self, image, perturbation_size, target):
        """Apply weighted ensemble perturbation."""
        perturbed_image = image.clone().detach().requires_grad_(True)
        total_grad = 0.0

        # Compute model weights
        weights = self.compute_weights()

        # Accumulate weighted gradients from each model
        for i, model in enumerate(self.models):
            model.eval()
            output = model(perturbed_image)
            loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target]).to(self.device))
            loss = -loss  # Maximize the target class probability

            model.zero_grad()
            loss.backward(retain_graph=True)

            # Accumulate the weighted gradients
            total_grad += weights[i] * perturbed_image.grad
            perturbed_image.grad.zero_()  # Clear gradients

        # Apply the perturbation
        perturbed_image = perturbed_image + perturbation_size * total_grad.sign()
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Clamp pixel values

        return perturbed_image.squeeze().detach().cpu().numpy()

    def apply_weighted_ensemble(self, image, perturbation_size):
        """Apply the weighted ensemble perturbation using stored perturbations."""
        if isinstance(image, str):
            file = Image.open(image)
            image = self.transform_image(file).to(self.device)
        else:
            image = transforms.ToTensor()(image).permute(1, 2, 0)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).to(self.device)
        # Compute model weights based on success probabilities
        weights = self.compute_weights()

        # Accumulate weighted perturbations
        total_grad = sum(w * p for w, p in zip(weights, self.perturbations))

        # Apply the perturbation to the original image
        perturbed_image = image + perturbation_size * total_grad.sign()
        perturbed_image = self.denormalize(perturbed_image)
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure valid pixel range

        return perturbed_image.squeeze().detach().cpu().numpy()

    def generate_individual_perturbations(self, image, perturbation_size, target):
        """Generate individual perturbations for each model and store them."""
        if isinstance(image, str):
            file = Image.open(image)
            image = self.transform_image(file)
        else:
            image = transforms.ToTensor()(image).permute(1, 2, 0)
            image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

        self.perturbations = []  # Reset stored perturbations
        perturbed_images = []

        for i, model in enumerate(self.models):
            perturbed_image = image.clone().detach().requires_grad_(True).to(self.device).unsqueeze(axis=0)
            perturbed_image.retain_grad()
            model.eval()

            # Forward pass and compute loss
            output = model(perturbed_image)
            loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target]).to(self.device))
            loss = -loss  # Maximize the target class probability

            # Backpropagate the loss to get gradients
            model.zero_grad()
            loss.backward()

            # Store the perturbation (gradient) for this model
            perturbation = perturbed_image.grad.clone().detach()
            self.perturbations.append(perturbation)
            perturbed_image = perturbed_image + perturbation_size * perturbation.sign()
            perturbed_image = torch.clamp(self.denormalize(perturbed_image), 0, 1)
            # Save the perturbed image for future use
            perturbed_images.append(perturbed_image.squeeze().detach().cpu().numpy())

            # Clear gradients for the next model
            model.zero_grad()

        return perturbed_images
