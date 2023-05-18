import torch
import torchvision
from torchvision import transforms
from typing import List, Tuple
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

from model_builder import TinyVGG


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parent_dir = Path(os.path.abspath(__file__)).parent.parent
    MODEL_PATH = os.path.join(parent_dir, "models", "05_going_modular_script_mode_tinyvgg_model.pth")
    IMAGE_PATH = os.path.join(parent_dir, "data", "04-pizza-dad.jpeg")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="path to the model", type=str, default=MODEL_PATH)
    parser.add_argument("--image_path", help="path to the custom image", type=str, default=IMAGE_PATH)
    args = parser.parse_args()
    return args


def pred_and_plot_single_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device=DEVICE
) -> Tuple[str, float]:
    """Makes a prediction on a target image with a trained model and plots the image and prediction."""
    # Load in the image
    original_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    original_image = original_image / 255.

    # Transform if necessary
    if transform:
        target_image = transform(original_image)

    # Make sure the model is on the target device
    model.to(device)

    # Turn on eval/inference mode and make a prediction
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image (this is the batch dimension, e.g. our model will predict on batches of 1x image)
        target_image = target_image.unsqueeze(0)

        # Make a prediction on the image with an extra dimension
        target_image_pred = model(target_image.to(device)) # make sure the target image is on the right device

    # Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert predction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot the image alongside the prediction and prediction probability
    plt.imshow(original_image.squeeze().permute(1, 2, 0)) # remove batch dimension and rearrange shape to be HWC
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    return class_names[target_image_pred_label.cpu()], target_image_pred_probs.max().cpu()


if __name__ == "__main__":
    args = parse_args()
    
    # transform
    custom_image_transform = transforms.Compose(
        [
            transforms.Resize(size=(64, 64))
        ]
    )

    class_names = ["pizza", "steak", "sushi"]

    # load model
    model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names))
    model.load_state_dict(torch.load(f=Path(args.model_path)))
    model.to(device=DEVICE)

    # Pred on our custom image
    pred_class, pred_prob = pred_and_plot_single_image(
        model=model,
        image_path=Path(args.image_path),
        class_names=class_names,
        transform=custom_image_transform,
        device=DEVICE
    )
    print(f"[INFO] Pred class: {pred_class}, Pred prob: {pred_prob:.3f}")