"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms


def parse_args() -> argparse.Namespace:
    # Hyperparameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    HIDDEN_UNITS = 10

    # Directories
    TRAIN_DIR = "../data/pizza_steak_sushi/train"
    TEST_DIR = "../data/pizza_steak_sushi/test"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="path to training data", type=str, default=TRAIN_DIR)
    parser.add_argument("--test_dir", help="path to testing data", type=str, default=TEST_DIR)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch_size", help="batch size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", help="number of epochs to train for", type=int, default=NUM_EPOCHS)
    parser.add_argument("--hidden_units", help="number of neurons in the hidden layer", type=int, default=HIDDEN_UNITS)
    args = parser.parse_args()
    return args


def train(args):
    # Setup hyperparameters
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    hidden_units = args.hidden_units

    # Setup directories
    train_dir = args.train_dir
    test_dir = args.test_dir

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
    ])

    # Create DataLoaders with help from data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=batch_size
    )

    # Create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=hidden_units,
        output_shape=len(class_names)
    ).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=learning_rate)

    # Start training with help from engine.py
    engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=num_epochs,
                device=device)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                    target_dir="../models",
                    model_name="05_going_modular_script_mode_tinyvgg_model.pth")


if __name__ == "__main__":
    train(parse_args())
    