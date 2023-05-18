import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm.auto import tqdm
from typing import Tuple, Dict, List
import pathlib
from PIL import Image
from timeit import default_timer as timer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_predictions(
        X_train,
        y_train,
        X_test,
        y_test,
        y_pred=None
    ):
    X_train=X_train.cpu().numpy()
    y_train=y_train.cpu().numpy()
    X_test=X_test.cpu().numpy()
    y_test=y_test.cpu().numpy()
    plt.scatter(X_train, y_train, c='b', s=5, label="Training data")
    plt.scatter(X_test, y_test, c='g', s=5, label="Test data")
    if y_pred is not None:
        y_pred = y_pred.cpu().numpy()
        plt.scatter(X_test, y_pred, c='r', s=5, label="Predictions")
    plt.legend()
    plt.show()


def plot_train_test_loss(
        epoch_count,
        train_loss=None,
        test_loss=None,
        test_accuracy=None
    ):
    """Plots train_loss, test_loss and test_accuracy over the epochs."""
    if train_loss:
        plt.plot(epoch_count, torch.tensor(train_loss).cpu().numpy(), label="Train loss")
    if test_loss:
        plt.plot(epoch_count, torch.tensor(test_loss).cpu().numpy(), label="Test loss")
    if test_accuracy:
        plt.plot(epoch_count, test_accuracy, label="Test accuracy")
    plt.legend()
    plt.show()


def print_train_time(start: float, end: float, device: torch.device=None):
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def eval_model_classification(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        accuracy_fn,
        device: torch.device=DEVICE
    ):
    """Returns a dictionary containing the results of model predicting on data_loader."""
    model.eval()
    loss, acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device=device), y.to(device=device)
            y_logit = model(X)
            y_prob = torch.softmax(y_logit, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)
            loss += loss_fn(y_logit, y)
            acc += accuracy_fn(y.cpu().numpy(), y_pred.cpu().numpy())
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_accuracy": acc
    }


def _train_step_classification(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim,
        device: torch.device=DEVICE,
    ) -> Tuple[float, float]:
    """Performs a training loop with model trying to learn on data_loader and returns (train_loss, train_acc)."""
    train_loss, train_acc = 0, 0
    model.to(device=device)
    model.train()

    # Add a loop to loop through the training batches
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device=device), y.to(device=device)

        # 1. Forward pass
        y_logit = model(X)
        y_prob = torch.softmax(y_logit, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)

        # 2. Calculate loss and accuracy (per batch)
        loss = loss_fn(y_logit, y)
        train_loss += loss.item()
        train_acc += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Divide total train loss and acc by length of train dataloader
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def _test_step_classification(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device=DEVICE,
    ) -> Tuple[float, float]:
    """Performs a testing loop step on model going over data_loader and returns (test_loss, test_acc)."""
    test_loss, test_acc = 0, 0
    model.to(device=device)
    model.eval()

    with torch.inference_mode():
        # Add a loop to loop through the testing batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device=device), y.to(device=device)

            # 1. Forward pass
            y_logit = model(X)
            y_prob = torch.softmax(y_logit, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)

            # 2. Calculate loss and accuracy (per batch)
            loss = loss_fn(y_logit, y)
            test_loss += loss.item()
            test_acc += accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())

        # Divide total test loss and acc by length of test dataloader
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim,
        loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
        epochs: int = 5, 
        device=DEVICE
    ) -> Dict[str, List[float]]:
    results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = _train_step_classification(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = _test_step_classification(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Print out what's happening
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def make_predictions(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = DEVICE) -> torch.tensor:
    """Return prediction tensor for the the data using the model."""
    model = model.to(device)
    model.eval()
    y_preds = []
    with torch.inference_mode():
        for batch, (X, y) in tqdm(enumerate(data_loader), desc="Making predictions..."):
            # Send the data and targets to target device
            X, y = X.to(device), y.to(device)

            # Do the forward pass
            y_logit = model(X)

            # Turn predictions from logits -> prediction probabilities -> prediction labels
            y_prob = torch.softmax(y_logit, dim=1)
            y_pred = torch.argmax(y_prob, dim=1)
            
            # Put prediction on CPU for evaluation
            y_preds.append(y_pred.cpu())

    return torch.cat(y_preds)


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary."""
    # Get the loss values of the results dictionary(training and test)
    train_loss = results["train_loss"]
    test_loss = results["test_loss"]

    # Get the accuracy values of the results dictionary (training and test)
    train_accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    # Figure out how mnay epochs there were
    epochs = range(len(results["train_loss"]))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend() 

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def pred_and_plot_single_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device=DEVICE
):
    """Makes a prediction on a target image with a trained model and plots the image and prediction."""
    # Load in the image
    original_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    original_image = original_image / 255.

    # Transform if necessary
    if transform:
        target_image = transform(original_image)
    else:
        target_image = original_image

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
    plt.show()


def pred_and_store(paths: List[pathlib.Path], 
                   model: torch.nn.Module,
                   transform: torchvision.transforms, 
                   class_names: List[str], 
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
    """A function to return a list of dictionaries with sample, truth label, prediction, prediction probability and prediction time"""
    
    # 2. Create an empty list to store prediction dictionaires
    pred_list = []
    
    # 3. Loop through target paths
    for path in tqdm(paths):
        
        # 4. Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # 5. Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name
        
        # 6. Start the prediction timer
        start_time = timer()
        
        # 7. Open image path
        img = Image.open(path)
        
        # 8. Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device) 
        
        # 9. Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()
        
        # 10. Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image) # perform inference on target sample 
            pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
            pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
            pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

            # 11. Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class
            
            # 12. End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        # 13. Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # 14. Add the dictionary to the list of preds
        pred_list.append(pred_dict)
    
    # 15. Return list of prediction dictionaries
    return pred_list


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


# Set seeds
def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)