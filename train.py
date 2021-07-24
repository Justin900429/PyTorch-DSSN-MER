import argparse
import os
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from dataloader import (
    MyDataset,
    get_loader,
    LOSO_sequence_generate
)
import network
from read_file import read_csv


def train(epochs: int, criterion: nn.Module, optimizer: torch.optim,
          model: nn.Module, train_loader: DataLoader, device: torch.device,
          weight_path: str, model_last_name: str):
    """Train the model

    Parameters
    ----------
    epochs : int
        Epochs for training the model
    model : DSSN
        Model to be trained
    train_loader : DataLoader
        DataLoader to load in the data
    device: torch.device
        Device to be trained on
    weight_path: str
        Place for weight to be saved
    model_last_name: str
        Last name for the model to be saved
    """
    best_accuracy = -1

    for epoch in range(epochs):
        train_loss = 0.0
        train_accuracy = 0.0

        # Set model in training mode
        model.train()
        for stream, labels in train_loader:
            if isinstance(stream, (tuple, list)):
                first_stream = stream[0].to(device)
                second_stream = stream[1].to(device)
                output = model(first_stream, second_stream)
            else:
                stream = stream.to(device)
                output = model(stream)
            labels = labels.to(device)

            # Compute the loss
            loss = criterion(output, labels)
            train_loss += loss.item()

            # Update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            train_accuracy += prediction.sum().item() / labels.size(0)

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        print(f"Epoch: {epoch + 1}")
        print(f"Loss: {train_loss}")
        print(f"Accuracy: {train_accuracy}")

        if train_accuracy > best_accuracy:
            torch.save(model.state_dict(), f"{weight_path}/model.pt")
            best_accuracy = train_accuracy
            print("Save model")

    torch.save(model.state_dict(), f"{weight_path}/{model_last_name}")
    print("Save the last model")


def evaluate(test_loader: DataLoader, model: nn.Module, device: torch.device):
    # Set into evaluation mode
    model.eval()
    test_accuracy = 0.0
    test_f1_score = 0.0

    with torch.no_grad():
        for stream, labels in test_loader:
            # Move data to device and compute the output
            if isinstance(stream, (tuple, list)):
                first_stream = stream[0].to(device)
                second_stream = stream[1].to(device)
                output = model(first_stream, second_stream)
            else:
                stream = stream.to(device)
                output = model(stream)
            labels = labels.to(device)

            # Compute the accuracy
            prediction = (output.argmax(-1) == labels)
            test_accuracy += prediction.sum().item() / labels.size(0)
            test_f1_score += f1_score(labels.cpu().numpy(), output.argmax(-1).cpu().numpy(),
                                      average="weighted")

    return test_accuracy / len(test_loader), test_f1_score / len(test_loader)


def LOSO_train(data: pd.DataFrame, sub_column: str, args,
               label_mapping: dict, device: torch.device):
    log_file = open("train.log", "w")
    # Create different DataFrame for each subject
    train_list, test_list = LOSO_sequence_generate(data, sub_column)
    test_accuracy = 0.0
    test_f1_score = 0.0

    image_mode = tuple(args.image_mode)
    for idx in range(len(train_list)):
        print(f"=================LOSO {idx + 1}=====================")
        train_csv = train_list[idx]
        test_csv = test_list[idx]

        # Create dataset and dataloader
        _, train_loader = get_loader(csv_file=train_csv,
                                     preprocess_path=args.pre,
                                     label_mapping=label_mapping,
                                     mode=image_mode,
                                     batch_size=args.batch_size,
                                     catego=args.catego)
        _, test_loader = get_loader(csv_file=test_csv,
                                    preprocess_path=args.pre,
                                    label_mapping=label_mapping,
                                    mode=image_mode,
                                    batch_size=args.batch_size,
                                    catego=args.catego)

        # Read in the model
        model = getattr(network, args.model)(num_classes=args.num_classes,
                                             freeze_k=args.freeze_k,
                                             mode=args.combination_mode,).to(device)

        # Create criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)

        # Train the data
        train(epochs=args.epochs,
              criterion=criterion,
              optimizer=optimizer,
              model=model,
              train_loader=train_loader,
              device=device,
              weight_path=args.weight_save_path,
              model_last_name=f"model_last_{idx}.pt")

        print(f"{idx + 1}")
        temp_test_accuracy, temp_f1_score = evaluate(test_loader=test_loader,
                                                     model=model,
                                                     device=device)
        print(f"In LOSO {idx + 1}, test accuracy: {temp_test_accuracy}, f1-score: {temp_f1_score}")
        log_file.write(f"LOSO {idx + 1}: Accuracy: {temp_test_accuracy}, F1-Score: {temp_f1_score}\n")
        test_accuracy += temp_test_accuracy
        test_f1_score += temp_f1_score

    print(f"LOSO accuracy: {test_accuracy / len(train_list)}, f1-score: {test_f1_score / len(train_list)}")
    log_file.write(f"Total: Accuracy {test_accuracy / len(train_list)}, F1-Score: {test_f1_score / len(train_list)}\n")
    log_file.close()


if __name__ == "__main__":
    # Argument parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        required=True,
                        help="Path for the csv file for training data")
    parser.add_argument("--pre",
                        type=str,
                        required=True,
                        help="Preprocess path for training")
    parser.add_argument("--catego",
                        type=str,
                        required=True,
                        help="SAMM or CASME dataset")
    parser.add_argument("--num_classes",
                        type=int,
                        default=5,
                        help="Classes to be trained")
    parser.add_argument("--combination_mode",
                        type=str,
                        default="add",
                        help="Mode to be used in combination")
    parser.add_argument("--image_mode",
                        nargs="+",
                        help="Image type to be used in training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Training batch size")
    parser.add_argument("--weight_save_path",
                        type=str,
                        default="model",
                        help="Path for the saving weight")
    parser.add_argument("--model",
                        type=str,
                        default="DSSN",
                        help="Model to used for training")
    parser.add_argument("--freeze_k",
                        type=int,
                        default=4,
                        help="Layer to freeze in AlexNet")
    parser.add_argument("--epochs",
                        type=int,
                        default=15,
                        help="Epochs for training the model")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate for training the model")
    args = parser.parse_args()

    # Check if freeze_k is smaller or equal to 5
    assert args.freeze_k <= 5, "freeze_k should smaller or equal to 5"

    # Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read in the data
    data, label_mapping = read_csv(args.path)

    # Create folders for the saving weight
    os.makedirs(args.weight_save_path, exist_ok=True)

    # Train the model
    LOSO_train(data=data,
               sub_column="Subject",
               label_mapping=label_mapping,
               args=args,
               device=device)

