import torch, torchvision
import numpy as np
from torch.optim import lr_scheduler
from collections import defaultdict
import os
import json
from torch import nn, optim

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_classes=3

def create_model(n_classes):
    model = models.resnet34(pretrained=True)

    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)

    return model.to(device)



def train_epoch(
        model,
        data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        n_examples
):
    model = model.train()

    losses = []
    correct_predictions = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(model, data_loaders, dataset_sizes, device,model_dir,n_epochs=10):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(n_epochs):

        print(f'Epoch {epoch + 1}/{n_epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            data_loaders['train'],
            loss_fn,
            optimizer,
            device,
            scheduler,
            dataset_sizes['train']
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            data_loaders['test'],
            loss_fn,
            device,
            dataset_sizes['test']
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            print(os.path.join(model_dir, 'model.pth'))
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
            best_accuracy = val_acc

    print(f'Best val accuracy: {best_accuracy}')

#     model.load_state_dict(torch.load('best_model_state.bin'))

    return

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__=="__main__":

    print("starting in main")
    args, unknown = _parse_args()

    DATA_DIR = args.train


    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    transforms = {'train': T.Compose([
        T.RandomResizedCrop(size=256),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]), 'test': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]),
    }


    DATASETS = ['train', 'test']

    image_datasets = {d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS}

    dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}
    class_names = image_datasets['train'].classes

    data_loaders = {d: DataLoader(image_datasets[d], batch_size=8, shuffle=True, num_workers=4) for d in DATASETS}

    base_model = create_model(len(class_names))

    train_model(base_model, data_loaders, dataset_sizes, device, model_dir=args.model_dir)



    ## Save model





