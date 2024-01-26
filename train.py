import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET

from utils import (
   load_checkpoint,
   save_checkpoint,
   get_loaders,
   check_accuracy,
   save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "Dataset/train_images/train"
TRAIN_MASK_DIR = "Dataset/train_masks/train_masks"
TEST_IMG_DIR = "Dataset/test_images"
TEST_MASK_DIR = "Dataset/test_masks"

def train_fnc(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        #forward
    with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        #backward
    # optimizer.zero_grad()
    # scaler.scale(loss).backward
    # scaler.step(optimizer)
    # scaler.update()
        scaler.scale(loss).backward()  # Scale the loss before backward
        scaler.step(optimizer)         # Step the optimizer
        scaler.update()                # Update the scaler
        optimizer.zero_grad(set_to_none=True)  # Zero out the gradients

        #update tqdm loop
    loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0,1.0,1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0,0.0,0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() #FOR THREE CHANNELS CROSSENTROPYLOSS  OU_CHANNELS ABOVE = 3
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        test_transforms,
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    
    check_accuracy(test_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        train_fnc(train_loader, model, optimizer, loss_fn, scaler)

    #save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
    #check accuracy
        check_accuracy(test_loader, model, device=DEVICE)

        save_predictions_as_imgs(test_loader, model, folders="saved_images/", device=DEVICE
                                 )
if __name__ == "__main__":
    main()

