import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset as BaseDataset
from torchvision import transforms as T
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import albumentations as albu
from PIL import Image
import cv2
import csv
import albumentations as albu
from sklearn.metrics import confusion_matrix
import time
import os
from tqdm import tqdm
from torchvision import transforms
import segmentation_models_pytorch as smp
DATA_DIR = 'dataset/'
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')

CLASSES = ['Background', 'LV', 'Myocardium', 'MI', 'MVO']
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(256, 256)
    ]
    return albu.Compose(test_transform)
# load best saved checkpoint
# model = smp.Unet( classes=len(CLASSES))
model = torch.load('UNet_mIoU-0.770.pt')

def calculate_metrics(predictions, targets, num_classes):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    true_positives = np.zeros(num_classes)
    false_positives = np.zeros(num_classes)
    false_negatives = np.zeros(num_classes)

    for i in range(num_classes):
        intersection[i] = np.sum((predictions == i) & (targets == i))
        union[i] = np.sum((predictions == i) | (targets == i))
        true_positives[i] = np.sum((predictions == i) & (targets == i))
        false_positives[i] = np.sum((predictions == i) & (targets != i))
        false_negatives[i] = np.sum((predictions != i) & (targets == i))

    precision = np.divide(true_positives, (true_positives + false_positives), out=np.zeros_like(true_positives), where=(true_positives + false_positives) != 0)
    recall = np.divide(true_positives, (true_positives + false_negatives), out=np.zeros_like(true_positives),
                          where=(true_positives + false_negatives) != 0)
    f1_score = 2 * np.divide((precision * recall), (precision + recall), out=np.zeros_like((precision * recall)),
                       where=(precision + recall) != 0)
    iou = 2 * np.divide(intersection, union, out=np.zeros_like(intersection),
                             where=union != 0)
    accuracy = true_positives / np.sum(targets == targets)

    return precision, recall, f1_score, iou,accuracy

class testDataset(BaseDataset):


    CLASSES = ['Background', 'LV', 'Myocardium', 'MI', 'MVO']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        img_id = self.ids[i]


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        #t = T.Compose([T.ToTensor()])
        #image = t(image)
        mask = torch.from_numpy(mask).long()

        return image, mask, img_id

    def __len__(self):
        return len(self.ids)

# create test dataset
test_dataset = testDataset(
    x_test_dir,
    y_test_dir,
    augmentation=get_validation_augmentation(),
    #preprocessing=get_preprocessing(preprocessing_fn),
    classes=CLASSES,
)

test_dataloader = DataLoader(test_dataset)

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=len(CLASSES)):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int32)

        for i in range(n_classes):
            for j in range(n_classes):
                confusion_matrix[i, j] = torch.logical_and(mask == i, pred_mask == j).sum().float().item()
            # Calculate F1 score for each class
            f1_scores = []
            precision_score = []
            recall_score = []
            for class_id in range(n_classes):
                true_positive = confusion_matrix[class_id, class_id]
                false_positive = np.sum(confusion_matrix[:, class_id]) - true_positive
                false_negative = np.sum(confusion_matrix[class_id, :]) - true_positive
                precision = true_positive / (true_positive + false_positive + 1e-10)
                recall = true_positive / (true_positive + false_negative + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                f1_scores.append(f1)
                precision_score.append(precision)
                recall_score.append(recall)


        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class), np.nanmean(f1_scores), np.nanmean(precision_score), np.nanmean(recall_score)

device = torch.device('cuda')
def predict_image_mask_miou(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor()])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)

    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        score, f1, pre, rec = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score, f1, pre, rec

def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy
def save_images(mask,pred_mask, img_id):
    pred_mask_np = pred_mask.numpy()
    mask_gt = mask.cpu().squeeze(0)
    mask_gt = mask_gt.numpy()
    label_to_color = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 255],
        3: [0, 0, 255],
        4: [0, 255, 0],
    }
    img_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    img_gt = np.zeros((128, 128, 3), dtype=np.uint8)
    for label, rgb in label_to_color.items():
        img_rgb[pred_mask_np == label, :] = rgb
        img_gt[mask_gt == label, :] = rgb
    img = Image.fromarray(img_rgb, "RGB")
    gt = Image.fromarray(img_gt, "RGB")
    img.save(f'pred/{img_id}')
    gt.save(f'GT/{img_id}')
j = 0
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    return plt
def predict_image_mask_pixel(model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    t = T.Compose([T.ToTensor()])
    image = t(image)
    model.to(device); image=image.to(device)
    mask = mask.to(device)
    with torch.no_grad():

        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc

def miou_score(model, test_set):
    score_iou = []
    f1_scores = []
    precision_score = []
    recall_score = []
    num_classes = 5
    all_metrics = []
    all_miou = []
    all_f1 = []
    all_precision = []
    all_recall = []
    all_accuracy = []

    for i in tqdm(range(len(test_set))):
        img, mask, img_id = test_set[i]
        pred_mask, score, f1, pre, rec = predict_image_mask_miou(model, img, mask)
        target = mask.squeeze().cpu().numpy()
        pred = pred_mask.squeeze().cpu().numpy()

        # Calculate metrics for this image
        precision, recall, f1_score, miou,acc = calculate_metrics(pred, target, num_classes)
        # all_metrics.append(metrics)
        all_recall.append(recall)
        all_precision.append(precision)
        all_f1.append(f1_score)
        all_miou.append(miou)
        all_accuracy.append(acc)




        fig=visualize(
            image=img,
            ground_truth=mask,
            predict_mask=pred_mask)
        fig.savefig(f'U-Net/Figs/{img_id}.png')

        pred = pred_mask.cpu().numpy()
        gt = mask.cpu().numpy()
        cv2.imwrite(f'U-Net/Pred_Indx/{img_id}',pred)
        cv2.imwrite(f'U-Net/GT_Indx/{img_id}', gt)

        score_iou.append(score)
        f1_scores.append(f1)
        precision_score.append(pre)
        recall_score.append(rec)
    # Calculate mean metrics over all images
    recall_array = np.array(all_recall)
    precision_array = np.array(all_precision)
    f1_array = np.array(all_f1)
    miou_array = np.array(all_miou)
    accuracy_array = np.array(all_accuracy)

    return recall_array, precision_array, f1_array, miou_array, accuracy_array
recall_array, precision_array, f1_array, miou_array, accuracy_array = miou_score(model, test_dataset)

# # Calculate the mean for each class across all images
class_means_rec = recall_array.mean(axis=0)
class_means_pre = precision_array.mean(axis=0)
class_means_f1 = f1_array.mean(axis=0)
class_means_miou = miou_array.mean(axis=0)
class_means_acc = accuracy_array.mean(axis=0)

# Print the mean for each class
for class_num in range(5):
    print(f"Class {class_num + 1} Mean Metrics:")
    print(f"  Recall: {class_means_rec[class_num]:.5f}")
    print(f"  Precision: {class_means_pre[class_num]:.5f}")
    print(f"  F1 Score: {class_means_f1[class_num]:.5f}")
    print(f"  Mean IoU: {class_means_miou[class_num]:.5f}")
    print(f"  Accuracy: {class_means_acc[class_num]:.5f}")
    print()




def pixel_acc(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask,_ = test_set[i]
        pred_mask, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return accuracy

mob_acc = pixel_acc(model, test_dataset)
print('Test Set Pixel Accuracy', np.mean(mob_acc))

