import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import torch.nn.functional as F

# Directories
image_dir = './dataset/image'
label_dir = './dataset/indexLabel'
downscale_factor = 4  # This can be adjusted

# Check if MPS is available
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# Load images and labels
def load_data(image_dir, label_dir):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

    images = []
    labels = []
    
    for img_file, lbl_file in zip(image_files, label_files):
        # Load image and label
        img = cv2.imread(os.path.join(image_dir, img_file))
        lbl = cv2.imread(os.path.join(label_dir, lbl_file), cv2.IMREAD_GRAYSCALE)

        # Downscale image and label
        img = cv2.resize(img, (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor), interpolation=cv2.INTER_AREA)
        lbl = cv2.resize(lbl, (lbl.shape[1] // downscale_factor, lbl.shape[0] // downscale_factor), interpolation=cv2.INTER_NEAREST)

        # Normalize image
        img = img.astype('float32') / 255.0
        
        # Append to lists
        images.append(img)
        labels.append(lbl)

    # Convert lists to arrays
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Display some images and their corresponding labels
def display_samples(images, labels, num_samples=2):
    for i in range(num_samples):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title('Image')
        
        plt.subplot(1, 2, 2)
        plt.imshow(labels[i], cmap='tab20')
        plt.title('Label')
        
        plt.show()

# Load data
images, labels = load_data(image_dir, label_dir)

# Check class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Display some samples from training data
print("Displaying samples from training data...")
display_samples(X_train, y_train)

print("Data is prepared for PyTorch.")

# Create a custom Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the U-Net model with Dropout layers
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(0.5),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(0.5),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc = self.encoder(x)
        bottleneck = self.bottleneck(enc)
        dec = self.decoder(bottleneck)
        return self.final(dec)

# Initialize the model weights
def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

num_classes = 19
model = UNet(num_classes)
model.apply(initialize_weights)
model = model.to(device)

# ACWE Loss (Active Contours Without Edges)
class ACWELoss(nn.Module):
    def __init__(self, lamda=1.0):
        super(ACWELoss, self).__init__()
        self.lamda = lamda

    def forward(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        labels = labels.long()  # Ensure labels are integers
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        # Compute c1 and c2
        c1 = torch.sum(preds * labels_one_hot, dim=(2, 3), keepdim=True) / (torch.sum(labels_one_hot, dim=(2, 3), keepdim=True) + 1e-6)
        c2 = torch.sum(preds * (1 - labels_one_hot), dim=(2, 3), keepdim=True) / (torch.sum(1 - labels_one_hot, dim=(2, 3), keepdim=True) + 1e-6)

        # Compute ACWE loss
        acwe_loss = self.lamda * (torch.sum((preds - c1) ** 2 * labels_one_hot, dim=(2, 3)) +
                                  torch.sum((preds - c2) ** 2 * (1 - labels_one_hot), dim=(2, 3)))
        
        # Extract boundaries
        boundaries = self.extract_boundaries(labels_one_hot)

        # Compute boundary weights
        boundary_weights = 1 + boundaries.sum(dim=1, keepdim=True)

        return torch.mean(acwe_loss), labels_one_hot, boundaries, boundary_weights

    def extract_boundaries(self, labels_one_hot):
        # Compute gradients
        grad_x = torch.abs(labels_one_hot[:, :, :, 1:] - labels_one_hot[:, :, :, :-1])
        grad_y = torch.abs(labels_one_hot[:, :, 1:, :] - labels_one_hot[:, :, :-1, :])

        # Pad the gradients to ensure the dimensions match
        grad_x = F.pad(grad_x, (0, 1, 0, 0))  # Pad right
        grad_y = F.pad(grad_y, (0, 0, 0, 1))  # Pad bottom

        boundaries = torch.clamp(grad_x + grad_y, 0, 1)
        return boundaries

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, preds, labels, boundary_weights):
        log_probs = torch.log_softmax(preds, dim=1)
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        loss = -labels_one_hot * log_probs
        loss = loss * boundary_weights
        return loss.sum(dim=(1, 2, 3)).mean()

def visualize_boundaries(inputs, boundaries, num_samples=1):
    for i in range(num_samples):
        input_img = np.transpose(inputs[i].cpu().numpy(), (1, 2, 0))
        boundary_img = boundaries[i].cpu().numpy()

        # Sum the boundary channels to get a single-channel image
        boundary_img_sum = np.sum(boundary_img, axis=0)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(input_img)
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        plt.imshow(boundary_img_sum, cmap='gray')
        plt.title('ACWE Boundaries')

        plt.show()

# Adjusted Dice Loss
def adjusted_dice_loss(preds, labels_one_hot, smooth=1e-6):
    preds = torch.softmax(preds, dim=1)
    intersection = (preds * labels_one_hot).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + labels_one_hot.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

# Adjusted mIoU Loss
def adjusted_miou_loss(preds, labels_one_hot, smooth=1e-6):
    preds = torch.argmax(preds, dim=1)
    labels = torch.argmax(labels_one_hot, dim=1)
    iou_list = []
    for cls in range(labels_one_hot.shape[1]):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou = (intersection + smooth) / (union + smooth)
        iou_list.append(iou)
    miou = torch.mean(torch.tensor(iou_list))
    return 1 - miou

# Combined Loss (Cross Entropy + Dice Loss + mIoU Loss + ACWE Loss)
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.custom_ce_loss = CustomCrossEntropyLoss()
        self.acwe_loss = ACWELoss(lamda=0.1)

    def forward(self, preds, labels):
        acwe, labels_one_hot, boundaries, boundary_weights = self.acwe_loss(preds, labels)
        ce_loss = self.custom_ce_loss(preds, labels, boundary_weights)
        dice = adjusted_dice_loss(preds, labels_one_hot)
        miou = adjusted_miou_loss(preds, labels_one_hot)
        print(f"CE Loss: {ce_loss.item()}, ACWE Loss: {acwe.item()}, Dice Loss: {dice.item()}, mIoU Loss: {miou.item()}")
        return ce_loss + dice + miou + acwe, boundaries

# IoU Metric
def iou_score(preds, labels, num_classes):
    preds = torch.argmax(preds, dim=1)
    iou_list = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_list.append(float('nan'))  # If there is no ground truth, do not include in average
        else:
            iou_list.append((intersection / union).item())
    return np.nanmean(iou_list)  # Return the mean IoU, ignoring NaN values

# Accuracy Metric
def accuracy(preds, labels):
    preds = torch.argmax(preds, dim=1)
    correct = (preds == labels).float().sum()
    total = labels.numel()
    return correct / total

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            print(f'Initial best loss: {val_loss}')
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'No improvement in val_loss: {val_loss}, counter: {self.counter}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print(f'Improved val_loss: {val_loss}')
            self.best_loss = val_loss
            self.counter = 0

# Save the U-Net model
def save_unet_model(model, path):
    torch.save(model.state_dict(), path)

# Load the weight matrix of the pre-trained U-Net model
def load_unet_weights(path):
    state_dict = torch.load(path, map_location=device)
    final_weight = state_dict['final.weight'].cpu().numpy()
    final_bias = state_dict['final.bias'].cpu().numpy()
    return final_weight, final_bias

# Convert weight matrix to SVD readable format
def prepare_svd(weight_matrix):
    original_shape = weight_matrix.shape
    
    # Reshape the weight tensor to 2D
    reshaped_weight = weight_matrix.reshape(original_shape[0], -1)
    
    # Perform SVD
    U, Sigma, Vt = np.linalg.svd(reshaped_weight, full_matrices=False)
    
    # Convert to torch tensors and make them trainable
    U = torch.tensor(U, dtype=torch.float32, requires_grad=True)
    Sigma = torch.tensor(Sigma, dtype=torch.float32, requires_grad=True)
    Vt = torch.tensor(Vt, dtype=torch.float32, requires_grad=True)
    
    return U, Sigma, Vt, original_shape

# Function to reconstruct the weight tensor from SVD components
def reconstruct_from_svd(U, Sigma, Vt, original_shape):
    # Reconstruct the weight tensor from SVD components
    Sigma_matrix = torch.diag(Sigma)
    reconstructed_weight_2d = torch.matmul(U, torch.matmul(Sigma_matrix, Vt))
    
    # Reshape the reconstructed weight tensor back to the original 4D shape
    reconstructed_weight = reconstructed_weight_2d.reshape(original_shape)
    
    return reconstructed_weight

# Input shape (height, width, channels)
input_shape = (378, 504, 3)
num_classes = 19

# Build the model
model = UNet(num_classes).to(device)

# Loss and optimizer
criterion = CombinedLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00015, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create DataLoaders
batch_size = 4  # Recommended batch size 
train_dataset = CustomDataset(X_train, y_train, transform=transform)
test_dataset = CustomDataset(X_test, y_test, transform=transform)

# Split the training data into training and validation sets (80% training, 20% validation)
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

train_dataset_split = CustomDataset(X_train_split, y_train_split, transform=transform)
val_dataset = CustomDataset(X_val, y_val, transform=transform)

train_loader_split = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Training the main model with validation and early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=200, patience=25):
    early_stopping = EarlyStopping(patience=patience)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # Ensure labels are integers
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss, boundaries = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_iou = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()  # Ensure labels are integers
                
                outputs = model(inputs)
                loss, boundaries = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, labels_one_hot, _, _ = ACWELoss()(outputs, labels)
                val_acc += accuracy(outputs, labels).item() * inputs.size(0)
                val_iou += iou_score(outputs, labels, num_classes) * inputs.size(0)
                val_dice += (1 - adjusted_dice_loss(outputs, labels_one_hot).item()) * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_iou /= len(val_loader.dataset)
        val_dice /= len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val mIoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}')
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Visualize boundaries for a few samples from validation set
        visualize_boundaries(inputs, boundaries, num_samples=1)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


# Train the main model with validation
train_model(model, train_loader_split, val_loader, criterion, optimizer, scheduler)

# Save the pre-trained U-Net model
pretrained_unet_path = 'pretrained_unet.pth'
save_unet_model(model, pretrained_unet_path)

# Evaluate the pre-trained U-Net model
def evaluate_model(model, test_loader, criterion, num_classes):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    test_iou = 0.0
    test_dice = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).long()  # Ensure labels are integers
            
            outputs = model(inputs)
            loss, boundaries = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            _, labels_one_hot, _, _ = ACWELoss()(outputs, labels)
            test_acc += accuracy(outputs, labels).item() * inputs.size(0)
            test_iou += iou_score(outputs, labels, num_classes) * inputs.size(0)
            test_dice += (1 - adjusted_dice_loss(outputs, labels_one_hot).item()) * inputs.size(0)
    
    test_loss /= len(test_loader.dataset)
    test_acc /= len(test_loader.dataset)
    test_iou /= len(test_loader.dataset)
    test_dice /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test mIoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}')

# Post-process the output to get class labels
def predict_and_display(model, data_loader, num_samples=1):
    model.eval()
    samples = iter(data_loader)
    with torch.no_grad():
        for i in range(num_samples):
            inputs, labels = next(samples)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(inputs.cpu().numpy()[0], (1, 2, 0)))
            plt.title('Input Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(torch.argmax(outputs[0], dim=0).cpu().numpy(), cmap='tab20')
            plt.title('Predicted Label')
            
            plt.subplot(1, 3, 3)
            plt.imshow(labels[0].cpu().numpy(), cmap='tab20')
            plt.title('True Label')
            
            plt.show()

# Evaluate the pre-trained U-Net model
print("Evaluating pre-trained U-Net model...")
evaluate_model(model, test_loader, criterion, num_classes)

# Display predictions for some test samples from pre-trained model
print("Displaying predictions for pre-trained U-Net model...")
predict_and_display(model, test_loader)
