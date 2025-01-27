import os, io, torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
Create model
'''
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # create the layers in the CNN model
        # Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max Pooling Layer: downsample by factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer 1 
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=128, out_features=4)

        # Added Dropout Layer
        self.dropout = nn.Dropout(p=0.3)    # 30% dropout rate

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolution + ReLU + pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten feature maps 
        x = x.view(x.size(0), -1)

        # Fully connected layers + ReLU
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # Apply dropout after fc1

        # Output layer
        x = self.fc2(x)

        return x

'''
Prepare data
'''
def load_filepaths_and_labels(target_dir):
    filepaths = []
    labels = []

    # get all the files in the directory
    files = os.listdir(target_dir)

    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):   # Filter valid image files
            filepaths.append(f"{target_dir}/{file}")

            # Extract label from filename (e.g., 'apple_1.jpg' -> 'apple')
            label = file.split('_')[0]
            labels.append(label)

    return np.array(filepaths), np.array(labels)

def prepare_data(target_dir):
    filepaths, labels = load_filepaths_and_labels(target_dir)

    # dictionary of mapping from label strings to integer values
    label_mapping = {
        'apple': 0,
        'banana': 1,
        'orange': 2,
        'mixed': 3
    }

    # Convert string labels to numeric labels using the mapping
    numeric_labels = [label_mapping[label] for label in labels]

    return filepaths, torch.tensor(numeric_labels)

def load_images(filepaths, is_train=True):
    if is_train:
        # Instantiate class to resize and transform image to tensor (Data Augmentations)
        to_tensor = transforms.Compose([
            transforms.Resize((64, 64)),        # Resize images to 64x64
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images horizontally
            transforms.RandomVerticalFlip(p=0.5),   # Randomly flip images vertically
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Adjust brightness, contrast, etc.
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),    # Apply slight blur to images
            transforms.ToTensor(),          # Convert image to tensor
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), value=0),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        to_tensor = transforms.Compose([
            transforms.Resize((64, 64)),        # Resize images to 64x64
            transforms.ToTensor(),              # Convert image to tensor
        ])

    tensors = []

    # List all files in directory
    for item in filepaths:
        try:
            with open(item, 'rb') as f:
                bin_data = f.read() # read raw bytes of image (compressed in JPEG)
                image = Image.open(io.BytesIO(bin_data))    # convert into a image
                image = image.convert("RGB")    # convert to RGB format for consistency
                img_tensor = to_tensor(image)   # convert into pytorch's tensor 
                tensors.append(img_tensor)
        except Exception as e:
            print(f"Skipping file {item}: {e}")

    if tensors:
        return torch.stack(tensors)
    else:
       raise RuntimeError("No valid images loaded. Check the filepaths or image formats.")
    
'''
Plot training metrics
'''
def plot(losses, accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot training losses
    axes[0].plot(range(1, len(losses) + 1), losses, label='Training Loss', color='red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')

    # Plot training accuracy
    axes[1].plot(range(1, len(accuracies) + 1), accuracies, label='Training Accuracy', color='green')
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


'''
Train model
'''
def train(model, criterion, optimizer, filepaths, labels):
    # hyper-parameters for training
    n_epochs = 45
    # number of samples in each iteration
    batch_size = 32

    # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []

    # Set model to training mode
    model.train()

    for epoch in range(n_epochs):
        # For tracking and printing training progress
        samples_trained = 0
        run_loss = 0
        correct_preds = 0
        total_samples = len(filepaths)

        # for shuffling of the samples/data
        # returns a random permutation of intergers from 0 to n-1
        permutation = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42))
        for i in range(0, total_samples, batch_size):
            indices = permutation[i : i + batch_size]
            batch_inputs = load_images(filepaths[indices], is_train=True)
            batch_labels = labels[indices]

            # Forward pass: compute predicted outputs
            outputs = model(batch_inputs)

            # Compute loss
            loss = criterion(outputs, batch_labels)
            run_loss += loss.item()

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Get probability distribution
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            # Update statistics
            samples_trained += len(indices)
            correct_preds += torch.sum(preds == batch_labels) # compare pred with label

        # Calculate epoch metrics
        avg_loss = run_loss / samples_trained
        accuracy = correct_preds / float(samples_trained) # cast to float to get in decimal

        # Append metrics for plotting
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy.item())

        print(f"Epoch {epoch + 1}/{n_epochs}: " +
            f"Loss={avg_loss:.5f}, Accuracy={accuracy:.5f}")
    
    plot(train_losses, train_accuracies)
     
'''
Test the model
'''
def test(model, filepaths, labels):
    batch_size = 10
    samples_tested = 0
    correct_preds = 0
    total_samples = len(filepaths)

    # To store preds and true labels for all batches
    all_preds = []
    all_labels = []

    # Set model to evaluation mode
    model.eval()    # Disables training behaviors like dropout

    for i in range(0, total_samples, batch_size):
        batch_inputs = load_images(filepaths[i : i + batch_size], is_train=False)
        batch_labels = labels[i : i + batch_size]

        # Forward pass: compute predicted outputs
        outputs = model(batch_inputs)

        # Get probability-distributions
        probs = torch.softmax(outputs, dim=1)
        max_probs, preds = torch.max(probs, dim=1)

        # Accumulate predictions and labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

        # Determine accuracy
        samples_tested += len(batch_labels)
        correct_preds += torch.sum(preds.eq(batch_labels))
        accuracy = correct_preds / float(samples_tested)

        print(f"({samples_tested}/{total_samples}): Accuracy={accuracy:.5f}")

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['apple', 'banana', 'orange', 'mixed'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

'''
Distribution Plot for Data
'''
def plot_data_distribution(train_labels, test_labels):
    # Convert tensors to numpy arrays
    train_labels = train_labels.numpy()
    test_labels = test_labels.numpy()

    # Count occurrences of each class
    train_counts = np.bincount(train_labels)
    test_counts = np.bincount(test_labels)

    class_names = ['Apple', 'Banana', 'Orange', 'Mixed']

    data = {
        'Class': class_names * 2,
        'Count': np.concatenate([train_counts, test_counts]),
        'Dataset': ['Train'] * len(class_names) + ['Test'] * len(class_names)
    }

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Count', hue='Dataset', data=data)
    plt.title('Distribution of Classes in Training and Testing Datasets')
    plt.xlabel('Fruit Class')
    plt.ylabel('Count')
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.show()

'''
Main Program
'''
if __name__ == "__main__":
    # Instantiate the model
    model = SimpleCNN()
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Adj lr from 0.01 to 0.001, added weight_decay

    # Train the model
    dir_train = '/Users/JIAMIN./Desktop/GDipSA59/SA4110_Machine Learning Application Development/10_CNN/02_Assignment/Team03/train/'
    train_filepaths, train_labels = prepare_data(dir_train)
    train(model, criterion, optimizer, train_filepaths, train_labels)

    # Test the model
    dir_test = '/Users/JIAMIN./Desktop/GDipSA59/SA4110_Machine Learning Application Development/10_CNN/02_Assignment/Team03/test/'
    test_filepaths, test_labels = prepare_data(dir_test)
    test(model, test_filepaths, test_labels)

    # Save trained model
    torch.save(model.state_dict(), "cnn_fruits_model.pth")

    # Plot distribution of dataset
    # plot_data_distribution(train_labels, test_labels)                                             