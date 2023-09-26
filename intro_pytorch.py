import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ])

    train_set = datasets.FashionMNIST("./data", train = True, download = True, transform = custom_transform)
    test_set = datasets.FashionMNIST("./data", train = False, transform = custom_transform)

    if(training):
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle = False)

    return loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(784, 128),
                          nn.ReLU(),
                          nn.Linear(128, 64),
                          nn.ReLU(),
                          nn.Linear(64, 10))

    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    model.train()

    for epoch in range(T):

        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item() * inputs.size(0)

        accuracy = round(100 * correct / total, 2)
        epoch_loss = round(running_loss / total, 3)
        print(f"Train Epoch: {epoch} Accuracy: {correct}/{total}({accuracy}%) Loss: {epoch_loss}")

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy

    RETURNS:
        None
    """
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():

        running_loss = 0.0
        for inputs, labels in test_loader:

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

        if(show_loss):
            print(f"Average loss: {round(running_loss / total, 4)}")
        print(f"Accuracy: {round(100 * correct / total, 2)}%")

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

    logits = model(test_images[index])
    prob = F.softmax(logits, dim = 1)[0]

    class_prob = dict()
    for i, value in enumerate(prob):
         class_prob[class_names[i]] = round(100 * value.item(), 2)

    sorted_class = sorted(class_prob.items(), key = lambda x:x[1], reverse = True)
    class_prob = dict(sorted_class)

    for i in range(3):
        class_name = sorted_class[i][0]
        print(f"{class_name}: {class_prob[class_name]}%")

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''

