# <ASSIGNMENT: Define and interact with your model. Motivate your choices in the docstrings and comments. This file
# contains a suggested structure; you are free to define your own structure, adjust function arguments etc. Don't forget
# to write appropriate tests for your functionality.>

import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from PIL import Image
from pasanet_nn import PasaNet
from data_sets import *
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Current file marks the root directory
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")  # Directory for loading training images
TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")  # Directory for loading test images

# if you want to work in wandb you should first create an account if you don't have one and then login using 'wandb login' on the command line, if you want to skip the log step comment al the wandb stuff out.

dtype = torch.float32 #data type that will be used

if torch.cuda.is_available(): #if someone is rich and can still afford a GPU in these times select gpu as device since this will shorten the training time
    device = torch.device('cuda')
else:                         #but if you are poor like me just train on your cpu
    device = torch.device('cpu')

print("Using device:", device)

def build_model():
    """
    Build a model.

    Returns
    -------
    model : model class
        Model structure to fit, as defined by build_model().
    """
    model = PasaNet()
    model.to(device)
    return model

def accuracy(loader, model, train): # function to measure accuracy
    """
    Measure the accuracy of the model.

    Arguments
    ---------
    loader : DataLoader
        Pytorch DataLoader that contains the data
    model : model class
        Model structure to fit, as defined by build_model().
    train : bool
        Notes if evaluated on train or test set

    Returns
    -------
    model : model class
        The trained model.
    """
    num_correct = 0
    num_samples = 0
    model.eval()  # evaluation mode
    with torch.no_grad(): #since we're evaluating we don't need to calculate the gradient and we can save some memory
        for val_tensor_x, val_tensor_y in loader:
            x = val_tensor_x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = val_tensor_y.to(device=device, dtype=torch.long)
            scores = model(x)

            # Use crossentropy as loss function
            loss = F.cross_entropy(scores, y)

            # Log loss in wandb
            if train:
                wandb.log({"val loss": loss})
            else:
                wandb.log({"test loss": loss})

            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = 100 * float(num_correct) / num_samples
        # log train acc in wandb
        if train:
            wandb.log({"accuracy": acc})
        else:
            wandb.log({"test accuracy": acc})
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))
        return acc


def train_model(model, n_validation, write_to_file=False):
    """
    Fit the model on the training data set.

    Arguments
    ---------
    model : model class
        Model structure to fit, as defined by build_model().
    n_validation : int
        Number of training examples used for cross-validation.
    write_to_file : bool
        Write model to file; can later be loaded through load_model().

    Returns
    -------
    model : model class
        The trained model.
    """
    # initialize wandb project, its recommended that you change the name of the run aftter each run to keep everything organized in the wandb dashboard
    wandb.init(project="pokerbot", entity="hakdemir", name='train_run: final') 

    #extract train and val data
    training_features, training_labels, validation_features, validation_labels = load_data_set(TRAINING_IMAGE_DIR, n_validation)
    
    #set some hyperparameters
    epochs = 10
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    batch_size = 64

    #convert train and val data into dataloaders
    train_tensor_x = torch.Tensor(np.array(training_features)).unsqueeze(1)
    train_tensor_y = torch.Tensor(training_labels)
    train_data_set = TensorDataset(train_tensor_x, train_tensor_y)
    loader_train = DataLoader(train_data_set, batch_size=batch_size)

    val_tensor_x = torch.Tensor(np.array(validation_features)).unsqueeze(1)
    val_tensor_y = torch.Tensor(validation_labels)
    val_data_set = TensorDataset(val_tensor_x, val_tensor_y)
    loader_val = DataLoader(val_data_set, batch_size=batch_size)

    # used for evaluation
    train=True

    # setup wandb parameters
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size
    }

    # watch the model in wandb
    wandb.watch(model, log="all")
    
    for e in range(epochs):
        for t, (train_tensor_x, train_tensor_y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = train_tensor_x.to(device=device, dtype=dtype)  
            y = train_tensor_y.to(device=device, dtype=torch.long)

            scores = model(x)
            # Use crossentropy as loss function
            loss = F.cross_entropy(scores, y)

            # Log loss in wandb
            wandb.log({"train loss": loss})

            # Zero out all of the gradients
            optimizer.zero_grad()

            # Do the backward pass
            loss.backward()

            # Update parameters using gradients calculated in backward pass
            optimizer.step()


            # Every 100 iterations report back and check on validation set
            if t % 100 == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                accuracy(loader_val, model, train)
                print()
    
    if write_to_file==True:
        #if true save the model
        torch.save(model.state_dict(), r'PasaNet')

    wandb.finish()   


def load_model(model, filename):
    """ Load the trained model.

    Args:
        model (Model class): Untrained model to load.
        filename (str): Name of the file to load the model from.

    Returns:
        Model: Model with parameters loaded from file.
    """
    model.load_state_dict(torch.load(filename))
    return model
    

def evaluate_model(model, path_to_data):
    """
    Evaluate model on the test set.

    Arguments
    ---------
    model : model class
        Trained model.

    Returns
    -------
    score : float
        A measure of model performance.
    """

    wandb.init(project="pokerbot", entity="hakdemir", name='test_run: final') 

    test_features, test_labels, _, _ = load_data_set(path_to_data) # load test data

    #prepare testdata into a DataLoader
    test_tensor_x = torch.Tensor(np.array(test_features)).unsqueeze(1) 
    test_tensor_y = torch.Tensor(test_labels)
    test_data_set = TensorDataset(test_tensor_x, test_tensor_y)
    loader_test = DataLoader(test_data_set)

    # used for evaluation
    score = accuracy(loader_test, model, train = False)
    wandb.finish()

    return score

def identify(image, model):
    """
    Use model to classify a single card image.

    Arguments
    ---------
    image : Image
        Image to classify.
    model : model class
        Trained model.

    Returns
    -------
    rank : str in ['J', 'Q', 'K', 'A'] 
        Estimated card rank.
    """
    pos_ranks = ['A', 'J', 'K', 'Q'] 

    # convert the image into features
    features = extract_features(image) 
    torch_tensor = model(torch.Tensor(features).unsqueeze(0).unsqueeze(0)) # perform inference
    np_arr = torch_tensor.cpu().detach().numpy() # convert the tensor into a numpy array

    # return the rank corresponding to the highest probability
    return pos_ranks[np.argmax(np_arr)]
    
    


