import torch
import torch.nn.functional as F
import os
import argparse

from model import build_model, train_model, evaluate_model
from data_sets import TEST_IMAGE_DIR

parser = argparse.ArgumentParser(description='Train and evaluate model')
parser.add_argument('-v', dest='n_val', help='number of validation samples', default=2000, type=int)
parser.add_argument('-s', dest='save_model', help='decide if model should be saved or not', default=False, type=bool)

args = parser.parse_args()
write_to_file = args.save_model
n_validation = args.n_val

def main():
    """
    Main function.
    """
    # supress wandb notifications
    os.environ["WANDB_SILENT"] = "true"

    # build the model
    model = build_model()
    
    # train the model
    train_model(model, n_validation=n_validation, write_to_file=write_to_file)

    # evaluate the model
    score = evaluate_model(model, TEST_IMAGE_DIR)

    print(f'Final accuracy on test set: {score}')


# if the file is run as a script, run the main function
if __name__ == '__main__':
    main()
