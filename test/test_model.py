import os
from model import build_model, load_model, evaluate_model, identify
from pasanet_nn import PasaNet
from PIL import Image

TEST_DIR = os.path.dirname(os.path.abspath(__file__))  # Mark the test root directory
TRAINING_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "training_images")
TEST_IMAGE_TEST_DIR = os.path.join(TEST_DIR, "data_sets", "test_images")


class TestModel:
    # <ASSIGNMENT: Test the build_model(), load_model() and evaluate_model() functions in model.py. You can use the
    # images under the test\data_sets\ directories for unit testing. You don't need to test train_model().>
    def test_build_model(self):
        """ Test the build_model() function in model.py. """
        model = build_model()
        assert type(model) == PasaNet
    
    def test_load_model(self, barebone_model):
        """ Test the load_model() function in model.py. """
        train_model = load_model(barebone_model, r'PasaNet')
        assert type(train_model) == PasaNet

    def test_evaluate_model(self, trained_model):
        """ Test the evaluate_model() function in model.py. """
        score = evaluate_model(trained_model, TEST_IMAGE_TEST_DIR)
        assert 0.0 <= score <= 100.0
    
    def test_identify(self, trained_model):
        """ Test the identify() function in model.py. """
        # import test images
        test_image_A = Image.open(os.path.join(TRAINING_IMAGE_TEST_DIR, "A_1289.png"))
        test_image_J = Image.open(os.path.join(TRAINING_IMAGE_TEST_DIR, "J_1022.png"))
        test_image_K = Image.open(os.path.join(TRAINING_IMAGE_TEST_DIR, "K_6982.png"))
        test_image_Q = Image.open(os.path.join(TRAINING_IMAGE_TEST_DIR, "Q_1846.png"))

        # do inference and compare to expected results
        assert identify(test_image_A, trained_model) == 'A'
        assert identify(test_image_J, trained_model) == 'J'
        assert identify(test_image_K, trained_model) == 'K'
        assert identify(test_image_Q, trained_model) == 'Q'
