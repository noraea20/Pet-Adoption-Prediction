from sklearn import preprocessing
import keras
import numpy as np


def preprocess_image(data, path_to_images='Images'):
    # Split the images into training and testing sets following 80/20 partition
    train_X, test_X, train_y, test_Y = split(data)
    trainPet_ID = train_X['PetID'].to_numpy()
    testPet_ID = test_X['PetID'].to_numpy()

    # Load and resize images for training, then saving them into numpy array and return it from the method
    trainImages = np.zeros(shape=(len(trainPet_ID), 32, 32, 3))  # 3????

    counter = 0
    for i in trainPet_ID:
        image = preprocessing.images.load_img(path_to_images + '/' + str(i) + "-1.jpg")
    image = images.resize(32, 32)
    trainImages[counter] = image

    # Load and resize images for testing, then saving them into numpy array and return it from the method
    testImages = np.zeros(shape=(len(testPet_ID), 32, 32, 3))  # 3????

    counter = 0
    for i in testPet_ID:
        image = preprocessing.images.load_img(path_to_images + '/' + str(i) + "-1.jpg")
    image = images.resize(32, 32)
    testImages[counter] = image

    return trainImages, testImages




path_to_data = "dataset.csv"
images_path="images/"
data = pd.read_csv(path_to_data)


# Data visualization
visualize (path_to_data)

# Split the dataset into training and testing sets following 80/20 partition
train_X, test_X, train_y, test_Y = split(data)

# Preprocess categorical and continuous data
pr_train_X, pr_test_x,pr_train_y, pr_test_y = preprocess_data(train_X, test_X, train_y, test_Y)

# Preprocess images
train_images, test_images = preprocess_image(train_X, test_X, images_path)

# Train MLP model
mlp_model = create_mlp(pr_train_X)


