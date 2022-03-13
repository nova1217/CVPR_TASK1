url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" #link of the dataset
download_dir = "./data"
download.maybe_download_and_extract(url,download_dir)

cifar10_dir = './data/cifar-10-batches-py' #dataset direction
X_train, y_train, X_test, y_test = data_utils.load_CIFAR10(cifar10_dir) #reading training and testing data

# Checking the size of the training and testing data
print('Training dataset shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Test dataset shape:', X_test.shape)
print('Test labels shape:', y_test.shape)

Training dataset shape: (50000, 32, 32, 3)
Training labels shape: (50000,)
Test dataset shape: (10000, 32, 32, 3)
Test labels shape: (10000,)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] #classes of the data
num_classes = len(classes) #number of class
samples_per_class = 5
#visualizing 5 sample of each class
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(y_train == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#taking 50000 training data and 1,000 test data
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 1000
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]

# reshaping data and placing into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


# KNN class
class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):  # traning the data
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):  # predicting class
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Value is invalid %d for num_loops' % num_loops)
        return self.predict_labels(dists, k=k)

    def compute_distances(self, X):  # calculating distances of the data
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(
            np.sum(np.square(self.X_train), axis=1) + np.sum(np.square(X), axis=1)[:, np.newaxis] - 2 * np.dot(X,
                                                                                                               self.X_train.T))
        pass
        return dists

    def predict_labels(self, dists, k=1):  # predicting class label according to the shape
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.y_train[sorted_dist[0:k]])
            pass
            y_pred[i] = (np.argmax(np.bincount(closest_y)))
            pass
        return y_pred