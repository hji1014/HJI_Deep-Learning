import keras
import numpy as np

# Datasets
ABS_Path = '/media/nelab/Hyun/posner/초기데이터_pipeline/ft_pipeline/Analysis_EEG/dl_input_dataset/'

annot_file_path = ABS_Path + '/ERP_source_whole_trials_12/mark.mat'
root_path = ABS_Path + 'ERP_source_whole_trials_12'


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, annot_file, root_path, batch_size, dim,
                 n_classes, shuffle=True, list_idx=None):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.root_dir = root_path
        self.marks = scio.loadmat(annot_file)['R_G_all']
        # self.labels = labels
        # self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indices = list_idx
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [(k, os.path.join(self.root_dir, self.marks[k, 0][0])) for k in indexes]

        # Generate data
        data_, labels_ = self.__data_generation(list_IDs_temp)
        # sample = {'data': data, 'label': labels}
        data = tf.convert_to_tensor(data_)
        labels = tf.convert_to_tensor(labels_)
        # labels = tf.keras.utils.to_categorical(y, num_classes=2)

        # return sample
        return (data, labels)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.indices is None:
            self.indexes = np.arange(len(self.marks))
        else:
            self.indexes = self.indices
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        # X = np.empty((self.batch_size, self.dim))
        y = np.empty((self.batch_size), dtype='float')

        # Generate data
        for i, (k, ID) in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = scio.loadmat(ID)['X_all'][:, :, :, np.newaxis]  # time_X_Y_channel

            # X[i,] = scio.loadmat(ID)['X_all'].reshape(-1) # for MLP test
            # Store class
            y[i,] = np.array(self.marks[k, 1][0] == 2, dtype='float')  # 0: CON, 1: RBD (annots: 1: CON, 2: RBD)

        return X, y
