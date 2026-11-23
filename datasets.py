import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    def __init__(self, data_name, data_X, data_Y):
        super(MultiViewDataset, self).__init__()
        self.data_name = data_name

        self.X = dict()
        self.num_views = data_X.shape[0]
        for v in range(self.num_views):
            self.X[v] = self.normalize(data_X[v])

        self.Y = data_Y
        self.Y = np.squeeze(self.Y)
        if np.min(self.Y) == 1:
            self.Y = self.Y - 1
        self.Y = self.Y.astype(dtype=np.int64)
        self.num_classes = len(np.unique(self.Y))
        self.dims = self.get_dims()

    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)
        target = self.Y[index]
        return data, target, index

    def __len__(self):
        return len(self.X[0])

    def get_dims(self):
        dims = []
        for view in range(self.num_views):
            dims.append([self.X[view].shape[1]])
        return np.array(dims)

    @staticmethod
    def normalize(x, min=0):
        if min == 0:
            scaler = MinMaxScaler((0, 1))
        else:
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def postprocessing(self, index, addNoise=False, sigma=0, ratio_noise=0.5, addConflict=False, ratio_conflict=0.5):
        if addNoise:
            self.addNoise(index, ratio_noise, sigma=sigma)
        if addConflict:
            self.addConflict(index, ratio_conflict)
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            views = np.random.choice(np.array(self.num_views), size=np.random.randint(self.num_views), replace=False)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        pass

    def addConflict(self, index, ratio):
        records = dict()
        for c in range(self.num_classes):
            i = np.where(self.Y == c)[0][0]
            temp = dict()
            for v in range(self.num_views):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.num_views)
            self.X[v][i] = records[(self.Y[i] + 1) % self.num_classes][v]
        pass


def HandWritten():
    data_path = "data/handwritten.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("HandWritten", data_X, data_Y)

def Scene():
    data_path = "data/scene15_mtv.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("Scene", data_X, data_Y)

def PIE():
    data_path = "data/PIE_face_10.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("PIE", data_X, data_Y)

def CUB():
    data_path = "data/CUB.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['gt']
    for v in range(len(data_X)):
        data_X[v] = data_X[v].T
    return MultiViewDataset("CUB", data_X, data_Y)

def LandUse():
    data_path = "data/LandUse_21.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("LandUse", data_X, data_Y)

def NUS_OBJ():
    data_path = "data/NUSWIDEOBJ.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("NUS-OBJ", data_X, data_Y)

def Leaves():
    data_path = "data/100Leaves.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("Leaves", data_X, data_Y)

def CCV():
    data_path = "data/CCV.mat"
    data = sio.loadmat(data_path)
    data_X = data['X'][0]
    data_Y = data['Y']
    return MultiViewDataset("CCV", data_X, data_Y)

def Caltech():
    data_path = "data/Caltech-5V.mat"
    data = sio.loadmat(data_path)
    keys = sorted([k for k in data.keys() if k.startswith('X') and k != 'X'],
                  key=lambda s: int(s[1:]))
    data_X = np.empty(len(keys), dtype=object)
    for i, k in enumerate(keys):
        data_X[i] = data[k]
    data_Y = data['Y']
    return MultiViewDataset("Caltech", data_X, data_Y)

def Fashion():
    data_path = "data/3V_Fashion_MV.mat"
    data = sio.loadmat(data_path)
    data_X = np.empty(3, dtype=object)
    for v in range(3):
        x = data[f'X{v+1}']
        data_X[v] = x.reshape(x.shape[0], -1)
    data_Y = data['Y']
    return MultiViewDataset("Fashion", data_X, data_Y)

def YoutubeFace_sel():
    import mat73
    data = mat73.loadmat("data/YoutubeFace_sel.mat")
    X = data['X']
    data_X = np.empty(len(X), dtype=object)
    for v in range(len(X)):
        data_X[v] = np.array(X[v])
    data_Y = np.array(data['Y'])
    return MultiViewDataset("YoutubeFace_sel", data_X, data_Y)

def AWA():
    import mat73
    data = mat73.loadmat("data/AwAfea.mat")
    X = data['X']
    data_X = np.empty(len(X), dtype=object)
    for v in range(len(X)):
        data_X[v] = np.array(X[v])
    data_Y = np.array(data['Y'])
    return MultiViewDataset("AWA", data_X, data_Y)
