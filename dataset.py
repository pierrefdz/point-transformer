# ModelNet40 dataloader
# source: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master/data_utils

import os
import random
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm


def pc_normalize(pc):
    """
    Normalize the point cloud
    Input:
        pc: pointcloud data, [N, D]
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(pc, n_sample):
    """
    Input:
        pc: pointcloud data, [N, D]
        n_sample: number of samples
    Return:
        centroids: sampled pointcloud index, [n_sample, D]
    """
    N, D = pc.shape
    xyz = pc[:, :3]
    centroids = np.zeros((n_sample,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_sample):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    pc = pc[centroids.astype(np.int32)]
    return pc


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * np.pi
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta),      0],
                               [np.sin(theta),  np.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud


class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud


class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud


def default_transforms():
    return transforms.Compose([RandomRotation_z(), RandomNoise()])


class ModelNetDataLoader(Dataset):
    def __init__(self, root, num_point=1024, transforms=default_transforms(), use_uniform_sample=True, use_normals=True,
                 num_category=40, split='train', process_data=False):
        self.root = root
        self.n_sample = num_point
        self.process_data = process_data
        self.uniform = use_uniform_sample
        self.use_normals = use_normals
        self.num_category = num_category
        self.transforms = transforms

        if self.num_category == 10:
            self.catfile = os.path.join(
                self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(
                self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(
                os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (
                self.num_category, split, self.n_sample))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (
                self.num_category, split, self.n_sample))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' %
                      self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath), position=0, leave=True):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(
                        fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(
                            point_set, self.n_sample)
                    else:
                        point_set = point_set[0:self.n_sample, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.n_sample)
            else:
                point_set = point_set[0:self.n_sample, :]

        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.transforms is not None:
            point_set[:, 0:3] = self.transforms(point_set[:, 0:3])

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    data = ModelNetDataLoader('data/modelnet40_normal_resampled/', 
                          num_category=40,
                          split='test', 
                          process_data=False)
    DataLoader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)