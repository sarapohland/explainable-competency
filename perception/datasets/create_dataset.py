import os
import torch
import shutil
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import _pickle as cPickle

from natsort import natsorted

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_clusters(data_path, clusters_path, N=10):
    # Create feature extractor from pretrained CNN
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = shufflenet_v2_x1_0(weights=weights)
    return_nodes = {'mean': 'mean'}
    # weights = ResNet50_Weights.DEFAULT
    # model = resnet50(weights=weights)
    # return_nodes = {'flatten': 'flatten'}
    preprocess = weights.transforms()
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Extract image features
    data = {}
    for file in tqdm(os.listdir(data_path)):
        if file.endswith('.jpg') or file.endswith('.png'):
            img = Image.open(os.path.join(data_path, file))
            img = np.array(img)
            img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
            img = np.expand_dims(img, axis=0)
            img = preprocess(torch.from_numpy(img))
            features = feature_extractor(img)['mean']
            # features = feature_extractor(img)['flatten']
            data[file] = features.detach().numpy().flatten()

    # Perform PCA
    print('Performing PCA...')
    pca = PCA(n_components=0.99, random_state=22)
    X = pca.fit_transform(np.array(list(data.values())))

    # Cluster feature vectors
    print('Clustering images...')
    kmeans = KMeans(n_clusters=N, random_state=22)
    kmeans.fit(X)

    # Save clusters
    for file, cluster in tqdm(zip(list(data.keys()), kmeans.labels_)):
        cluster_path = os.path.join(clusters_path, str(cluster))
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        shutil.copyfile(os.path.join(data_path, file), os.path.join(cluster_path, file))

def save_dataset(id_data_dir, ood_data_dir, save_dir, height=None, width=None):
    # Collect in-distribution data
    data, labels = [], []
    _, cluster_names, _ = list(os.walk(id_data_dir))[0]
    cluster_names = natsorted(cluster_names)
    num_id_classes = len(cluster_names)
    for idx, cluster_name in enumerate(cluster_names):
        cluster_dir = os.path.join(id_data_dir, cluster_name)
        for file in os.listdir(cluster_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = Image.open(os.path.join(cluster_dir, file))
                height = height if height is not None else img.size[1]
                width  = width  if width  is not None else img.size[0]
                img = img.resize((width, height))
                img = np.array(img)
                img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
                img = np.expand_dims(img, axis=0)
                data.append(img)
                labels.append(idx)
    data = np.vstack(data)
    labels = np.vstack(labels)

    # Separate ID data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.8, random_state=22, shuffle=True)

    # Collect out-of-distribution data
    data, labels = [], []
    _, cluster_names, _ = list(os.walk(ood_data_dir))[0]
    cluster_names = natsorted(cluster_names)
    for idx, cluster_name in enumerate(cluster_names):
        cluster_dir = os.path.join(ood_data_dir, cluster_name)
        for file in os.listdir(cluster_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = Image.open(os.path.join(cluster_dir, file))
                height = height if height is not None else img.size[1]
                width  = width  if width  is not None else img.size[0]
                img = img.resize((width, height))
                img = np.array(img)
                img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])
                img = np.expand_dims(img, axis=0)
                data.append(img)
                labels.append(num_id_classes + idx)
    X_ood = np.vstack(data)
    y_ood = np.vstack(labels)

    # Convert integer array to floats
    X_train = X_train / 255
    X_test  = X_test  / 255
    X_ood   = X_ood   / 255

    X_train = np.float16(X_train)
    X_test = np.float16(X_test)
    X_ood = np.float16(X_ood)

    y_train = np.int16(y_train)
    y_test = np.int16(y_test)
    y_ood = np.int16(y_ood)

    # Save all data to pickle file
    train = {'data': X_train, 'labels': y_train}
    test  = {'data': X_test,  'labels': y_test}
    ood   = {'data': X_ood,   'labels': y_ood}

    dataset = {'train': train, 'test': test, 'ood': ood}
    file = os.path.join(save_dir, 'dataset.p')
    pickle.dump(dataset, open(file, 'wb'))


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('data_path', type=str)
    parser.add_argument('--cluster_data', action='store_true', default=False)
    parser.add_argument('--save_data', action='store_true', default=False)
    parser.add_argument('--num_clusters', type=int, default=10)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    args = parser.parse_args()

    clusters_path = os.path.join(args.data_path, 'images/ID/')
    id_data_path  = os.path.join(args.data_path, 'images/unsorted/')
    ood_data_path = os.path.join(args.data_path, 'images/OOD/')

    if args.cluster_data:
        if not os.path.exists(clusters_path):
            os.makedirs(clusters_path)
        create_clusters(id_data_path, clusters_path, args.num_clusters)

    if args.save_data:
        save_dataset(clusters_path, ood_data_path, args.data_path, height=args.height, width=args.width)
    

if __name__ == "__main__":
    main()