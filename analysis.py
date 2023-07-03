import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import plotly.express as px
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
import copy
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import gzip


def load_data(folders):
    '''
    Load the data from the folders

    Parameters
    ----------
    folders : str or list of str
        The path(s) to the folder containing the data


    Returns
    -------
    df : pandas dataframe
        The dataframe containing the data
    '''

    df = pd.DataFrame()

    if type(folders) == str:
        folders = [folders]

    else:
        for folder in folders:
            # Depth search the folder and load the first 10 .parquet file in each folder
            for root, dirs, files in os.walk(folder):
                if file.endswith('.parquet'):
                    df = pd.concat(
                        [df, pd.read_parquet(os.path.join(root, file))])

    return df


def load_subset_data(liste):
    '''
    Load the data from the folders

    Parameters
    ----------
    liste : list of str
        The class of the alert to load


    Returns
    -------
    df : pandas dataframe
        The dataframe containing the data
    '''

    # define df as an empty dataframe
    df = pd.DataFrame()

    dossier = '/home/centos/data/data_march'

    for nom_fichier in liste:
        chemin = os.path.join(dossier, "finkclass=" + nom_fichier)

        temp = pd.read_parquet([chemin + "/" + os.listdir(chemin)[i]
                               for i in range(min(10, len(os.listdir(chemin))))])

        # concatenate the dataframes
        df = pd.concat([df, temp], ignore_index=True)

    kilonova = pd.read_parquet(
        '/home/centos/data/balanced_data/finkclass=Kilonova candidate')
    ambiguous = pd.read_parquet(
        '/home/centos/data/balanced_data/finkclass=Ambiguous')
    SN = pd.read_parquet(
        '/home/centos/data/balanced_data/finkclass=SN candidate')

    # Add a column 'finkclass' with the value Kilonova candidate
    kilonova['finkclass'] = 'Kilonova candidate'
    ambiguous['finkclass'] = 'Ambiguous'
    SN['finkclass'] = 'SN candidate'
    df = pd.concat([df, kilonova, ambiguous, SN], ignore_index=True)

    return df


def define_meta_class(ele, dic):
    '''
    Define the meta class of an element

    Parameters
    ----------
    ele : str
        The element to define

    dic : dict
        The dictionary containing the meta classes

    Returns
    -------
    ele : str
        The meta class of the element
    '''
    for key in dic.keys():
        if ele in dic[key]:
            ele = key

    return ele


def color_map(name_meta):
    '''
    Create a color map for the meta classes

    Parameters
    ----------
    name_meta : array
        The array containing the meta classes

    Returns
    -------
    colors : dict
        The dictionary containing the colors of the meta classes
    '''
    dic = {}

    # create a list with len(name) different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(name_meta)))

    for k in range(len(colors)):
        dic[name_meta[k]] = colors[k]

    return dic


def feature_choice(df, columns):
    '''
    Process the data by taking only the column that are numerical

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the data

    columns : list of: str or list [name, table]
        The columns to keep

    Returns
    -------
    df : pandas dataframe
        The dataframe containing the data
    '''
    cols = []

    to_concat = []

    for col in columns:
        if type(col) == str:
            if col == 'lc_features_g':
                to_concat.append(pd.DataFrame(
                    df['lc_features_g'].tolist()).add_suffix('_g'))

            elif col == 'lc_features_r':
                to_concat.append(pd.DataFrame(
                    df['lc_features_r'].tolist()).add_suffix('_r'))

            else:
                cols.append(col)

        elif type(col) == list:
            name, tab = col[0], col[1]
            to_concat.append(pd.DataFrame(df[name].tolist(), columns=tab))

        else:
            raise TypeError('The type of the column is not valid')

    to_concat.append(df[cols])

    return pd.concat(to_concat, axis=1)


def normalize_data(df):
    '''
    Normalize the data with by susbtracting the mean and dividing by the standard deviation

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the data

    Returns
    -------
    df : pandas dataframe
        The dataframe containing the data
    '''
    # in the column isdiffpos, replace t by 1 and f by -1
    if 'isdiffpos' in df.columns:
        df['isdiffpos'] = df['isdiffpos'].replace({'t': 1, 'f': -1})

    # Normalizing data
    df = (df - df.mean()) / df.std()

    # replace None or NaN by 0
    df = df.fillna(0)

    return df


def keep_important_variables(df, n_components=20, threshold=0.5):
    '''
    Find the most important variables with PCA and keep only those

    Parameters
    ----------
    df : pandas dataframe
        The dataframe containing the data

    n_components : int
        The number of components to keep

    threshold : float between 0 and 1
        The threshold to keep the components

    Returns
    -------
    df : pandas dataframe
        The dataframe containing the data
    '''
    pca = PCA(n_components=n_components)
    pca.fit(df)
    df_pca = pca.transform(df)

    # select the variable with the highest absolute value in the first principal component
    max_ind = np.argmax(np.abs(pca.components_[0]))

    # select variable with a high absolute value of at least half of the principal component with the highest absolute value
    selected_variables = df.columns[np.abs(
        pca.components_[0]) > np.abs(pca.components_[0][max_ind]/2)]

    # create a new df with the selected variables
    return df[selected_variables]


def create_pairs(x, y):
    '''
    Create the pairs of data

    Parameters
    ----------
    x : array
        The array containing the data

    y : array
        The array containing the labels

    Returns
    -------
    x_pairs : array
        The array containing the pairs of data

    y_pairs : array
        The array containing 1 if labels are the same and 0 if not
    '''
    x_pairs = []
    y_pairs = []

    for i in range(len(x)):
        for j in range(i+1, len(x)):
            x_pairs.append([x.iloc[i].to_list(),
                           x.iloc[j].to_list()])
            y_pairs.append(1 if y.iloc[i] == y.iloc[j] else 0)

    return x_pairs, y_pairs


class dataset(Dataset):
    '''
    Define the dataset class to be exploited by DataLoader 
    '''

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


class Net(nn.Module):
    def __init__(self, nb_variables):
        super(Net, self).__init__()

        self.stack = nn.Sequential(nn.Linear(2*nb_variables, 4*nb_variables),
                                   nn.ReLU(),
                                   nn.Linear(4*nb_variables, 2*nb_variables),
                                   nn.ReLU(),
                                   nn.Linear(2*nb_variables, 1),
                                   nn.Sigmoid())

        self.nb_variables = nb_variables

    def forward(self, x):
        x = x.view(-1, 2*self.nb_variables)
        x = self.stack(x)

        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    Train the model

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader containing the data

    model : torch.nn.Module
        The model to train

    loss_fn : torch.nn.Module
        The loss function

    optimizer : torch.optim
        The optimizer

    Returns
    -------
    None
    '''

    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred[:, 0], y)

        # Backpropagation (always in three steps)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    '''
    Test the model

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader containing the datatest

    model : torch.nn.Module
        The model trained

    loss_fn : torch.nn.Module
        The loss function

    Returns
    -------
    error : float
        The error of the model
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred[:, 0], y).item()
            pred = torch.tensor(
                [1 if pred[i] > 0.5 else 0 for i in range(len(pred))])
            correct += (pred == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return 100*(1-correct)


def global_loop(trainloader, testloader, model, loss_fn, optimizer, epochs=50):
    '''
    Train the model

    Parameters
    ----------
    trainloader : DataLoader
        The dataloader containing the training data

    testloader : DataLoader
        The dataloader containing the test data

    model : torch.nn.Module
        The model to train

    loss_fn : torch.nn.Module
        The loss function

    optimizer : torch.optim
        The optimizer

    epochs : int
        The number of epochs

    Returns
    -------
    error : list
        The list of the errors

    '''

    error = []

    for t in range(epochs):
        print(f"Epoch {t+1}-----------------")
        # Use train_loop and test_loop functions
        train_loop(trainloader, model, loss_fn, optimizer)
        x = test_loop(testloader, model, loss_fn)
        error.append(x)
    print("Done!")

    return error
