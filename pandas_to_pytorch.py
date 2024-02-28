from typing import List
from collections import OrderedDict
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class Datasets(Dataset):
    def __init__(self, x, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        user_id = self.X[idx, 0]
        movie_id = self.X[idx, 1]
        genres = self.X[idx, 1]
        gender = self.X[idx, 1]
        age = self.X[idx, 1]
        occupation = self.X[idx, 1]
        zip_code = self.X[idx, 1]
        label = self.y[idx]
        return (user_id, movie_id, genres, gender, age, occupation, zip_code, label)
    
    def __len__(self):
        return len(self.X)

X = torch.tensor(df.iloc[:,:-1].values)
y = torch.tensor(df.iloc[:,-1].values)
train_dataset = Datasets(X,y)

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

next(iter(train_dataloader))

class Model(nn.Module):
    def __init__(self, emb_dim=16):
        super(Model, self).__init__()
        
        # sparse embedings
        self.user_emb = nn.Embedding(235, emb_dim)
        self.gender_emb = nn.Embedding(2, emb_dim)
        self.occupation_emb = nn.Embedding(111, emb_dim)
        self.zip_code_emb = nn.Embedding(526, emb_dim)
        self.movie_emb = nn.Embedding(21, emb_dim)
        self.genres_emb = nn.Embedding(50, emb_dim)
        
    def forward(self, user_id, movie_id, genres, gender, age, occupation, zip_code, label=None):
        
        # user embedding
        
        user_e = self.user_emb(user_id)
        gender_e = self.gender_emb(gender)
        occupation_e = self.occupation_emb(occupation)
        zip_code_e = self.zip_code_emb(zip_code)
        movie_id_e = self.movie_emb(movie_id)
        genres_e = self.genres_emb(genres)
        
        output = torch.cat([user_e, gender_e, occupation_e, zip_code_e, movie_id_e, genres_e, age], dim=-1)
        
        return output
    
        
def get_vocabularies(df: pd.DataFrame, categorical_columns: List):
    vocab_size = {}
    for cat in categorical_columns:
        vocab_size[cat] = df[cat].max() + 1
    
    return vocab_size

categorical_features = ['uid', 'ugender', 'iid', 'igenre']
vocab_sizes = get_vocabularies(df, categorical_features)

def get_embedding_dim_dict(ategorical_features, embedding_dim):
    return {cat: embedding_dim for cat in ategorical_features}


embedding_dim_dict = get_embedding_dim_dict(categorical_features, 6)
embedding_dim_dict
# {'uid': 6, 'ugender': 6, 'iid': 6, 'igenre': 6}


@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int
    embedding_name: str = None
    dtype: str = torch.long
    
    def __post_init__(self):
        """Auto fill embedding_name"""
        if self.embedding_name is None:
            self.embedding_name = self.name

embedding_dim = 8
uid_sparse_feat = SparseFeat(name='uid', vocabulary_size=vocab_size['uid'], embedding_dim=embedding_dim)
# get vocabulary size for uid
uid_sparse_feat.vocabulary_size
# get embedding dim for uid
uid_sparse_feat.embedding_dim

@dataclass
class DenseFeat:
    name: str
    dimension: int
    dtype: str = torch.float
    
dense_feat = DenseFeat(name='score', dimension=1)
dense_feat
# DenseFeat(name='score', dimension=1, dtype=torch.float32)


categorical_features = ['uid', 'ugender', 'iid', 'igenre']
sparse_features = [SparseFeat(name=cat,
                              vocabulary_size=vocab_sizes[cat],
                              embedding_dim=embedding_dim_dict[cat]) for cat in categorical_features]

numerical_features = ['score']
# create list of numerical features
dense_features = [DenseFeat(name=col, dimension=1)
                  for col in numerical_features]

feature_columns = sparse_features + dense_features


def build_input_features(feature_columns):

    features = OrderedDict()
    start = 0
    for feat in feature_columns:
        if isinstance(feat, DenseFeat):
            features[feat.name] = (start, start + feat.dimension)
            start += feat.dimension

        elif isinstance(feat, SparseFeat):
            features[feat.name] = (start, start + 1)
            start += 1

        else:
            raise TypeError('Invalid feature columns type, got', type(feat))
    return features


feature_positions = build_input_features(feature_columns)
feature_positions
# OrderedDict([('uid', (0, 1)),
#              ('ugender', (1, 2)),
#              ('iid', (2, 3)),
#              ('igenre', (3, 4)),
#              ('score', (4, 5))])


def build_torch_dataset(df: pd.DataFrame, feature_columns: List):
    """ Create a torch tensor from the pandas dataframe according to the order of the features in feature_columns
    Cannot just use torch.tensor(df.values) because for variable length columns, it contains a list.
    Args:
        df (pandas.DataFrame): dataframe containing the features
        feature_columns (List)
    Returns:
        (torch.Tensor): pytorch tensor from df according to the order of feature_columns
    """
    tensors = []
    df = df.copy()
    feature_length_names = []
    for feat in feature_columns:
        tensor = torch.tensor(df[feat.name].values, dtype=feat.dtype)
        tensors.append(tensor.reshape(-1, 1))
    return torch.concat(tensors, dim=1)


torch_df = build_torch_dataset(df, feature_columns)
torch_df
# tensor([[0.0000, 0.0000, 1.0000, 1.0000, 0.1000],
#         [1.0000, 1.0000, 2.0000, 2.0000, 0.2000],
#         [2.0000, 0.0000, 3.0000, 1.0000, 0.3000]])


def build_embedding_dict(all_sparse_feature_columns, init_std=0.001):
    embedding_dict = nn.ModuleDict(
        {feat.name: nn.Embedding(feat.vocabulary_size,
                                 feat.embedding_dim) for feat in all_sparse_feature_columns})
    if init_std is not None:
        for tensor in embedding_dict.values():
            # nn.init is in_place
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict


embedding_dict = build_embedding_dict(sparse_features)
embedding_dict
# ModuleDict(
#   (uid): Embedding(3, 6)
#   (ugender): Embedding(2, 6)
#   (iid): Embedding(4, 6)
#   (igenre): Embedding(3, 6)
# )


def embedding_lookup(X,
                     feature_positions,
                     embedding_dict,
                     sparse_feature_columns,
                     return_feat_list=()):

    embeddings_list = []
    for feat in sparse_feature_columns:
        feat_name = feat.name
        embedding_name = feat.embedding_name
        if feat_name in return_feat_list or len(return_feat_list) == 0:
            lookup_idx = feature_positions[feat_name]
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            embedding = embedding_dict[embedding_name](input_tensor)
            embeddings_list.append(embedding)
    return embeddings_list


categorical_embeddings = embedding_lookup(torch_df,
                                          feature_positions,
                                          embedding_dict,
                                          sparse_features,
                                          return_feat_list=['uid', 'genre'])
categorical_embeddings
# [tensor([[[-9.1713e-04,  6.5061e-05, -8.2737e-04, -6.2794e-04,  3.2218e-04,
#            -9.5998e-04]],

#          [[-3.6192e-04, -7.2849e-04, -4.4335e-04,  5.4883e-04, -6.2344e-04,
#            -5.5105e-04]],

#          [[ 4.9634e-04,  2.3615e-04, -1.2853e-03, -2.9909e-04,  1.2274e-03,
#            -2.2752e-04]]], grad_fn=<EmbeddingBackward0>)]


def dense_lookup(X, feature_positions, dense_features, return_feat_list=()):
    dense_list = []
    for feat in dense_features:
        feat_name = feat.name
        lookup_idx = feature_positions[feat_name]
        tensor = X[:, lookup_idx[0]:lookup_idx[1]]
        dense_list.append(tensor)
    return dense_list


dense_feats = dense_lookup(torch_df,
                           feature_positions,
                           dense_features,
                           return_feat_list=['score'])
dense_feats
# [tensor([[0.1000],
#          [0.2000],
#          [0.3000]])]
