from argparse import ArgumentParser
from typing import cast, List
from omegaconf import OmegaConf, DictConfig
import json
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
from os import cpu_count
from src.utils import PAD, MASK, UNK
from tqdm import tqdm
from multiprocessing import cpu_count, Manager, Pool
import functools
from src.utils import read_gpickle

SPECIAL_TOKENS = [PAD, UNK, MASK]
USE_CPU = max(1, cpu_count() // 2)  


class LossLogger(CallbackAny2Vec):
    """Callback to log loss during training"""
    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print(f'Loss after epoch {self.epoch}: {loss_now}')
        self.epoch += 1


def process_parallel(path: str, split_token: bool):
    """

    Args:
        path:

    Returns:

    """
    xfg = read_gpickle(path)
    tokens_list = list()
    for ln in xfg:
        code_tokens = xfg.nodes[ln]["code_sym_token"]

        if len(code_tokens) != 0:
            tokens_list.append(code_tokens)

    return tokens_list


def train_word_embedding(config_path: str):
    """
    train word embedding using word2vec

    Args:
        config_path:

    Returns:

    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name
    root = config.data_folder
    train_json = f"{root}/{cweid}/train.json"
    with open(train_json, "r") as f:
        paths = json.load(f)
    tokens_list = list()
    with Manager():
        pool = Pool(USE_CPU)

        process_func = functools.partial(process_parallel,
                                         split_token=config.split_token)
        tokens: List = [
            res
            for res in tqdm(
                pool.imap_unordered(process_func, paths),
                desc=f"xfg paths: ",
                total=len(paths),
            )
        ]
        pool.close()
        pool.join()
    
    for token_l in tokens:
        tokens_list.extend(token_l)


    print("training w2v...")
    num_workers = cpu_count(
    ) if config.num_workers == -1 else config.num_workers
   
    model = Word2Vec(
        sentences=tokens_list,
        vector_size=config.gnn.embed_size,      # Match model embedding size
        window=8,                               # Larger window for code context
        min_count=2,                            # Lower min_count for domain terms
        workers=num_workers,
        sg=1,                                   # Skip-gram (better for rare words)
        hs=0,                                   # Use negative sampling
        negative=10,                            # More negative samples
        sample=1e-4,                            # Subsampling frequent words
        epochs=15,                              # More epochs for better quality
        alpha=0.025,                            # Learning rate
        min_alpha=0.0001,                       # Final learning rate
        compute_loss=True,                      # Track loss
        callbacks=[LossLogger()],               # Log training progress
        max_vocab_size=config.dataset.token.vocabulary_size,
        seed=config.seed
    )

    # model = Word2Vec(sentences=tokens_list, min_count=3, size=config.gnn.embed_size,
    #                  max_vocab_size=config.dataset.token.vocabulary_size, workers=num_workers, sg=1)
    

    model.wv.save(f"{root}/{cweid}/w2v.wv")


def load_wv(config_path: str):
    """

    Args:
        config_path:

    Returns:

    """
    config = cast(DictConfig, OmegaConf.load(config_path))
    cweid = config.dataset.name

    model = KeyedVectors.load(f"{config.data_folder}/{cweid}/w2v.wv", mmap="r")

    print(model)


if __name__ == '__main__':
    __arg_parser = ArgumentParser()
    __arg_parser.add_argument("-c",
                              "--config",
                              help="Path to YAML configuration file",
                              default="configs/dwk.yaml",
                              type=str)
    __args = __arg_parser.parse_args()
    train_word_embedding(__args.config)
    load_wv(__args.config)
