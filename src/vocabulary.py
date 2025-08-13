import pickle
from dataclasses import dataclass
from os.path import exists
from typing import Dict, List
from src.utils import PAD, UNK, MASK
from gensim.models import KeyedVectors

TOKEN_TO_ID = "token_to_id"

@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]

    @staticmethod
    def build_from_w2v(w2v_path: str, speicial_tokens: List[str] = [PAD, UNK, MASK]):
        """
        build vocabulary from word2vec wv

        Args:
            w2v_path: path to word2vec wv
            speicial_tokens:


        Returns:

        """
        assert exists(w2v_path), f"{w2v_path} not exists!"
        model = KeyedVectors.load(w2v_path, mmap="r")
        attr = dict()
        
        # Add special tokens first
        for idx, tk in enumerate(speicial_tokens):
            attr[tk] = idx
        
        # Handle different Gensim versions
        try:
            # Gensim 4.x and later
            if hasattr(model, 'key_to_index'):
                for wd in model.key_to_index:
                    attr[wd] = model.key_to_index[wd] + len(speicial_tokens)
            else:
                # Fallback for older versions
                for wd in model.index_to_key:
                    attr[wd] = model.get_index(wd) + len(speicial_tokens)
        except AttributeError:
            try:
                # Gensim 3.x and earlier
                for wd in model.vocab:
                    attr[wd] = model.vocab[wd].index + len(speicial_tokens)
            except AttributeError:
                # Another fallback method
                for i, wd in enumerate(model.index_to_key):
                    attr[wd] = i + len(speicial_tokens)
        
        return Vocabulary(token_to_id=attr)

    @staticmethod
    def load_vocabulary(vocabulary_path: str) -> "Vocabulary":
        if not exists(vocabulary_path):
            raise ValueError(f"Can't find vocabulary in: {vocabulary_path}")
        with open(vocabulary_path, "rb") as vocabulary_file:
            vocabulary_dicts = pickle.load(vocabulary_file)
        token_to_id = vocabulary_dicts[TOKEN_TO_ID]
        return Vocabulary(token_to_id=token_to_id)

    def dump_vocabulary(self, vocabulary_path: str):
        with open(vocabulary_path, "wb") as vocabulary_file:
            vocabulary_dicts = {
                TOKEN_TO_ID: self.token_to_id,
            }
            pickle.dump(vocabulary_dicts, vocabulary_file)

    def convert_token_to_id(self, token: str):
        return self.token_to_id.get(token, self.token_to_id[UNK])

    def convert_tokens_to_ids(self, tokens: List[str]):
        return [self.convert_token_to_id(token) for token in tokens]

    def get_vocab_size(self):
        return len(self.token_to_id)

    def get_pad_id(self):
        return self.convert_token_to_id(PAD)
    
if __name__ == "__main__":
    w2v_path = "/home/linh/Documents/code/DeepWukong/data/CWE119/w2v.wv"  # Replace with your actual path
    if not exists(w2v_path):
        raise ValueError(f"Can't find word2vec model in: {w2v_path}")
    # Load the word2vec model
    model = KeyedVectors.load(w2v_path, mmap="r")
    speicial_tokens= [PAD, UNK, MASK]
    attr = dict()
    for idx, tk in enumerate(speicial_tokens):
        attr[tk] = idx
    import torch

    if hasattr(model, 'key_to_index'):
        for wd in model.key_to_index:
            print("wd: ",  wd)
            print("type wd: ", type(wd))
            print("model[wd]: ", model[wd])
            print("type model[wd]: ", type(model[wd]))
            print("convert to torch", torch.from_numpy(model[wd]))
            print("type torch.from_numpy(model[wd]): ", type(torch.from_numpy(model[wd])))
            break
    #         wd_torch = torch.from_numpy(model[wd])
    #         print("wd_torch: ", wd_torch)
            # # print("model.key_to_index[wd]: ", model.key_to_index[wd])
            # attr[wd] = model.key_to_index[wd] + len(speicial_tokens)
            # # print("attr[wd]: ", attr[wd])
    
    # print("Vocabulary size:", len(attr))

    # print("torch_string: ", torch.from_numpy("h"))         
    # print("torch_string: ", torch.from_numpy("hello world"))        
