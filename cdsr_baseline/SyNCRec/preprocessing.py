import os
from os import path

import pandas as pd

from dataset.vocab import WordVocab


class Preprocessing:
    def __init__(self, strd_ym):
        self.strd_ym = strd_ym

    def pretrain_get_data(self, data_name):
        print("Data Loading Start with", data_name)

        root_path = "data/" + data_name
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        data_path = root_path + "/" + f"{data_name}_{self.strd_ym}" + ".pkl"
        if not path.exists(data_path):
            seq = pd.read_table(f"dataset/{data_name}/{data_name}_seq.txt")
            seq = seq.drop(columns="Unnamed: 0")
            seq.to_pickle(data_path)
        else:
            print("File already exists!")
            seq = pd.read_pickle(data_path)

        return seq

    def build_voca(self, voca_raw):
        voca = []
        for i in voca_raw:
            line = i.split(",")
            voca.extend(line)

        print("Voca_len: ", len(set(voca)))
        return voca, len(set(voca))

    def make_voca(self, data_name):
        root_path = "data/" + data_name
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        data_path = root_path + "/" + f"{data_name}_voca_item_" + self.strd_ym + ".ep"

        if not path.exists(data_path):
            print("No file!")

            columns = ["item", "type"]
            voca_set = []
            for i in columns:
                print("voca dataset loading...")

                voca = pd.read_table(f"dataset/{data_name}/voca_{i}.txt")
                voca = voca.drop(columns="Unnamed: 0")
                min_freq = 0

                print("Vocabularay Building...")
                voca = WordVocab(voca, min_freq=min_freq)

                output_voca_path = root_path + "/" + data_name + "_voca_" + i + "_" + self.strd_ym + ".ep"
                voca.save_vocab(output_voca_path)
                voca = WordVocab.load_vocab(output_voca_path)
                voca_set.append(voca)
        else:
            print("Vocab already exist!")
            columns = ["item", "type"]

            voca_set = []
            for i in columns:
                output_voca_path = root_path + "/" + f"{data_name}_voca_{i}_" + self.strd_ym + ".ep"
                voca = WordVocab.load_vocab(output_voca_path)
                voca_set.append(voca)

        return voca_set

    def pretrain_loader(self, data_name):
        data = self.pretrain_get_data(data_name)
        print(data["item"].head())

        voca = self.make_voca(data_name)

        return data, voca
