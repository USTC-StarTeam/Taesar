import argparse
import pickle

from preprocessing import Preprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--strd_ym", type=str, default="202509", help="ym")
parser.add_argument("--data_name", type=str, default="ESTT")


args = parser.parse_args()
print(args)


def voca_mapping(sequence, voca, timestamp):
    sequence = sequence.split(",")
    if timestamp:
        sequence_index = ",".join([str(i) for i in sequence])  # str
    else:
        sequence_index = ",".join([str(voca.stoi.get(i, voca.unk_index)) for i in sequence])

    return sequence_index


def save_array(data, column_list):
    final_list = []
    for column in column_list:
        temp_seq = data[column].tolist()
        final_list.append(temp_seq)
    return final_list


def main(strd_ym="202303", data_name="amazon"):
    # 1. data loading
    loader = Preprocessing(strd_ym=strd_ym)
    data_total, voca = loader.pretrain_loader(data_name)

    print(
        "Avg. Length of " + data_name + " seq:",
        data_total["item"].apply(lambda x: len(x.split(","))).mean(),
    )

    # 2. voca mapping column
    print("Total dataset Vocabulary Mapping...")
    data_total["timestamp"] = data_total["unix_time"].apply(lambda x: voca_mapping(x, voca[0], timestamp=True))
    data_total["item_index"] = data_total["item"].apply(lambda x: voca_mapping(x, voca[0], timestamp=False))
    data_total["type_index"] = data_total["type"].apply(lambda x: voca_mapping(x, voca[1], timestamp=False))

    data_total.to_pickle(f"data/{data_name}/{data_name}_voca_mapping_{strd_ym}.pkl")

    column_list_ = ["item_index", "type_index", "timestamp"]
    final_data = save_array(data=data_total, column_list=column_list_)

    # 3. save data
    with open(f"data/{data_name}/{data_name}_list_dataset_{strd_ym}.pkl", "wb") as fp:
        pickle.dump(final_data, fp)
    fp.close()


if __name__ == "__main__":
    main(strd_ym=args.strd_ym, data_name=args.data_name)
