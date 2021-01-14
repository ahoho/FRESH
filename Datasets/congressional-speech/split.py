import argparse
import pickle
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from stratified_group_kfold import StratifiedGroupKFold

if __name__ == "__main__":
    variables = [
        "party",
        "congress",
        "chamber",
        "speaker_id",
        "gender",
        "district",
        "state",
        "is_voting",
        "date",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("--label", required=True, choices=variables)
    parser.add_argument("--label_filter", nargs="+", default=None)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--dev_split", type=float, default=0.1)
    parser.add_argument("--split_by", nargs="+", choices=variables, default=[])
    parser.add_argument("--seed", type=int, default=11235)
    parser.add_argument("--make_default_data_source", action="store_true")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if Path(args.output_dir, "train.jsonl").exists():
        raise ValueError("train.jsonl exists in this directory")

    data = pd.read_json(args.input_file, dtype={'id': str}, lines=True)

    # Cleanup
    data.loc[(data.first_name=="BERNARD") & (data.last_name == "SANDERS"), "party"] = "D"
    data = data[[args.label, "id", "text"] + args.split_by]
    if args.label_filter:
        data = data.loc[data[args.label].isin(args.label_filter)]

    # Create splits
    held_out_prop = args.test_split + args.dev_split
    if args.split_by == ["date"]:
        data = data.sort_values('date')
        train_idx = int(len(data) * (1 - held_out_prop))
        dev_idx = int(len(data) * (1 - args.test_split))
        
        train = data.iloc[:train_idx]
        dev = data.iloc[train_idx:dev_idx]
        test = data.iloc[dev_idx:]
    elif args.split_by:
        groups = data.groupby(args.split_by).ngroup()

        cv = StratifiedGroupKFold(
            n_splits=max(2, int(1 / held_out_prop)),
            shuffle=True,
            random_state=args.seed
        )
        train_idx, held_out_idx = list(cv.split(data, data[args.label], groups=groups))[0]
        train = data.iloc[train_idx]
        held_out = data.iloc[held_out_idx]

        cv = StratifiedGroupKFold(
            n_splits=max(2, int(held_out_prop / args.test_split)),
            shuffle=True,
            random_state=args.seed
        )
        dev_idx, test_idx = list(cv.split(held_out, held_out[args.label], groups=groups.iloc[held_out_idx]))[0]
        dev = held_out.iloc[dev_idx]
        test = held_out.iloc[test_idx]
    else:
        train, held_out = train_test_split(data, test_size=held_out_prop, random_state=args.seed)
        dev, test = train_test_split(held_out, test_size=args.test_split / held_out_prop, random_state=args.seed)
    # TODO: cut out non-predictive / highly common text

    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        grouped = split.groupby(args.label).size()
        grouped /= grouped.sum()

        print(f"==={name}===")
        print(f"Split prop: {len(split) / len(data):0.2f}")
        print(f"Label prop: {name}")
        print(grouped.round(2))

        split = split.rename({"id": "annotation_id", "text": "document", args.label: "label"}, axis=1)
        split[["annotation_id", "document", "label"] + args.split_by].to_json(Path(args.output_dir, f"{name}.jsonl"), lines=True, orient="records")
    with open(Path(args.output_dir, f"hparams.pkl"), "wb") as outfile:
        pickle.dump(args, outfile)
        
    if args.make_default_data_source:
        os.symlink(str(Path(args.output_dir)), str("./data"))
