import numpy as np
import os
import glob
import argparse


def opts():
    opt = argparse.ArgumentParser(description="This program splits each video's feature to each files")
    opt.add_argument("--input", default="tmp/embeddings.npy", type=str, help="Path to embedding file")
    opt.add_argument("--output", default="tmp/features/", type=str, help="Path to output directory")
    opt.add_argument("--overwrite", default=False, action="store_true", help="force process")
    args = opt.parse_args()

    return args


def split_feature(args):
    emb_file = args.input
    out_dir = args.output

    os.makedirs(out_dir, exist_ok=True)
    if len(glob.glob(os.path.join(out_dir, "*"))) > 0 and not args.overwrite:
        raise ValueError("File exists in output directory."
                         "Please provide other output directory or pass --overwrite while launching script.")
    embs = np.load(emb_file, allow_pickle=True, encoding='bytes').item()
    print('Split features')
    for i in range(len(embs["names"])):
        file_name = embs["names"][i][0].decode('UTF-8')
        features = np.array(embs["embs"][i])
        save_file = os.path.join(out_dir, file_name + '.npy')
        np.save(save_file, features)
        print('save', file_name, 'feature')


if __name__ == '__main__':
    arg = opts()
    split_feature(arg)
