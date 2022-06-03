import argparse
from mapping_cuisine_id import make_dict
from same_ingred_dict import get_same_ingred
from embedding import embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='사용법 테스트입니다.')
    parser.add_argument('--s', required=False, help='save directory path', default='../Container')
    parser.add_argument('--d', required=False, help='data directory path', default='../')
    args = parser.parse_args()
    data_path = args.d
    save_path = args.s
    make_dict(data_path, save_path)
    get_same_ingred(data_path, save_path)
    embedding(data_path, save_path, rm_same=False)