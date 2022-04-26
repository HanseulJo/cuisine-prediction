from mapping_cuisine_id import make_dict
from same_ingred_dict import get_same_ingred
from embedding import embedding

if __name__ == "__main__":
    data_path = "../"
    save_path = "./container"
    make_dict(data_path, save_path)
    get_same_ingred(data_path, save_path)
    embedding(data_path, save_path, rm_same=False)