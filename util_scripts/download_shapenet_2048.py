import shutil
import urllib
from os import makedirs, remove, listdir
from os.path import exists, join
from zipfile import ZipFile

from core.arg_parser import parse_config


def main(config):
    dataset_config: dict = config['dataset']
    dataset_path: str = dataset_config['path']

    if exists(dataset_path):
        raise Exception(f'directory {dataset_path} already exists')

    makedirs(dataset_path)

    url = 'https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=1'

    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2][:-5]
    file_path = join(dataset_path, filename)
    with open(file_path, mode='wb') as f:
        d = data.read()
        f.write(d)

    print('Extracting...')
    with ZipFile(file_path, mode='r') as zip_f:
        zip_f.extractall(dataset_path)

    remove(file_path)

    extracted_dir = join(dataset_path,
                         'shape_net_core_uniform_samples_2048')
    for d in listdir(extracted_dir):
        shutil.move(src=join(extracted_dir, d),
                    dst=dataset_path)

    shutil.rmtree(extracted_dir)


if __name__ == '__main__':
    main(parse_config())
