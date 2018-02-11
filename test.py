from hyperdash import monitor
import time
import utils
from functools import partial



HYPERDASH_PATH = '/Users/realnabe/.hyperdash/hyperdash.json'


get_api_key_from_file_hyperdash = partial(utils.get_api_key_from_file, json_path=HYPERDASH_PATH)


@monitor("dogs vs. cats", api_key_getter=get_api_key_from_file_hyperdash)
def main():
  print("Epoch 1, accuracy: 50%")
  time.sleep(2)
  print("Epoch 2, accuracy: 75%")
  time.sleep(2)
  print("Epoch 3, accuracy: 100%")


if __name__ == '__main__':
    main()