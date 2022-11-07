#!/usr/bin/python3

import logging

import numpy as np
import tqdm


def generate_example(filename: str, upper_limit: int, num_samples: int):
    logging.info(f"Generating {filename}...")

    batch_size = 10**7
    num_batches = num_samples // batch_size
    last_batch_size = num_samples % batch_size
    if last_batch_size == 0:
        last_batch_size = batch_size
        num_batches -= 1

    with open(filename, "w") as f, tqdm.tqdm(total=num_batches + 1) as pbar:
        for i_batch in range(num_batches):
            example_data = np.random.randint(1, upper_limit, (batch_size,))
            f.write(" ".join(map(str, example_data)))
            f.write(" ")
            pbar.update(1)

        example_data = np.random.randint(1, upper_limit, (last_batch_size,))
        f.write(" ".join(map(str, example_data)))
        pbar.update(1)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

    np.random.seed(0x80)

    generate_example("example_xs.txt", 100, 50)
    generate_example("example_s.txt", 10**7, 10**4)
    generate_example("example_m.txt", 10**7, 3*10**4)
    generate_example("example_l.txt", 10**7, 6*10**4)
    generate_example("example_xl.txt", 10**7, 10**5)
    generate_example("example_xxl.txt", 10**9, 5*10**5)
    generate_example("example_xxxl.txt", 10**9, 10**6)
