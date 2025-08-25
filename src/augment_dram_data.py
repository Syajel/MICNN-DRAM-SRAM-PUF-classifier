import os
import numpy as np

def corrupt(data_dir):
    """
    Corrupts all the files in data_dir by setting the last 20% to zeros
    """
    for root, _, files in os.walk(data_dir):
        files = sorted([f for f in files if f.startswith("run")])
        for file_name in files:
            file_path = os.path.join(root, file_name)
            with open(file_path, 'rb') as f:
                data = f.read()
            corrupt_point = int(len(data) * 0.8)  # Last 20% to zeros
            corrupted_data = data[:corrupt_point] + b'\x00' * (len(data) - corrupt_point)
            with open(file_path, 'wb') as f:
                f.write(corrupted_data)
        print(f"✅ Processed {root}")


def noise(data_dir,z,o):
    """
    Adds asymmetric probabilistic noise to the files in data_dir given majority-to-minority probability 'z' and minority-to-majority probability 'o'
    """
    p_relevant = 0
    n = 0
    for root, _, files in os.walk(data_dir):
        files = sorted([f for f in files if f.startswith("run")])
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, 'rb') as f:
                byte_data = f.read()

            # Convert byte data to bit array
            bit_array = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
            total_bits = bit_array.size
            num_zeros = np.sum(bit_array == 0)
            num_ones = np.sum(bit_array == 1)
            n += 1

            # Account for majority bits being either zeros or ones
            if num_ones > num_zeros:
                flip_prob_z2o = o
                flip_prob_o2z = z
                perc_zeros = (num_zeros / total_bits) * 100
                print(f"{file_name} → Zeroes: {perc_zeros:.2f}%")
                p_relevant += perc_zeros
            else:
                flip_prob_z2o = z
                flip_prob_o2z = o
                perc_ones = (num_ones / total_bits) * 100
                print(f"{file_name} → Ones: {perc_ones:.2f}%")
                p_relevant += perc_ones



            
            flip_mask = np.where(bit_array == 0, np.random.rand(bit_array.size) < flip_prob_z2o, np.random.rand(bit_array.size) < flip_prob_o2z)            # Generate random mask: True where bit should be flipped


            flipped_bits = np.bitwise_xor(bit_array, flip_mask.astype(np.uint8))            # XOR flip the bits (0^1=1, 1^1=0)


            flipped_bytes = np.packbits(flipped_bits)           # Repack bits into bytes

            with open(file_path, 'wb') as f:
                f.write(flipped_bytes.tobytes())
    print(p_relevant/n)
    

    

#corrupt("data/dram/train/corr")
#corrupt("data/dram/test/corr")
#noise("data/dram/test/noise", 0.002, 0.2)
#noise("data/dram/train/noise", 0.002, 0.2)

print("Done")
