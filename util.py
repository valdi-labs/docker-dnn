import os
import gzip
import numpy as np


def create_uncompressed_file(compressed_filename, uncompressed_filename):
    with gzip.open(compressed_filename, 'rb') as f:
        binary_data = f.read()
    with open(uncompressed_filename, 'wb') as f:
        f.write(binary_data)


def read_mnist(data_filename, label_filename):
    # unzip data if necessary
    data_filename_no_ext, data_file_ext = os.path.splitext(data_filename)
    if data_file_ext == '.gz':
        create_uncompressed_file(data_filename, data_filename_no_ext)
    else:  # assume not compressed
        data_filename_no_ext = data_filename

    # read and verify the data headers
    with open(data_filename_no_ext, 'rb') as f:
        data_headers = np.fromfile(f, dtype='>u4', count=4)
    assert data_headers[0] == 2051, 'Unexpected data magic number'  # from info here: http://yann.lecun.com/exdb/mnist/
    num_samples = data_headers[1]
    row_dim = data_headers[2]
    col_dim = data_headers[3]

    # unzip labels if necessary
    label_filename_no_ext, label_file_ext = os.path.splitext(label_filename)
    if label_file_ext == '.gz':
        create_uncompressed_file(label_filename, label_filename_no_ext)
    else:  # assume not compressed
        label_filename_no_ext = label_filename

    # read and verify the label headers
    with open(label_filename_no_ext, 'rb') as f:
        label_headers = np.fromfile(f, dtype='>u4', count=2)

    assert label_headers[0] == 2049, 'Unexpected label magic number'
    num_labels = label_headers[1]
    assert num_samples == num_labels, 'Number of data samples does not match number of labels'

    # read and reshape the data
    with open(data_filename_no_ext, 'rb') as f:
        raw_data = np.fromfile(f, dtype='>u1', offset=16)
    assert len(raw_data) == num_samples * row_dim * col_dim, 'Unexpected number of data samples'
    data = np.reshape(raw_data, (num_samples, row_dim, col_dim)) / 255.0

    # read the labels
    with open(label_filename_no_ext, 'rb') as f:
        raw_labels = np.fromfile(f, dtype='>u1', offset=8)
    assert len(raw_labels) == num_labels, 'Unexpected number of labels'

    # The slow way of reshaping the array, for posterity
    # data = np.zeros((num_samples, row_dim, col_dim), dtype=np.float_)
    # for sample_idx in range(num_samples):
    #     for row_idx in range(row_dim):
    #         data[sample_idx, row_idx, :] = \
    #             raw_data[sample_idx * row_dim * col_dim + row_idx * row_dim:
    #                      sample_idx * row_dim * col_dim + row_idx * row_dim + row_dim]

    return data, raw_labels
