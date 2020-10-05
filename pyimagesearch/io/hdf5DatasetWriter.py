# import the necessary packages
import h5py
import os


class HDF5DatasetWriter():
    def __init__(self, dims, outputPath, dataKey="images", bufferSize=1000):
        if os.path.exists(outputPath):
            raise ValueError(
                "the output path already exists remove the database manually ", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, shape=dims, dtype="float")
        self.labels = self.db.create_dataset(
            "labels", shape=(dims[0],), dtype="float")

        self.bufferSize = 1000
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels into buffer(memory)
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        if len(self.buffer["data"]) >= self.bufferSize:
            self.flush()

    def flush(self):
        # move the data from memory to disk
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", shape=(len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # if the buffer is not empty, flush it.
        if len(self.buffer["data"]) > 0:
            self.flush()
        self.db.close()
