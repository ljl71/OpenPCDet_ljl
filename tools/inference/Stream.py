

import numpy as np
import torch



# -----------------------------------------------------------

# sex age   city     income  job
# -1  0.39  0  0  1  0.0100  2
#  1  0.59  1  0  0  0.0200  0
# -1  0.41  1  0  0  0.0300  1
# . . .
# -1  0.43  0  1  0  0.4000  0

# -----------------------------------------------------------

class EmployeeStreamLoader():

    def __init__(self, fn, bat_size, buff_size,shuffle=False, seed=0):
        if buff_size % bat_size != 0:
            raise Exception("buff_size must be evenly div by bat_size")

        self.bat_size = bat_size
        self.buff_size = buff_size
        self.shuffle = shuffle

        self.rnd = np.random.RandomState(seed)

        self.ptr = 0  # points into x_data and y_data

        self.fin = open(fn, "r")  # line-based text file

        self.the_buffer = []  # list of numpy vectors
        self.xy_mat = None  # NumPy 2-D version of buffer
        self.x_data = None  # predictors as Tensors
        self.y_data = None  # targets as Tensors

        self.dataset = None

        self.reload_buffer()

    def reload_buffer(self):
        self.the_buffer = []
        self.ptr = 0
        ct = 0  # number of lines read
        while ct < self.buff_size:
            line = self.fin.readline()
            if line == "":
                self.fin.seek(0)
                return -1  # reached EOF
            else:
                line = line.strip()  # remove trailing newline
                np_vec = np.fromstring(line, sep="\t")
                self.the_buffer.append(np_vec)
                ct += 1

        if len(self.the_buffer) != self.buff_size:
            return -2  # buffer was not fully loaded

        if self.shuffle == True:
            self.rnd.shuffle(self.the_buffer)  # in-place

        self.xy_mat = np.array(self.the_buffer)  # 2-D array
        self.x_data = torch.tensor(self.xy_mat[:, 0:6],dtype=torch.float32)
        self.y_data = torch.tensor(self.xy_mat[:, 6], dtype=torch.int64)

        return 0  # buffer successfully loaded

    def __iter__(self):
        return self

    def __next__(self):  # next batch as a tuple
        res = 0

        if self.ptr + self.bat_size > self.buff_size:  # reload
            print(" ** reloading buffer ** ")
            res = self.reload_buffer()
            # 0 = success, -1 = hit eof, -2 = not fully loaded

        if res == 0:
            start = self.ptr
            end = self.ptr + self.bat_size


            x = self.x_data[start:end, :]
            y = self.y_data[start:end]
            self.ptr += self.bat_size
            return (x, y)

        # reached end-of-epoch (EOF), so signal no more
        self.reload_buffer()  # prepare for next epoch
        raise StopIteration


# -----------------------------------------------------------

def main():
    print("\nBegin streaming data loader demo ")
    np.random.seed(1)

    fn = ""  # 40 lines of data
    bat_size = 3
    buff_size = 12  # a multiple of bat_size
    emp_ldr = EmployeeStreamLoader(fn, bat_size=bat_size, buff_size=buff_size,shuffle=False)

    max_epochs = 4
    for epoch in range(max_epochs):
        print("\n == Epoch: " + str(epoch) + " ==")
        for (b_idx, batch) in enumerate(emp_ldr):
            print("epoch: " + str(epoch) + "   batch: " + str(b_idx))
            print(batch[0])  # predictors
            print(batch[1])  # labels
            print("")
    emp_ldr.fin.close()

    print("End demo ")


if __name__ == "__main__":
    main()