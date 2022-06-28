import numpy as np


class SeamCarve():
    __max_energy = 1000000.0

    def __init__(self, img):
        self.__arr = img.astype(int)
        self.__height, self.__width = img.shape[:2]
        self.__energy_arr = np.empty((self.__height, self.__width))
        self.__compute_energy_arr()


    def __swapaxes(self):
        self.__energy_arr = np.swapaxes(self.__energy_arr, 0, 1)
        self.__arr = np.swapaxes(self.__arr, 0, 1)
        self.__height, self.__width = self.__width, self.__height

    def __compute_energy_arr(self):
        self.__energy_arr[[0, -1], :] = self.__max_energy
        self.__energy_arr[:, [0, -1]] = self.__max_energy

        self.__energy_arr[1:-1, 1:-1] = np.add.reduce(
            np.abs(self.__arr[:-2, 1:-1] - self.__arr[2:, 1:-1]), -1)
        self.__energy_arr[1:-1, 1:-1] += np.add.reduce(
            np.abs(self.__arr[1:-1, :-2] - self.__arr[1:-1, 2:]), -1)

    def __compute_seam(self, horizontal=False):
        if horizontal:
            self.__swapaxes()

        energy_sum_arr = np.empty_like(self.__energy_arr)

        energy_sum_arr[0] = self.__energy_arr[0]
        for i in range(1, self.__height):
            energy_sum_arr[i, :-1] = np.minimum(
                energy_sum_arr[i - 1, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i, 1:] = np.minimum(
                energy_sum_arr[i, :-1], energy_sum_arr[i - 1, 1:])
            energy_sum_arr[i] += self.__energy_arr[i]

        seam = np.empty(self.__height, dtype=int)
        seam[-1] = np.argmin(energy_sum_arr[-1, :])
        seam_energy = energy_sum_arr[-1, seam[-1]]

        for i in range(self.__height - 2, -1, -1):
            l, r = max(0, seam[i + 1] -
                        1), min(seam[i + 1] + 2, self.__width)
            seam[i] = l + np.argmin(energy_sum_arr[i, l: r])



        return (seam_energy, seam)

    def __carve(self, horizontal=False, seam=None, remove=True):
        if horizontal:
            self.__swapaxes()
        
        if seam is None:
            seam = self.__compute_seam()[1]
            
        if remove:
            self.__width -= 1
        else:
            self.__width += 1

        new_arr = np.empty((self.__height, self.__width, 3))
        new_energy_arr = np.empty((self.__height, self.__width))
        mp_deleted_count = 0

        for i, j in enumerate(seam):
            if remove:
                if self.__energy_arr[i, j] < 0:
                    mp_deleted_count += 1
                new_energy_arr[i] = np.delete(
                    self.__energy_arr[i], j)
                new_arr[i] = np.delete(self.__arr[i], j, 0)


        self.__arr = new_arr
        self.__energy_arr = new_energy_arr


        return mp_deleted_count



    def remove_mask(self, mask):
        mp_count = np.count_nonzero(mask)

        self.__energy_arr[mask] *= -(self.__max_energy ** 2)
        self.__energy_arr[mask] -= (self.__max_energy ** 2)

        while mp_count:
            v_seam_energy, v_seam = self.__compute_seam(False)
            h_seam_energy, h_seam = self.__compute_seam(True)

            horizontal, seam = False, v_seam

            if v_seam_energy > h_seam_energy:
                horizontal, seam = True, h_seam

            mp_count -= self.__carve(horizontal, seam)


    def image(self):
        return self.__arr.astype(np.uint8)
