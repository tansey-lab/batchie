import numpy as np
from batchie.interfaces import Plate


class ComboPlate(Plate):
    def __init__(
        self,
        idx: np.ndarray,
        cline: np.ndarray,
        dd1: np.ndarray,
        dd2: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.idx = idx
        self.cline = cline
        self.dd1 = np.maximum(dd1, dd2)
        self.dd2 = np.minimum(dd1, dd2)

    ## Concatenates current plate with all plates in plate_list
    ## Only considers unique triplets
    def combine(self, plate_list: list["ComboPlate"], **kwargs) -> "ComboPlate":
        current_set = set()
        idx = []
        cline = []
        dd1 = []
        dd2 = []
        for i, c, d1, d2 in zip(self.idx, self.cline, self.dd1, self.dd2):
            if ((c, d1, d2) not in current_set) and ((c, d2, d1) not in current_set):
                idx.append(i)
                cline.append(c)
                dd1.append(d1)
                dd2.append(d2)
                current_set.add((c, d1, d2))

        for p in plate_list:
            for i, c, d1, d2 in zip(p.idx, p.cline, p.dd1, p.dd2):
                if ((c, d1, d2) not in current_set) and (
                    (c, d2, d1) not in current_set
                ):
                    idx.append(i)
                    cline.append(c)
                    dd1.append(d1)
                    dd2.append(d2)
                    current_set.add((c, d1, d2))

        idx = np.array(idx, dtype=np.int32)
        cline = np.array(cline, dtype=np.int32)
        dd1 = np.array(dd1, dtype=np.int32)
        dd2 = np.array(dd2, dtype=np.int32)
        plate = ComboPlate(idx, cline, dd1, dd2)
        return plate

    def size(self):
        return len(self.idx)

    def split(self, new_size: int) -> tuple["ComboPlate", "ComboPlate"]:
        n = self.size()
        assert new_size <= n, "Attempted to subsample more indices than exist!"

        indices = np.random.choice(n, size=new_size, replace=False)
        idx = self.idx[indices]
        cline = self.cline[indices]
        dd1 = self.dd1[indices]
        dd2 = self.dd2[indices]
        plate1 = ComboPlate(idx, cline, dd1, dd2)

        indices_rem = np.setdiff1d(np.arange(n), indices)
        idx_rem = self.idx[indices_rem]
        cline_rem = self.cline[indices_rem]
        dd1_rem = self.dd1[indices_rem]
        dd2_rem = self.dd2[indices_rem]
        plate2 = ComboPlate(idx_rem, cline_rem, dd1_rem, dd2_rem)
        return (plate1, plate2)

    def subsample(self, new_size: int) -> "ComboPlate":
        plate1, _ = self.split(new_size)
        return plate1
