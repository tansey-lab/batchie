from batchie.data import Screen
import numpy as np

import string


def generate():
    plate_names = np.random.choice(
        [x for x in string.ascii_letters], replace=True, size=1000000
    )
    names = np.random.choice(
        ["a", "b", "c", "d", "e", "control"], replace=True, size=1000000
    )
    names2 = np.random.choice(
        ["a", "b", "c", "d", "e", "control"], replace=True, size=1000000
    )
    doses = np.random.choice([0.0, 1.0, 2.0, 3.0, 4.0, 0.0], replace=True, size=1000000)
    doses2 = np.random.choice(
        [0.0, 1.0, 2.0, 3.0, 4.0, 0.0], replace=True, size=1000000
    )

    Screen(
        treatment_names=np.vstack([names, names2]).T,
        treatment_doses=np.vstack([doses, doses2]).T,
        plate_names=plate_names,
        sample_names=plate_names,
        control_treatment_name="control",
    )


if __name__ == "__main__":
    for i in range(10):
        generate()
