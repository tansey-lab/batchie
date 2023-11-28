import argparse
import glob
import json
import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

ch = logging.StreamHandler()

logger.setLevel(logging.DEBUG)
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_script_location():
    return os.path.abspath(os.path.dirname(os.path.realpath(__file__)))


def get_nextflow_dir():
    return os.path.abspath(os.path.join(get_script_location(), ".."))


def get_base_config():
    return os.path.abspath(os.path.join(get_nextflow_dir(), "..", "nextflow.config"))


def get_repository_root():
    return os.path.abspath(os.path.join(get_script_location(), "..", ".."))


def get_main_nf_file():
    return os.path.abspath(os.path.join(get_repository_root(), "main.nf"))


def get_args():
    parser = argparse.ArgumentParser(
        description="Run retrospective simulation, all arguments are passed to nextflow"
    )
    parser.add_argument(
        "--screen",
        type=str,
        required=True,
        help="Path to screen",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many plates to select each iteration (using the same trained model)",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="Path to output directory"
    )
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def validate_initial_output_dir_and_get_result_files_as_dict(output_dir):
    test_screen_glob = list(glob.glob(os.path.join(output_dir, "*", "test.screen.h5")))

    training_screen_glob = list(
        glob.glob(os.path.join(output_dir, "*", "training.screen.h5"))
    )

    screen_metadata = list(
        glob.glob(os.path.join(output_dir, "*", "screen_metadata.json"))
    )

    if len(training_screen_glob) == 0 or len(screen_metadata) == 0:
        return None

    test_screen = test_screen_glob[0]
    training_screen = training_screen_glob[0]
    screen_metadata = screen_metadata[0]

    with open(screen_metadata, "r") as f:
        screen_metadata_obj = json.load(f)

    return {
        "test_screen": test_screen,
        "training_screen": training_screen,
        "screen_metadata": screen_metadata_obj,
    }


def get_screen_from_job_output(output_dir):
    advanced_screen_glob = list(
        glob.glob(os.path.join(output_dir, "*", "advanced_screen.h5"))
    )

    training_screen_glob = list(
        glob.glob(os.path.join(output_dir, "*", "training.screen.h5"))
    )

    if len(advanced_screen_glob) == 0 and len(training_screen_glob) == 0:
        return None
    if len(advanced_screen_glob) > 0:
        return advanced_screen_glob[0]
    else:
        return training_screen_glob[0]


def validate_job_dir_and_return_meta(output_dir):
    screen_metadata = list(
        glob.glob(os.path.join(output_dir, "*", "screen_metadata.json"))
    )

    if len(screen_metadata) == 0:
        return None

    screen_metadata = screen_metadata[0]

    with open(screen_metadata, "r") as f:
        screen_metadata_obj = json.load(f)

    return screen_metadata_obj


def get_theta_and_dist_chunks(output_dir):
    thetas = list(glob.glob(os.path.join(output_dir, "*", "thetas*.h5")))
    dist_chunks = list(
        glob.glob(os.path.join(output_dir, "*", "distance_matrix_chunk*.h5"))
    )

    if len(thetas) == 0 or len(dist_chunks) == 0:
        raise ValueError("No thetas or dist_chunks found")

    return {
        "thetas": os.path.join(output_dir, "*", "thetas*.h5"),
        "dist_chunks": os.path.join(output_dir, "*", "distance_matrix_chunk*.h5"),
    }


def run_initial_plate(output_dir, screen, experiment_name, extra_args):
    subprocess.check_call(
        [
            "nextflow",
            "run",
            get_main_nf_file(),
            "--mode",
            "retrospective",
            "--screen",
            screen,
            "--name",
            experiment_name,
            "--outdir",
            output_dir,
            "--initialize",
            "true",
            "-work-dir",
            os.path.join(output_dir, "work"),
        ]
        + extra_args,
        cwd=get_repository_root(),
    )


def run_first_batch_plate(output_dir, screen, experiment_name, extra_args):
    subprocess.check_call(
        [
            "nextflow",
            "run",
            get_main_nf_file(),
            "--mode",
            "retrospective",
            "--screen",
            screen,
            "--name",
            experiment_name,
            "--outdir",
            output_dir,
            "--initialize",
            "false",
            "-work-dir",
            os.path.join(output_dir, "work"),
        ]
        + extra_args,
        cwd=get_repository_root(),
    )


def run_subsequent_batch_plate(
    output_dir, screen, thetas, dist_chunks, experiment_name, extra_args
):
    subprocess.check_call(
        [
            "nextflow",
            "run",
            get_main_nf_file(),
            "--mode",
            "next_plate",
            "--reveal",
            "true",
            "--screen",
            screen,
            "--thetas",
            thetas,
            "--distance_matrix",
            dist_chunks,
            "--name",
            experiment_name,
            "--outdir",
            output_dir,
            "-work-dir",
            os.path.join(output_dir, "work"),
        ]
        + extra_args,
        cwd=get_repository_root(),
    )


def dir_sort_key(x):
    return int(os.path.basename(x).split("_")[1])


def examine_output_dir_to_determine_current_iteration(output_dir, batch_size):
    # list all directories in output directory
    contents_of_output_directory = glob.glob(output_dir + "/iter_*")
    # filter to directories
    iter_dirs = [x for x in contents_of_output_directory if os.path.isdir(x)]

    iter_dirs = sorted(iter_dirs, key=dir_sort_key)
    last_successful_run_meta = None
    current_iter_index = None
    current_plate_idx = None

    for iter_dir in iter_dirs:
        contents_of_iter_directory = glob.glob(iter_dir + "/plate_*")
        plate_dirs = [x for x in contents_of_iter_directory if os.path.isdir(x)]

        plate_dirs = sorted(plate_dirs, key=dir_sort_key)

        current_plate_idx = 0

        for idx, plate_dir in enumerate(plate_dirs):
            plate_idx = dir_sort_key(plate_dir)

            if validate_job_dir_and_return_meta(plate_dir) is None:
                raise RuntimeError(
                    f"Found job dir with invalid structure. "
                    f"Consider deleting this directory to continue simulation: {plate_dir}"
                )

            if plate_idx != idx:
                raise RuntimeError(
                    f"Found job dir with no apparent ancestor. "
                    f"Consider deleting this directory to continue simulation: {plate_dir}"
                )

            current_plate_idx = plate_idx
            current_iter_index = dir_sort_key(iter_dir)
            last_successful_run_meta = validate_job_dir_and_return_meta(plate_dir)

    if last_successful_run_meta is None:
        return 0, 0, None, None

    if current_plate_idx >= batch_size - 1:
        next_iter_index = current_iter_index + 1
        next_plate_index = 0
    else:
        next_iter_index = current_iter_index
        next_plate_index = current_plate_idx + 1

    return (
        next_iter_index,
        next_plate_index,
        last_successful_run_meta,
        get_screen_from_job_output(plate_dir),
    )


def run_next_step(output_dir, input_screen, extra_args, batch_size):
    os.makedirs(output_dir, exist_ok=True)

    experiment_name, _ = os.path.splitext(os.path.basename(input_screen))

    (
        current_iter_index,
        current_plate_idx,
        last_successful_run_meta,
        current_screen,
    ) = examine_output_dir_to_determine_current_iteration(output_dir, batch_size)

    if last_successful_run_meta is not None:
        plates_remaining = last_successful_run_meta["n_unobserved_plates"]

        logger.info(f"Plates remaining: {plates_remaining}")

        if plates_remaining <= 0:
            logger.info(f"0 unobserved plates remaining, exiting")
            return False

    job_output_dir = os.path.join(
        output_dir, f"iter_{current_iter_index}", f"plate_{current_plate_idx}"
    )

    # clear job output dir incase some partial results were written
    shutil.rmtree(job_output_dir, ignore_errors=True)
    os.makedirs(job_output_dir, exist_ok=True)

    logger.info(f"Running iteration {current_iter_index}, plate {current_plate_idx}")

    if (current_iter_index, current_plate_idx) == (0, 0):
        run_initial_plate(
            output_dir=job_output_dir,
            screen=input_screen,
            experiment_name=experiment_name,
            extra_args=extra_args,
        )
    elif current_plate_idx == 0:
        run_first_batch_plate(
            output_dir=job_output_dir,
            screen=current_screen,
            experiment_name=experiment_name,
            extra_args=extra_args,
        )
    else:
        first_plate_of_iter_output_dir = os.path.join(
            output_dir, f"iter_{current_iter_index}", f"plate_{0}"
        )
        theta_and_dist_chunks = get_theta_and_dist_chunks(
            first_plate_of_iter_output_dir
        )

        run_subsequent_batch_plate(
            output_dir=job_output_dir,
            screen=current_screen,
            experiment_name=experiment_name,
            extra_args=extra_args,
            thetas=theta_and_dist_chunks["thetas"],
            dist_chunks=theta_and_dist_chunks["dist_chunks"],
        )
    return True


def main():
    args, remaining_args = get_args()
    while True:
        should_run_again = run_next_step(
            output_dir=os.path.abspath(args.outdir),
            input_screen=os.path.abspath(args.screen),
            extra_args=remaining_args,
            batch_size=args.batch_size,
        )

        if not should_run_again:
            break


if __name__ == "__main__":
    main()
