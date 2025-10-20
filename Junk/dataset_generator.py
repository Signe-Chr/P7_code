import yaml
import logging
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tools import RoomSimulator
from copy import deepcopy

logger = logging.getLogger(__name__)


class DatasetGenerator:
    def __init__(self, dataset_params: dict | None = None):
        """
        Initialize the DatasetGenerator.
        """
        self.parse_params(**dataset_params)
        self.save_params(dataset_params)
        self.random_gen = np.random.default_rng(seed=self.seed)

    def parse_params(self, **kwargs):
        self.name: str = kwargs.get("name", "unnamed_run")
        self.splits: dict = kwargs.get(
            "splits",
            {
                "train": {
                    "num_rooms": 1,
                    "num_phone_pos": 1,
                },
                "val": {
                    "num_rooms": 1,
                    "num_phone_pos": 1,
                },
                "test": {
                    "num_rooms": 1,
                    "num_phone_pos": 1,
                },
            },
        )
        # Ensure each split has num_rooms and num_phone_pos
        for split, params in self.splits.items():
            assert "num_rooms" in params, f"Split {split} must have num_rooms"
            assert "num_phone_pos" in params, f"Split {split} must have num_phone_pos"

        self.room_params: dict = kwargs.get(
            "room_params",
            {
                "fs": None,
                "n_mics": None,
                "mic_radius": None,
                "shape": "shoebox",
                "signal": None,
            },
        )
        # Ensure room_params has fs, n_mics, mic_radius, and signal
        assert "fs" in self.room_params, "room_params must have fs"
        assert "n_mics" in self.room_params, "room_params must have n_mics"
        assert "mic_radius" in self.room_params, "room_params must have mic_radius"
        assert "shape" in self.room_params, "room_params must have shape"
        assert "signal" in self.room_params, "room_params must have signal"

        self.shape: str = self.room_params["shape"]
        self.root: Path = Path(kwargs.get("root", "dataset")) / self.shape / self.name
        self.root.mkdir(parents=True, exist_ok=False)

        self.regularizer: str = kwargs.get("regularizer", "rt60")
        # Ensure regularizer is valid
        assert self.regularizer in [
            "rt60",
            "maxlen",
        ], "Regularizing must be either 'rt60' or 'maxlen'"

        self.dtype: type = kwargs.get("dtype", np.float32)
        self.seed: int | None = kwargs.get("seed", None)
        return True

    def save_params(self, params: dict):
        """
        Save the parameters to a yaml file.
        """
        logger.info(f"Saving parameters to {self.root}/config.yaml")

        params_to_save = deepcopy(params)  # TODO: This is a lazy way of getting around the signal problem
        params_to_save["room_params"]["signal"] = "idk, probably relaxing guitar loop :)"

        with open(self.root / "dataset_params.yaml", "w") as file:
            yaml.dump(params_to_save, file)
        return True

    # Must be a static method to be used in multiprocessing, thus no self
    @staticmethod
    def simulate_room(
        room_params,
        regularizer: str,
        num_phone_pos: int = 1,
        save_dir: Path | None = None,
        dtype=np.float32,
        plot=False,
        seed: int | None = None,
    ):
        """
        Simulate the room using the RoomSimulator class.
        """
        assert regularizer in ["rt60", "maxlen"], "Regularizer must be either 'rt60' or 'maxlen'"
        assert num_phone_pos > 0, "num_phone_pos must be greater than 0"
        
        logger.info(f"Simulating room with {num_phone_pos} phone positions...")
        
        # Create the simulator
        simulator = RoomSimulator(seed=seed)
        
        # Generate room
        simulator.compose_room(**room_params)
        if plot:
            simulator.plot_room()

        room_config_dict = {
            "num_phone_pos": num_phone_pos,
            "bbox": simulator.room.get_bbox()[:, 1].tolist(),
            "volume": simulator.room.get_volume().tolist(),
            "phone_positions": {},
        }

        # Do for each phone position
        for i in range(num_phone_pos):

            # Generate phone position
            simulator.randomize_room()

            # Compute the RIRs and RT60 times then regularize them
            if regularizer == "rt60":
                rirs, rt60s = simulator.compute_rir(rt60=True)
                reg_rirs = simulator.regularize_rir(rirs, rt60s, dtype=dtype)

            elif regularizer == "maxlen":
                rirs = simulator.compute_rir(rt60=False)
                reg_rirs = simulator.regularize_rir(rirs, dtype=dtype)

            # Split the RIRs into bright and dark zones
            bz_rir, dz_rir = simulator.get_zones(rirs=reg_rirs)

            # Save the RIRs
            if save_dir is not None:
                save_path = save_dir / f"{i}".rjust(4, "0")
                logger.info(f"Saving RIRs to {save_path}")

                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(save_path.absolute(), bz_rir=bz_rir, dz_rir=dz_rir)
                logger.info(f"RIRs saved to {save_path.absolute()}")

                room_config_dict["phone_positions"][f"pos_{i}"] = simulator.get_phone_pos()

        if save_dir is not None:
            config_path = save_dir / "room_config.yaml"
            with open(config_path, "w") as file:
                yaml.dump(room_config_dict, file)
            logger.info(f"Saved room config to {config_path.absolute()}")

        return True  # Return True to indicate success


    def start(self):
        """
        Start the dataset generation process.
        """
        logger.info("Starting dataset generation...")

        # Generate data for each split
        for split, params in self.splits.items():
            logger.warning(f"Generating {split} split...")

            # Generate rooms dataset
            for i in range(params["num_rooms"]):
                logger.info(f"Generating room {i}/{params['num_rooms']}...")
                save_dir = self.root / split / f"room_{i}"

                self.simulate_room(
                    self.room_params,
                    regularizer=self.regularizer,
                    num_phone_pos=params["num_phone_pos"],
                    save_dir=save_dir,
                    dtype=self.dtype,
                    plot=False,
                    seed=self.random_gen.integers(0, 10000),
                )
                logger.warning(
                    f"Successfully generated room {i+1}/{params['num_rooms']}!"
                )

            logger.warning(f"Successfully generated {split} split!")

        logger.warning("Dataset generation completed!")
        
        
    def start_mp(self, utilization=0.25):
        """
        Start the dataset generation process.
        """
        minimum_utilization = (1/mp.cpu_count())
        assert utilization > minimum_utilization, f"Utilization must be greater than {minimum_utilization:.3f} for your CPU to allocate at least one thread!"
        assert utilization <= 1, "Utilization cannot be greater than 1!"
        
        allocated_threads = int(mp.cpu_count()*utilization)
        logger.warning(f"Starting multiprocessed dataset generation with {allocated_threads}/{mp.cpu_count()} threads allocated...")
        
        # Generate data for each split
        for split, params in self.splits.items():
            logger.warning(f"Generating {split} split:\n - num_rooms={params['num_rooms']}\n - num_phone_pos={params['num_phone_pos']}")
            if allocated_threads > params["num_rooms"]: 
                threads = params["num_rooms"]
                logger.warning(f"Reducing number of threads for '{split}' split to {threads} to match number of rooms...")
            else:
                threads = allocated_threads

            # Generate rooms dataset
            args_list = [
                (
                    self.room_params,
                    self.regularizer,
                    params["num_phone_pos"],
                    self.root / split / f"room_{i}",
                    self.dtype,
                    False,
                    self.random_gen.integers(0, 10000),
                )
                for i in range(params["num_rooms"])
            ]

            with mp.Pool(processes=threads) as pool:
                results = pool.starmap(self.simulate_room, args_list) # starmap because we need to pass multiple arguments to the function
                
            assert all(results), "One or more room simulations failed!"

            logger.warning(f"Successfully generated {split} split!")

        logger.warning("Dataset generation completed!")


def main():

    # Set logging configuration
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    )

    # Specify signal source
    from scipy.io import wavfile

    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono

    # Example usage
    dataset_params = {
        "name": "run_post_hand_in",
        "root": Path("dataset"),
        "splits": {
            "train": {
                "num_rooms": 1000,
                "num_phone_pos": 10,
            },
            "val": {
                "num_rooms": 200,
                "num_phone_pos": 10,
            },
            "test": {
                "num_rooms": 100,
                "num_phone_pos": 10,
            },
        },
        "room_params": {
            "fs": fs,
            "n_mics": 12, # number of microphones in the microphone circle, the amount of microphones for the phone cannot be changed and is fixed at 4 (ear=1, phone=3)
            "mic_radius": 0.5, # m
            "shape": "shoebox",  # "shoebox", "l_room", "t_room"
            "signal": signal,
            "room_bounds": {
                "min_width": 3.0, # m
                "max_width": 10.0, # m
                "min_length": 3.0, # m
                "max_length": 10.0, # m
                "min_extrude": 3.0, # m
                "max_extrude": 6.0, # m
            },
            # "desired_rt60": 0.5,
            "material_properties_bounds": {  #  # If desired_rt60 is None, this will be used
                "energy_absorption": (0.05, 0.4),
                "scattering": (0.05, 0.3),
            },
            # "ray_tracing_params": {
            #     "receiver_radius": 0.1,
            #     "n_rays": 10000,
            #     "energy_thres": 0.01,
            # },
        },
        "regularizer": "rt60",
        "dtype": np.float32,
        "seed": 69,
    }

    dataset_generator = DatasetGenerator(dataset_params)
    import time
    start_time = time.time()
    #dataset_generator.start()
    dataset_generator.start_mp(utilization=0.95)  # 0.25 for 25% CPU utilization
    end_time = time.time()
    print(f"Dataset generation took {end_time - start_time:.3f} seconds.")


if __name__ == "__main__":
    main()
