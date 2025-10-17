import numpy as np
from tools import RoomGenerator
from tools import MicrophoneCircle
from tools import Phone
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class RoomSimulator:
    def __init__(
        self,
        seed: int | None = None,
    ):
        """Simulate room :)

        Args:
            seed (int, optional): The random seed used. Defaults to None.
        """

        # Define room properties
        self.signal = None
        self.room_params = None
        self.random_gen = np.random.default_rng(seed=seed)

    def compose_room(
        self,
        fs: int,
        n_mics: int = 12,
        mic_radius: float = 0.5,
        phone_rotation: np.ndarray = np.array([0, -90, 0]),
        signal: np.ndarray | None = None,
        shape: str = "shoebox",
        desired_rt60: float | None = None,
        material_properties_bounds: dict | None = None,
        ray_tracing_params: dict | None = None,
        generate_new_room: bool = True,
        room_bounds: dict | None = None,
    ):
        """Generate a room with Phone and Microphone Circle
            Sets the `self.room` and `self.room_dims` variables

        Args:
            n_mics (int, optional): Number of mics in circle. Defaults to 12.
            fs (int): Sampling frequency.
            mic_radius (float, optional): The radius of the mic circle. Defaults to 0.5.
            phone_rotation (np.ndarray, optional): The rotation of the phone in Euler angles (XYZ). Defaults to np.array(0, -90, 0) (Standing).
            signal (np.ndarray, optional): The signal to be used. Defaults to None.
            shape (str, optional): The shape of the room. Defaults to "shoebox".
            desired_rt60 (float, optional): The desired RT60 time. Defaults to None, thus using the defined material_properties_bounds.
            material_properties_bounds (dict, optional): The bounds of the randomly generated material. Defaults to None (Randomized).
            ray_tracing_params (dict, optional): Set the ray tracing parameters. If not set, ray tracing is not used. Defaults to None.
            room_bounds (dict, optional): The bounds of the room. Defaults to (3,10,3,10,2,5).
        """
        if self.signal is None or signal is not None:
            logging.debug("Setting signal...")
            self.signal = signal

        if generate_new_room:
            # Generate a random room
            logging.info("Generating new room...")
            self._room_gen = RoomGenerator(
                shape=shape,
                desired_rt60=desired_rt60,
                material_properties_bounds=material_properties_bounds,
                ray_tracing_params=ray_tracing_params,
                fs=fs,
            )
            self.seed = self.random_gen.integers(0, 10000)
            room, _ = self._room_gen.generate_room(
                seed=self.seed, room_bounds=room_bounds
            )
        else:
            # Generate same room, essentially obtaining an empty room :)
            logging.info("Generating same room...")
            room, _ = self._room_gen.generate_room(
                seed=self.seed, room_bounds=room_bounds
            )

        # Add mic circle
        room_bbox = room.get_bbox()  # in m
        self.phone_pos = [
            self.random_gen.uniform(
                room_bbox[0, 0] + mic_radius + 0.05, room_bbox[0, 1] - mic_radius - 0.05
            ),
            self.random_gen.uniform(
                room_bbox[1, 0] + mic_radius + 0.05, room_bbox[1, 1] - mic_radius - 0.05
            ),
            self.random_gen.normal(loc=1.7, scale=0.1),
        ]
        # Check if the phone position is inside the room
        while (
            self.phone_pos[2] < (room_bbox[2, 0] + 0.1) or self.phone_pos[2] > (room_bbox[2, 1] - 0.1)
        ):
            self.phone_pos[2] = self.random_gen.normal(loc=1.7, scale=0.1)

        mic_circle_gen = MicrophoneCircle(
            center=self.phone_pos, n_mics=n_mics, radius=mic_radius
        )
        dark_zone_mics = mic_circle_gen.get_microphone_positions()
        room.add_microphone_array(dark_zone_mics.T)

        # Add phone mics and speakers
        phone_gen = Phone(position=self.phone_pos, orientation=phone_rotation, unit="m")
        bright_zone_mics = phone_gen.get_mic_positions()
        phone_speakers = phone_gen.get_speaker_positions()

        room.add_microphone_array(bright_zone_mics.T)

        for pos in phone_speakers:
            # room.add_source(pos, signal=self.signal)
            room.add_source(pos, signal=self.signal/len(phone_speakers))
        logger.info(f"Added {len(phone_speakers)} speakers to the room!") 
        
        # Set the room and its params
        self.room = room
        self.room_params = {
            "fs": fs,
            "n_mics": n_mics,
            "mic_radius": mic_radius,
            "phone_rotation": phone_rotation,
            "signal": signal,
            "desired_rt60": desired_rt60,
            "material_properties_bounds": material_properties_bounds,
            "ray_tracing_params": ray_tracing_params,
            "room_bounds": room_bounds,
        }

        return True

    def randomize_room(self):
        """Randomize the room by generating a new one with the same properties"""
        assert (
            self.room_params is not None
        ), "Room parameters are not set. Please call compose_room() first."

        # Randomize the room
        logger.info("Randomizing room...")
        self.compose_room(generate_new_room=False, **self.room_params)
        logger.info("Randomized room!")
        return True

    def compute_rir(self, rt60=False, plot=False):
        """Computes the Room Impulse Responses and RT60 times for the room
        The RIRs are computed for each microphone and source in the room.
        The RT60 times are computed for each microphone and source in the room.

        Args:
            plot (bool, optional): Plot the RIRs. Defaults to False.

        Returns:
            tuple: Contains Rirs and RT60 times
        """
        logger.info("Computing room impulse response(s)...")
        self.room.compute_rir()
        logger.info("Computed room impulse response(s)!")

        if plot:
            logger.info("Plotting room impulse response(s)...")
            fig, ax = self.room.plot_rir()
            plt.show()

        # Calculate the RT60 times
        if rt60:
            logger.info("Calculating RT60 times...")
            # Calculate the RT60 times for each microphone
            rt60 = self.room.measure_rt60()
            logger.info("Calculated RT60 times!")
            return self.room.rir, rt60

        # If RT60 is not requested, return only the RIRs
        return self.room.rir  # [mic, source, samples]

    def get_zones(self, rirs, rt60: list | None = None):
        """Splits the RIRs into dark and bright zones and RT60 times if passed.

        Args:
            rirs (list): List containing all RIRs
            rt60 (list, optional): List containing all RT60 times. Defaults to None.

        Returns:
            tuple: If only RIRs are passed, a single tuple containing (BZ, DZ) RIRs is returned. If RT60 times are passed, two tuples are returned containing (BZ, DZ) for both RIRs and RT60 times.
        """
        logger.info("Splitting RIRs into bright and dark zones...")

        # Extract the RIRs for dark and bright zones
        dz_rir = rirs[: self.room_params["n_mics"]]
        bz_rir = rirs[self.room_params["n_mics"] :]

        # Extract the RT60 times for dark and bright zones, if RT60 is given
        if rt60 is not None:
            logger.info("Splitting RT60 times into bright and dark zones...")
            dz_rt60 = rt60[: self.room_params["n_mics"]]
            bz_rt60 = rt60[self.room_params["n_mics"] :]
            return (bz_rir, dz_rir), (bz_rt60, dz_rt60)

        # If RT60 is not given, return only RIRs
        return (
            bz_rir,
            dz_rir,
        )  # TODO: Should this also return a second (None, None) tuple?

    def regularize_rir(self, rirs, rt60: list | None = None, dtype=np.float64):
        """Regularize the RIR by cutting it to the longest RIR length or RT60 time if defined.

        Args:
            rirs (list): List containing all RIRs.
            rt60 (list, optional): List containing all RT60 times. Defaults to None.
            dtype (np.dtype, optional): The data type to use for the RIRs. Defaults to np.float64.

        Returns:
            np.ndarray: A numpy array containing the regularized RIRs. [mic, source, samples]
        """

        logger.info("Regularizing RIRs...")

        # Get cutoff index
        if rt60 is None:
            cutoff_idx = max([max([len(source) for source in mic]) for mic in rirs])
            logger.info("Using max length")
        else:
            cutoff_idx = int(self.room_params["fs"] * max([max(mic) for mic in rt60]))
            logger.info("Using RT60 length")

        # Go through each RIR
        rirs_np = np.zeros((len(rirs), len(rirs[0]), cutoff_idx))

        for n_mic, mic in enumerate(rirs):
            for n_source, source in enumerate(mic):
                # Cut the RIR to the cutoff_idx
                source = source[:cutoff_idx]

                # Pad the RIR to the cutoff_idx
                source = np.pad(
                    source,
                    (0, cutoff_idx - len(source)),
                    "constant",
                    constant_values=(0, 0),
                )

                # Add the RIR to the array
                rirs_np[n_mic, n_source] = source

        logger.info(f"Regularized RIRs to shape {rirs_np.shape}!")

        return rirs_np.astype(dtype=dtype)

    def get_phone_pos(self):
        return self.phone_pos

    def plot_room(self):
        """Plot the room"""
        fig, ax = self.room.plot()
        ax.set_xlim([0, 10])
        ax.set_ylim([0, 10])
        ax.set_zlim([0, 10])
        plt.show()
        return True


def main():
    from scipy.io import wavfile

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    )

    # specify signal source
    fs, signal = wavfile.read("wav_files/relaxing-guitar-loop-v5-245859.wav")
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # Average channels to convert to mono

    room_params = {
        "fs": fs,
        "n_mics": 6,
        "mic_radius": 0.5,
        "shape": "shoebox",
        "signal": signal,
        "room_bounds": {
            "min_width": 3.0,
            "max_width": 10.0,
            "min_length": 3.0,
            "max_length": 10.0,
            "min_extrude": 2.0,
            "max_extrude": 5.0,
        },
        "desired_rt60": 0.2,  # in seconds
        "material_properties_bounds": {
            "energy_absorption": (0.6, 0.9),
            "scattering": (0.05, 0.1),
        },
        "ray_tracing_params": {
            "receiver_radius": 0.4,
            "n_rays": 10000,
            "energy_thres": 1e-7,
        },
    }

    room_sim = RoomSimulator(seed=42)

    # Generate a room with the specified parameters, all non-specified parameters are randomized
    room_sim.compose_room(**room_params)
    room_sim.plot_room()

    # Randomize the phone position
    # room_sim.randomize_room()
    # room_sim.plot_room()

    # Compute the RIRs and RT60 times
    rirs, rt60s = room_sim.compute_rir(rt60=True, plot=False)

    # Regularize the RIRs to the RT60 length
    reg_rirs = room_sim.regularize_rir(rirs, rt60s)

    # Split the RIRs and RT60s into bright and dark zones
    (bz_rir, dz_rir), (bz_rt60s, dz_rt60s) = room_sim.get_zones(
        rirs=reg_rirs, rt60=rt60s
    )
    print("BZ RIR shape:", bz_rir.shape)
    print("DZ RIR shape:", dz_rir.shape)

    # This can also be done the other way around, get_zones -> regularize_rir
    # (bz_rir, dz_rir), (bz_rt60s, dz_rt60s) = room_sim.get_zones(rirs=rirs, rt60=rt60s)
    # bz_rir = room_sim.regularize_rir(bz_rir, bz_rt60s)
    # dz_rir = room_sim.regularize_rir(dz_rir, dz_rt60s)

    # Plot the RIRs for
    for n, (rir, rt60) in enumerate(zip(bz_rir[0], bz_rt60s[0])):
        print("RIR shape:", rir.shape)
        print("RT60:", rt60)
        plt.figure()
        plt.plot(rir)
        plt.title(f"Impulse Response for Microphone [{0},{n}] in Bright Zone")
        plt.axvline(x=rt60 * fs, color="r", linestyle="--", label="RT60 Cutoff")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()

