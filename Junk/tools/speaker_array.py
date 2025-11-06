import numpy as np

class SpeakerArray:
    def __init__(self, n_speakers, start_position, distance, orientation=np.array([1, 0, 0])):
        """
        Initialize a 1D speaker array.

        Parameters:
        n_speakers (int): Number of speakers in the array.
        start_position (list or np.array): Position of the first speaker [x, y, z].
        distance (float): Distance between adjacent speakers.
        x_orientation (list or np.array, optional): Orientation (x,y,z) (default: [1,0,0]).
        """
        self.n_speakers = n_speakers
        self.start_position = np.array(start_position)
        self.distance = distance
        self.orientation = np.array(orientation)


        # Normalize orientation vectors
        self.orientation = self.orientation / np.linalg.norm(self.orientation)

        # Generate speaker positions
        self.speaker_array = self._generate_speaker_positions()

    def _generate_speaker_positions(self):
        """
        Generate speaker positions
        """
        positions = []
        for i in range(self.n_speakers):
            position = (self.start_position +
                        i * self.distance * self.orientation)  # Move in the direction of the orientation vector
            positions.append(position)
        return np.array(positions)

    def get_speaker_positions(self):
        """
        Return the speaker positions as a NumPy matrix.
        """
        return self.speaker_array

    def __repr__(self):
        return (f"SpeakerArray(n_speakers={self.n_speakers}, start_position={self.start_position.tolist()}, "
                f"distance={self.distance}, orientation={self.orientation.tolist()})")
