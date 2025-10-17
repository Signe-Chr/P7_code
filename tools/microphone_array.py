import numpy as np

class MicrophoneArray:
    def __init__(self, rows, cols, start_position, distance, x_orientation=np.array([1, 0, 0]), y_orientation=np.array([0, 1, 0])):
        """
        Initialize a 2D microphone array (grid).

        Parameters:
            rows (int): Number of rows in the microphone array.
            cols (int): Number of columns in the microphone array.
            start_position (list or np.array): Position of the first microphone [x, y, z].
            distance (float): Distance between adjacent microphones.
            x_orientation (list or np.array, optional): Orientation along the x-axis (default: [1,0,0]).
            y_orientation (list or np.array, optional): Orientation along the y-axis (default: [0,1,0]).
        """
        self.rows = rows
        self.cols = cols
        self.start_position = np.array(start_position)
        self.distance = distance
        self.x_orientation = np.array(x_orientation)
        self.y_orientation = np.array(y_orientation)

        # Normalize orientation vectors
        self.x_orientation = self.x_orientation / np.linalg.norm(self.x_orientation)
        self.y_orientation = self.y_orientation / np.linalg.norm(self.y_orientation)

        # Generate microphone positions as a matrix
        self.microphone_array = self._generate_microphone_positions()

    def _generate_microphone_positions(self):
        """
        Generate a 2D grid of microphone positions as a (rows*cols, 3) matrix.
        """
        positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                position = (self.start_position +
                            r * self.distance * self.y_orientation +  # Move in y-direction
                            c * self.distance * self.x_orientation)   # Move in x-direction
                positions.append(position)
        return np.array(positions)

    def get_microphone_positions(self):
        """
        Return the microphone positions as a NumPy matrix.
        """
        return self.microphone_array

    def __repr__(self):
        return (f"MicrophoneArray(rows={self.rows}, cols={self.cols}, start_position={self.start_position.tolist()}, "
                f"distance={self.distance}, x_orientation={self.x_orientation.tolist()}, "
                f"y_orientation={self.y_orientation.tolist()})")

if __name__ == "__main__":
    # Example usage: Create a 4x4 microphone grid
    rows = 4
    cols = 4
    start_position = [0, 0, 0]  # Bottom-left corner
    distance = 0.1
    x_orientation = [1, 0, 0]  # Along x-axis
    y_orientation = [0, 1, 0]  # Along y-axis

    mic_array = MicrophoneArray(rows, cols, start_position, distance, x_orientation, y_orientation)
    print(mic_array.get_microphone_positions())
