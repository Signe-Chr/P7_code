import numpy as np

class MicrophoneCircle:
    def __init__(self, center: list | np.ndarray, radius: float, n_mics: int, sphere=False) -> np.ndarray:
        """
        Initialize a circle of microphones around a defined center.

        Parameters:
            center (list or np.array): Position of the microhpne circle center [x, y, z]
            radius (float): Distance from center to microphones
            n_mics (int): Number of microphones in the circle
        """
        self.center = center
        self.radius = radius
        self.n_mics = n_mics

        # Generate microphone positions as a matrix
        if sphere: self.microphone_array = self._generate_microphone_positions_sphere()
        else: self.microphone_array = self._generate_microphone_positions()

    def _generate_microphone_positions(self, add_center=False):
        """
        Generate a circle of microphone positions as a (rows*cols, 3) matrix.
        """
        positions = []
        
        # Add the center microphone as the first index
        if add_center: positions.append(self.center)
        
        # Calculate the positions of microphones in a circle
        angles = np.linspace(0, 2 * np.pi, self.n_mics, endpoint=False) # Endpoint ensures the last mic is not sampled at the same position as the first, 0deg = 360deg
        for angle in angles:
            x = self.center[0] + self.radius * np.cos(angle)
            y = self.center[1] + self.radius * np.sin(angle)
            z = self.center[2]
            positions.append([x, y, z])
        
        return np.array(positions)
    
    def _generate_microphone_positions_sphere(self, add_center=False):
        """
        Generate a sphere of microphone positions as a (n_mics, 3) matrix.
        """
        positions = []
        
        # Add the center microphone as the first index
        if add_center: positions.append(self.center)
        
        # Generate spherical coordinates for microphone positions
        phi = np.linspace(0, np.pi, int(np.sqrt(self.n_mics)))  # Elevation angles
        theta = np.linspace(0, 2 * np.pi, int(np.sqrt(self.n_mics)), endpoint=False)  # Azimuthal angles
        
        # Create a grid of spherical coordinates
        phi, theta = np.meshgrid(phi, theta)
        phi = phi.flatten()
        theta = theta.flatten()
        
        # Ensure the number of microphones matches n_mics
        if len(phi) > self.n_mics - 1:
            phi = phi[:self.n_mics - 1]
            theta = theta[:self.n_mics - 1]
        
        # Convert spherical coordinates to Cartesian coordinates
        for p, t in zip(phi, theta):
            x = self.center[0] + self.radius * np.sin(p) * np.cos(t)
            y = self.center[1] + self.radius * np.sin(p) * np.sin(t)
            z = self.center[2] + self.radius * np.cos(p)
            positions.append([x, y, z])
        
        return np.array(positions)

    def plot_microphone_positions(self, plot=True, fig=None, ax=None):
        """
        Plot the microphone positions in 3D.
        """
        import matplotlib.pyplot as plt
        
        # Create a new figure and axis if not provided
        create_figure = fig is None and ax is None
        if create_figure:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot the microphones
        if not create_figure: print("Using provided figure and axis for 'microphone_circle'.")
        ax.scatter(self.microphone_array[:, 0], self.microphone_array[:, 1], self.microphone_array[:, 2], c='r', marker='o')
        
        if create_figure:
            # Set labels
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        
        if plot: 
            # Plot fig
            plt.show()
        
        return fig, ax
    
    def get_microphone_positions(self):
        """
        Return the microphone positions as a NumPy matrix.
        """
        return self.microphone_array

    def __repr__(self):
        return (f"MicrophoneCircle(center={self.center}, radius={self.radius}, n_mics={self.n_mics})")


if __name__ == "__main__":
    # Example usage: Create a 100 microphone circle
    
    center = [0, 0, 5] # x, y, z
    radius = 50.0
    n_mics = 20

    mic_array = MicrophoneCircle(center=center, radius=radius, n_mics=n_mics, sphere=False)
    mic_array.plot_microphone_positions()
    print(mic_array)
    
    # Print the microphone positions
    print(mic_array.get_microphone_positions())
