import numpy as np
from scipy.spatial.transform import Rotation

class Phone:
    def __init__(self, position: list | np.ndarray = [0,0,0], orientation: list | np.ndarray = [0,0,0], unit: str = "mm"):
        """
        Initialize a phone speaker system.
        
        Parameters:
            position (list or np.array, optional): Global position of the phone in mm.
                                                   Default is [0, 0, 0] (origin).
                                                   
            orientation (list or np.array, optional): Euler angles (roll, pitch, yaw) in degrees.
                                                      Default is [0, 0, 0] (no rotation).
            
            unit (str, optional): The measuring unit used for the phone.
                                  Default is "mm".
                                  The allowed units are "mm", "cm" and "m".
        """
        
        # Set position
        self.position = np.array(position, dtype=float)
        
        # Set orientation (Euler angles in degrees)
        self.orientation = np.radians(orientation)  # Convert to radians
        
        # Define unit dict
        self.unit_dict = {"mm": 1, "cm": 0.1, "m": 0.001}
        assert unit in self.unit_dict, f"Unit '{unit}' is not supported. Choose from {list(self.unit_dict.keys())}."
        self.unit = unit
        
        # Define phone dimensions (height, width, depth)
        self.phone_dimensions = np.array([161, 76, 9])*self.unit_dict[unit]
        self.phone_center = self.phone_dimensions / 2
        
        # Compute rotation matrix
        self.rotation_matrix = Rotation.from_euler('xyz', self.orientation).as_matrix()  # 3x3 matrix

        # Generate speaker and microphone positions
        self.speaker_array = self._generate_speaker_positions()
        self.mic_array = self._generate_mic_positions()
        # Compute the position of the ear microphone, which represents a virtual mic placed near the ear.
        # This mic is intentionally inserted at the beginning of the mic_array for specific downstream processing.
        self.ear_mic = self._generate_ear_mic_position(ear_offset=50.0)
        self.mic_array = np.insert(self.mic_array, 0, self.ear_mic, axis=0)
        

    def _generate_speaker_positions(self):
        """
        Generate speaker positions
        """
        h, w, d = self.phone_dimensions
        relative_speaker_positions = np.array([
            [h*0.0, w*0.2, d*0.5], 
            [h,     w*0.5, d*0.0], 
            [h,     w*0.8, d*0.5]
        ])
        
        # Apply rotation to relative positions (relative to the phone center)
        speaker_positions = []
        for pos in relative_speaker_positions:
            rotated_position = self.rotation_matrix @ (pos - self.phone_center)  # Apply rotation relative to center
            global_position = rotated_position + self.position  # Translate to global position
            speaker_positions.append(global_position)
        
        return np.array(speaker_positions)


    def get_speaker_positions(self):
        """
        Return the speaker positions as a NumPy matrix.
        """
        return self.speaker_array
    
    
    def _generate_mic_positions(self):
        """
        Generate microphone positions
        """
        h, w, d = self.phone_dimensions
        relative_mic_positions = np.array([
            [h,      w*0.3,  d*0.5], 
            [h*0.65, w*0.7,  d*0.0], 
            [h*0.0,  w*0.55, d*0.5]
        ])
        
        # Apply rotation to relative positions (relative to the phone center)
        mic_positions = []
        for pos in relative_mic_positions:
            rotated_position = self.rotation_matrix @ (pos - self.phone_center)  # Apply rotation relative to center
            global_position = rotated_position + self.position  # Translate to global position
            mic_positions.append(global_position)
        
        return np.array(mic_positions)
    
    
    def _generate_ear_mic_position(self, ear_offset: float = 50.0):
        """
        Generate ear microphone position
        
        Parameters:
            ear_offset (float): Offset from the speaker position to the ear microphone position, given in mm (method will auto fix unit to match phone dims).
                                Default is 50.0 mm.
        
        """

        h, w, d = self.phone_dimensions
        orthogonal_speaker = np.array([h, w*0.5, d*0.0]) # Matches the top-middle speaker position (loudspeaker 2 [idx=1])
        
        ear_offset = ear_offset * self.unit_dict[self.unit]  # Convert to the same unit as the phone dimensions
        relative_ear_mic_position = orthogonal_speaker + np.array([0, 0, -ear_offset])
        
        # Apply rotation to relative positions (relative to the phone center)
        rotated_position = self.rotation_matrix @ (relative_ear_mic_position - self.phone_center)
        global_position = rotated_position + self.position
        
        return global_position
    
    
    def get_mic_positions(self):
        """
        Return the microphone positions as a NumPy matrix.
        """
        return self.mic_array
    
    
    def get_phone_info(self):
        """
        Return all phone info as a dict.
        """
        return {
            'name': self.__class__.__name__,
            'phone_dimensions': self.phone_dimensions,
            'phone_center': self.phone_center,
            'global_position': self.position,
            'orientation': self.orientation,
            'global_speaker_positions': self.speaker_array,
            'global_mic_positions': self.mic_array,
            'rotation_matrix': self.rotation_matrix,
        }
    
    
    def plot_phone(self, plot=True, fig=None, ax=None, ax_lim:int=None):
        """Plot the phone, speakers, and microphones in 3D with fixed axes."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Create a new figure and axis if not provided
        create_figure = fig is None and ax is None
        if create_figure:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot speakers
        ax.scatter(self.speaker_array[:,0], self.speaker_array[:,1], self.speaker_array[:,2], c='r', marker='o', label='Speakers')

        # Plot microphones
        ax.scatter(self.mic_array[:,0], self.mic_array[:,1], self.mic_array[:,2], c='b', marker='^', label='Microphones')
        
        # Plot center
        ax.scatter(self.position[0], self.position[1], self.position[2], c='k', marker='x', label='Phone Center')
        
        # Phone body as a wireframe cube
        h, w, d = self.phone_dimensions
        phone_corners = np.array([
            [0, 0, 0], [h, 0, 0], [h, w, 0], [0, w, 0],  # Bottom face
            [0, 0, d], [h, 0, d], [h, w, d], [0, w, d]   # Top face
        ]) - self.phone_center
        
        # Apply rotation to phone corners
        # Rotate the phone corners relative to the center
        rotated_phone_corners = (self.rotation_matrix @ phone_corners.T).T
        
        # Translate to global position
        global_phone_corners = rotated_phone_corners + self.position
        
        # Define edges of the cube
        edges = [
            [global_phone_corners[i] for i in [0, 1, 2, 3]],  # Bottom
            [global_phone_corners[i] for i in [4, 5, 6, 7]],  # Top
            [global_phone_corners[i] for i in [0, 1, 5, 4]],  # Side
            [global_phone_corners[i] for i in [2, 3, 7, 6]],  # Side
            [global_phone_corners[i] for i in [0, 3, 7, 4]],  # Side
            [global_phone_corners[i] for i in [1, 2, 6, 5]],  # Side
        ]
        
        # Add the phone body as a wireframe cube
        ax.add_collection3d(Poly3DCollection(edges, alpha=0.2, linewidths=1, edgecolors='k'))
        
        # Add quiver pointing in the height direction
        height_orientation_vector = self.rotation_matrix @ np.array([1, 0, 0])
        ax.quiver(
            self.position[0], self.position[1], self.position[2], # Origin
            height_orientation_vector[0], height_orientation_vector[1], height_orientation_vector[2], # Orientation
            length=50, color="r", linewidth=2, arrow_length_ratio=0.2, label='Height Orientation Vector', # Length of the arrow
        )
        
        # Add quiver pointing in the width direction
        width_orientation_vector = self.rotation_matrix @ np.array([0, 1, 0])
        ax.quiver(
            self.position[0], self.position[1], self.position[2], # Origin
            width_orientation_vector[0], width_orientation_vector[1], width_orientation_vector[2], # Orientation
            length=25, color="g", linewidth=2, arrow_length_ratio=0.2, label='Width Orientation Vector', # Length of the arrow
        )
        
        if create_figure:
            # Set axes
            x, y, z = self.position
            if ax_lim is None: ax_lim = 150
            ax.set_xlim([-ax_lim, ax_lim] + x)
            ax.set_ylim([-ax_lim, ax_lim] + y)
            ax.set_zlim([-ax_lim, ax_lim] + z)
            
            ax.set_xlabel(f"X ({self.unit})")
            ax.set_ylabel(f"Y ({self.unit})")
            ax.set_zlabel(f"Z ({self.unit})")
            
            ax.set_title("3D Phone Model")
            ax.legend()
        
        # Plot fig
        if plot:
            plt.show()
        
        return fig, ax

    def __repr__(self):
        return (f"PhoneSpeaker(position={self.position.tolist()}, "
                f"orientation={self.orientation.tolist()})")


if __name__ == "__main__":
    # phone = PhoneSpeaker(np.array([0,0,0]))
    phone = Phone(np.array([20, 30, 40]), np.array([45,-90,0])) # pos in mm, orientation in degrees (roll, pitch, yaw)
    
    #print(phone.get_phone_info())
    phone.plot_phone(plot=True)