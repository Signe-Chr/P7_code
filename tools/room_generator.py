import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

class RoomGenerator:
    def __init__(self,
                 shape: str = "shoebox",
                 desired_rt60: float = None, 
                 material_properties_bounds={"energy_absorption": None, "scattering": None}, 
                 fs=48000, 
                 ray_tracing_params=None,):
        """
        Initialize the RoomGenerator.

        Parameters:
            shape (str): Shape of the room. Options are "shoebox", "t_room", "l_room".
            material_properties_bounds (dict): Dictionary with 'energy_absorption' and 'scattering' bounds.
            fs (int): Sampling frequency.
            ray_tracing_params (dict): Dictionary with 'receiver_radius', 'n_rays', and 'energy_thres'.
        """
        assert shape in ["shoebox", "t_room", "l_room"], f"Invalid room shape: {shape}. Must be one of ['shoebox', 't_room', 'l_room']."
        self.shape = shape
        self.desired_rt60 = desired_rt60
        self.material_properties_bounds = material_properties_bounds
        self.fs = fs
        self.use_ray_tracing = True if ray_tracing_params is not None else False
        self.ray_tracing_params = ray_tracing_params


    # TODO: Needs a v2
    def _compose_random_shoebox(self, np_random_generator: np.random.Generator, min_width = 3.0, max_width = 10.0, min_length = 3.0, max_length = 10.0, min_extrude = 2.0, max_extrude = 5.0, **kwargs) -> tuple[np.ndarray, float]:
        """
        Compose a shoebox room.
        """
        logger.info("Generating random shoebox room")
        logger.info(f"Bounds: {min_width}, {max_width}, {min_length}, {max_length}, {min_extrude}, {max_extrude}")
        
        # Generate random width and length
        width = np_random_generator.uniform(min_width, max_width)
        length = np_random_generator.uniform(min_length, max_length)
        logger.info(f"Width: {width}, Length: {length}")
        corners = np.array([[0, 0], [0, width], [length, width], [length, 0]])
        
        extrude_height = np_random_generator.uniform(min_extrude, max_extrude)
        return corners, extrude_height
    
    
    # FIXME: Make this
    def _compose_random_troom(self,) -> tuple[np.ndarray, float]:
        """
        Compose a T-shaped room.
        """
        raise NotImplementedError("T-shaped room composition is not implemented.")
        return corners, extrude_height
    
    
    # FIXME: Make this
    def _compose_random_lroom(self,) -> tuple[np.ndarray, float]:
        """
        Compose a L-shaped room.
        """
        raise NotImplementedError("L-shaped room composition is not implemented.")
        return corners, extrude_height
    
    
    def generate_room(self, seed=None, room_bounds=None) -> tuple[pra.Room, dict]:
        """
        Generate a random room with the specified room_bounds.
        """
        
        # Set randomness generator seed
        random_gen = np.random.default_rng(seed=seed)
        
        
        # Generate random corners
        if self.shape == "shoebox":
            corners, extrude_height = self._compose_random_shoebox(np_random_generator=random_gen, **room_bounds)
        elif self.shape == "t_room":
            corners, extrude_height = self._compose_random_troom()
        elif self.shape == "l_room":
            corners, extrude_height = self._compose_random_lroom()

        # TODO: Implement it for ray tracing https://github.com/LCAV/pyroomacoustics/blob/master/examples/room_from_rt60.py
        if (self.desired_rt60 is not None):
            # If desired RT60 is set, use it to calculate material properties
            logger.info(f"Desired RT60: {self.desired_rt60}")
            room_dim = [corners[2,0], corners[2,1], extrude_height,] # x, y, z
            e_absorption, max_order = pra.inverse_sabine(rt60=self.desired_rt60, room_dim=room_dim,)
            logger.info(f"Calculated e_absorption: {e_absorption}, max_order: {max_order}")
            
            # Create the room
            if not self.use_ray_tracing:
                room = pra.Room.from_corners(
                    corners.T,  
                    fs=self.fs,
                    materials=pra.Material(e_absorption), 
                    max_order=max_order if max_order < 10 else 10,
                    use_rand_ism=True,
                    air_absorption=True, 
                )
            elif self.use_ray_tracing:
                # If ray tracing is used, set the maximum order for ray tracing
                room = pra.Room.from_corners(
                    corners.T, 
                    fs=self.fs,
                    materials=pra.Material(e_absorption),
                    max_order=3, 
                    ray_tracing=self.use_ray_tracing, 
                    air_absorption=True, 
                )
            else: raise ValueError("Invalid room generation parameters.")
            
        else:
            # If material properties are not set, use default values
            if self.material_properties_bounds is None:
                material_properties = {
                    'energy_absorption': random_gen.uniform(0.4, 0.6),
                    'scattering': random_gen.uniform(0.4, 0.6)
                }
            else:
                material_properties = {
                    'energy_absorption': random_gen.uniform(self.material_properties_bounds['energy_absorption'][0], self.material_properties_bounds['energy_absorption'][1]),
                    'scattering': random_gen.uniform(self.material_properties_bounds['scattering'][0], self.material_properties_bounds['scattering'][1])
                }
            logger.info(f"Material properties: {material_properties}")
            
            # Create the room with the specified material properties
            material = pra.Material(energy_absorption=material_properties['energy_absorption'], scattering=material_properties['scattering'])
            room = pra.Room.from_corners(corners.T, materials=material, fs=self.fs, ray_tracing=self.use_ray_tracing, air_absorption=True, max_order=10)
        
        # If ray tracing parameters are not set, use default values
        if self.use_ray_tracing:
            logger.info("Using ray tracing")
            ray_tracing_params = {
                'receiver_radius': self.ray_tracing_params['receiver_radius'],
                'n_rays': self.ray_tracing_params['n_rays'],
                'energy_thres': self.ray_tracing_params['energy_thres']
            }
        

        # Extrude the room to create a 3D room
        room.extrude(extrude_height)
        
        if self.use_ray_tracing:
            logger.info(f"Ray tracing parameters: {ray_tracing_params}")
            room.set_ray_tracing(
                receiver_radius=ray_tracing_params['receiver_radius'],
                # n_rays=ray_tracing_params['n_rays'], # FIXME: REMEMBER TO UNCOMMENT THIS
                # energy_thres=ray_tracing_params['energy_thres'],
            )
        
        dimensions = {
            'width': room.get_bbox()[0,1],
            'length': room.get_bbox()[1,1],
            'height': extrude_height
        }

        return room, dimensions
    
if __name__ == "__main__":
    room_generator = RoomGenerator()
    room1, room1_dimensions = room_generator.generate_room()
    print(room1_dimensions['width'], room1_dimensions['length'], room1_dimensions['height'])
    fig, ax = room1.plot()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    plt.show()
    
    room2, room2_dimensions = room_generator.generate_room()
    print(room2_dimensions['width'], room2_dimensions['length'], room2_dimensions['height'])
    fig, ax = room2.plot()
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 10])
    plt.show()
