# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

import matplotlib.pyplot as plt
from simpa import Tags
import simpa as sp
import numpy as np

# FIXME temporary workaround for newest Intel architectures
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

path_manager = sp.PathManager()

VOLUME_TRANSDUCER_DIM_IN_MM = 15
VOLUME_PLANAR_DIM_IN_MM = 15
VOLUME_HEIGHT_IN_MM = 15
SPACING = 0.5
RANDOM_SEED = 471
VOLUME_NAME = "MyVolumeName_"+str(RANDOM_SEED)
SAVE_REFLECTANCE = False
SAVE_PHOTON_DIRECTION = False
WAVELENGTH = 808

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True

RADIUS = 7.5


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    background_dictionary = sp.Settings()
    background_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(1e-4, 1e-4, 0.9)
    background_dictionary[Tags.STRUCTURE_TYPE] = Tags.BACKGROUND


    vessel_1_dictionary = sp.Settings()
    vessel_1_dictionary[Tags.PRIORITY] = 3
    vessel_1_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                    VOLUME_PLANAR_DIM_IN_MM/2,
                                                    RADIUS]
    vessel_1_dictionary[Tags.STRUCTURE_RADIUS_MM] = RADIUS
    vessel_1_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.constant(0.24, 3.37/(1-0.92), 0.92)
    vessel_1_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    vessel_1_dictionary[Tags.STRUCTURE_TYPE] = Tags.SPHERICAL_STRUCTURE

    tissue_dict = sp.Settings()
    tissue_dict[Tags.BACKGROUND] = background_dictionary
    tissue_dict["vessel_1"] = vessel_1_dictionary
    return tissue_dict


# Seed the numpy random configuration prior to creating the global_settings file in
# order to ensure that the same volume
# is generated with the same random seed every time.

np.random.seed(RANDOM_SEED)

general_settings = {
    # These parameters set the general properties of the simulated volume
    Tags.RANDOM_SEED: RANDOM_SEED,
    Tags.VOLUME_NAME: VOLUME_NAME,
    Tags.SIMULATION_PATH: path_manager.get_hdf5_file_save_path(),
    Tags.SPACING_MM: SPACING,
    Tags.DIM_VOLUME_Z_MM: VOLUME_HEIGHT_IN_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_TRANSDUCER_DIM_IN_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_PLANAR_DIM_IN_MM,
    Tags.WAVELENGTHS: [WAVELENGTH],
    Tags.DO_FILE_COMPRESSION: True
}

settings = sp.Settings(general_settings)

settings.set_volume_creation_settings({
    Tags.SIMULATE_DEFORMED_LAYERS: True,
    Tags.STRUCTURES: create_example_tissue()
})
settings.set_optical_settings({
    Tags.OPTICAL_MODEL_NUMBER_PHOTONS: 5e7,
    Tags.OPTICAL_MODEL_BINARY_PATH: path_manager.get_mcx_binary_path(),
    Tags.COMPUTE_DIFFUSE_REFLECTANCE: SAVE_REFLECTANCE,
    Tags.COMPUTE_PHOTON_DIRECTION_AT_EXIT: SAVE_PHOTON_DIRECTION
})
settings["noise_model_1"] = {
    Tags.NOISE_MEAN: 1.0,
    Tags.NOISE_STD: 0.1,
    Tags.NOISE_MODE: Tags.NOISE_MODE_MULTIPLICATIVE,
    Tags.DATA_FIELD: Tags.DATA_FIELD_INITIAL_PRESSURE,
    Tags.NOISE_NON_NEGATIVITY_CONSTRAINT: True
}

if not SAVE_REFLECTANCE and not SAVE_PHOTON_DIRECTION:
    pipeline = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapter(settings),
        sp.GaussianNoise(settings, "noise_model_1")
    ]
else:
    pipeline = [
        sp.ModelBasedVolumeCreationAdapter(settings),
        sp.MCXAdapterReflectance(settings),
    ]


class ExampleDeviceSlitIlluminationLinearDetector(sp.PhotoacousticDevice):
    """
    This class represents a digital twin of a PA device with a slit as illumination next to a linear detection geometry.

    """

    def __init__(self):
        super().__init__(device_position_mm=np.asarray([VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                        VOLUME_PLANAR_DIM_IN_MM/2, 0]))
        # self.set_detection_geometry(sp.LinearArrayDetectionGeometry())
        self.add_illumination_geometry(sp.GaussianBeamIlluminationGeometry(beam_radius_mm=VOLUME_TRANSDUCER_DIM_IN_MM/2))


device = ExampleDeviceSlitIlluminationLinearDetector()

sp.simulate(pipeline, settings, device)

if Tags.WAVELENGTH in settings:
    WAVELENGTH = settings[Tags.WAVELENGTH]

initial_pressure = sp.load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                                      sp.Tags.DATA_FIELD_FLUENCE, WAVELENGTH)

xdim, ydim, zdim = np.shape(initial_pressure)
depth_energy = initial_pressure[int(xdim/2), int(ydim/2), :]
depth_energy = (depth_energy / np.max(depth_energy)) * 100

plt.figure(layout="constrained", figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Energy Distribution Visual [logscale]")
plt.imshow(np.log(initial_pressure[:, int(ydim/2), :]).T)
plt.subplot(1, 2, 2)
plt.title("Depth-dependent energy loss")
plt.semilogy(np.linspace(0, len(depth_energy), len(depth_energy)) * SPACING, depth_energy)
plt.yticks([100, 1, 0.01, 0.0001], ["100%", "1%", "0.01%", "0.0001%"])
plt.ylabel("Remaining Laser Energy [%]")
plt.xlabel("Depth [mm]")

plt.show()