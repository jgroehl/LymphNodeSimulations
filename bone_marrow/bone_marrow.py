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

VOLUME_TRANSDUCER_DIM_IN_MM = 40
VOLUME_PLANAR_DIM_IN_MM = 40
VOLUME_HEIGHT_IN_MM = 40
SPACING = 0.5
RANDOM_SEED = 471
VOLUME_NAME = "MyVolumeName_"+str(RANDOM_SEED)
SAVE_REFLECTANCE = False
SAVE_PHOTON_DIRECTION = False
WAVELENGTH = 808

# If VISUALIZE is set to True, the simulation result will be plotted
VISUALIZE = True


def create_example_tissue():
    """
    This is a very simple example script of how to create a tissue definition.
    It contains a muscular background, an epidermis layer on top of the muscles
    and a blood vessel.
    """
    muscle_dictionary = sp.Settings()
    muscle_dictionary[Tags.PRIORITY] = 1
    muscle_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    muscle_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 100]
    muscle_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.muscle(background_oxy=0.7,
                                                                            blood_volume_fraction=0.01)
    muscle_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    muscle_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    muscle_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    epidermis_dictionary = sp.Settings()
    epidermis_dictionary[Tags.PRIORITY] = 8
    epidermis_dictionary[Tags.STRUCTURE_START_MM] = [0, 0, 0]
    epidermis_dictionary[Tags.STRUCTURE_END_MM] = [0, 0, 0.5]
    epidermis_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.epidermis(0.001)
    epidermis_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    epidermis_dictionary[Tags.ADHERE_TO_DEFORMATION] = False
    epidermis_dictionary[Tags.STRUCTURE_TYPE] = Tags.HORIZONTAL_LAYER_STRUCTURE

    bone_dictionary = sp.Settings()
    bone_dictionary[Tags.PRIORITY] = 4
    bone_dictionary[Tags.STRUCTURE_START_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                    0,
                                                    VOLUME_HEIGHT_IN_MM/2]
    bone_dictionary[Tags.STRUCTURE_END_MM] = [VOLUME_TRANSDUCER_DIM_IN_MM/2,
                                                  VOLUME_PLANAR_DIM_IN_MM,
                                                  VOLUME_HEIGHT_IN_MM/2]
    bone_dictionary[Tags.STRUCTURE_RADIUS_MM] = 15
    bone_dictionary[Tags.MOLECULE_COMPOSITION] = sp.TISSUE_LIBRARY.bone()
    bone_dictionary[Tags.CONSIDER_PARTIAL_VOLUME] = False
    bone_dictionary[Tags.STRUCTURE_TYPE] = Tags.CIRCULAR_TUBULAR_STRUCTURE



    tissue_dict = sp.Settings()
    tissue_dict["muscle"] = muscle_dictionary
    tissue_dict["epidermis"] = epidermis_dictionary
    tissue_dict["bone"] = bone_dictionary
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

fluence = sp.load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                                      sp.Tags.DATA_FIELD_FLUENCE, WAVELENGTH)

initial_pressure = sp.load_data_field(path_manager.get_hdf5_file_save_path() + "/" + VOLUME_NAME + ".hdf5",
                                      sp.Tags.DATA_FIELD_INITIAL_PRESSURE, WAVELENGTH)

xdim, ydim, zdim = np.shape(initial_pressure)
depth_energy = fluence[int(xdim/2), int(ydim/2), :]
depth_energy = (depth_energy / np.max(depth_energy)) * 100

plt.figure(layout="constrained", figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Energy Distribution Visual [logscale]")
plt.imshow(np.log(fluence[:, int(ydim/2), :]).T)
plt.subplot(1, 3, 2)
plt.title("Deposited Energy [logscale]")
plt.imshow(np.log(initial_pressure[:, int(ydim/2), :]).T)
plt.subplot(1, 3, 3)
plt.title("Depth-dependent energy loss")
plt.semilogy(np.linspace(0, len(depth_energy), len(depth_energy)) * SPACING, depth_energy)
plt.yticks([100, 1, 0.01, 0.0001], ["100%", "1%", "0.01%", "0.0001%"])
plt.ylabel("Remaining Laser Energy [%]")
plt.xlabel("Depth [mm]")

plt.show()