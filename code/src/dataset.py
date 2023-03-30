import cv2
import torch
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lenstronomy.Data.psf import PSF
from colossus.cosmology import cosmology
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.simulation_util import data_configure_simple


class GalaxyDewarperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_pixels, pixel_width, psf_fwhm, z_source, source_model_class):
        # Define member variables
        self.psf_fwhm = psf_fwhm
        self.z_source = z_source
        self.num_pixels = num_pixels
        self.pixel_width = pixel_width
        self.source_model_class = source_model_class
        self.cosmos = cosmology.setCosmology('planck18')
        self.convert = plt.get_cmap(matplotlib.cm.magma)

        # Create a list of image names
        self.galaxy_images = pd.read_csv(dataset_path)['asset_id'].tolist()
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.galaxy_images)

    def _log_transform(self, image):
        # Get the max and min values for log scaler
        vmax = image.max()
        vmin = vmax * 1e-3

        # Scale the values and assign colors based on log scale
        norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        return cv2.cvtColor((self.convert(norm(image.clip(vmin, None).T))[:, :, 2::-1]).astype(np.float32), cv2.COLOR_BGR2GRAY)[None]

    def __getitem__(self, idx):
        # Read the image in grayscale mode
        img = cv2.cvtColor(cv2.imread(self.galaxy_images[idx]), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Set lens parameters
        kwargs_lens = ({
            'theta_E': np.random.uniform(15, 25),
            'gamma': 2 + 0.1 * np.random.randn(),
            'e1':np.random.uniform(-0.05, 0.05),
            'e2':np.random.uniform(-0.05, 0.05),
            'center_x': 0.01 * np.random.randn(),
            'center_y': 0.01 * np.random.randn()
         }, )
        
        # Set the source parameters
        kwargs_source =  [{
            'image': img,
            'center_x': 0,
            'center_y': 0,
            'phi_G': 0,
            'scale': self.pixel_width * (self.cosmos.angularDiameterDistance(1.5) / self.cosmos.angularDiameterDistance(self.z_source))
        }]

        # Generate the lensed image (warped)
        lensed = (ImageModel(
            data_class=ImageData(**data_configure_simple(numPix=self.num_pixels, deltaPix=self.pixel_width)),
            psf_class=PSF(psf_type='GAUSSIAN', fwhm=self.psf_fwhm),
            lens_model_class=LensModel(('PEMD', ), ),
            source_model_class=LightModel(self.source_model_class)
        ).image(kwargs_lens=kwargs_lens, kwargs_source=kwargs_source))

        # Generate the unlensed image (blank lens)
        unlensed = (ImageModel(
            data_class=ImageData(**data_configure_simple(numPix=self.num_pixels, deltaPix=self.pixel_width)),
            psf_class=PSF(psf_type='GAUSSIAN', fwhm=self.psf_fwhm),
            lens_model_class=LensModel(('SIS', ), ),
            source_model_class=LightModel(self.source_model_class)
        ).image(kwargs_lens=({'theta_E': 0}, ), kwargs_source=kwargs_source))
    
        return self._log_transform(lensed), self._log_transform(unlensed)


class GalaxyClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, num_pixels, pixel_width, psf_fwhm, z_source, source_model_class):
        # Define member variables
        self.psf_fwhm = psf_fwhm
        self.z_source = z_source
        self.num_pixels = num_pixels
        self.pixel_width = pixel_width
        self.source_model_class = source_model_class
        self.cosmos = cosmology.setCosmology('planck18')
        self.convert = plt.get_cmap(matplotlib.cm.magma)

        # Load the csv dataset
        self.dataset = pd.read_csv(dataset_path)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.dataset)

    def _log_transform(self, image):
        # Get the max and min values for log scaler
        vmax = image.max()
        vmin = vmax * 1e-3

        # Scale the values and assign colors based on log scale
        norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
        return cv2.cvtColor((self.convert(norm(image.clip(vmin, None).T))[:, :, 2::-1]).astype(np.float32), cv2.COLOR_BGR2GRAY)[None]

    def __getitem__(self, idx):
        # Read the labels
        labels = torch.from_numpy(self.dataset.iloc[idx, 1: -1].numpy().astype(np.float32))

        # Read the image in grayscale mode
        img = cv2.cvtColor(cv2.imread(self.dataset.iloc[idx]['asset_id']), cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Set the source parameters
        kwargs_source =  [{
            'image': img,
            'center_x': 0,
            'center_y': 0,
            'phi_G': 0,
            'scale': self.pixel_width * (self.cosmos.angularDiameterDistance(1.5) / self.cosmos.angularDiameterDistance(self.z_source))
        }]

        # Generate the unlensed image (blank lens)
        unlensed = (ImageModel(
            data_class=ImageData(**data_configure_simple(numPix=self.num_pixels, deltaPix=self.pixel_width)),
            psf_class=PSF(psf_type='GAUSSIAN', fwhm=self.psf_fwhm),
            lens_model_class=LensModel(('SIS', ), ),
            source_model_class=LightModel(self.source_model_class)
        ).image(kwargs_lens=({'theta_E': 0}, ), kwargs_source=kwargs_source))

        return torch.from_numpy(self._log_transform(unlensed)), labels

warped_galaxy_dataset = GalaxyDewarperDataset('/N/slate/lramesh/AI_galaxy_dewarping/data/gz2_train.csv', 256, 0.396217, 0.1, 1.5, ['INTERPOL'])
result = warped_galaxy_dataset.__getitem__(400)
print(result[0].shape)
lensed = (result[0] * 255).astype(np.uint8)
unlensed = (result[1] * 255).astype(np.uint8)
cv2.imwrite('lensed.png', lensed)
cv2.imwrite('unlensed.png', unlensed)