import os
import time
import argparse
import datetime

import pkbar
import torch
from torch.utils.tensorboard import SummaryWriter

from src import model
from src import dataset


class Trainer:
    def __init__(self, device, num_pixels, pixel_width, psf_fwhm, z_source, source_model_class):
        # Define member variables
        self.device = device
        self.psf_fwhm = psf_fwhm
        self.z_source = z_source
        self.num_pixels = num_pixels
        self.pixel_width = pixel_width
        self.source_model_class = [source_model_class]

        # Define the train dataloader
        self.train_loader = torch.utils.data.DataLoader(
            dataset.GalaxyDewarperDataset(
                os.path.join(FLAGS.dataset_path, 'gz2_train.csv'),
                self.num_pixels, self.pixel_width, self.psf_fwhm, self.z_source, self.source_model_class),
            batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers)

        # Define the val dataloader
        self.val_loader = torch.utils.data.DataLoader(
            dataset.GalaxyDewarperDataset(
                os.path.join(FLAGS.dataset_path, 'gz2_val.csv'),
                self.num_pixels, self.pixel_width, self.psf_fwhm, self.z_source, self.source_model_class),
            batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers)

        # Define the autoencoder model
        self.model = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=32, pretrained=False
        ).to(self.device)

        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), FLAGS.learning_rate, weight_decay=1e-5)

        # Define the loss function
        self.loss_fn = torch.nn.MSELoss()

        # Create a progress bar object
        self.progress_bar = None

    def _learning_step(self, data, subset, step=None):
        '''
        Implements the train or val cycle for a batch
        '''
        # Load the demand matrix input
        input = data[0].to(self.device)
        gt = data[1].to(self.device)

        # Perform forward pass to get the reconstructed demand matrix
        pred = self.model(input)

        # Compute the loss
        loss = self.loss_fn(pred, gt)

        # Extra steps for the train cycle
        if subset == 'train':
            # Perform backpropagation
            loss.backward()

            # Update the progress bar
            self.progress_bar.update(step, values=[('train_mse_loss', loss.item())])

        return loss.item()

    def train(self):
        '''
        Implements the training script
        '''
        # Create the tensorboard summary writer
        run_folder = f'runs/autoencoder/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        writer = SummaryWriter(run_folder)

        # Execute the training loop
        best_val_loss = float('inf')
        for epoch in range(FLAGS.epochs):
            # Create a progress bar object
            self.progress_bar = pkbar.Kbar(
                target=len(self.train_loader),
                epoch=epoch,
                num_epochs=FLAGS.epochs,
                width=8,
                always_stateful=False
            )

            # Execute the train cycle
            train_loss = 0
            self.model.train()
            for step, data in enumerate(self.train_loader):
                # Zero out the gradients before forward pass
                self.optimizer.zero_grad()

                # Perform forward pass and backpropagation
                train_loss += self._learning_step(data, 'train', step)

                # Update the weights
                self.optimizer.step()

            # Aggregate the train loss
            train_loss /= (step + 1)

            # Execute the val cycle if required
            val_loss = 0
            self.model.eval()
            with torch.no_grad():
                for step, data in enumerate(self.val_loader):
                    val_loss += self._learning_step(data, 'val')

            # Aggregate the val loss
            val_loss /= (step + 1)

            # If the val loss reduces, update the best val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                # Store the checkpoint
                torch.save(self.model.state_dict(), os.path.join(
                    run_folder, f'model_{epoch}_{best_val_loss:.4f}.pth'))

            # Write the tensorboard scalars
            writer.add_scalar(f'mse_reconstruction_loss/train', train_loss, epoch + 1)
            writer.add_scalar(f'mse_reconstruction_loss/val', val_loss, epoch + 1)

            # Update the progress bar
            self.progress_bar.add(1, values=[('val_mse_loss', val_loss)])

        # Print the final validation loss
        print(f'\nTraining completed\nBest val loss: {best_val_loss}')


def main():
    # Define the trainer object
    trainer = Trainer(torch.device('cuda'))

    # Train the u-net autoencoder model
    trainer.train()


def parse_arguments():
    '''
    Parse command-line arguments
    '''
    parser = argparse.ArgumentParser(description='Train the u-net autoencoder model to dewarp galaxies')

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/N/slate/lramesh/AI_galaxy_dewarping/data',
        help='''Path to the Galaxy Zoo dataset'''
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='''Batch size used in training'''
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=12,
        help='''Number of workers for the dataloader'''
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='''Number of training iterations'''
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='''Learning rate of the optimizer'''
    )

    parser.add_argument(
        '--num_pixels',
        type=int,
        default=256,
        help='''Width and height of the input images'''
    )

    parser.add_argument(
        '--pixel_width',
        type=float,
        default=0.396217,
        help='''pixel width of the *produced* images in arcsec'''
    )

    parser.add_argument(
        '--psf_fwhm',
        type=float,
        default=0.1,
        help='''Full width half maximum of the Gaussian point spread function of the observation'''
    )

    parser.add_argument(
        '--z_source',
        type=float,
        default=1.5,
        help='''Redshift at which to place the source galaxy'''
    )

    parser.add_argument(
        '--source_model_class',
        type=str,
        default='INTERPOL',
        help='''Source model class'''
    )

    return parser.parse_args()


if __name__ == '__main__':
    # Parse arguments
    FLAGS = parse_arguments()

    # Enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Train the u-net autoencoder model
    main()