import imgaug.augmenters as iaa

DEFAULT_AUGMENTATION = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                       iaa.Flipud(0.5),  # vertically flip 50% of all images,
                                       # iaa.PadToFixedSize(*run_params['image_resolution'], position='center')
                                       ])

ADVANCED_AUGMENTATION = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                        iaa.Flipud(0.5),  # vertically flip 50% of all images,
                                        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),  # Change brightness
                                        iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),
                                        # Zoom in / zoom out
                                        iaa.Sometimes(0.5, iaa.Affine(rotate=(-20, 20), fit_output=True)),  # Rotate
                                        # iaa.PadToFixedSize(*run_params['image_resolution'], position='center')
                                        ])

COMMON_PARAMS = {
    'data_source': 'XR_HAND_PHOTOSHOP',
    'pipeline': {
        'hist_equalisation': True,
        'otsu_filter': False,
        'adaptive_hist_equilization': False,
    },
    'random_seed': [42, 4242, 424242, 42424242],
}

MODEL_SPECIFIC_PARAMS = {
    'BaselineAutoencoder': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 1000,
        'batch_normalisation': True,
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    },
    'BottleneckAutoencoder': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': True,
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    },
    'SkipConnection': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': True,
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    },
    'VAE': {
        'augmentation': ADVANCED_AUGMENTATION,
        'batch_size': 16,
        'image_resolution': (128, 128),
        'num_epochs': 300,
        'batch_normalisation': False,
        'soft_delta': 0.05,
        'z_dim': 512,
        'h_dim': 18432,
        'lr': 0.0001,

    },
    'DCGAN': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 128,
        'image_resolution': (512, 512),
        'num_epochs': 300,
        'batch_normalisation': True,
        'spectral_normalisation': True,
        'soft_labels': True,
        'glr': 0.001,
        'dlr': 0.00001,
        'soft_delta': 0.05,
        'z_dim': 2048,
        'checkpoint_frequency': 100,
    },
    'AlphaGan': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 16,
        'image_resolution': (128, 128),
        'num_epochs': 300,
        'glr': 0.001,
        'dlr': 0.00001,
        'checkpoint_frequency': 100,
    },
    'SAGAN': {
        'augmentation': DEFAULT_AUGMENTATION,
        'batch_size': 16,
        'image_resolution': (128, 128),
        'num_epochs': 300,
        'glr': 0.001,
        'dlr': 0.00001,
        'adv_loss': 'hinge',
        'checkpoint_frequency': 100,
    }
}
