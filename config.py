import torchvision

NETS = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
    'resnet152': torchvision.models.resnet152,
    'densenet121': torchvision.models.densenet121,
    'densenet161': torchvision.models.densenet161,
    'densenet169': torchvision.models.densenet169,
    'densenet201': torchvision.models.densenet201
}


DSETS = {'imagenet': {'src': 'Datasets/imagenet',
                         'stats': (.485, .456, .406, .229, .224, .225),
                         'nr_classes': 1000,
                         'size': {'train': 50000,
                                  'test': 1}},
            'caltech': {'src': 'Datasets/caltech/256_ObjectCategories',
                        'stats': (.517, .5015, .4736, .315, .3111, .324),
                        'nr_classes': 256,
                        'size': {'train': 21424,
                                 'test': 9183}},
            'sun': {'src': 'Datasets/sun',
                    'stats': (.472749174938, .461143867394, .432053035945, .265962445818, .263875783693, .289179359433),
                    'nr_classes': 100,
                    'size': {'train': 4994,
                             'test': 1}},
            'indoors': {'src': 'Datasets/indoors',
                        'stats': (.485, .456, .406, .229, .224, .225),  # imagenet stats
                        'nr_classes': 67,
                        'size': {'train': 10934,
                                 'test': 4686}}
         }


DATA_DIR = 'data'
SPLITTING_DIR = 'splittings'
FEATURE_DIR = 'features'
NET_DIR = 'nets'
LABEL_DIR = 'labels'
RESULT_DIR = 'results'
NUM_EXPS = 1