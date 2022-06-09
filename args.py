import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of MLDL project feature classifier with temporal aggregation")

parser.add_argument('--source', type=str, choices=['1', '2', '3'],
                    default='1', help="list of source domains")
parser.add_argument('--target', type=str, choices=['1', '2', '3'], default='1', help="list of target domains")

parser.add_argument('--aggregation', type=str, choices=['avgpool', 'trm'],
                    default='avgpool', help="list of temporal aggregations")

parser.add_argument('--modality', type=str, choices=['RGB', 'Flow'],
                    default='RGB', help="list of modalities")

parser.add_argument('--backbone', type=str, choices=['i3d', 'tsm'],
                    default='i3d', help="list backbones from which features have been extracted")
              