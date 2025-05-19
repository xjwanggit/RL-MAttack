import argparse

def configure():
    parser = argparse.ArgumentParser(description='DDPG hyper-parameters')

    parser.add_argument('-e', '--epoch', default=10000, help="Running episodes.")
    parser.add_argument('-s', '--training_step', default=200)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('-ac_lr', default=5e-4, help='learning rate for actor network')
    parser.add_argument('-cr_lr', default=1e-3, help='learning rate for critic network')
    parser.add_argument('-t', '--tau', default=0.1, help='learning rate for soft-updates')
    parser.add_argument('-gamma', default=0.9)
    parser.add_argument('--epsilon', default=1.0)
    parser.add_argument('--buffer', default=10000, help='experience replay buffer')
    parser.add_argument('--update', default=50, help='update times')
    parser.add_argument('--dataset', nargs='?', default='amazon_men',
                        help='dataset path: amazon_men, amazon_women')
    parser.add_argument('--index', nargs='?', default='8',
                        help='the index of the target image', type=int)
    parser.add_argument('--target_classes', nargs='?', default='770',
                        help='the classes of the target image', type=int)
    parser.add_argument('--original_classes', nargs='?', default='806',
                        help='the classes of the target image', type=int)
    parser.add_argument('--weight_path', default='../../rec_model_weights/amazon_men/original_images/epoch2000.npy',
                        help='the classes of the target image', type=str)
    parser.add_argument('--image_feature_path', default='../../data/amazon_men/original_images/features.npy',
                        help='the classes of the target image', type=str)

    args = parser.parse_args()
    return args