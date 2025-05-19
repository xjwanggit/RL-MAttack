import argparse
import os

def args_parser():
    base_dir = '/home/lxh/wcy/TAaMR-master/data'
    parser = argparse.ArgumentParser("description config of the project")

    # dataset info
    parser.add_argument('--dataset', type=str, default='amazon_women', help='the name of dataset')
    parser.add_argument('--weight', type=str, default='', help='the path to the recommender feature extraction model')
    parser.add_argument('--attack', type=str, default='EXPA', help='the name of attack method')
    parser.add_argument('--GPU', type=int, default=1, choices=[0, 1, 2, 3, 4, 5, 6, 7], help='the No. of the GPU')
    parser.add_argument('--user_number', default=5, type=int, help='the number of the target user')
    parser.add_argument('--save_prob', action='store_true', default=False, help='Whether to save the probability distribution')
    parser.add_argument('--adv', action='store_true', default=False, help='whether use the defense model or not')
    parser.add_argument('--model', type=str, default='VBPR', help='recommender models: VBPR, DVBPR, AMR')
    parser.add_argument('--use_ensemble', action='store_true', default=True, help='Whether to use the perturbation ensemble method or not')
    parser.add_argument('--continue_running', action='store_true', default=False, help='Continue running from the last break out point when dealing with images')


    ## env info
    parser.add_argument('--arch', default='resnet50', help="feature extraction model used by reinforcement learning")
    parser.add_argument('--rec_arch', default='resnet50', help="feature extraction model used by recommender system") # DVBPR uses tensorflow_file to represent
    parser.add_argument('--eps', type=float, default=100, help='HyperParameters used to balance score and SSIM')


    ## store file
    parser.add_argument('--save_path', default='./results', help="the path to save encryption models")
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Whether store the info using wandb or not')
    args = parser.parse_args()

    args.pos_txt = os.path.join(base_dir, args.dataset, 'pos.txt')  # 'the path of the interactive file'
    args.file_path = os.path.join(base_dir, args.dataset, 'original_images', 'images')  # the path of the target picture
    return args