import argparse
from recommendation.recommender_utils.Solver import Solver
from time import time
import os



def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--dataset', nargs='?', default='amazon_women',
                        help='dataset path')
    parser.add_argument('--experiment_name', nargs='?', default='original_images',
                        help='original_images, fgsm_***, cw_***, pgd_***')
    parser.add_argument('--model', nargs='?', default='DeepStyle',
                        help='recommender models: VBPR, AMR, DVBPR, DeepStyle')
    parser.add_argument('--emb1_K', type=int, default=64, help='size of embeddings')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size') # VBPR用512 DVBPR用1024 DeepStyle用2048
    parser.add_argument('--validation_batch_size', type=int, default=2048, help='the batch size that used in the validation set')
    parser.add_argument('--lr', nargs='?', default='[0.01,1e-3, 1e-3, 1e-3, 1e-4]', help='learning rate') # VBPR [0.01,1e-4,1e-3]  DVBPR[0.01,1e-3,1e-3] DeepStyle [0.01,1e-3, 1e-3, 1e-3, 1e-4]
    parser.add_argument('--verbose', type=int, default=500, help='verbose')
    parser.add_argument('--epoch', type=int, default=4000, help='epochs')
    parser.add_argument('--regs', nargs='?', default='[1e-3,1e-2,1e-4]', help='lambdas for regularization') # VBPR [1e-1,1e-3,0] DVBPR [1e-3,1.2e-3,0] DeepStyle [1e-3,1e-2,1e-4]
    parser.add_argument('--lmd', type=float, default=0.01, help='lambda for balance the common loss and adversarial loss')
    parser.add_argument('--keep_prob', type=float, default=0.6, help='keep probability of dropout layers')
    parser.add_argument('--adv', type=int, default=0, help='adversarial training')
    parser.add_argument('--adv_type', nargs='?', default='rand', help='adversarial training type: grad, rand')
    parser.add_argument('--cnn', nargs='?', default='resnet', help='cnn type: resnet50')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon for adversarial')
    parser.add_argument('--weight_dir', nargs='?', default='rec_model_weights', help='directory to store the weights')
    parser.add_argument('--result_dir', nargs='?', default='rec_results', help='directory to store the predictions')
    parser.add_argument('--filter_cold_users', action='store_false', default=True, help='whether filter the cold users whose interactions are less that 5')

    parser.add_argument('--tp_k_predictions', type=int, default=1000,
                        help='top k predictions to store before the evaluation')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)  # 指定gpu
    print('Device gpu: {0}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    solver = Solver(args)

    # print(parse_args())
    print(args)

    # solver.original_test("123")
    start_time = time()

    print('START Training of the Recommender Model at {0}.'.format(start_time))
    solver.train()  # 这里的做法是先进行模型的训练然后再存储这个预测的情况
    print('END Training of the Recommender Model in {0} secs.'.format(time() - start_time))
