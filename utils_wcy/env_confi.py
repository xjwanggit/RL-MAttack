import argparse

def configure():
    parser = argparse.ArgumentParser(description='hyper-parameters')

    parser.add_argument('-e', '--epoch', default=200, help="Running episodes.") # VBPR 600, DVBPR, 200
    parser.add_argument('-s', '--training_step', default=10) # 15->5
    parser.add_argument('--batch_size', default=30)  # VBPR 30
    parser.add_argument('-ac_lr', default=5e-4, help='learning rate for actor network')  # 5e-4
    parser.add_argument('-cr_lr', default=1e-4, help='learning rate for critic network')  # 1e-3
    parser.add_argument('-t', '--tau', default=0.01, help='learning rate for soft-updates')
    parser.add_argument('-gamma', default=0.9)
    parser.add_argument('-gae_lambda', default=0.95)
    parser.add_argument('--epsilon', default=1.0)
    parser.add_argument('--buffer', default=60, help='experience replay buffer') # VBPR60, DVBPR, 30
    parser.add_argument('--eps', default=[3000, 1500, 150], help='Disturbance size')
    parser.add_argument('--update_net', type=int, default=3, help='After how many weight updates before updating the target network')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='coefficient for the entropy bonus (to encourage exploration)')
    parser.add_argument('--num_workers', type=int, default=1, help='the number of the processes')


    args = parser.parse_args()
    return args