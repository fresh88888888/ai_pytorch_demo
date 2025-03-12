import datetime
import argparse

current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
num = 4e4
print(num)
print(current_time)

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--algo', default='AC')
    parse.add_argument('--env', default='halfcheetah-expert-v2')
    parse.add_argument('--seed', default=1, type=int)
    parse.add_argument('--eval_freq', default=4e4,type=int)
    parse.add_argument('--max_time', default=1e6, type=int)
    parse.add_argument('--save_model',action='store_true')
    parse.add_argument('--batch_size',default=256,type=int)
    parse.add_argument('--discount',default=0.99, type=float)
    parse.add_argument('--tau',default=0.005,type=float)
    parse.add_argument('--policy_noise',default=0.2, type=float)
    parse.add_argument('--noise_clip', default=0.5, type=float)
    parse.add_argument('--policy_freq',default=2, type=int)
    parse.add_argument('--use_discrete',default=False, type=bool)
    parse.add_argument('--use_epsilon', default=False, type=bool)
    parse.add_argument('--epsilon', default=0.0, type=float)
    parse.add_argument('--bc_weight', default=1.0, type=float)
    parse.add_argument('--use_cuda', default=True, type=bool)
    parse.add_argument('--vae_step', default=200000, type=int)
    parse.add_argument('--alpha', default=1.0, type=float)
    parse.add_argument('--normalize', default=True, type=bool)
    parse.add_argument('--reward_tune', default='None')
    parse.add_argument('--clip', default=False, type=bool)
    parse.add_argument('--in_sample', default=False,type=bool)
    
    args = parse.parse_args()
    
    print("env: ", args.env)
    print("seed: ", args.seed)
    print("save_model: ", args.save_model)
