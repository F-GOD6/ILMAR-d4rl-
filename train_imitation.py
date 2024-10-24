import os
import argparse
import torch
import numpy as np
from datetime import datetime
from cail.env import make_env
from cail.buffer import SerializedBuffer
from cail.algo.algo import ALGOS
from cail.trainer import Trainer
from cail.utils import return_range
from cail.env import get_dataset
import wandb
def get_device():
    # 检查 CUDA_VISIBLE_DEVICES 环境变量
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if visible_devices is not None and torch.cuda.is_available():
        print(f"Using GPU: {visible_devices}")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")
    
    return device
def run(args):
    """Train Imitation Learning algorithms"""

    expert_dataset = get_dataset(dirname=args.dirname,env_id=args.env_id, dataname="expert",num_trajectories=args.expert_num_trajectories)
    suboptimal_dataset = {
            'init_states': [],
            'states': [],
            'actions': [],
            'next_states': [],
            'dones': [],
            'rewards': []
        }
    union_dataset = {
            'init_states': [],
            'states': [],
            'actions': [],
            'next_states': [],
            'dones': [],
            'rewards': []
        }

    expert_dataset_name = "expert"
    if len(args.suboptimal_dataset_names) > 0:
        for suboptimal_datatype_idx, (suboptimal_dataset_name, suboptimal_num_traj) in enumerate(
                zip(args.suboptimal_dataset_names, args.suboptimal_num_trajs)):
            start_idx = args.expert_num_trajectories if (expert_dataset_name == suboptimal_dataset_name) else 0

            dataset = get_dataset(args.dirname, args.env_id, suboptimal_dataset_name, suboptimal_num_traj, start_idx=start_idx)
            suboptimal_dataset["init_states"].append(dataset["init_states"])
            suboptimal_dataset["states"].append(dataset["states"])
            suboptimal_dataset["actions"].append(dataset["actions"])
            suboptimal_dataset["next_states"].append(dataset["next_states"])
            suboptimal_dataset["dones"].append(dataset["dones"])
            suboptimal_dataset["rewards"].append(dataset["rewards"])

    suboptimal_dataset["init_states"] = np.concatenate(suboptimal_dataset["init_states"]).astype(np.float32)
    suboptimal_dataset["states"] = np.concatenate(suboptimal_dataset["states"]).astype(np.float32)
    suboptimal_dataset["actions"] = np.concatenate(suboptimal_dataset["actions"]).astype(np.float32)
    suboptimal_dataset["next_states"] = np.concatenate(suboptimal_dataset["next_states"]).astype(np.float32)
    suboptimal_dataset["dones"] = np.concatenate(suboptimal_dataset["dones"]).astype(np.float32)
    suboptimal_dataset["rewards"] = np.concatenate(suboptimal_dataset["rewards"]).astype(np.float32)

    union_dataset["init_states"] = np.concatenate([suboptimal_dataset["init_states"], expert_dataset["init_states"]]).astype(np.float32)
    union_dataset["states"] = np.concatenate([suboptimal_dataset["states"], expert_dataset["states"]]).astype(np.float32)
    union_dataset["actions"] = np.concatenate([suboptimal_dataset["actions"], expert_dataset["actions"]]).astype(np.float32)
    union_dataset["next_states"] = np.concatenate([suboptimal_dataset["next_states"], expert_dataset["next_states"]]).astype(np.float32)
    union_dataset["dones"] = np.concatenate([suboptimal_dataset["dones"], expert_dataset["dones"]]).astype(np.float32)
    union_dataset["rewards"] = np.concatenate([suboptimal_dataset["rewards"], expert_dataset["rewards"]]).astype(np.float32)
    print('# of expert demonstraions: {}'.format(expert_dataset["states"].shape[0]))
    print('# of imperfect demonstraions: {}'.format(suboptimal_dataset["states"].shape[0]))
     # normalize
    shift = -np.mean(suboptimal_dataset["states"], 0)
    scale = 1.0 / (np.std(suboptimal_dataset["states"], 0) + 1e-3)
    union_init_states = (union_dataset["init_states"] + shift) * scale
    expert_dataset["states"] = (expert_dataset["states"] + shift) * scale
    expert_dataset["next_states"] = (expert_dataset["next_states"] + shift) * scale
    union_dataset["states"] = (union_dataset["states"] + shift) * scale
    union_dataset["next_states"] = (union_dataset["next_states"] + shift) * scale
    env = make_env(args.env_id,normalize=True, shift=shift, scale=scale)
    env_test = env
    union_dataset["actions"]=union_dataset["actions"] / env.action_scale
    expert_dataset["actions"]=expert_dataset["actions"] / env.action_scale
    device = get_device()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # buffer_exp = SerializedBuffer(
    #     path=args.buffer_exp,
    #     device=device,
    #     label_ratio=args.label,
    #     use_mean=args.use_transition
    # )
    state_shape=env.observation_space.shape 
    if 'Ant' in env.spec.id: 
        state_shape = (27,)
    algo_type=True
    if args.algo == 'cail':
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
            lr_conf=args.lr_conf,
            pretrain_steps=args.pre_train,
            use_transition=args.use_transition
        )
    elif args.algo == 'drex':
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
            env=env
        )
    elif args.algo == 'ssrr':
        algo = ALGOS[args.algo](
            buffer_exp=buffer_exp,
            state_shape=state_shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
            env=env,
            airl_actor_path=args.airl_actor,
            airl_discriminator_path=args.airl_disc,
        )
    elif args.algo == 'iswbc' or args.algo == 'demodice' or args.algo =="metaiswbc" or args.algo =="metademodice" or args.algo =="ilas" or args.algo =="iswbcg" or args.algo == 'mybc':
        algo_type=False
        if args.use_union:
            algo = ALGOS[args.algo](
                buffer_exp=expert_dataset,
                buffer_union=union_dataset,
                state_shape=state_shape,
                action_shape=env.action_space.shape,
                device=device,
                seed=args.seed,
                batch_size= args.batch_size,
            )
        else:
            algo = ALGOS[args.algo](
                buffer_exp=expert_dataset,
                buffer_union=suboptimal_dataset,
                state_shape=state_shape,
                action_shape=env.action_space.shape,
                device=device,
                seed=args.seed,
                batch_size= args.batch_size,
            )           
    else:
        algo = ALGOS[args.algo](
            buffer_exp=union_dataset,
            state_shape=state_shape,
            action_shape=env.action_space.shape,
            device=device,
            seed=args.seed,
            rollout_length=args.rollout_length,
        )

    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        'logs', args.env_id, args.algo, f'seed{args.seed}-{time}')
    total_dir = os.path.join(
        'logs', args.env_id)

    if args.algo=="bc" or args.algo=="iswbc" or args.algo == "mybc" or args.algo =="demodice" or args.algo=="metaiswbc" or args.algo =="metademodice" or args.algo =="iswbcg":
        algo_type=False
    trainer = Trainer(
        env=env,
        env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_epi,
        seed=args.seed,
        algo_type=algo_type,
        total_dir =  total_dir             #启动问题
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()

    # required
    p.add_argument('--buffer_exp', type=str, required=False,
                   help='path to the demonstration buffer')
    p.add_argument('--buffer_union', type=str, required=False,
                   help='path to the union demonstration buffer')
    p.add_argument('--env-id', type=str, required=True,
                   help='name of the environment')
    p.add_argument('--algo', type=str, required=True,
                   help='Imitation Learning algorithm to be trained')

    # custom
    p.add_argument('--rollout-length', type=int, default=10000,
                   help='rollout length of the buffer')
    p.add_argument('--num-steps', type=int, default=10**6,
                   help='number of steps to train')
    p.add_argument('--eval-interval', type=int, default=10**4,
                   help='time interval between evaluations')
    # for CAIL
    p.add_argument('--lr-conf', type=float, default=0.1,
                   help='learning rate of confidence for CAIL')
    p.add_argument('--pre-train', type=int, default=20000000,
                   help='pre-train steps for CAIL')
    p.add_argument('--use-transition', action='store_true', default=False,
                   help='use state transition reward for cail')

    # for SSRR
    p.add_argument('--airl-actor', type=str,
                   help='path to pre-trained AIRL actor for SSRR')
    p.add_argument('--airl-disc', type=str,
                   help='path to pre-trained AIRL discriminator for SSRR')

    # default
    p.add_argument('--num-eval-epi', type=int, default=10,
                   help='number of episodes for evaluation')
    p.add_argument('--seed', type=int, default=0,
                   help='random seed')
    p.add_argument('--label', type=float, default=0.05,
                   help='ratio of labeled data')
    p.add_argument('--batch_size', type=int, default=512,
                   help='batch_size')
    p.add_argument('--dirname', type=str, default="/home/fanjiangdong/.d4rl/datasets",
                   help='batch_size')
    p.add_argument('--expert_num_trajectories', type=int, default=1,
                   help='expert_num_trajectories')
    p.add_argument('--suboptimal_dataset_names', type=str, nargs='+', default=["full_replay"],
                   help='batch_size')
    p.add_argument('--suboptimal_num_trajs', type=int, nargs='+',default=[4000],
                   help='batch_size')
    p.add_argument('--use_union', type=bool, default=True,
                   help='use_union')
    args = p.parse_args()
    wandb.init(project="ILMAR", entity="f-god666", 
                   name=f"{args.algo}_{args.env_id}_seed_{args.seed}")
    wandb.config.update(vars(args))
    run(args)
    wandb.finish()
