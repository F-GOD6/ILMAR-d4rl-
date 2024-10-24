dataset_dir = "/home/fanjiangdong/.d4rl/datasets"
from cail.env import get_dataset
import numpy as np
env_id = "Hopper-v2"
expert_dataset = get_dataset(dirname=dataset_dir,env_id="Hopper-v2", dataname="expert",num_trajectories=1)
suboptimal_dataset = {
        'init_states': [],
        'states': [],
        'actions': [],
        'next_states': [],
        'dones': []
    }
union_dataset = {
        'init_states': [],
        'states': [],
        'actions': [],
        'next_states': [],
        'dones': []
    }
suboptimal_dataset_names = ["full_replay","random"]
expert_dataset_name = "expert"
suboptimal_num_traj = [50,50]
if len(suboptimal_dataset_names) > 0:
    for suboptimal_datatype_idx, (suboptimal_dataset_name, suboptimal_num_traj) in enumerate(
            zip(suboptimal_dataset_names, suboptimal_num_traj)):
        start_idx = expert_num_traj if (expert_dataset_name == suboptimal_dataset_name) else 0
        dataset = get_dataset(dataset_dir, env_id, suboptimal_dataset_name, suboptimal_num_traj, start_idx=start_idx)
        suboptimal_dataset["init_states"].append(dataset["init_states"])
        suboptimal_dataset["states"].append(dataset["states"])
        suboptimal_dataset["actions"].append(dataset["actions"])
        suboptimal_dataset["next_states"].append(dataset["next_states"])
        suboptimal_dataset["dones"].append(dataset["dones"])
suboptimal_dataset["init_states"] = np.concatenate(suboptimal_dataset["init_states"]).astype(np.float32)
suboptimal_dataset["states"] = np.concatenate(suboptimal_dataset["states"]).astype(np.float32)
suboptimal_dataset["actions"] = np.concatenate(suboptimal_dataset["actions"]).astype(np.float32)
suboptimal_dataset["next_states"] = np.concatenate(suboptimal_dataset["next_states"]).astype(np.float32)
suboptimal_dataset["dones"] = np.concatenate(suboptimal_dataset["dones"]).astype(np.float32)

union_dataset["init_states"] = np.concatenate([suboptimal_dataset["init_states"], expert_dataset["init_states"]]).astype(np.float32)
union_dataset["states"] = np.concatenate([suboptimal_dataset["states"], expert_dataset["states"]]).astype(np.float32)
union_dataset["actions"] = np.concatenate([suboptimal_dataset["actions"], expert_dataset["actions"]]).astype(np.float32)
union_dataset["next_states"] = np.concatenate([suboptimal_dataset["next_states"], expert_dataset["next_states"]]).astype(np.float32)
union_dataset["dones"] = np.concatenate([suboptimal_dataset["dones"], expert_dataset["dones"]]).astype(np.float32)

print('# of expert demonstraions: {}'.format(expert_dataset["states"].shape[0]))
print('# of imperfect demonstraions: {}'.format(suboptimal_dataset["states"].shape[0]))