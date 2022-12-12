import os
import shutil
from datetime import datetime
import torch
import config as cfg
args=cfg.read_config()
epoch=10


print('just a test????????')

cpt_path = os.path.join(args.ckpt_dir,
                          f"{args.name}_{datetime.now().strftime('%m%d%H%M%S')}.pth")  # datetime.now().strftime('%Y%m%d_%H%M%S')

# Save checkpoint
def save_checkpoint(state,cpt_path,model=None):
    if model is not None:
        save_checkpoint_mod(cpt_path, model, state)
        return

    torch.save(state, cpt_path)
    print(f"Model saved to {cpt_path}")

def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

def save_checkpoint_mod(cpt_path, model, state):
    torch.save({
        'epoch': state + 1,
        'state_dict': model.module.state_dict()
    }, cpt_path)
    print(f"Model saved to {cpt_path}")
    return

save_checkpoint()
with open(os.path.join(args.log_dir, f"checkpoint_path.txt"), 'w+') as f:
    f.write(f'{cpt_path}')