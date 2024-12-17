import os, torch, yaml, time
import colorful as c
from torch.utils.data.dataloader import DataLoader
from my_utils import set_seed, save_img
from metric.metric import psnr, mae,ssim
from models.model import inpaint_model
from dataset import get_dataset
import numpy as np


# set seed
set_seed(1234)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
save_mask=True

# open config file
with open('/w/331/abdulbasit/transformer_inpaint/code/config/model_config.yml', 'r') as config:
    args = yaml.safe_load(config)

# Define the model
Inpaint_model = inpaint_model(args)



# Define the dataset
test_dataset = get_dataset(args['test_path'], test_mask_path=args['test_mask_1~60_path'],is_train=False, image_size=args['image_size'])

# set initial
iterations = 0
Total_time = []
# loaded_ckpt
if os.path.exists(args['test_ckpt']):
    data = torch.load(args['test_ckpt'],map_location=device)
    Inpaint_model.load_state_dict(data['state_dict'],strict=False)
    #Inpaint_model = Inpaint_model.to(args['gpu'])
    Inpaint_model = Inpaint_model.to(device)
    print('Finished reloading the Epoch '+c.yellow(str(data['epoch']))+' model')
    # Optimizer
    raw_model = Inpaint_model.module if hasattr(Inpaint_model, "module") else Inpaint_model 
    iterations = data['iterations']
    previous_epoch = data['epoch']
    print(c.blue('------------------------------------------------------------------'))
    print('resume training with iterations: '+c.yellow(str(iterations))+ ', previous_epoch: '+c.yellow(str(previous_epoch)))
    print(c.blue('------------------------------------------------------------------'))
else:
    raise Exception('Warnning: There is no test model found.')
# DataLoaders
test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True,
                        batch_size=args['batch_size'] // args['world_size'],
                        num_workers=args['num_workers'])



epoch_start = time.time()
# start EVAL
print(c.blue('-----------------------------')+c.cyan(' test ! ')+c.blue('-------------------------------------'))
# Evaluation (Validation)
raw_model.eval()    # eval MODE
loader = test_loader
PSNR_center = []
SSIM_center = []
Mae_center = []
Fid_center=[]
for val_it, val_items in enumerate(loader):
    for k in val_items:
        if type(val_items[k]) is torch.Tensor:
            val_items[k] = val_items[k].to(device)

    # in to the model (no_grad)
    with torch.no_grad():
        # pred_img, pred_edge = raw_model(val_items['img'], val_items['edge'], val_items['mask'])
        if save_mask:
            masked_image=val_items['img']*(1-val_items['mask'])
            save_img(masked_image,args['save_mask_path'],[img_name.split(".")[0]+"_masked.jpg" for img_name in val_items['name']])
        pred_img = raw_model(val_items['img'], val_items['mask'])
        comp_imgs = (1 - val_items['mask']) * val_items['img'] + val_items['mask'] * pred_img
        images=list(img.cpu().numpy().transpose((1, 2, 0)) for img in val_items['img'] )
        #pred_images=list(img.cpu().numpy().transpose((1, 2, 0)) for img in pred_img )
        comp_images=list(img.cpu().numpy().transpose((1, 2, 0)) for img in comp_imgs )
        PSNR_center.append(psnr(images,comp_images))
        SSIM_center.append(ssim(images,comp_images))
        Mae_center.append(mae(images,comp_images))
        # FID_center.append(FID(val_items['img'], pred_img))
        save_img(comp_imgs,args['save_img_path_comps'], val_items['name'],val_items['mask'])

PSNR_center = np.mean(PSNR_center).item()
SSIM_center = np.mean(SSIM_center).item()
Mae_center = np.mean(Mae_center).item()
FID_center = np.mean(Fid_center).item()
print(c.green('---------------------------------------------------------------------------------'))
print(c.blue('------------------------------')+c.cyan(' PSNR: ')+c.magenta(PSNR_center)+c.blue(' --------------------------------------'))
print(c.blue('------------------------------')+c.cyan(' SSIM: ')+c.magenta(SSIM_center)+c.blue(' --------------------------------------'))
print(c.blue('-----------------------------')+c.cyan(' Mae: ')+c.magenta(Mae_center)+c.blue(' --------------------------------------'))
print(f"\n FID:{FID_center}")
print(c.green('---------------------------------------------------------------------------------'))

epoch_time = (time.time() - epoch_start)  # teain one epoch time
print(c.blue('------------------------')+f"This epoch cost {epoch_time:.5f} seconds!"+c.blue('-----------------------'))
print(c.blue('----------------------------------')+c.cyan(' End EVAL! ')+c.blue('-----------------------------------'))

  
