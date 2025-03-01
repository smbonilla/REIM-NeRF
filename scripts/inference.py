import os
import cv2
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from reimnerf.models.rendering import render_rays
from reimnerf.models.nerf import *

from reimnerf.utils import load_ckpt

from reimnerf.datasets import dataset_dict
from reimnerf.datasets.pfm_io import *
from collections import defaultdict
from reimnerf import metrics
torch.backends.cudnn.benchmark = True

import lpips


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'reim_json', 'reim_json_render'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test',
                        help='test or test_train')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[256, 256],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--N_emb_xyz', type=int, default=10,
                        help='number of frequencies in xyz positional encoding')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of frequencies in dir positional encoding')
    parser.add_argument('--N_emb_light_xyz', type=int, default=4,
                        help='number of frequencies in light source location positional encoding')
    parser.add_argument('--N_samples', type=int, default=64,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=128,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--chunk', type=int, default=32*1024*4,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='pretrained checkpoint path to load')
    
    ## extensions 
    parser.add_argument('--variant', help='which variant of the nerf model to load',
                        choices=['nerf','ls_loc'], default='nerf')
    ##

    parser.add_argument('--save_depth', default=False, action="store_true",
                        help='whether to save depth prediction')
    parser.add_argument('--depth_format', type=str, default='pfm',
                        choices=['pfm', 'bytes'],
                        help='which format to save')
    parser.add_argument('--fps', type=int, default=30)

    return parser.parse_args()


@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        compute_normals=False,
                        normal_pertrube=False,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()
    w, h = args.img_wh

    kwargs = {'root_dir': args.root_dir,
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name in ['llff']:
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)

    embedding_xyz = Embedding(args.N_emb_xyz)
    embedding_dir = Embedding(args.N_emb_dir)
    embeddings = defaultdict(lambda : None)
    embeddings['xyz'] = embedding_xyz
    embeddings['dir'] = embedding_dir
    color_mlp_in = 6*args.N_emb_dir+3
    if args.variant == 'ls_loc':
        embedding_light_xyz = Embedding(args.N_emb_light_xyz)
        embeddings['light_loc'] = embedding_light_xyz
        color_mlp_in += (6*args.N_emb_light_xyz)+3




    nerf_coarse = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=color_mlp_in)
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    nerf_coarse.cuda().eval()

    models = {'coarse': nerf_coarse}

    if args.N_importance > 0:
        nerf_fine = NeRF(in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=color_mlp_in)
        load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')
        nerf_fine.cuda().eval()
        models['fine'] = nerf_fine

    # Initialize the LPIPS model
    lpips_model = lpips.LPIPS(net='vgg').cuda().eval()

    imgs, depth_maps, psnrs, lpi_values = [], [], [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        rays = sample['rays'].cuda()
        results = batched_inference(models, embeddings, rays,
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        img_pred = np.clip(results[f'rgb_{typ}'].view(h, w, 3).cpu().numpy(), 0, 1)

        if args.save_depth:
            depth_pred = results[f'depth_{typ}'].view(h, w).cpu().numpy()
            depth_maps += [depth_pred]
            if args.depth_format == 'pfm':
                save_pfm(os.path.join(dir_name, f'depth_{i:03d}.pfm'), depth_pred)
            else:
                with open(os.path.join(dir_name, f'depth_{i:03d}'), 'wb') as f:
                    f.write(depth_pred.tobytes())

        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(torch.FloatTensor(img_pred), img_gt).item()]
            img_pred_tensor = torch.FloatTensor(img_pred).permute(2, 0, 1).unsqueeze(0).cuda() # Adds a batch dimension and moves to GPU
            img_gt_tensor = torch.FloatTensor(img_gt).permute(2, 0, 1).unsqueeze(0).cuda() # Same for the ground truth
            lpips_val = metrics.calc_lpips(img_pred_tensor, 
                                           img_gt_tensor, 
                                           lpips_model)
            lpi_values.append(lpips_val.item())
        else:
            print('No rgbs in the dataset, skipping PSNR and LPIPS calculation')

    imageio.mimsave(os.path.join(dir_name, f'rgb.gif'), imgs, fps=args.fps)

    if args.save_depth:
        min_depth = np.min(depth_maps)
        max_depth = np.max(depth_maps)
        depth_imgs = (depth_maps - np.min(depth_maps)) / (max(np.max(depth_maps) - np.min(depth_maps), 1e-8))
        depth_imgs_ = [cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET) for img in depth_imgs]
        imageio.mimsave(os.path.join(dir_name, 'depth.gif'), depth_imgs_, fps=args.fps)

    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')
    
    if lpi_values:
        mean_lpi = np.mean(lpi_values)
        print(f'Mean LPIPS : {mean_lpi:.4f}')
