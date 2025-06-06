#%%
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.tools import patchextraction as PE
from tiatoolbox.tools import stainnorm
import numpy as np
import matplotlib.pyplot as plt

from src.misc.utils import recur_find_ext


root_dir = '/mnt/romesco_lab_share_stainFuser/atypiaComp/mitos_atypia_2014_training_'

aperio = f'{root_dir}aperio/'
hamamatsu = f'{root_dir}hamamatsu/'

# %%
sf_path = '/mnt/romesco_cloud_workspace/atypiaTest/output_2.npy'
mac_path = 'output/macenko_test.npy'
aperio_path = 'output/aperio.npy'
nst_path = '/home/robj/Projects/Diffusion/outputs/debug2/h_target/NST_test.npy'
gt_path = 'output/hamamatsu.npy'
# %%
sf = np.load(sf_path, mmap_mode='r')
mac = np.load(mac_path, mmap_mode='r')
ap = np.load(aperio_path, mmap_mode='r')
nst = np.load(nst_path, mmap_mode='r')
gt = np.load(gt_path, mmap_mode='r')
target = imread('output/h_target.png')

# %%
idx = 223
fig, ax = plt.subplots(1,6)
ax[0].imshow(ap[idx])
ax[0].set_title('Aperio\n(Source)')
ax[0].axis('off')
ax[1].imshow(target)
ax[1].set_title('Target')
ax[1].axis('off')
ax[2].imshow(gt[idx])
ax[2].set_title('Hamamatsu\n(GT)')
ax[2].axis('off')
ax[3].imshow(sf[idx])
ax[3].set_title('StainFuser')
ax[3].axis('off')
ax[4].imshow(mac[idx])
ax[4].set_title('Macenko')
ax[4].axis('off')
ax[5].imshow(nst[idx])
ax[5].set_title('NST')
ax[5].axis('off')
plt.show()


# %%
from tqdm import tqdm
aperio = np.load('/home/robj/Projects/stainFuser/output/aperio_train.npy', mmap_mode='r')
# hamamatsu = np.load('/home/robj/Projects/stainFuser/output/hamamatsu.npy', mmap_mode='r')
target = imread('/home/robj/Projects/stainFuser/output/h_target.png')
stain_norm = stainnorm.MacenkoNormalizer()
stain_norm.fit(target)

output = np.zeros(aperio.shape, dtype=np.uint8)

for i in tqdm(range(aperio.shape[0]), total=aperio.shape[0]):
    # stain_norm = stainnorm.MacenkoNormalizer()
    # stain_norm.fit(hamamatsu[i].copy())
    output[i] = stain_norm.transform(aperio[i].copy())

np.save('output/macenko_train.npy', output)

# %%

normed_path = '/mnt/romesco_cloud_workspace/atypiaTest/output_2.npy'
normed_path2 = '/home/robj/Projects/Diffusion/outputs/debug/h_target/h_target.npy'
normed_path2 = '/home/robj/Projects/Diffusion/outputs/debug2/h_target/NST_test.npy'
gt_path = 'output/hamamatsu.npy'

# %%
sf = np.load(normed_path, mmap_mode='r')
nst = np.load(normed_path2, mmap_mode='r')
gt = np.load(gt_path, mmap_mode='r')

# %%
idx = 89
fig, ax = plt.subplots(1,3)
ax[0].imshow(sf[idx])
ax[1].imshow(nst[idx])
ax[2].imshow(gt[idx])
plt.show()

# %%
ap = np.load('output/aperio.npy', mmap_mode='r')
ham = np.load('output/hamamatsu.npy', mmap_mode='r')

# %%
idx = 187
fig, ax = plt.subplots(1,2)
ax[0].imshow(ap[idx])
ax[0].axis('off')
ax[1].imshow(ham[idx])
ax[1].axis('off')
plt.show()

# %%
aperio_files  = recur_find_ext(aperio, ['.tiff'])
hamamatsu_files = recur_find_ext(hamamatsu, ['.tiff'])

def get_reader(img, scanner):
    reader = WSIReader.open(img)
    if scanner == 'aperio':
        reader.info.mpp = 0.2455 * 2
    elif scanner == 'hamamatsu':
        reader.info.mpp = (0.227299 * 2, 0.227531 * 2)
    return reader

# %%
h_tiles, a_tiles = [], []
idx = 0
for img_h, img_a in zip(hamamatsu_files, aperio_files):
    reader_h = get_reader(img_h, 'hamamatsu')
    reader_a = get_reader(img_a, 'aperio')
    
    extractor_h = PE.get_patch_extractor(
        input_img=reader_h,
        method_name='slidingwindow',
        patch_size=(512, 512),
        stride=(512, 512),
        within_bound=True,
        resolution=0.5,
        units='mpp'
    )
    extractor_a = PE.get_patch_extractor(
        input_img=reader_a,
        method_name='slidingwindow',
        patch_size=(512, 512),
        stride=(512, 512),
        within_bound=True,
        resolution=0.5,
        units='mpp'
    )
    assert(len(extractor_a) == len(extractor_h))
    for patch in extractor_h:
        h_tiles.append(patch)
        idx += 1
    for patch in extractor_a:
        a_tiles.append(patch)

# %%
arr_h = np.array(h_tiles)
arr_a = np.array(a_tiles)

# %%
np.save('output/hamamatsu_train.npy', arr_h)
np.save('output/aperio_train.npy', arr_a)
# with open('output/hamamatsu_256.npy', 'wb') as f:
#     np.save(f, arr_h)
    
# with open('output/aperio_256.npy', 'wb') as f:
#     np.save(f, arr_a)
    
# %%
t = np.load('output/aperio.npy', mmap_mode='r')
w = np.load('output/hamamatsu.npy', mmap_mode='r')

# %%

fig, ax = plt.subplots(1,2)
ax[0].imshow(t[24])
ax[1].imshow(w[24])
plt.show()
# %%
reader = WSIReader.open(aperio_files[0])
reader.info.mpp = 0.2455 * 2

q = reader.slide_thumbnail(resolution=0.5, units='mpp')

# %%
reader2 = WSIReader.open(hamamatsu_files[0])
reader2.info.mpp = (0.227299 * 2, 0.227531 * 2)
# reader2.info.mpp = (0.227531 * 2, 0.227299 * 2)

w = reader2.slide_thumbnail(resolution=0.5, units='mpp')

# %%
for i in range(5):
    img_a = imread(aperio_files[i])
    img_h = imread(hamamatsu_files[i])
    print(img_a.shape, img_h.shape)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_a)
    ax[1].imshow(img_h)
    plt.show()
    # assert(img_a.shape == img_h.shape)


# %%


extractor = patchextraction.get_patch_extractor(
    input_img=img_a,
    method_name='slidingwindow',
    patch_size=(512, 512),
    stride=(512, 512),
    within_bound=True,
)

extractor_h = patchextraction.get_patch_extractor(
    input_img=img_h,
    method_name='slidingwindow',
    patch_size=(512, 512),
    stride=(512, 512),
    within_bound=True,
)
# %%
fig, ax = plt.subplots(2, 6)
for i, (patch_h, patch_a) in enumerate(zip(extractor_h, extractor)):
    ax[0,i].imshow(patch_h)
    ax[0,i].axis('off')
    ax[1,i].imshow(patch_a)
    ax[1,i].axis('off')
plt.show()


# %%
n_tiles = 0
h_tiles, a_tiles = [], []
idx = 0
for img_h, img_a in zip(hamamatsu_files, aperio_files):
    # if idx == 500:
    #     break
    extractor_h = patchextraction.get_patch_extractor(
        input_img=img_h,
        method_name='slidingwindow',
        patch_size=(512, 512),
        stride=(512, 512),
        within_bound=True
    )
    extractor_a = patchextraction.get_patch_extractor(
        input_img=img_a,
        method_name='slidingwindow',
        patch_size=(512, 512),
        stride=(512, 512),
        within_bound=True
    )
    assert(len(extractor_a) == len(extractor_h))
    for patch in extractor_h:
        h_tiles.append(patch)
        idx += 1
    for patch in extractor_a:
        a_tiles.append(patch)
    # tiles.append(img)
    # n_tiles += len(extractor)
    # break

# %%
arr_h = np.array(h_tiles[:500])
arr_a = np.array(a_tiles[:500])

# %%
fig, ax = plt.subplots(2, 5)
for i in range(5):
    ax[0,i].imshow(arr_h[i])
    ax[0,i].axis('off')
    ax[1,i].imshow(arr_a[i])
    ax[1,i].axis('off')
plt.show()
# %%
with open('output/hamamatsu.npy', 'wb') as f:
    np.save(f, arr_h)

# %%

t = np.load('output/aperio.npy', mmap_mode='r')
w = np.load('output/hamamatsu.npy', mmap_mode='r')

# %%

fig, ax = plt.subplots(1,2)
ax[0].imshow(t[52])
ax[1].imshow(w[52])
plt.show()

# ! need to deal with different mpps here
# %%
# import torchvision.transforms as T

# transform = T.Compose([T.ToTensor()])

# model = segmentor.model

# output = model.infer_batch(model, transform(tiles[0]).permute(1,2,0).unsqueeze(0), on_gpu=False)

# # %%
# res = output[0][0,:,:]
# pred = np.argmax(res, axis=-1)
# # res = output[0,:].permute(1, 2, 0).detach().numpy()
# # pred = np.argmax(res, axis=-1)

# # %%
# from tiatoolbox.models.engine.semantic_segmentor import SemanticSegmentor
# from src.misc.utils import rmdir

# segmentor = SemanticSegmentor(
#     pretrained_model='fcn_resnet50_unet-bcss',
#     num_loader_workers=4,
#     batch_size=4,
#     # resolution=1.,
#     # units='baseline',
#     # patch_input_shape=[512,512],
#     # patch_output_shape=[512,512],
#     # stride_shape=[512,512]
# )
# rmdir('debug/')

# output = segmentor.predict(
#     tiles[0:1],
#     mode='tile',
#     on_gpu=False,
#     save_dir='debug/'
# )
# # %%
# pred_raw = np.load('debug/0.raw.0.npy', mmap_mode='r')
# pred = np.argmax(pred_raw, axis=-1)

# fig = plt.figure()
# label_names_dict = {
#     0: "Tumour",
#     1: "Stroma",
#     2: "Inflamatory",
#     3: "Necrosis",
#     4: "Others",
# }

# for i in range(5):
#     ax = plt.subplot(1, 5, i+1)
#     (
#         plt.imshow(pred_raw[:,:,i]),
#         plt.xlabel(label_names_dict[i]),
#         ax.axes.xaxis.set_ticks([]),
#         ax.axes.yaxis.set_ticks([]),
#     )

# fig2 = plt.figure()
# tile = imread(tiles[0])
# ax1 = plt.subplot(1, 2, 1), plt.imshow(tile), plt.axis("off")
# ax2 = plt.subplot(1, 2, 2), plt.imshow(pred), plt.axis("off")
# # %%
