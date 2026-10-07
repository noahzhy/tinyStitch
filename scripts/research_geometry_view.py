"""Plot saved first-stage estimates against evaluation-only synthetic labels."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path(__file__).resolve().parents[1]
s=root/'data/research/mixed-bays-v1/738018'
p=root/'artifacts/research/mixed-bays-stable-test18-v1/60'
out=root/'artifacts/research/mixed-bays-optimization-v1'
T=np.load(s/'labels/geometry.npz')['T']
d=np.stack([np.load(s/'labels'/f'{i:04}.npy') for i in range(60)])
anchor=len(T)//2
gt=T@np.linalg.inv(T[anchor])
valid=d[anchor][d[anchor]>0];scale=np.partition(valid,(len(valid)-1)//2)[(len(valid)-1)//2]
gt[:,:3,3]/=scale
gd=d/scale
pred=np.load(p/'cameras.npy');pd=np.load(p/'depth.npy')
centers=lambda t:-np.einsum('nji,nj->ni',t[:,:3,:3],t[:,:3,3])
a,b=centers(gt),centers(pred)
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
for arr,color,label in [(a,'black','GT'),(b,'tab:orange','Estimated')]:
 axes[0,0].plot(arr[:,0],arr[:,2],'.-',color=color,label=label)
 directions=(gt if label=='GT' else pred)[:,2,:3]
 ids=np.arange(0,60,5)
 axes[0,0].quiver(arr[ids,0],arr[ids,2],directions[ids,0],directions[ids,2],color=color,angles='xy',scale_units='xy',scale=5,width=.004)
axes[0,0].set_ylim(-.12,.28)
axes[0,0].set(title='Camera path and viewing direction (top view)',xlabel='X (canonical units)',ylabel='Z (canonical units)');axes[0,0].legend()
for k,label in enumerate(['X','Y','Z']):axes[0,1].plot(b[:,k]-a[:,k],label=label)
axes[0,1].set(title='Camera center error by frame',xlabel='Frame',ylabel='Canonical units');axes[0,1].legend()
rel=pred[:,:3,:3]@gt[:,:3,:3].transpose(0,2,1)
err=np.rad2deg(np.arccos(np.clip((np.trace(rel,axis1=1,axis2=2)-1)/2,-1,1)))
axes[1,0].plot(err);axes[1,0].set(title='Absolute orientation error',xlabel='Frame',ylabel='Degrees')
absrel=np.array([np.mean(np.abs(x[y>0]-y[y>0])/y[y>0])*100 for x,y in zip(pd,gd)])
axes[1,1].plot(absrel);axes[1,1].set(title='Source depth AbsRel',xlabel='Frame',ylabel='Percent')
for ax in axes.flat:ax.grid(alpha=.25)
fig.suptitle('Stage 1 geometry - unseen synthetic shelf 738018 / 60 input views')
fig.savefig(out/'geometry-estimation.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(3,4,figsize=(13,7),layout='constrained')
lo,hi=np.percentile(gd,[1,99])
for row,i in enumerate([5,30,54]):
 axes[row,0].imshow(Image.open(s/'rgb'/f'{i:04}.png'));axes[row,0].set_title(f'Input frame {i}')
 axes[row,1].imshow(gd[i],vmin=lo,vmax=hi,cmap='viridis');axes[row,1].set_title('GT depth')
 axes[row,2].imshow(pd[i],vmin=lo,vmax=hi,cmap='viridis');axes[row,2].set_title('Estimated depth')
 im=axes[row,3].imshow(np.where(gd[i]>0,np.abs(pd[i]-gd[i])/np.maximum(gd[i],1e-8)*100,np.nan),vmin=0,vmax=10,cmap='magma');axes[row,3].set_title('Relative error (0-10%)')
 for ax in axes[row]:ax.axis('off')
fig.colorbar(im,ax=axes[:,3],shrink=.7,label='Percent');fig.savefig(out/'geometry-depth.png',dpi=150)
print(json.loads((p/'report.json').read_text())['stage1'])
