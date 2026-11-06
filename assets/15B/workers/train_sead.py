from pathlib import Path
import copy,hashlib,json,math,platform,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def require(ok,message):
    if not bool(ok):raise ValueError(message)


def write_json(path,obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))
    temp.replace(path)


def save_model(path,model,epoch):
    temp=path.with_suffix('.tmp')
    torch.save({'model':model.state_dict(),'epoch':epoch},temp)
    temp.replace(path)


class SEADAdapted(nn.Module):
    """Released affine architecture. No silently added nonlinearities or sigmoid.

    Adaptation changes device handling and guards zero residual direction.
    Raw residual magnitude and residual direction are returned separately.
    """
    def __init__(self,input_dim=21,latent_dim=64,widths=(256,128)):
        super().__init__()
        w1,w2=widths
        self.encoder=nn.Sequential(nn.Linear(input_dim,w1),nn.Linear(w1,w2),nn.Linear(w2,latent_dim))
        self.decoder=nn.Sequential(nn.Linear(latent_dim,w2),nn.Linear(w2,w1),nn.Linear(w1,input_dim))
        self.projector=nn.Linear(latent_dim,2)
        self.head1=nn.Linear(input_dim+latent_dim+1,128)
        self.head2=nn.Linear(128+1,32)
        self.head3=nn.Linear(32+1,1)

    def forward(self,x):
        z=self.encoder(x)
        residual=self.decoder(z)-x
        error=torch.linalg.vector_norm(residual,dim=1,keepdim=True)
        direction=residual/error.clamp_min(1e-12)
        h=self.head1(torch.cat([direction,z,error],dim=1))
        h=self.head2(torch.cat([h,error],dim=1))
        score=self.head3(torch.cat([h,error],dim=1)).flatten()
        return z,self.projector(z),residual,error.flatten(),direction,score


def stage2_loss(score,error,direction,weak_y,margin,variant):
    e=error if variant=='raw_error' else torch.linalg.vector_norm(direction,dim=1)
    return ((1-weak_y)*(score.abs()+e)+weak_y*(F.relu(margin-score)+F.relu(margin-e))).mean()


def average_precision(y,scores):
    # Tie-aware AP, independently checked against sklearn before delivery.
    require(np.isfinite(scores).all(),'Nonfinite selection scores')
    order=np.argsort(-scores,kind='stable');s=np.asarray(scores)[order];label=np.asarray(y)[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(label,dtype=np.float64)[ends]
    require(tp[-1]>0,'No positive selection labels')
    return float(np.sum(np.diff(np.r_[0,tp])*(tp/(ends+1)))/tp[-1])


def schedule(epoch,total,peak,minimum,warmup):
    if epoch<warmup:
        return 1e-6+(peak-1e-6)*epoch/max(1,warmup)
    fraction=(epoch-warmup)/max(1,total-warmup-1)
    return minimum+(peak-minimum)*0.5*(1+math.cos(math.pi*fraction))


def augment(x,noise_std,mask_probability):
    # Sensor positions are never permuted/reversed/cropped.
    # Zero is the fit-set sensor mean after standardization.
    return (x+noise_std*torch.randn_like(x))*(torch.rand_like(x)>=mask_probability)


@torch.no_grad()
def infer(model,x,output='score',batch=512):
    model.eval();parts=[]
    for a in x.split(batch):
        if output=='latent':v=model.encoder(a)
        elif output=='score':v=model(a)[-1]
        elif output=='mse':v=(model.decoder(model.encoder(a))-a).square().mean(1)
        else:raise ValueError(output)
        parts.append(v.cpu().numpy())
    values=np.concatenate(parts)
    require(np.isfinite(values).all(),'Nonfinite inference '+output)
    return values


def optimizer_step(loss,optimizer,parameters):
    require(torch.isfinite(loss),'Nonfinite loss')
    loss.backward()
    require(all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters),'Nonfinite gradient')
    optimizer.step()


def train(out,plan):
    started=time.perf_counter();p=plan['training']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(plan['seed'])
    data=np.load(out/'prepared_data.npz',allow_pickle=False)
    require(not any('outer' in k for k in data.files),'Outer data present in training export')
    X={r:torch.from_numpy(data['X_'+r].copy()) for r in ['fit','early_stop','selection']}
    weak=torch.from_numpy(data['weak_y_fit'].copy()).float()
    ysel=data['y_selection'].copy()
    require(set(np.unique(ysel))=={0,1} and int(weak.sum())==30,'Label budget/reference mismatch')
    model=SEADAdapted(21,p['latent_dim'],tuple(p['widths']))
    key=copy.deepcopy(model)
    for value in key.parameters():value.requires_grad=False
    queue=F.normalize(torch.randn(p['queue_size'],2),dim=1);ptr=0
    require(p['queue_size']%p['batch_size']==0,'Queue/batch mismatch')
    params=list(model.encoder.parameters())+list(model.decoder.parameters())+list(model.projector.parameters())
    optimizer=torch.optim.Adam(params,lr=p['learning_rate'],weight_decay=p['weight_decay'])
    history=[];best_loss=math.inf;best_epoch=None
    initial_encoder={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
    for epoch in range(p['pretrain_epochs']):
        model.train()
        lr=schedule(epoch,p['pretrain_epochs'],p['learning_rate'],1e-5,p['warmup_epochs'])
        for group in optimizer.param_groups:group['lr']=lr
        sums=np.zeros(3)
        for _ in range(p['batches_per_epoch']):
            ids=torch.randint(len(X['fit']),(p['batch_size'],))
            a=augment(X['fit'][ids],p['noise_std'],p['mask_probability'])
            b=augment(X['fit'][ids],p['noise_std'],p['mask_probability'])
            optimizer.zero_grad(set_to_none=True)
            z=model.encoder(a);q=F.normalize(model.projector(z),dim=1)
            residual=model.decoder(z)-a
            with torch.no_grad():
                for qm,km in [(model.encoder,key.encoder),(model.projector,key.projector)]:
                    for source,target in zip(qm.parameters(),km.parameters()):
                        target.mul_(p['momentum']).add_(source,alpha=1-p['momentum'])
                k=F.normalize(key.projector(key.encoder(b)),dim=1)
            logits=torch.cat([(q*k).sum(1,keepdim=True),q@queue.detach().clone().T],dim=1)/p['temperature']
            nce=F.cross_entropy(logits,torch.zeros(len(q),dtype=torch.long))
            reconstruction=residual.square().mean() # released-code MSE convention, not Eq.(21) L2
            loss=p['contrastive_weight']*nce+(1-p['contrastive_weight'])*reconstruction
            optimizer_step(loss,optimizer,params)
            with torch.no_grad():
                queue[ptr:ptr+len(k)]=k;ptr=(ptr+len(k))%len(queue)
            sums+=np.array([loss.item(),reconstruction.item(),nce.item()])
        val=float(infer(model,X['early_stop'],'mse').mean())
        record=dict(stage='pretrain',epoch=epoch+1,learning_rate=lr,loss=float(sums[0]/p['batches_per_epoch']),
                    reconstruction_MSE=float(sums[1]/p['batches_per_epoch']),InfoNCE=float(sums[2]/p['batches_per_epoch']),early_stop_clean_MSE=val)
        history.append(record)
        if val<best_loss:
            best_loss=val;best_epoch=epoch+1;save_model(out/'sclm_best.pt',model,best_epoch)
        write_json(out/'training_history.json',history)
        if epoch==0 or (epoch+1)%10==0 or epoch+1==p['pretrain_epochs']:
            print(f'ON_EGITIM {epoch+1}/{p["pretrain_epochs"]} MSE={record["reconstruction_MSE"]:.6g} erken_kume_MSE={val:.6g}',flush=True)
    state=torch.load(out/'sclm_best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(state['model'])
    delta=float(sum((v-initial_encoder[k]).square().sum().item() for k,v in model.encoder.state_dict().items()))
    require(delta>0,'Encoder parameters did not change')
    representations={f'Z_{r}':infer(model,X[r],'latent') for r in ['fit','selection']}
    representations['keys_fit']=data['keys_fit'];representations['keys_selection']=data['keys_selection']
    representations['y_selection']=ysel
    variants={}
    # Same pretrained encoder, head initialization, batch draws and 30 labels.
    # The source-loss control distinguishes a necessary Eq.(24) repair from an
    # unreported algorithm change. Primary variant is fixed before results.
    for variant in p['stage2_variants']:
        torch.manual_seed(plan['seed']+1000)
        finetuned=SEADAdapted(21,p['latent_dim'],tuple(p['widths']))
        finetuned.encoder.load_state_dict(model.encoder.state_dict())
        finetuned.decoder.load_state_dict(model.decoder.state_dict())
        params=list(finetuned.parameters())
        optimizer=torch.optim.Adam(params,lr=p['learning_rate'],weight_decay=p['weight_decay'])
        pos=torch.flatnonzero(weak==1) if hasattr(torch,'flatnonzero') else torch.nonzero(weak==1,as_tuple=True)[0]
        unl=torch.nonzero(weak==0,as_tuple=True)[0]
        best_ap=-math.inf;chosen_epoch=None;epoch_scores=[];variant_history=[]
        for epoch in range(p['finetune_epochs']):
            finetuned.train()
            lr=schedule(epoch,p['finetune_epochs'],p['learning_rate'],1e-4,p['warmup_epochs'])
            for group in optimizer.param_groups:group['lr']=lr
            loss_sum=0.0
            for _ in range(p['batches_per_epoch']):
                half=p['batch_size']//2
                ids=torch.stack([unl[torch.randint(len(unl),(half,))],pos[torch.randint(len(pos),(half,))]],dim=1).flatten()
                optimizer.zero_grad(set_to_none=True)
                _,_,_,error,direction,score=finetuned(X['fit'][ids])
                loss=stage2_loss(score,error,direction,weak[ids],p['margin'],variant)
                optimizer_step(loss,optimizer,params)
                loss_sum+=loss.item()
            values=infer(finetuned,X['selection'])
            ap=average_precision(ysel,values);epoch_scores.append(values)
            rec=dict(stage=variant,epoch=epoch+1,learning_rate=lr,loss=loss_sum/p['batches_per_epoch'],selection_AP=ap)
            history.append(rec);variant_history.append(rec)
            if ap>best_ap:
                best_ap=ap;chosen_epoch=epoch+1;save_model(out/f'sead_{variant}_best.pt',finetuned,chosen_epoch)
            write_json(out/'training_history.json',history)
            if epoch==0 or (epoch+1)%10==0 or epoch+1==p['finetune_epochs']:
                print(f'INCE_AYAR {variant} {epoch+1}/{p["finetune_epochs"]} kayip={rec["loss"]:.6g}',flush=True)
        epoch_scores=np.stack(epoch_scores)
        selected=epoch_scores[chosen_epoch-1]
        checkpoint=torch.load(out/f'sead_{variant}_best.pt',map_location='cpu',weights_only=True)
        finetuned.load_state_dict(checkpoint['model'])
        repeated=infer(finetuned,X['selection'])
        require(np.array_equal(selected,repeated),'Saved model score reproduction failed')
        representations[f'APM_{variant}_all_epoch_scores']=epoch_scores
        representations[f'APM_{variant}_selection']=selected
        fine_delta=float(sum((v-model.encoder.state_dict()[k]).square().sum().item() for k,v in finetuned.encoder.state_dict().items()))
        require(fine_delta>0,'Stage2 encoder did not update')
        variants[variant]=dict(selected_epoch=chosen_epoch,selection_AP=best_ap,encoder_squared_parameter_change=fine_delta,
                               score_min=float(selected.min()),score_max=float(selected.max()),checkpoint_score_reproduced=True)
    np.savez_compressed(out/'trained_outputs.npz',**representations)
    result=dict(status='PASS_TRAINING',fold=plan['fold'],seed=plan['seed'],pretrain_best_epoch=best_epoch,
                pretrain_early_stop_MSE=best_loss,pretrain_encoder_squared_parameter_change=delta,
                stage2=variants,outer_evaluation_performed=False,device='cpu',threads=1,
                python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                elapsed_seconds=time.perf_counter()-started,architecture=p,
                trained_outputs_sha256=hashlib.sha256((out/'trained_outputs.npz').read_bytes()).hexdigest())
    write_json(out/'training_summary.json',result)
    print('EGITIM_TAMAM',flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]);plan=json.loads((out/'plan.json').read_text())
    train(out,plan)
