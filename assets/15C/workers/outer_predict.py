from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
from train_sead import SEADAdapted,infer


def predict(source,out):
    plan=json.loads((source/'plan.json').read_text());p=plan['training']
    training=json.loads((source/'training_summary.json').read_text())
    if hashlib.sha256((source/'trained_outputs.npz').read_bytes()).hexdigest()!=training['trained_outputs_sha256']:raise ValueError('Training arrays changed')
    data=np.load(source/'prepared_data.npz',allow_pickle=False)
    recorded=np.load(source/'trained_outputs.npz',allow_pickle=False)
    outer=np.load(out/'outer_inputs.npz',allow_pickle=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    Xsel=torch.from_numpy(data['X_selection'].copy());Xout=torch.from_numpy(outer['X_outer'].copy())
    model=SEADAdapted(21,p['latent_dim'],tuple(p['widths']))
    checkpoint=torch.load(source/'sclm_best.pt',map_location='cpu',weights_only=True)
    if checkpoint['epoch']!=training['pretrain_best_epoch']:raise ValueError('SCLM epoch mismatch')
    model.load_state_dict(checkpoint['model'])
    if not np.array_equal(infer(model,Xsel,'latent'),recorded['Z_selection']):raise ValueError('SCLM selection representation differs')
    result={'Z_outer':infer(model,Xout,'latent'),'keys':outer['keys']}
    for variant in p['stage2_variants']:
        state=torch.load(source/f'sead_{variant}_best.pt',map_location='cpu',weights_only=True)
        if state['epoch']!=training['stage2'][variant]['selected_epoch']:raise ValueError('SEAD epoch mismatch')
        model.load_state_dict(state['model'])
        if not np.array_equal(infer(model,Xsel),recorded[f'APM_{variant}_selection']):raise ValueError('SEAD selection scores differ '+variant)
        result['SEAD_'+variant]=infer(model,Xout)
    np.savez_compressed(out/'outer_neural_outputs.npz',**result)
    report=dict(status='PASS_NEURAL_INFERENCE',selection_reproduction=True,training_performed=False,
                outer_rows=len(Xout),torch=torch.__version__,numpy=np.__version__,device='cpu',
                output_sha256=hashlib.sha256((out/'outer_neural_outputs.npz').read_bytes()).hexdigest())
    (out/'outer_predict_summary.json').write_text(json.dumps(report,indent=2))
    print('DIS_AG_SKORLARI_HAZIR',flush=True)


if __name__=='__main__':
    out=Path(sys.argv[1]);context=json.loads((out/'evaluation_context.json').read_text())
    predict(Path(context['source']),out)
