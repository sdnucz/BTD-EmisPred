"""Recompute reviewer 1(a)/(b) tables and figures from saved prediction files."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

TARGET = 'λem (nm)'
ORDER = ['CAT','XGB','LGBM','RF','GBR','KNN','KRR','SVR']
PANELS = ['CAT','SVR','KRR','RF','LGBM','GBR','XGB','KNN']
NAMES = dict(zip(ORDER,['CatBoost','XGBoost','LightGBM','RF','GBR','KNN','KRR','SVR']))
COLORS = dict(zip(ORDER,['#DCD5F6','#FADAB5','#C2E8DE','#E6D4E9','#C4DDEE','#F5B7BF','#E0E7BC','#E5E5E5']))

def metrics(d):
    if d.empty:
        raise ValueError('An evaluation subset has no records.')
    y=d[TARGET].to_numpy(float); p=d.prediction_nm.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite labels or predictions.')
    error=p-y; denominator=np.sum((y-y.mean())**2)
    return dict(n_records=len(d),n_molecules=d.canonical_smiles.nunique(),
                MAE=float(np.mean(abs(error))),RMSE=float(np.sqrt(np.mean(error**2))),
                R2=float(1-np.sum(error**2)/denominator) if denominator>0 else None)

def validate(oof,test,candidates,choices):
    for d in [oof,test]:
        if d.record_id.duplicated().any(): raise ValueError('Duplicate evaluation record IDs.')
        metrics(d)
    if set(oof.canonical_smiles)&set(test.canonical_smiles):
        raise ValueError('Training/test molecule overlap.')
    if set(oof.record_id)&set(test.record_id):raise ValueError('Training/test record overlap.')
    if oof.groupby('canonical_smiles').outer_fold.nunique().max()!=1:
        raise ValueError('A molecule spans outer validation folds.')
    folds=set(oof.outer_fold)
    if folds!=set(range(1,11)):raise ValueError('Expected 10 outer folds.')
    if set(candidates.algorithm)!=set(ORDER):raise ValueError('Expected eight candidates.')
    if choices.outer_fold.duplicated().any() or set(choices.outer_fold)!=folds:
        raise ValueError('Invalid fold-selection table.')
    reference=oof.sort_values('record_id').reset_index(drop=True)
    for name,g in candidates.groupby('algorithm'):
        g=g.sort_values('record_id').reset_index(drop=True)
        pd.testing.assert_frame_equal(g[['record_id','canonical_smiles','outer_fold',TARGET]],
                                      reference[['record_id','canonical_smiles','outer_fold',TARGET]])
        metrics(g)
    for row in choices.itertuples():
        a=oof[oof.outer_fold.eq(row.outer_fold)].sort_values('record_id')
        b=candidates[candidates.outer_fold.eq(row.outer_fold)&candidates.algorithm.eq(row.algorithm)].sort_values('record_id')
        np.testing.assert_allclose(a.prediction_nm,b.prediction_nm,rtol=0,atol=1e-7)

def save(fig,output,name):
    for ax in fig.axes:
        if ax.axison:
            low,high=ax.get_ylim()
            ax.set_yticks([v for v in ax.get_yticks() if low <= v <= high])
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        if not ax.axison:continue
        for t in ax.findobj(matplotlib.text.Text):
            if not t.get_visible() or not t.get_text():continue
            b=t.get_window_extent(renderer)
            if b.x0<-1 or b.y0<-1 or b.x1>fig.bbox.x1+1 or b.y1>fig.bbox.y1+1:
                raise ValueError(f'Text outside canvas in {name}: {t.get_text()}')
    for ext in ['pdf','svg','png']:
        fig.savefig(output/f'{name}.{ext}',dpi=400)
    fig.savefig(output/f'{name}_preview.png',dpi=140)
    plt.close(fig)

def build(run,output,font_dir=None):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    if font_dir:
        for f in font_dir.glob('*.ttf'):font_manager.fontManager.addfont(str(f))
    font='Arial' if any(f.name=='Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
    plt.rcParams.update({'font.family':font,'font.size':10,'axes.labelsize':10,
                         'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    paths={k:run/v for k,v in {
        'oof':'Nested_OOF_Predictions.csv','test':'final/Predictions.csv',
        'candidates':'Candidate_Outer_OOF_Predictions.csv',
        'choices':'Outer_Algorithm_Selections.csv'}.items()}
    frames={k:pd.read_csv(p) for k,p in paths.items()}
    oof,test,allpred,choices=[frames[k] for k in ['oof','test','candidates','choices']]
    validate(oof,test,allpred,choices)
    inner_tables=[]
    for row in choices.itertuples():
        scores=pd.read_csv(run/'outer'/f'{row.outer_fold-1:02d}'/'Candidate_Inner_Scores.csv')
        if set(scores.algorithm)!=set(ORDER) or scores.algorithm.duplicated().any():
            raise ValueError('Incomplete inner candidate scores.')
        selected=scores[scores.algorithm.eq(row.algorithm)].mean_inner_RMSE.iloc[0]
        np.testing.assert_allclose(selected,scores.mean_inner_RMSE.min(),rtol=0,atol=1e-9)
        np.testing.assert_allclose(selected,row.mean_inner_RMSE,rtol=0,atol=1e-9)
        scores['outer_fold']=row.outer_fold;inner_tables.append(scores)
    pd.concat(inner_tables).to_csv(output/'R1b_Inner_Selection_Scores_By_Outer_Fold.csv',index=False)
    longrows=[]; sources=[]
    for label,d in [('Nested outer OOF',oof),('Independent test',test)]:
        sub=d[d[TARGET].ge(1000)].copy()
        longrows.append(dict(evaluation=label,threshold='experimental emission >= 1000 nm',**metrics(sub)))
        sub['evaluation']=label;sub['absolute_error_nm']=abs(sub.prediction_nm-sub[TARGET]);sources.append(sub)
    long=pd.DataFrame(longrows);long.to_csv(output/'R1a_Long_Wavelength_Metrics.csv',index=False)
    pd.concat(sources).to_csv(output/'R1a_Long_Wavelength_Source_Data.csv',index=False)
    pooled=pd.DataFrame([dict(algorithm=a,**metrics(allpred[allpred.algorithm.eq(a)])) for a in ORDER])
    folds=pd.DataFrame([dict(algorithm=a,outer_fold=int(f),**metrics(g)) for (a,f),g in allpred.groupby(['algorithm','outer_fold'])])
    pooled.to_csv(output/'R1b_Pooled_Outer_Metrics.csv',index=False)
    folds.to_csv(output/'R1b_Individual_Fold_Metrics.csv',index=False)
    selected=[]
    for r in choices.sort_values('outer_fold').itertuples():
        g=oof[oof.outer_fold.eq(r.outer_fold)]
        selected.append(dict(outer_fold=r.outer_fold,algorithm=r.algorithm,
                             mean_inner_RMSE=r.mean_inner_RMSE,**metrics(g)))
    pd.DataFrame(selected).to_csv(output/'R1b_Outer_Algorithm_Selections.csv',index=False)
    allpred.to_csv(output/'R1b_Eight_Algorithm_Outer_Predictions.csv',index=False)
    oof.to_csv(output/'R1b_Selected_Procedure_OOF.csv',index=False)
    summary=[]
    for a in ORDER:
        g=folds[folds.algorithm.eq(a)]
        row={'algorithm':a,'n_folds':len(g)}
        for m in ['RMSE','MAE','R2']:
            row[m+'_mean']=g[m].mean();row[m+'_SD']=g[m].std(ddof=1)
        summary.append(row)
    summary=pd.DataFrame(summary);summary.to_csv(output/'R1b_Fold_Mean_SD.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.0,4.1),layout='constrained')
    for ax,m in zip(axes,['MAE','RMSE']):
        ax.bar([0,1],long[m],width=.55,color=['#C4DDEE','#FADAB5'],edgecolor='#777',linewidth=.7)
        for i,row in long.iterrows():ax.text(i,row[m]+2,f'{row[m]:.2f}',ha='center',fontsize=11)
        ax.set_xticks([0,1],[f'Outer OOF\n{long.n_records.iloc[0]} records / {long.n_molecules.iloc[0]} molecules',f'Test set\n{long.n_records.iloc[1]} records / {long.n_molecules.iloc[1]} molecules'])
        ax.set_ylabel(f'{m} (nm)');ax.set_ylim(0,max(long[m])*1.25)
        ax.grid(axis='y',linestyle=':',alpha=.3);ax.set_axisbelow(True)
    fig.suptitle('Experimental emission ≥ 1000 nm',fontsize=13)
    save(fig,output,'R1a_Long_Wavelength_Performance')
    fig,axes=plt.subplots(3,3,figsize=(10.2,10.4))
    fig.subplots_adjust(left=.09,right=.98,bottom=.08,top=.93,wspace=.38,hspace=.43)
    fig.suptitle('8-model 10-fold CV: Validation folds (OOF)',fontsize=14,y=.977)
    upper=max(1600,100*np.ceil(max(allpred[TARGET].max(),allpred.prediction_nm.max())/100))
    lower=min(-100,100*np.floor(allpred.prediction_nm.min()/100))
    for ax,a in zip(axes.flat,PANELS):
        d=allpred[allpred.algorithm.eq(a)];m=metrics(d)
        ax.scatter(d[TARGET],d.prediction_nm,s=7,c=COLORS[a],edgecolors='#777777',linewidths=.2,alpha=.6)
        ax.plot([lower,upper],[lower,upper],ls='--',color='#555',lw=.8)
        ax.set(xlim=(lower,upper),ylim=(lower,upper),xticks=np.arange(0,upper+1,400),yticks=np.arange(0,upper+1,400))
        ax.set_aspect('equal');ax.set_title(NAMES[a]);ax.set_xlabel(r'Experimental $\lambda_{em}$ (nm)');ax.set_ylabel(r'Predicted $\lambda_{em}$ (nm)')
        ax.text(.025,.97,f'$R^2$ = {m["R2"]:.4f}\nRMSE = {m["RMSE"]:.1f} nm\nMAE = {m["MAE"]:.1f} nm',transform=ax.transAxes,va='top',fontsize=9,bbox={'facecolor':'white','edgecolor':'#ccc','pad':3})
    axes[2,2].axis('off');save(fig,output,'Supplementary_Fig_2')
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5))
    fig.subplots_adjust(left=.085,right=.98,bottom=.11,top=.91,wspace=.25,hspace=.40)
    fig.text(.085,.975,'Bars: mean ± SD across 10 outer folds; circles: individual folds',va='top',fontsize=12)
    for ax,m in zip([axes[0,0],axes[0,1],axes[1,0]],['RMSE','MAE','R2']):
        vals=summary[m+'_mean'].to_numpy();sd=summary[m+'_SD'].to_numpy()
        ax.bar(range(8),vals,width=.68,color=[COLORS[a] for a in ORDER],edgecolor='#969696',linewidth=.7)
        ax.errorbar(range(8),vals,yerr=sd,fmt='none',ecolor='#888',capsize=3,lw=.9)
        for i,a in enumerate(ORDER):
            v=folds[folds.algorithm.eq(a)].sort_values('outer_fold')[m].to_numpy()
            ax.scatter(i+np.linspace(-.20,.20,10),v,s=22,facecolors='white',edgecolors='#888',lw=.8,zorder=4)
            ax.text(i,vals[i]*.4,f'{vals[i]:.4f}' if m=='R2' else f'{vals[i]:.1f}',ha='center',va='top',fontsize=9,weight='bold')
        low=min(0,folds[m].min()-.04 if m=='R2' else 0)
        high=max(folds[m].max(),max(vals+sd))*1.16
        if m=='R2':high=max(1.08,high)
        ax.set_ylim(low,high);ax.set_ylabel('$R^2$' if m=='R2' else f'{m} (nm)')
        ax.set_xticks(range(8),[NAMES[a] for a in ORDER],rotation=25,ha='right',fontsize=9)
        ax.grid(axis='y',ls=':',alpha=.35);ax.set_axisbelow(True)
        ax.text(.99,.98,'higher is better' if m=='R2' else 'lower is better',transform=ax.transAxes,ha='right',va='top',color='#777',fontsize=9)
    axes[1,1].axis('off');save(fig,output,'Supplementary_Fig_3')
    provenance={'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
                'threshold':'>= 1000 nm based on experimental wavelength','font':font,
                'scatter_metrics':'pooled outer OOF','bar_metrics':'unweighted fold mean +/- sample SD (ddof=1)',
                'selected_procedure_metrics':metrics(oof),'tests_passed':['10 folds','8 candidates','no molecule overlap','selected OOF matches selected candidate in each fold']}
    (output/'Verification.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False))
    print(long.to_string(index=False));print(pd.DataFrame(selected).to_string(index=False));print('VERIFIED')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--font-dir',type=Path)
    args=p.parse_args();build(args.run,args.output,args.font_dir)
