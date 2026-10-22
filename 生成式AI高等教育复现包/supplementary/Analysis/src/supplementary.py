from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from common import config, design,  quantile, read_scored, read_csv, write_json, validate_design
from inference import fit_export, slope, vertex


def icc_oneway(y, groups):
    y=np.asarray(y,float);groups=np.asarray(groups)
    if y.shape!=groups.shape or not np.isfinite(y).all():
        raise ValueError('Invalid outcome/group vectors.')
    levels=np.unique(groups);n=len(y);g=len(levels)
    if not 1<g<n:raise ValueError('ICC requires at least two nonempty replicated groups.')
    blocks=[y[groups==v] for v in levels];sizes=np.array([len(a) for a in blocks])
    msb=sum(len(a)*(a.mean()-y.mean())**2 for a in blocks)/(g-1)
    msw=sum(np.sum((a-a.mean())**2) for a in blocks)/(n-g)
    n0=(n-np.sum(sizes**2)/n)/(g-1)
    vb=(msb-msw)/n0
    return {'n':n,'groups':g,'effective_cluster_size':n0,'MS_between':msb,
            'MS_within':msw,'between_variance':vb,'within_variance':msw,
            'ICC':vb/(vb+msw) if vb+msw>0 else np.nan}


def rcs_nonlinear(x, knots):
    x=np.asarray(x,float);t=np.asarray(knots,float)
    if len(t)!=5 or np.any(np.diff(t)<=0):raise ValueError('Five distinct increasing knots required.')
    plus=lambda z: np.maximum(z,0)**3
    return np.column_stack([(plus(x-tj)-plus(x-t[-2])*(t[-1]-tj)/(t[-1]-t[-2])
        +plus(x-t[-1])*(t[-2]-tj)/(t[-1]-t[-2]))/(t[-1]-t[0])**2 for tj in t[:-2]])


def htmt_value(A, B):
    A=np.asarray(A,float);B=np.asarray(B,float)
    if A.ndim!=2 or B.ndim!=2 or len(A)!=len(B) or min(A.shape[1],B.shape[1])<2:
        raise ValueError('HTMT needs two item matrices with at least two columns.')
    z=np.c_[A,B]
    if not np.isfinite(z).all() or np.any(z.std(axis=0)==0):return np.nan
    r=np.abs(np.corrcoef(z,rowvar=False));a=A.shape[1];b=B.shape[1]
    den=np.sqrt(r[:a,:a][np.triu_indices(a,1)].mean()*r[a:,a:][np.triu_indices(b,1)].mean())
    return float(r[:a,a:].mean()/den) if den>0 else np.nan


def htmt_bootstrap(d, itemsets, B, seed, return_draws=False):
    if not isinstance(B,(int,np.integer)) or isinstance(B,bool) or B<1:
        raise ValueError('HTMT bootstrap B must be a positive integer.')
    names=list(itemsets);pairs=list(combinations(names,2))
    if len(names)<2 or any(len(v)<2 for v in itemsets.values()):
        raise ValueError('HTMT requires at least two constructs with at least two indicators each.')
    allitems=sum([list(v) for v in itemsets.values()],[])
    if len(allitems)!=len(set(allitems)): raise ValueError('Compared HTMT constructs must have disjoint item sets.')
    X=d[allitems].to_numpy(float)
    if len(X)<3 or not np.isfinite(X).all(): raise ValueError('HTMT needs complete finite item rows.')
    indices={k:[allitems.index(c) for c in itemsets[k]] for k in names}
    def evaluate(x):
        if np.any(x.std(axis=0)==0):
            varying=x.std(axis=0)>0
        else: varying=np.ones(x.shape[1],dtype=bool)
        with np.errstate(divide='ignore',invalid='ignore'):
            r=np.abs(np.corrcoef(x,rowvar=False))
        v=[]
        for a,b in pairs:
            ia,ib=indices[a],indices[b]
            if not varying[ia+ib].all(): v.append(np.nan);continue
            aa=r[np.ix_(ia,ia)];bb=r[np.ix_(ib,ib)]
            den=np.sqrt(aa[np.triu_indices(len(ia),1)].mean()*bb[np.triu_indices(len(ib),1)].mean())
            v.append(r[np.ix_(ia,ib)].mean()/den if den>0 else np.nan)
        return np.asarray(v)
    estimates=evaluate(X);draws=np.full((B,len(pairs)),np.nan);rng=np.random.default_rng(seed)
    for draw in range(B): draws[draw]=evaluate(X[rng.integers(0,len(X),len(X))])
    rows=[]
    for j,(a,b) in enumerate(pairs):
        v=draws[:,j];v=v[np.isfinite(v)]
        ci=quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
        rows.append({'construct_a':a,'construct_b':b,'HTMT':estimates[j],
                     'CI_low':ci[0],'CI_high':ci[1],'requested':B,'valid':len(v),'seed':seed})
    frame=pd.DataFrame(rows)
    if return_draws:
        detail=pd.DataFrame(draws,columns=[a+'__'+b for a,b in pairs])
        detail.insert(0,'draw',np.arange(1,B+1));return frame,detail
    return frame


def wald_columns(fit, columns, use_f=True):
    names=list(fit.params.index);R=np.zeros((len(columns),len(names)))
    for i,c in enumerate(columns):R[i,names.index(c)]=1
    w=fit.wald_test(R,use_f=use_f,scalar=True)
    if use_f:
        return {'F':float(w.statistic),'df_num':len(columns),'df_den':float(fit.df_resid),
                'p':float(w.pvalue),'reference_distribution':'F'}
    return {'W':float(w.statistic),'df':len(columns),'p':float(w.pvalue),
            'reference_distribution':'chi_square'}


def pruning_items(load, expected_items, threshold=.60):
    indicators=load.loc[(load.op=='=~') & load.rhs.isin(expected_items)]
    if (set(indicators.rhs)!=set(expected_items) or indicators.rhs.duplicated().any()
            or not np.isfinite(indicators['est.std']).all()):
        raise ValueError('Incomplete or nonunique focal loadings for pruning.')
    return set(indicators.loc[indicators['est.std'].abs()<threshold,'rhs'])


def fitted_shape(fit, X, centres, d):
    q=quantile(d.literacy,[.25,.5,.75]);bounds=quantile(d.use,[.05,.95]);rows=[]
    for label,l in zip(['low','median','high'],q):
        lo=slope(fit,list(X),centres,float(bounds[0]),l)
        hi=slope(fit,list(X),centres,float(bounds[1]),l)
        observed_lo=slope(fit,list(X),centres,float(d.use.min()),l)
        observed_hi=slope(fit,list(X),centres,float(d.use.max()),l)
        maximum=vertex(fit.params,list(X),centres,l)
        rows.append({'literacy_level':label,'maximum':maximum,
                     'lower_use':bounds[0],'upper_use':bounds[1],
                     'lower_slope':lo['slope'],'upper_slope':hi['slope'],
                     'p_U':max(lo['p_positive'],hi['p_negative']),
                     'observed_lower_slope':observed_lo['slope'],'observed_upper_slope':observed_hi['slope'],
                     'p_U_observed':max(observed_lo['p_positive'],observed_hi['p_negative']),
                     'interior_observed':bool(np.isfinite(maximum) and d.use.min()<maximum<d.use.max()),
                     'interior_Q05_Q95':bool(np.isfinite(maximum) and bounds[0]<maximum<bounds[1])})
    return pd.DataFrame(rows)


def run_role(d,cfg,role,out,B,measurement=None):
    out.mkdir(parents=True,exist_ok=True);X,centres=design(d,cfg,role);diagnostics=[];inst=[]
    for outcome in ['capability','dependence']:
        y=d[outcome].to_numpy(float);path=out/outcome
        full=fit_export(X,y,path/'expanded');ordinary=sm.OLS(y,X).fit()
        influence=ordinary.get_influence();h=influence.hat_matrix_diag;cook=influence.cooks_distance[0]
        cut_h,cut_c=quantile(h,[.99])[0],quantile(cook,[.99])[0];keep=(h<=cut_h)&(cook<=cut_c)
        trimmed=fit_export(X.loc[keep],y[keep],path/'influence_trimmed')
        fitted_shape(trimmed,X.loc[keep],centres,d).to_csv(path/'influence_trimmed'/'shape.csv',index=False)
        dx=X.copy();dx['fitted_sq']=ordinary.fittedvalues**2;dx['fitted_cube']=ordinary.fittedvalues**3
        reset=sm.OLS(y,dx).fit(cov_type='HC3',use_t=True);rr=wald_columns(reset,['fitted_sq','fitted_cube'])
        bp=het_breuschpagan(ordinary.resid,X,robust=False)
        knots=quantile(d.use,[.05,.275,.5,.725,.95]);xs=X.drop(columns='use_sq').copy()
        basis=rcs_nonlinear(d.use,knots)
        for j in range(3):xs[f'rcs_{j+1}']=basis[:,j]
        sp=fit_export(xs,y,path/'restricted_cubic_spline')
        spline_terms=['rcs_1','rcs_2','rcs_3','literacy_use']
        sw=wald_columns(sp,spline_terms,use_f=False)
        sf=wald_columns(sp,spline_terms,use_f=True)
        write_json(path/'restricted_cubic_spline'/'wald_test.json',sw)
        write_json(path/'restricted_cubic_spline'/'basis_definition.json',{'knots':knots,'quantile_type':7,'joint_terms':['rcs_1','rcs_2','rcs_3','literacy_use']})
        write_json(path/'influence_trimmed'/'selection.json',{'leverage_Q99':cut_h,'Cooks_D_Q99':cut_c,'rule':'union_strictly_above_either_threshold','original_centres':centres,'kept_n':int(keep.sum()),'removed_n':int((~keep).sum())})
        diagnostics.append({'outcome':outcome,'RESET_F':rr['F'],'RESET_p':rr['p'],
            'BP_LM':bp[0],'BP_p':bp[1],'RCS_Wald_W':sw['W'],'RCS_Wald_df':sw['df'],
            'RCS_Wald_p':sw['p'],'RCS_Wald_reference':sw['reference_distribution'],
            'RCS_Wald_F':sf['F'],'RCS_Wald_F_p':sf['p'],
            'removed_n':int((~keep).sum()),'max_leverage':h.max(),'max_Cooks_D':cook.max(),
            'cases_Cooks_D_above_4_over_n':int((cook>4/len(d)).sum()),
            'exact_boundary_n':int(((y==1)|(y==7)).sum()),
            'exact_boundary_percent':100*float(((y==1)|(y==7)).mean()),
            'max_abs_DFBETA_squared':abs(influence.dfbetas[:,list(X).index('use_sq')]).max(),
            'max_abs_DFBETA_interaction':abs(influence.dfbetas[:,list(X).index('literacy_use')]).max()})
        pd.DataFrame({'respondent_id':d.respondent_id,'leverage':h,'Cooks_D':cook,'retained':keep}).to_csv(path/'influence_private.csv',index=False)
        record=icc_oneway(y,d.institution);record['outcome']=outcome;inst.append(record)
        levels=sorted(d.institution.unique(),key=str);xfe=X.copy()
        for site in levels[1:]:xfe['site__'+str(site)]=(d.institution==site).astype(float)
        if np.linalg.matrix_rank(xfe)<xfe.shape[1]:raise ValueError('Institution FE design rank deficient.')
        fit_export(xfe,y,path/'institution_FE')
        for site_index,site in enumerate(levels,1):
            ds=d.loc[d.institution!=site].copy().reset_index(drop=True);xx,cc=design(ds,cfg,role)
            ff=fit_export(xx,ds[outcome].to_numpy(float),path/'leave_one_site_out'/f'site_{site_index:02d}')
            fitted_shape(ff,xx,cc,ds).to_csv(path/'leave_one_site_out'/f'site_{site_index:02d}'/'shape.csv',index=False)
        if measurement is not None:
            fp=Path(measurement)/f'{role}_literacy_second'/'factor_scores_private.csv'
            if not fp.exists():raise FileNotFoundError(f'Lavaan literacy factor scores missing: {fp}')
            score=read_csv(fp)
            if 'cohort' not in score or set(score.cohort)!={role}: raise ValueError('CFA score file belongs to another cohort.')
            if set(score.respondent_id.astype(str))!=set(d.respondent_id.astype(str)): raise ValueError('CFA score file respondent set differs from the regression sample.')
            ds=d.merge(score[['respondent_id','literacy_factor_score_rescaled']],on='respondent_id',validate='one_to_one',how='left')
            if ds.literacy_factor_score_rescaled.isna().any():raise ValueError('Incomplete CFA score merge.')
            xx,cc=design(ds,cfg,role,literacy_col='literacy_factor_score_rescaled')
            ff=fit_export(xx,ds[outcome].to_numpy(),path/'literacy_factor_score')
            dshape=ds.copy();dshape['literacy']=ds.literacy_factor_score_rescaled
            fitted_shape(ff,xx,cc,dshape).to_csv(path/'literacy_factor_score'/'shape.csv',index=False)
        if measurement is not None:
            lp=Path(measurement)/f'{role}_focal_first'/'standardized_loadings.csv'
            load=pd.read_csv(lp)
            expected_items=sum([list(cfg['scales'][k]) for k in ['literacy','use','trust','habit','risk','capability_'+role,'dependence_'+role]],[])
            bad=pruning_items(load,expected_items)
            ds=d.copy();mapping={'literacy':'literacy','use':'use','trust':'trust','habit':'habit','risk':'risk',
                                 'capability':f'capability_{role}','dependence':f'dependence_{role}'}
            retained={}
            for col,key in mapping.items():
                items=[a for a in cfg['scales'][key] if a not in bad]
                if not items:raise ValueError(f'No retained indicators for {col}.')
                ds[col]=ds[items].mean(axis=1);retained[col]=items
            xx,cc=design(ds,cfg,role);ff=fit_export(xx,ds[outcome].to_numpy(),path/'indicator_pruning')
            (path/'indicator_pruning'/'items.json').write_text(json.dumps({'threshold':.60,'rule':'absolute_standardized_loading_strictly_below',
                'source_model':f'{role}_focal_first','removed':sorted(bad),'retained':retained},indent=2))
            fitted_shape(ff,xx,cc,ds).to_csv(path/'indicator_pruning'/'shape.csv',index=False)
    pd.DataFrame({'folder':[f'site_{i:02d}' for i in range(1,len(levels)+1)],'omitted_institution':levels}).to_csv(out/'site_index_private.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv',index=False)
    pd.DataFrame(inst).to_csv(out/'institution_ICC.csv',index=False)
    sets={x:cfg['scales'][x] for x in ['literacy','use','trust','habit','risk']}
    sets['capability']=cfg['scales']['capability_'+role];sets['dependence']=cfg['scales']['dependence_'+role]
    ht,draws=htmt_bootstrap(d,sets,B,cfg['seed'],return_draws=True)
    ht.to_csv(out/'HTMT.csv',index=False);draws.to_csv(out/'HTMT_bootstrap.csv',index=False)
    ht,draws=htmt_bootstrap(d,{'belief_trust':cfg['scales']['belief_trust'],'dependence':sets['dependence']},B,cfg['seed'],return_draws=True)
    ht.to_csv(out/'belief_HTMT.csv',index=False);draws.to_csv(out/'belief_HTMT_bootstrap.csv',index=False)


def pooled_contrasts(data,cfg,out):
    from scipy.linalg import block_diag
    fits={};rows=[]
    for outcome in ['capability','dependence']:
        arrays=[];ys=[];names=[]
        for role in ['student','teacher']:
            d=data[role];X,_=design(d,cfg,role);z=X.copy()
            for c in z:
                if c!='Intercept':z[c]=(z[c]-z[c].mean())/z[c].std(ddof=1)
            arrays.append(z.to_numpy());ys.append((d[outcome]-d[outcome].mean())/d[outcome].std(ddof=1))
            names.extend(role+'__'+c for c in z)
        xx=pd.DataFrame(block_diag(*arrays),columns=names);y=np.concatenate(ys);f=fit_export(xx,y,out/outcome)
        for term in ['use_sq','literacy_use','literacy_c','trust']:
            c=np.zeros(len(names));c[names.index('student__'+term)]=-1;c[names.index('teacher__'+term)]=1
            est=float(c@f.params);se=float(np.sqrt(c@f.cov_params()@c));crit=st.t.ppf(.975,f.df_resid)
            rows.append({'outcome':outcome,'term':term,'teacher_minus_student':est,'HC3_SE':se,
                         'CI_low':est-crit*se,'CI_high':est+crit*se,'p':2*st.t.sf(abs(est/se),f.df_resid)})
    pd.DataFrame(rows).to_csv(out/'cross_role_contrasts.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--measurement');p.add_argument('--config');p.add_argument('--bootstrap',type=int,default=5000)
    a=p.parse_args()
    if a.bootstrap<1:p.error('bootstrap must be positive')
    cfg=config(a.config);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);datasets={}
    for role in ['student','teacher']:
        d=read_scored(a.data,role,cfg);datasets[role]=d
        run_role(d,cfg,role,out/role,a.bootstrap,a.measurement)
    pooled_contrasts(datasets,cfg,out/'cross_role')

if __name__=='__main__':main()
