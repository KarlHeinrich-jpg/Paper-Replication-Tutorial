from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold,GridSearchCV
import joblib
from common import config,controls,pooled_metrics,read_scored,write_json,validate_controls
from transforms import UseTerms

def model_spec(family, seed):
    if family in ["ridge_linear", "ridge_nonlinear"]:
        return Ridge(), {"model__alpha": [.01, .1, 1., 10., 100.]}
    if family == "rf":
        model = RandomForestRegressor(random_state=seed, n_jobs=1)
        grid = {"model__n_estimators": [300, 600], "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [2, 5, 10], "model__max_features": ["sqrt", .5]}
        return model, grid
    if family == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1)
        grid = {"model__n_estimators": [300, 600], "model__max_depth": [2, 3, 5],
                "model__learning_rate": [.02, .05, .10], "model__subsample": [.7, .8, 1.],
                "model__colsample_bytree": [.7, .8, 1.]}
        return model, grid
    if family == "lightgbm":
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(n_estimators=600, random_state=seed, n_jobs=1, verbosity=-1)
        grid = {"model__num_leaves": [15, 31, 63], "model__min_child_samples": [10, 20, 40],
                "model__learning_rate": [.02, .05], "model__colsample_bytree": [.8, 1.]}
        return model, grid
    raise ValueError(f"Unknown model family: {family}")

def pipeline(family,num,cat,seed):
    nonlinear=family=='ridge_nonlinear';numeric=num+(['use_sq','literacy_use'] if nonlinear else [])
    preprocess=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)]).set_output(transform='pandas')
    model,grid=model_spec(family,seed)
    return Pipeline([('terms',UseTerms(nonlinear)),('preprocess',preprocess),('model',model)]),grid

def parent_name(encoded,cat):
    if encoded.startswith('num__'):return encoded[5:]
    for c in cat:
        if encoded.startswith('cat__'+c+'_'):return c
    raise ValueError(f'Cannot map transformed feature {encoded} to original predictor.')

def shap_for_fold(pipe,Xtest,ids,role,outcome,fold,model_id,cat,predictions=None):
    import shap
    transformed=pipe['preprocess'].transform(pipe['terms'].transform(Xtest));names=pipe['preprocess'].get_feature_names_out();model=pipe['model']
    exp=shap.TreeExplainer(model,model_output='raw',feature_perturbation='tree_path_dependent')(transformed,check_additivity=True)
    values=np.asarray(exp.values);base=np.asarray(exp.base_values).reshape(-1)
    pred=np.asarray(pipe.predict(Xtest) if predictions is None else predictions)
    if pred.shape!=(len(Xtest),) or not np.isfinite(pred).all():raise ValueError('Invalid saved OOF predictions.')
    if base.size==1: base=np.repeat(base,len(Xtest))
    if values.shape!=(len(Xtest),len(names)) or base.shape!=(len(Xtest),): raise AssertionError('Unexpected TreeSHAP output dimensions.')
    if not np.allclose(base+values.sum(axis=1),pred,rtol=2e-5,atol=2e-5):raise AssertionError('SHAP additivity does not match the saved OOF prediction.')
    groups={}
    for j,n in enumerate(names):groups.setdefault(parent_name(n,cat),[]).append(j)
    rows=[]
    for i,rid in enumerate(ids):
        row={'respondent_id':rid,'cohort':role,'outcome':outcome,'fold':fold,'model_id':model_id,'base_value':float(base[i]),'prediction':float(pred[i])}
        for name,idx in groups.items():row['SHAP__'+name]=float(values[i,idx].sum())
        rows.append(row)
    return pd.DataFrame(rows)

def run_role(d,cfg,role,out,families,jobs=1):
    out.mkdir(parents=True,exist_ok=True);num0,cat=controls(cfg,role,expanded=True);num=['literacy','use','trust','habit','risk']+num0;X=d[num+cat].copy()
    validate_controls(d,cfg,role,allow_missing=True)
    if not families or len(families)!=len(set(families)): raise ValueError('Specify distinct, nonempty model families.')
    if len(d)<10: raise ValueError('At least ten observations are required for the declared 5x4 CV.')
    for c in ['literacy','use','trust','habit','risk']:
        if X[c].isna().any():raise ValueError('Focal scores are required.')
    splits=list(KFold(5,shuffle=True,random_state=cfg['seed']).split(X));oofrows=[];metrics=[];foldmetrics=[];selected_models=[];shaprows=[]
    split_rows=[]
    for fold,(tr,te) in enumerate(splits,1):
        for idx in te: split_rows.append({'respondent_id':str(d.respondent_id.iloc[idx]),'cohort':role,'outer_fold':fold,'row_index':int(idx)})
    pd.DataFrame(split_rows).sort_values('row_index').to_csv(out/'outer_folds_private.csv',index=False)
    for outcome in ['capability','dependence']:
        y=d[outcome].to_numpy(float)
        for family in families:
            pred=np.full(len(d),np.nan);seen=np.zeros(len(d),int)
            for f,(tr,te) in enumerate(splits):
                seed=cfg['seed']+f+1;pipe,grid=pipeline(family,num,cat,seed);inner=KFold(4,shuffle=True,random_state=seed)
                inner_splits=list(inner.split(X.iloc[tr]))
                if any(set(itr)&set(ite) for itr,ite in inner_splits): raise AssertionError('Inner fold overlap.')
                search=GridSearchCV(pipe,grid,scoring='neg_root_mean_squared_error',cv=inner_splits,refit=True,n_jobs=jobs,error_score='raise',return_train_score=False)
                search.fit(X.iloc[tr],y[tr]);best=search.best_estimator_;p=best.predict(X.iloc[te]);pred[te]=p;seen[te]+=1
                model_id=f'{role}_{outcome}_{family}_fold{f+1}';mp=out/'models'/f'{model_id}.joblib';mp.parent.mkdir(exist_ok=True);joblib.dump(best,mp)
                pd.DataFrame(search.cv_results_).to_csv(out/'models'/f'{model_id}_inner_search.csv',index=False)
                selected_models.append({'model_id':model_id,'cohort':role,'outcome':outcome,
                    'family':family,'fold':f+1,'seed':seed,'train_n':len(tr),'test_n':len(te),
                    'best_params':search.best_params_,'inner_RMSE':-search.best_score_,
                    'training_centres':{'literacy':best['terms'].L_mean_,'use':best['terms'].U_mean_}})
                fm=pooled_metrics(y[te],p);fm.update(cohort=role,outcome=outcome,family=family,fold=f+1);foldmetrics.append(fm)
                for idx,pi in zip(te,p):oofrows.append({'respondent_id':d.respondent_id.iloc[idx],'cohort':role,'outcome':outcome,'family':family,'fold':f+1,'model_id':model_id,'y':y[idx],'prediction':pi})
                if family=='xgboost':shaprows.append(shap_for_fold(best,X.iloc[te],d.respondent_id.iloc[te].to_numpy(),role,outcome,f+1,model_id,cat,predictions=p))
            if not np.all(seen==1) or not np.isfinite(pred).all():raise AssertionError('Every respondent must have exactly one OOF prediction per family/outcome.')
            m=pooled_metrics(y,pred);m.update(cohort=role,outcome=outcome,family=family);metrics.append(m)
    pd.DataFrame(oofrows).to_csv(out/'oof_predictions_private.csv',index=False);pd.DataFrame(metrics).to_csv(out/'pooled_metrics.csv',index=False);pd.DataFrame(foldmetrics).to_csv(out/'fold_metrics.csv',index=False);write_json(out/'best_parameters.json',selected_models)
    if shaprows:
        allshap=pd.concat(shaprows,ignore_index=True)
        shapcols=[c for c in allshap if c.startswith('SHAP__')]
        if allshap[shapcols].isna().any().any(): raise AssertionError('Missing fold-level feature attributions.')
        allshap.to_csv(out/'heldout_shap_private.csv',index=False);rows=[]
        for outcome,g in allshap.groupby('outcome'):
            for c in allshap:
                if c.startswith('SHAP__'):rows.append({'cohort':role,'outcome':outcome,'feature':c[6:],'mean_absolute_SHAP':float(g[c].abs().mean()),'n':len(g)})
        pd.DataFrame(rows).sort_values(['outcome','mean_absolute_SHAP'],ascending=[True,False]).to_csv(out/'global_shap.csv',index=False)

    winners=[]
    for outcome in ['capability','dependence']:
        mm=[r for r in metrics if r['outcome']==outcome]
        best=max(mm,key=lambda r:r['R2'])
        fm=[r['R2'] for r in foldmetrics if r['outcome']==outcome and r['family']==best['family']]
        winners.append({**best,'fold_R2_min':min(fm),'fold_R2_max':max(fm),
                        'selection_interpretation':'descriptive_family_comparison_not_external_validation'})
    pd.DataFrame(winners).to_csv(out/'best_family_summary.csv',index=False)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.add_argument("--families", nargs="+", choices=["ridge_linear", "ridge_nonlinear", "rf", "xgboost", "lightgbm"],
                   default=["ridge_linear", "ridge_nonlinear", "rf", "xgboost", "lightgbm"])
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()
    if a.jobs == 0:
        p.error("jobs must be a positive integer or -1.")
    cfg, out = config(a.config), Path(a.out)
    for role in ["student", "teacher"]:
        run_role(read_scored(a.data, role, cfg), cfg, role, out / role, a.families, a.jobs)
if __name__=='__main__':main()
