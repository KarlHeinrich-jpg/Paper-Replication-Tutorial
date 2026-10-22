#!/usr/bin/env Rscript

need_packages <- function() {
  for (pkg in c('lavaan', 'jsonlite')) {
    if (!requireNamespace(pkg, quietly=TRUE)) stop(paste('Missing R package:', pkg))
  }
}

write_json <- function(x, file) {
  jsonlite::write_json(x, file, auto_unbox=TRUE, pretty=TRUE, digits=NA, na='null', null='null')
}

fit_model <- function(syntax, d, spec, do.fit=TRUE) {
  vars <- lavaan::lavNames(lavaan::lavaanify(syntax), 'ov')
  if (!all(vars %in% names(d))) stop(paste('Missing indicators:', paste(setdiff(vars,names(d)),collapse=', ')))
  if (anyNA(d[vars])) stop('CFA indicators contain missing data; no silent listwise deletion.')
  good <- vapply(d[vars], function(x) is.numeric(x) && all(is.finite(x)) && all(x>=1 & x<=7 & x==round(x)), logical(1))
  if (!all(good)) stop(paste('CFA items must be integer codes 1--7:', paste(vars[!good],collapse=', ')))
  if (any(vapply(d[vars], stats::var, numeric(1)) <= 0)) stop('Constant CFA indicator.')
  args <- list(model=syntax, data=d, estimator='ML', meanstructure=TRUE,
               std.lv=isTRUE(spec$std_lv), auto.cov.lv.x=FALSE, auto.cov.y=FALSE,
               fixed.x=FALSE, missing='listwise', do.fit=do.fit)
  if (!is.null(spec$group)) {
    args$group <- 'cohort'
    args$group.label <- c('student','teacher')
    args$group.equal <- unlist(spec$group_equal, use.names=FALSE)
  }
  do.call(lavaan::cfa, args)
}

alpha_raw <- function(x) {
  k <- ncol(x); S <- stats::cov(x)
  if (k < 2 || sum(S) <= 0) return(NA_real_)
  k/(k-1) * (1-sum(diag(S))/sum(S))
}

reliability <- function(fit, d, factors, id, out) {
  est <- lavaan::lavInspect(fit, 'est')
  L <- est$lambda; Theta <- est$theta; B <- est$beta
  if (is.null(B)) B <- matrix(0, ncol(L), ncol(L), dimnames=list(colnames(L),colnames(L)))
  M <- solve(diag(nrow(B))-B); dimnames(M) <- dimnames(B)
  Phi <- M %*% est$psi %*% t(M)
  Common <- L %*% Phi %*% t(L); Sigma <- Common + Theta
  std <- lavaan::standardizedSolution(fit, type='std.all')
  write.csv(Common, file.path(out,'common_covariance.csv'))
  write.csv(Theta, file.path(out,'residual_covariance.csv'))
  rows <- list()
  for (f in names(factors)) {
    items <- unlist(factors[[f]], use.names=FALSE)
    ss <- std[std$op=='=~' & std$lhs==f & std$rhs %in% items, ]
    if (nrow(ss)!=length(items) || anyDuplicated(ss$rhs)) stop('Nonunique/missing factor loadings for reliability.')
    lam <- ss$est.std[match(items,ss$rhs)]
    theta <- diag(Theta)[items] / diag(Sigma)[items]
    if (any(!is.finite(c(lam,theta))) || any(theta<0)) stop('Invalid reliability inputs.')
    total <- sum(Sigma[items,items,drop=FALSE]); common <- sum(Common[items,items,drop=FALSE])
    rows[[length(rows)+1L]] <- data.frame(model_id=id,factor=f,items=length(items),
      alpha=alpha_raw(d[items]),omega_total=common/total,
      raw_common_sum=common,raw_total_sum=total,
      CR_standardized=sum(lam)^2/(sum(lam)^2+sum(theta)),
      AVE=sum(lam^2)/(sum(lam^2)+sum(theta)),mean_squared_loading=mean(lam^2))
    write.csv(data.frame(item=items,lambda_std=lam,theta_std=theta),
              file.path(out,paste0('reliability_inputs_',f,'.csv')),row.names=FALSE)
  }
  write.csv(do.call(rbind,rows),file.path(out,'reliability.csv'),row.names=FALSE)
  if ('G' %in% colnames(L)) {
    items <- unlist(factors, use.names=FALSE)
    gpath <- as.vector(L[items,,drop=FALSE] %*% M[,'G'])
    omega_h <- sum(gpath)^2*est$psi['G','G']/sum(Sigma[items,items,drop=FALSE])
    so <- std[std$lhs=='G' & std$op=='=~',]$est.std
    write.csv(data.frame(item=items,general_loading_raw=gpath),file.path(out,'general_factor_paths.csv'),row.names=FALSE)
    high <- data.frame(model_id=id,omega_h=omega_h,general_factor_variance=est$psi['G','G'],
      raw_total_sum=sum(Sigma[items,items,drop=FALSE]),
      mean_squared_second_order_loading=mean(so^2),
      CR_second_order=sum(so)^2/(sum(so)^2+sum(1-so^2)))
    write.csv(high,file.path(out,'higher_order_reliability.csv'),row.names=FALSE)
    write.csv(std[std$lhs=='G' & std$op=='=~',],file.path(out,'second_order_loadings.csv'),row.names=FALSE)
  }
  invisible(rows)
}

export_fit <- function(fit, syntax, spec, out, warnings=character()) {
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  id <- spec$model_id
  writeLines(syntax,file.path(out,'model.lav'),useBytes=TRUE)
  saveRDS(fit,file.path(out,'fit_private.rds'))
  pt <- lavaan::parTable(fit)
  write.csv(pt,file.path(out,'parameter_table.csv'),row.names=FALSE)
  converged <- isTRUE(lavaan::lavInspect(fit,'converged'))
  post <- if(converged) isTRUE(suppressWarnings(lavaan::lavInspect(fit,'post.check'))) else FALSE
  measures <- tryCatch(lavaan::fitMeasures(fit),error=function(e) numeric())
  df <- if('df' %in% names(measures)) unname(measures['df']) else NA_real_
  p <- length(lavaan::lavNames(fit,'ov')); G <- lavaan::lavInspect(fit,'ngroups')
  moments <- G*(p*(p+1)/2+p)
  ok <- converged && post && is.finite(df)
  info <- list(model_id=id,indicators=lavaan::lavNames(fit,'ov'),nobs=lavaan::lavInspect(fit,'nobs'),
    raw_free_parameter_count=lavaan::lavInspect(fit,'npar'),
    equality_constraint_rows=sum(pt$op=='=='),sample_moments=moments,
    effective_free_parameters=if(is.finite(df))moments-df else NULL,
    df=df,converged=converged,post_check=post,valid_for_export=ok,
    estimator='ML',meanstructure=TRUE,std_lv=spec$std_lv,group_equal=spec$group_equal,
    warnings=unique(warnings))
  write_json(info,file.path(out,'model_inventory.json'))
  write.csv(as.data.frame(as.list(measures)),file.path(out,'fit_indices.csv'),row.names=FALSE)
  tryCatch(capture.output(summary(fit,fit.measures=TRUE,standardized=TRUE,rsquare=TRUE),file=file.path(out,'complete_fit.txt')),
           error=function(e) writeLines(conditionMessage(e),file.path(out,'complete_fit.txt')))
  if (!ok) return(info)
  write.csv(lavaan::parameterEstimates(fit,standardized=TRUE,ci=TRUE),file.path(out,'estimates.csv'),row.names=FALSE)
  write.csv(lavaan::standardizedSolution(fit,type='std.all'),file.path(out,'standardized_loadings.csv'),row.names=FALSE)
  V <- stats::vcov(fit)
  if(any(!is.finite(V))) stop(paste('Nonfinite parameter covariance:',id))
  write.csv(V,file.path(out,'parameter_covariance.csv'))
  samp <- lavaan::lavInspect(fit,'sampstat')
  implied <- stats::fitted(fit)
  latent <- lavaan::lavInspect(fit,'cov.lv')
  if(G==1L) {
    write.csv(samp$cov,file.path(out,'sample_covariance.csv'))
    write.csv(implied$cov,file.path(out,'model_implied_covariance.csv'))
    write.csv(latent,file.path(out,'latent_covariance.csv'))
  } else {
    for(g in names(samp)) {
      write.csv(samp[[g]]$cov,file.path(out,paste0('sample_covariance_',g,'.csv')))
      write.csv(implied[[g]]$cov,file.path(out,paste0('model_implied_covariance_',g,'.csv')))
      write.csv(latent[[g]],file.path(out,paste0('latent_covariance_',g,'.csv')))
    }
  }
  info
}

main <- function(args=commandArgs(trailingOnly=TRUE)) {
  if(length(args)<2L || length(args)>4L) stop('Usage: Rscript R/measurement.R scored.csv out_dir [config.json] [models_dir]')
  need_packages(); options(digits=17)
  scriptarg <- grep('^--file=',commandArgs(),value=TRUE)
  base <- if(length(scriptarg)) dirname(dirname(normalizePath(sub('^--file=','',scriptarg[1])))) else getwd()
  cfgpath <- if(length(args)>=3L) args[3] else file.path(base,'config','analysis.json')
  modeldir <- if(length(args)>=4L) args[4] else file.path(base,'models')
  cfg <- jsonlite::read_json(cfgpath,simplifyVector=FALSE)
  if(!identical(cfg$analysis_choices$cfa$estimator,'ML') || !isTRUE(cfg$analysis_choices$cfa$meanstructure)) stop('Unsupported estimator or meanstructure; review implementation rather than silently substituting methods.')
  specs <- jsonlite::read_json(file.path(modeldir,'model_inventory.json'),simplifyVector=FALSE)
  dat <- read.csv(args[1],check.names=FALSE,stringsAsFactors=FALSE,na.strings='',
                  colClasses=c(respondent_id='character',cohort='character',institution='character'))
  if(!all(c('cohort','respondent_id','institution') %in% names(dat)) || anyNA(dat[c('cohort','respondent_id','institution')])) stop('Missing row identifiers.')
  if(anyDuplicated(dat[c('cohort','respondent_id')])) stop('Duplicate respondent keys.')
  if(!setequal(unique(dat$cohort),c('student','teacher'))) stop('Both cohorts are required.')
  dat$cohort <- factor(dat$cohort,levels=c('student','teacher'))
  out <- args[2]; dir.create(out,recursive=TRUE,showWarnings=FALSE)
  statuses <- list(); fits <- list()
  for(spec in specs) {
    id <- spec$model_id; modelpath <- file.path(modeldir,paste0(id,'.lav'))
    syntax <- paste(readLines(modelpath,warn=FALSE),collapse='\n')
    d <- if(is.null(spec$role)) dat else dat[dat$cohort==spec$role,,drop=FALSE]
    warnings <- character()
    attempt <- tryCatch(withCallingHandlers({
      fit <- fit_model(syntax,d,spec)
      info <- export_fit(fit,syntax,spec,file.path(out,id),warnings)
      if(!isTRUE(info$valid_for_export)) stop('Nonconvergence or inadmissible fit; inspect this model summary.')
      if(isTRUE(spec$reliability)) reliability(fit,d,spec$factors,id,file.path(out,id))
      if(grepl('_literacy_second$',id)) {
        fs <- as.numeric(lavaan::lavPredict(fit,method='regression')[,'G'])
        obs <- rowMeans(d[unlist(cfg$scales$literacy,use.names=FALSE)])
        if(any(!is.finite(fs)) || stats::sd(fs)<=0 || stats::sd(obs)<=0) stop('Invalid literacy factor scores.')
        adjusted <- mean(obs)+(fs-mean(fs))*stats::sd(obs)/stats::sd(fs)
        write.csv(data.frame(respondent_id=d$respondent_id,cohort=as.character(d$cohort),
          literacy_factor_score_raw=fs,literacy_factor_score_rescaled=adjusted),
          file.path(out,id,'factor_scores_private.csv'),row.names=FALSE)
      }
      fits[[id]] <- fit
      list(model_id=id,status='passed',df=info$df,warnings=unique(warnings))
    },warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')}),
    error=function(e) list(model_id=id,status='failed',error=conditionMessage(e),warnings=unique(warnings)))
    statuses[[length(statuses)+1L]] <- attempt
    write_json(statuses,file.path(out,'model_status.json'))
  }
  if(!all(vapply(statuses,function(x)identical(x$status,'passed'),logical(1)))) stop('One or more CFA models did not produce an admissible fit.')
  for(role in c('student','teacher')) {
    free <- fits[[paste0(role,'_trust_free')]]; f85 <- fits[[paste0(role,'_trust_085')]]; one <- fits[[paste0(role,'_trust_one')]]
    capture.output(lavaan::lavTestLRT(free,f85),file=file.path(out,paste0(role,'_trust_085_LRT.txt')))
    write.csv(as.data.frame(lavaan::lavTestLRT(free,f85)),file.path(out,paste0(role,'_trust_085_LRT.csv')))
    write.csv(data.frame(comparison='collapsed_one_factor_minus_correlated_two_factor',
      delta_chisq=unname(lavaan::fitMeasures(one,'chisq')-lavaan::fitMeasures(free,'chisq')),
      delta_df=unname(lavaan::fitMeasures(one,'df')-lavaan::fitMeasures(free,'df')),
      interpretation='descriptive_boundary_contrast_no_regular_LRT_p_value'),
      file.path(out,paste0(role,'_trust_one_descriptive_contrast.csv')),row.names=FALSE)
  }
  inv <- do.call(rbind,lapply(c('configural','metric','scalar'),function(n){
    x <- lavaan::fitMeasures(fits[[paste0('invariance_',n)]],c('chisq','df','cfi','tli','rmsea','srmr'))
    data.frame(model=n,t(x),row.names=NULL)
  }))
  inv$delta_CFI <- c(NA,diff(inv$cfi)); inv$delta_RMSEA <- c(NA,diff(inv$rmsea))
  write.csv(inv,file.path(out,'invariance_comparison.csv'),row.names=FALSE)
  cat('All',length(specs),'CFA models estimated.\n')
}

if (sys.nframe()==0L) main()
