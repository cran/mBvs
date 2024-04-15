mmzipBvs <- function(Y,
lin.pred,
data,
offset = NULL,
zero_cutoff = 0.05,
hyperParams,
startValues,
mcmcParams)
{
    ###
    n	<- dim(Y)[1]
    q	<- dim(Y)[2]
    
    Xmat0 <- model.frame(lin.pred[[1]], data=data)
    Xmat1 <- model.frame(lin.pred[[2]], data=data)
    Xmat.adj <- model.frame(lin.pred[[3]], data=data)
    
    p0 <- ncol(Xmat0)
    p1 <- ncol(Xmat1)
    p_adj <- ncol(Xmat.adj)
    
    p1_all <- p1+p_adj
    p0_all <- p0+p_adj
    
    if(p_adj > 0)
    {
        Xmat0 <- cbind(Xmat0, Xmat.adj)
        Xmat1 <- cbind(Xmat1, Xmat.adj)
    }
    
    covNames.z = c(colnames(Xmat0))
    covNames.x = c(colnames(Xmat1))
    
    gamma_beta <- startValues$gamma_beta
    if(p_adj > 0)
    {
        for(i in 1:p_adj) gamma_beta[p1_all - i + 1, ] <- 1
    }
    
    if(mcmcParams$vs$count == 0)
    {
        for(i in 1:p1_all) gamma_beta[i, ] <- 1
    }
    
    gamma_alpha <- startValues$gamma_alpha
    if(p_adj > 0)
    {
        for(i in 1:p_adj) gamma_alpha[p0_all - i + 1, ] <- 1
    }
    
    if(mcmcParams$vs$binary == 0)
    {
        for(i in 1:p0_all) gamma_alpha[i, ] <- 1
    }
    
    if(!is.null(offset))
    {
        offs <- offset
    }else
    {
        offs <- rep(1, n)
    }
    
    if(is.null(mcmcParams$tuning$tuneM))
    {
        mcmcParams$tuning$tuneM <- 0
    }
    
    if(!is.null(mcmcParams$tuning$M_B))
    {
        M_B <- mcmcParams$tuning$M_B
    }else
    {
        M_B <- matrix(mcmcParams$tuning$Mvar_group, p1_all, q)
    }
    
    if(!is.null(mcmcParams$tuning$M_beta0))
    {
        M_beta0 <- mcmcParams$tuning$M_beta0
    }else
    {
        M_beta0 <- rep(mcmcParams$tuning$Mvar_group, q)
    }
    
    if(!is.null(mcmcParams$tuning$M_V))
    {
        M_V <- mcmcParams$tuning$M_V
    }else
    {
        M_V <- matrix(mcmcParams$tuning$Mvar_group, n, q)
    }
    
    if(!is.null(mcmcParams$tuning$M_A))
    {
        M_A <- mcmcParams$tuning$M_A
    }else
    {
        M_A <- matrix(mcmcParams$tuning$Mvar_group, p0_all, q)
    }
    
    if(!is.null(mcmcParams$tuning$M_alpha0))
    {
        M_alpha0 <- mcmcParams$tuning$M_alpha0
    }else
    {
        M_alpha0 <- rep(mcmcParams$tuning$Mvar_group, q)
    }
    
    if(!is.null(mcmcParams$tuning$M_m))
    {
        M_m <- mcmcParams$tuning$M_m
    }else
    {
        M_m <- rep(mcmcParams$tuning$Mvar_m, q)
    }
    
    if(is.null(mcmcParams$tuning$eps_V))
    {
        mcmcParams$tuning$eps_V <- mcmcParams$tuning$eps_group
    }
    
    if(is.null(mcmcParams$tuning$PtuneEps))
    {
        mcmcParams$tuning$PtuneEps <- mcmcParams$run$burninPerc
    }
    
    if(is.null(mcmcParams$tuning$PtuneM))
    {
        mcmcParams$tuning$PtuneM <- mcmcParams$run$burninPerc
    }
    
    if(is.null(colnames(Y)))
    {
        Y.names <- colnames(matrix(NA, 1, q), do.NULL=FALSE, prefix = "Outcome.")
    }else
    {
        Y.names <- colnames(Y)
    }
    
    ind_allNZ <- which(as.vector(apply(Y != 0, 2, sum))/n >= 1-zero_cutoff)
    q_allNZ <- length(ind_allNZ)
    if(q_allNZ > 0)
    {
        cat(paste(q_allNZ, " outcome(s) with more than ", round((1-zero_cutoff)*100), "% nonzero observations...", sep = ""), cat("\n"))
        cat(paste("Y and relevant parameters have been rearranged...", sep = ""), cat("\n"))
        cat(paste("......", sep = ""), cat("\n"))
        Y <- cbind(Y[,-ind_allNZ], Y[,ind_allNZ])
        Y.names <- c(Y.names[-ind_allNZ], Y.names[ind_allNZ])
        
        gamma_beta <- cbind(gamma_beta[,-ind_allNZ], gamma_beta[,ind_allNZ])
        startValues$B <- cbind(startValues$B[,-ind_allNZ], startValues$B[,ind_allNZ])
        startValues$beta0 <- c(startValues$beta0[-ind_allNZ], startValues$beta0[ind_allNZ])
        startValues$V <- cbind(startValues$V[,-ind_allNZ], startValues$V[,ind_allNZ])
        startValues$SigmaV <- cbind(startValues$SigmaV[,-ind_allNZ], startValues$SigmaV[,ind_allNZ])
        startValues$SigmaV <- rbind(startValues$SigmaV[-ind_allNZ,], startValues$SigmaV[ind_allNZ,])
        
        gamma_alpha <- gamma_alpha[,-ind_allNZ]
        startValues$A <- startValues$A[,-ind_allNZ]
        startValues$alpha0 <- startValues$alpha0[-ind_allNZ]
        startValues$m  <- startValues$m[-ind_allNZ]
        startValues$W <- startValues$W[,-ind_allNZ]
        
        hyperParams$v_beta <- c(hyperParams$v_beta[-ind_allNZ], hyperParams$v_beta[ind_allNZ])
        hyperParams$mu_beta0 <- c(hyperParams$mu_beta0[-ind_allNZ], hyperParams$mu_beta0[ind_allNZ])
        
        hyperParams$v_alpha <- hyperParams$v_alpha[-ind_allNZ]
        hyperParams$mu_alpha0 <- hyperParams$mu_alpha0[-ind_allNZ]
        hyperParams$mu_m <- hyperParams$mu_m[-ind_allNZ]
        
        M_B <- cbind(M_B[,-ind_allNZ], M_B[,ind_allNZ])
        M_beta0 <- c(M_beta0[-ind_allNZ], M_beta0[ind_allNZ])
        M_V <- cbind(M_V[,-ind_allNZ], M_V[,ind_allNZ])
        
        M_A <- M_A[,-ind_allNZ]
        M_alpha0 <- M_alpha0[-ind_allNZ]
        M_m <- M_m[-ind_allNZ]
    }
    
    ###
    hyperP  <- as.vector(c(hyperParams$v_beta, hyperParams$omega_beta, hyperParams$a_beta, hyperParams$b_beta, hyperParams$mu_beta0, hyperParams$a_beta0, hyperParams$b_beta0, hyperParams$v_alpha, hyperParams$omega_alpha,  hyperParams$a_alpha, hyperParams$b_alpha, hyperParams$mu_alpha0, hyperParams$a_alpha0, hyperParams$b_alpha0, hyperParams$rho0, hyperParams$Psi0, hyperParams$mu_m, hyperParams$v_m))
    
    start.time <- proc.time()
    
    mcmcP   <- as.vector(c(mcmcParams$tuning$L_group, mcmcParams$tuning$eps_group, mcmcParams$tuning$Mvar_group, mcmcParams$tuning$beta_prop_var, 1, 1, 1, mcmcParams$tuning$alpha_prop_var, 1, 1, 1, mcmcParams$tuning$L_m, mcmcParams$tuning$eps_m, mcmcParams$tuning$Mvar_m, mcmcParams$tuning$eps_V, mcmcParams$tuning$PtuneEps, mcmcParams$tuning$PtuneM, mcmcParams$tuning$tuneM))
    
    startVal <- as.vector(c(startValues$B, startValues$A, startValues$beta0, startValues$alpha0, startValues$m, startValues$sigSq_alpha0, startValues$sigSq_alpha, startValues$sigSq_beta0, startValues$sigSq_beta, startValues$V, startValues$W, startValues$SigmaV))
    
    storeV <- mcmcParams$storage$storeV
    storeW <- mcmcParams$storage$storeW
    store <- as.numeric(c(storeV, storeW))
    
    VS <- c(mcmcParams$vs$count, mcmcParams$vs$binary)
    
    numReps     <- mcmcParams$run$numReps
    thin        <- mcmcParams$run$thin
    burninPerc  <- mcmcParams$run$burninPerc
    nStore      <- numReps/thin * (1 - burninPerc)
    nStore <- round(nStore)
    
    if(storeV == TRUE | q < 10)
    {
        sV		    <- rep(0, nStore*n*q)
    }else
    {
        sV 			<- rep(0, nStore*20*10)
    }
    if(storeW == TRUE | q-q_allNZ < 10)
    {
        sW				<- rep(0, nStore*n*(q-q_allNZ))
    }else
    {
        sW 			<- rep(0, nStore*20*10)
    }
    
    mcmcRet     <- .C("MMZIPmcmc",
    Ymat            = as.double(as.matrix(Y)),
    Xmat0          = as.double(as.matrix(Xmat0)),
    Xmat1          = as.double(as.matrix(Xmat1)),
    offs            = as.double(offs),
    n				= as.integer(n),
    q				= as.integer(q),
    q_allNZ         = as.integer(q_allNZ),
    p0_all			= as.integer(p0_all),
    p1_all			= as.integer(p1_all),
    p_adj			= as.integer(p_adj),
    hyperP          = as.double(hyperP),
    mcmcP           = as.double(mcmcP),
    startValues 	= as.double(startVal),
    startGamma_beta = as.double(gamma_beta),
    startGamma_alpha = as.double(gamma_alpha),
    numReps			= as.integer(numReps),
    thin			= as.integer(thin),
    burninPerc      = as.double(burninPerc),
    store           = as.double(store),
    VS           = as.double(VS),
    samples_B       = as.double(rep(0, nStore*p1_all*q)),
    samples_gamma_beta   = as.double(rep(0, nStore*p1_all*q)),
    samples_beta0	= as.double(rep(0, nStore*q)),
    samples_A       = as.double(rep(0, nStore*p0_all*(q-q_allNZ))),
    samples_gamma_alpha   = as.double(rep(0, nStore*p0_all*(q-q_allNZ))),
    samples_alpha0	= as.double(rep(0, nStore*(q-q_allNZ))),
    samples_V       = as.double(sV),
    samples_W       = as.double(sW),
    samples_SigmaV_diag		= as.double(rep(0, nStore*q)),
    samples_m  		= as.double(rep(0, nStore*(q-q_allNZ))),
    samples_sigSq_beta = as.double(rep(0, nStore*p1_all)),
    samples_sigSq_beta0 = as.double(rep(0, nStore*1)),
    samples_sigSq_alpha = as.double(rep(0, nStore*p0_all)),
    samples_sigSq_alpha0 = as.double(rep(0, nStore*1)),
    samples_misc    = as.double(rep(0, 1+p1_all*q+1+p0_all*(q-q_allNZ)+p0_all*(q-q_allNZ)+1+1)), # count, B_ssvs,  alpha0, A_ssvs, A_hmc, m
    logpost = as.double(rep(0, nStore*1)),
    SigV_mean = as.double(rep(0, q*q)),
    SigV_var = as.double(rep(0, q*q)),
    M_B_ini = as.double(M_B),
    M_beta0_ini = as.double(M_beta0),
    M_V_ini = as.double(M_V),
    M_A_ini = as.double(M_A),
    M_alpha0_ini = as.double(M_alpha0),
    M_m_ini = as.double(M_m),
    final_SigmaV = as.double(rep(0, q*q)),
    final_V= as.double(rep(0, n*q)),
    final_W= as.double(rep(0, n*(q-q_allNZ))),
    final_eps = as.double(rep(0, 5)))
    
    B.p	 <- array(as.vector(mcmcRet$samples_B), c(p1_all, q, nStore))
    gamma_beta.p	<- array(as.vector(mcmcRet$samples_gamma_beta), c(p1_all, q, nStore))
    beta0.p         <- matrix(as.vector(mcmcRet$samples_beta0), nrow=nStore, byrow=T)
    
    A.p                <- array(as.vector(mcmcRet$samples_A), c(p0_all, (q-q_allNZ), nStore))
    gamma_alpha.p       <- array(as.vector(mcmcRet$samples_gamma_alpha), c(p0_all, (q-q_allNZ), nStore))
    alpha0.p        <- matrix(as.vector(mcmcRet$samples_alpha0), nrow=nStore, byrow=T)
    
    if(storeV == TRUE | q < 10)
    {
        V.p				<- array(as.vector(mcmcRet$samples_V), c(n, q, nStore))
    }else
    {
        V.p 			<- array(as.vector(mcmcRet$samples_V), c(20, 10, nStore))
    }
    if(storeW == TRUE | q < 10)
    {
        W.p				<- array(as.vector(mcmcRet$samples_W), c(n, (q-q_allNZ), nStore))
    }else
    {
        W.p 			<- array(as.vector(mcmcRet$samples_W), c(20, 10, nStore))
    }
    
    SigmaV_diag.p		<- matrix(as.vector(mcmcRet$samples_SigmaV_diag), nrow=nStore, byrow=T)
    m.p            		<- matrix(as.vector(mcmcRet$samples_m), nrow=nStore, byrow=T)
    
    Sigma_V.last		<- matrix(as.vector(mcmcRet$final_SigmaV), nrow=q, byrow=F)
    
    V.last              <- matrix(as.vector(mcmcRet$final_V), nrow=n, byrow=F)
    W.last              <- matrix(as.vector(mcmcRet$final_W), nrow=n, byrow=F)
    
    SigV.mean              <- matrix(as.vector(mcmcRet$SigV_mean), nrow=q, byrow=F)
    SigV.var              <- matrix(as.vector(mcmcRet$SigV_var), nrow=q, byrow=F)
    
    sigSq_beta.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta), nrow=nStore, byrow=T)
    sigSq_beta0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta0), nrow=nStore, byrow=T)
    sigSq_alpha.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha), nrow=nStore, byrow=T)
    sigSq_alpha0.p		<- matrix(as.vector(mcmcRet$samples_sigSq_alpha0), nrow=nStore, byrow=T)
    
    accept.group        <- as.vector(mcmcRet$samples_misc[1])
    accept.B_ssvs        <- matrix(as.vector(mcmcRet$samples_misc[(1+1):(1+p1_all*q)]), nrow = p1_all, byrow = FALSE)
    accept.A_ssvs		<- matrix(as.vector(mcmcRet$samples_misc[(1+p1_all*q+1+1):(1+p1_all*q+1+p0_all*(q-q_allNZ))]), nrow = p0_all, byrow = FALSE)
    accept.m    <- as.vector(mcmcRet$samples_misc[1+p1_all*q+1+p0_all*(q-q_allNZ)+p0_all*(q-q_allNZ)+1])
    accept.V    <- as.vector(mcmcRet$samples_misc[1+p1_all*q+1+p0_all*(q-q_allNZ)+p0_all*(q-q_allNZ)+1+1])
    
    logpost        <- matrix(as.vector(mcmcRet$logpost), nrow=nStore, byrow=T)
    
    lastValues <- list()
    lastValues$B <- B.p[,,nStore]
    lastValues$gamma_beta <- gamma_beta.p[,,nStore]
    lastValues$beta0 <- beta0.p[nStore,]
    lastValues$V <- V.last
    lastValues$SigmaV <- Sigma_V.last
    
    lastValues$A <- A.p[,,nStore]
    lastValues$gamma_alpha <- gamma_alpha.p[,,nStore]
    lastValues$alpha0 <- alpha0.p[nStore,]
    lastValues$W <- W.last
    lastValues$m <- m.p[nStore,]
    
    
    M_B <- matrix(as.vector(mcmcRet$M_B_ini), nrow=p1_all, byrow=F)
    M_beta0 <- as.vector(mcmcRet$M_beta0_ini)
    M_V <- matrix(as.vector(mcmcRet$M_V_ini), nrow=n, byrow=F)
    M_A <- matrix(as.vector(mcmcRet$M_A_ini), nrow=p0_all, byrow=F)
    M_alpha0 <- as.vector(mcmcRet$M_alpha0_ini)
    M_m <- as.vector(mcmcRet$M_m_ini)
    
    if(length(ind_allNZ) !=0)
    {
        for(ii in 1:length(ind_allNZ))
        {
            if(ind_allNZ[ii] != 1)
            {
                lastValues$A <- cbind(lastValues$A[,1:(ind_allNZ[ii]-1)], NA, lastValues$A[,ind_allNZ[ii]:dim(lastValues$A)[2]])
                
                lastValues$gamma_alpha <- cbind(lastValues$gamma_alpha[,1:(ind_allNZ[ii]-1)], NA, lastValues$gamma_alpha[,ind_allNZ[ii]:dim(lastValues$gamma_alpha)[2]])
                lastValues$alpha0 <- c(lastValues$alpha0[1:(ind_allNZ[ii]-1)], NA, lastValues$alpha0[ind_allNZ[ii]:length(lastValues$alpha0)])
                
                lastValues$W <- cbind(lastValues$W[,1:(ind_allNZ[ii]-1)], NA, lastValues$W[,ind_allNZ[ii]:dim(lastValues$W)[2]])
                
                lastValues$m <- c(lastValues$m[1:(ind_allNZ[ii]-1)], NA, lastValues$m[ind_allNZ[ii]:length(lastValues$m)])
                
                lastValues$B <- cbind(lastValues$B[,1:(ind_allNZ[ii]-1)], lastValues$B[,q-q_allNZ+ii], lastValues$B[,ind_allNZ[ii]:dim(lastValues$B)[2]])
                lastValues$B <- lastValues$B[,-(q-q_allNZ+ii+1)]
                
                lastValues$gamma_beta <- cbind(lastValues$gamma_beta[,1:(ind_allNZ[ii]-1)], lastValues$gamma_beta[,q-q_allNZ+ii], lastValues$gamma_beta[,ind_allNZ[ii]:dim(lastValues$gamma_beta)[2]])
                lastValues$gamma_beta <- lastValues$gamma_beta[,-(q-q_allNZ+ii+1)]
                
                lastValues$beta0 <- c(lastValues$beta0[1:(ind_allNZ[ii]-1)], lastValues$beta0[q-q_allNZ+ii], lastValues$beta0[ind_allNZ[ii]:length(lastValues$beta0)])
                lastValues$beta0 <- lastValues$beta0[-(q-q_allNZ+ii+1)]
                
                lastValues$V <- cbind(lastValues$V[,1:(ind_allNZ[ii]-1)], lastValues$V[,q-q_allNZ+ii], lastValues$V[,ind_allNZ[ii]:dim(lastValues$V)[2]])
                lastValues$V <- lastValues$V[,-(q-q_allNZ+ii+1)]
                
                lastValues$SigmaV <- cbind(lastValues$SigmaV[,1:(ind_allNZ[ii]-1)], lastValues$SigmaV[,q-q_allNZ+ii], lastValues$SigmaV[,ind_allNZ[ii]:dim(lastValues$SigmaV)[2]])
                lastValues$SigmaV <- lastValues$SigmaV[,-(q-q_allNZ+ii+1)]
                lastValues$SigmaV <- rbind(lastValues$SigmaV[1:(ind_allNZ[ii]-1),], lastValues$SigmaV[q-q_allNZ+ii,], lastValues$SigmaV[ind_allNZ[ii]:dim(lastValues$SigmaV)[2],])
                lastValues$SigmaV <- lastValues$SigmaV[-(q-q_allNZ+ii+1),]
                
                M_B <- cbind(M_B[,1:(ind_allNZ[ii]-1)], M_B[,q-q_allNZ+ii], M_B[,ind_allNZ[ii]:dim(M_B)[2]])
                M_B <- M_B[,-(q-q_allNZ+ii+1)]
                M_beta0 <- c(M_beta0[1:(ind_allNZ[ii]-1)], M_beta0[q-q_allNZ+ii], M_beta0[ind_allNZ[ii]:length(M_beta0)])
                M_beta0 <- M_beta0[-(q-q_allNZ+ii+1)]
                M_V <- cbind(M_V[,1:(ind_allNZ[ii]-1)], M_V[,q-q_allNZ+ii], M_V[,ind_allNZ[ii]:dim(M_V)[2]])
                M_V <- M_V[,-(q-q_allNZ+ii+1)]
                
                M_A <- cbind(M_A[,1:(ind_allNZ[ii]-1)], NA, M_A[,ind_allNZ[ii]:dim(M_A)[2]])
                M_alpha0 <- c(M_alpha0[1:(ind_allNZ[ii]-1)], NA, M_alpha0[ind_allNZ[ii]:length(M_alpha0)])
                M_m <- c(M_m[1:(ind_allNZ[ii]-1)], NA, M_m[ind_allNZ[ii]:length(M_m)])
            }else
            {
                lastValues$A <- cbind(NA, lastValues$A)
                
                lastValues$gamma_alpha <- cbind(NA, lastValues$gamma_alpha)
                lastValues$alpha0 <- c(NA, lastValues$alpha0)
                
                lastValues$W <- cbind(NA, lastValues$W)
                
                lastValues$m <- c(NA, lastValues$m)
                
                lastValues$B <- cbind(lastValues$B[,q-q_allNZ+ii], lastValues$B)
                lastValues$B <- lastValues$B[,-(q-q_allNZ+ii+1)]
                
                lastValues$gamma_beta <- cbind(lastValues$gamma_beta[,q-q_allNZ+ii], lastValues$gamma_beta)
                lastValues$gamma_beta <- lastValues$gamma_beta[,-(q-q_allNZ+ii+1)]
                
                lastValues$beta0 <- c(lastValues$beta0[q-q_allNZ+ii], lastValues$beta0)
                lastValues$beta0 <- lastValues$beta0[-(q-q_allNZ+ii+1)]
                
                lastValues$V <- cbind(lastValues$V[,q-q_allNZ+ii], lastValues$V)
                lastValues$V <- lastValues$V[,-(q-q_allNZ+ii+1)]
                
                lastValues$SigmaV <- cbind(lastValues$SigmaV[,q-q_allNZ+ii], lastValues$SigmaV)
                lastValues$SigmaV <- lastValues$SigmaV[,-(q-q_allNZ+ii+1)]
                lastValues$SigmaV <- rbind(lastValues$SigmaV[q-q_allNZ+ii,], lastValues$SigmaV)
                lastValues$SigmaV <- lastValues$SigmaV[-(q-q_allNZ+ii+1),]
                
                M_B <- cbind(M_B[,q-q_allNZ+ii], M_B)
                M_B <- M_B[,-(q-q_allNZ+ii+1)]
                M_beta0 <- c(M_beta0[q-q_allNZ+ii], M_beta0)
                M_beta0 <- M_beta0[-(q-q_allNZ+ii+1)]
                M_V <- cbind(M_V[,q-q_allNZ+ii], M_V)
                M_V <- M_V[,-(q-q_allNZ+ii+1)]
                
                M_A <- cbind(NA, M_A)
                M_alpha0 <- c(NA, M_alpha0)
                M_m <- c(NA, M_m)
            }
            
            
        }
    }
 
    lastValues$sigSq_alpha0 <- sigSq_alpha0.p[nStore]
    lastValues$sigSq_alpha <- sigSq_alpha.p[nStore,]
    lastValues$sigSq_beta0 <- sigSq_beta0.p[nStore]
    lastValues$sigSq_beta <- sigSq_beta.p[nStore,]
    
    lastEps <- list()
    lastEps$eps_group <- mcmcRet$final_eps[1]
    lastEps$eps_alp <- mcmcRet$final_eps[2]
    lastEps$eps_alp0 <- mcmcRet$final_eps[3]
    lastEps$eps_m <- mcmcRet$final_eps[4]
    lastEps$eps_V <- mcmcRet$final_eps[5]
    
    lastM <- list()
    lastM$M_B <- M_B
    lastM$M_beta0 <- M_beta0
    lastM$M_V <- M_V
    lastM$M_A <- M_A
    lastM$M_alpha0 <- M_alpha0
    lastM$M_m <- M_m
    
    ret <- list(lin.pred=lin.pred, offset=offset, hyperParams=hyperParams, startValues=startValues, mcmcParams=mcmcParams, B.p = B.p, gamma_beta.p=gamma_beta.p,  beta0.p=beta0.p, A.p = A.p, gamma_alpha.p=gamma_alpha.p, alpha0.p=alpha0.p, V.p=V.p, W.p=W.p, SigmaV_diag.p=SigmaV_diag.p, m.p=m.p, sigSq_beta.p=sigSq_beta.p, sigSq_beta0.p=sigSq_beta0.p, sigSq_alpha.p=sigSq_alpha.p, sigSq_alpha0.p=sigSq_alpha0.p, accept.B_ssvs=accept.B_ssvs, accept.A_ssvs=accept.A_ssvs, accept.group=accept.group, accept.m=accept.m, accept.V=accept.V, lastValues=lastValues, Y=Y, Y.names=Y.names, covNames.z=covNames.z, covNames.x=covNames.x, Xmat0=Xmat0, Xmat1=Xmat1, Xmat.adj=Xmat.adj, SigmaV_mean=SigV.mean, SigmaV_var=SigV.var, logpost=logpost, ind_allNZ=ind_allNZ, lastEps=lastEps, lastM=lastM, Ctime=proc.time()-start.time)
    
    class(ret) <- "mmzipBvs"
    return(ret)
}
