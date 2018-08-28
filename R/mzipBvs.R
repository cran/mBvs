
mzipBvs <- function(Y,
lin.pred,
data,
model="generalized",
offset = NULL,
hyperParams,
startValues,
mcmcParams)
{
    numReps     <- mcmcParams$run$numReps
    thin        <- mcmcParams$run$thin
    burninPerc  <- mcmcParams$run$burninPerc
    nStore      <- numReps/thin * (1 - burninPerc)
    
    if((numReps / thin * burninPerc) %% 1 == 0)
    {
        nChain <- length(startValues)
        
        chain = 1
        ret <- list()
        
        ###
        n    <- dim(Y)[1]
        q    <- dim(Y)[2]
        
        Xmat0 <- model.frame(lin.pred[[1]], data=data)
        Xmat1 <- model.frame(lin.pred[[2]], data=data)
        Xmat.adj <- model.frame(lin.pred[[3]], data=data)
        
        while(chain <= nChain)
        {
            cat("chain: ", chain, "\n")
            nam = paste("chain", chain, sep="")
            temp <- startValues[[chain]]
            
            if(model == "generalized")
            {
                p0 <- ncol(Xmat0) + ncol(Xmat.adj)
                p1 <- ncol(Xmat1) + ncol(Xmat.adj)
                p_adj <- ncol(Xmat.adj)

                if(p_adj > 0){
                    Xmat0 <- cbind(Xmat0, Xmat.adj)
                    Xmat1 <- cbind(Xmat1, Xmat.adj)
                }
                
                covNames.z = c(colnames(Xmat0))
                covNames.x = c(colnames(Xmat1))
                
                ###
                gamma_beta <- temp$B
                gamma_beta[gamma_beta !=0] <- 1
                
                gamma_alpha <- temp$A
                gamma_alpha[gamma_alpha !=0] <- 1
                
                if (p_adj > 0) {
                    for (i in 1:p_adj) {
                        gamma_beta[p1 - i + 1, ] <- 1
                        gamma_alpha[p0 - i + 1, ] <- 1
                    }
                }
                
                alpha0.prop.var    <- 0.5 # not used (place holder)
                W.prop.var    <- 0.5 # not used (place holder)
                S.prop.var    <- 0.005 # not used (place holder)
                rho.s    <- 100000 # not used (place holder)
                rho.r    <- 100000 # not used (place holder)
                
                rhoR     <- q + 3 + 1 # not used (place holder)
                PsiR    <- diag(3, q) # not used (place holder)
                
                phi     <- rep(1, n) # not used (place holder)
                S        <- rep(0.1, q) # not used (place holder)
                nu_t <- 1 # not used (place holder)
                sigSq_t <- 1 # not used (place holder)
                
                startV <- as.vector(c(temp$B, temp$A, temp$V, temp$W, temp$beta0, temp$alpha0, temp$R, S, temp$sigSq_alpha0, temp$sigSq_alpha, temp$sigSq_beta0, temp$sigSq_beta, phi, gamma_beta, gamma_alpha, temp$Sigma_V))
                
                
                if(!is.null(offset))
                {
                    offs <- offset
                }else
                {
                    offs <- rep(1, n)
                }
                
                muS    <- rep(log(1), q)
                PsiS <- diag(10^2, q)
                
                ###
                hyperP  <- as.vector(c(muS, PsiS, hyperParams$mu_alpha0, hyperParams$mu_alpha, hyperParams$mu_beta0, hyperParams$mu_beta, hyperParams$a_alpha0, hyperParams$b_alpha0, hyperParams$a_alpha, hyperParams$b_alpha, hyperParams$a_beta0, hyperParams$b_beta0, hyperParams$a_beta, hyperParams$b_beta, nu_t, sigSq_t, hyperParams$rho0, hyperParams$Psi0, hyperParams$v_beta, hyperParams$omega_beta, hyperParams$v_alpha, hyperParams$omega_alpha, rhoR, PsiR))
                
                ###
                mcmcP   <- as.vector(c(mcmcParams$tuning$beta0.prop.var, mcmcParams$tuning$beta.prop.var, mcmcParams$tuning$V.prop.var, alpha0.prop.var, mcmcParams$tuning$alpha.prop.var, W.prop.var, S.prop.var, rho.s, rho.r))
                
                storeV <- mcmcParams$storage$storeV
                storeW <- mcmcParams$storage$storeW
                
                store <- as.numeric(c(storeV, storeW))
                
                mcmcRet     <- .C("mzipBVS_general_mcmc",
                Ymat            = as.double(as.matrix(Y)),
                Xmat0          = as.double(as.matrix(Xmat0)),
                Xmat1          = as.double(as.matrix(Xmat1)),
                offs            = as.double(offs),
                n                = as.integer(n),
                q                = as.integer(q),
                p0                = as.integer(p0),
                p1                = as.integer(p1),
                p_adj            = as.integer(p_adj),
                hyperP          = as.double(hyperP),
                mcmcP           = as.double(mcmcP),
                startValues     = as.double(startV),
                startGamma_beta      = as.double(gamma_beta),
                startGamma_alpha      = as.double(gamma_alpha),
                numReps            = as.integer(numReps),
                thin            = as.integer(thin),
                burninPerc      = as.double(burninPerc),
                store           = as.double(store),
                samples_beta0    = as.double(rep(0, nStore*q)),
                samples_B       = as.double(rep(0, nStore*p0*q)),
                samples_V       = as.double(rep(0, nStore*n*q)),
                samples_alpha0    = as.double(rep(0, nStore*q)),
                samples_A       = as.double(rep(0, nStore*p0*q)),
                samples_gamma_beta   = as.double(rep(0, nStore*p1*q)),
                samples_gamma_alpha   = as.double(rep(0, nStore*p0*q)),
                samples_W       = as.double(rep(0, nStore*n*q)),
                samples_R        = as.double(rep(0, nStore*q*q)),
                samples_S          = as.double(rep(0, nStore*q)),
                samples_Sigma_V = as.double(rep(0, nStore*q*q)),
                samples_sigSq_alpha0 = as.double(rep(0, nStore*1)),
                samples_sigSq_beta0 = as.double(rep(0, nStore*1)),
                samples_sigSq_alpha = as.double(rep(0, nStore*p0)),
                samples_sigSq_beta = as.double(rep(0, nStore*p1)),
                samples_misc    = as.double(rep(0, p0*q+p1*q+n*q+n*q+q+q+1+q+1)))
                
                A.p                <- array(as.vector(mcmcRet$samples_A), c(p0, q, nStore))
                B.p                <- array(as.vector(mcmcRet$samples_B), c(p1, q, nStore))
                gamma_beta.p        <- array(as.vector(mcmcRet$samples_gamma_beta), c(p1, q, nStore))
                gamma_alpha.p    <- array(as.vector(mcmcRet$samples_gamma_alpha), c(p0, q, nStore))
                if(storeV == TRUE)
                {
                    V.p                <- array(as.vector(mcmcRet$samples_V), c(n, q, nStore))
                }else
                {
                    V.p <- NULL
                }
                if(storeW == TRUE)
                {
                    W.p                <- array(as.vector(mcmcRet$samples_W), c(n, q, nStore))
                }else
                {
                    W.p <- NULL
                }
                
                R.p            <- array(as.vector(mcmcRet$samples_R), c(q, q, nStore))
                S.p            <- matrix(as.vector(mcmcRet$samples_S), nrow=nStore, byrow=T)
                
                Sigma_V.p            <- array(as.vector(mcmcRet$samples_Sigma_V), c(q, q, nStore))
                
                alpha0.p        <- matrix(as.vector(mcmcRet$samples_alpha0), nrow=nStore, byrow=T)
                beta0.p         <- matrix(as.vector(mcmcRet$samples_beta0), nrow=nStore, byrow=T)
                
                sigSq_alpha0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha0), nrow=nStore, byrow=T)
                sigSq_beta0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta0), nrow=nStore, byrow=T)
                
                sigSq_alpha.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha), nrow=nStore, byrow=T)
                sigSq_beta.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta), nrow=nStore, byrow=T)
                
                accept.A        <- matrix(as.vector(mcmcRet$samples_misc[1:(p0*q)]), nrow = p0, byrow = FALSE)
                accept.B        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+1):(p0*q+p1*q)]), nrow = p1, byrow = FALSE)
                accept.V        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+p1*q+1):(p0*q+p1*q+n*q)]), nrow = n, byrow = FALSE)
                accept.W        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+1):(p0*q+p1*q+n*q+n*q)]), nrow = n, byrow = FALSE)
                accept.alpha0 <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+1):(p0*q+p1*q+n*q+n*q+q)])
                accept.beta0 <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+1):(p0*q+p1*q+n*q+n*q+q+q)])
                accept.R <- mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1)]
                accept.S <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1+1):(p0*q+p1*q+n*q+n*q+q+q+1+q)])
                accept.SigmaV <- mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1+q+1)]
                
                ret[[nam]] <- list(A.p = A.p, B.p = B.p, gamma_beta.p=gamma_beta.p, gamma_alpha.p=gamma_alpha.p, W.p=W.p, V.p=V.p, alpha0.p=alpha0.p, beta0.p=beta0.p, R.p=R.p, S.p=S.p, Sigma_V.p=Sigma_V.p, sigSq_alpha0.p=sigSq_alpha0.p, sigSq_beta0.p=sigSq_beta0.p, sigSq_alpha.p=sigSq_alpha.p, sigSq_beta.p=sigSq_beta.p, accept.A = accept.A, accept.B = accept.B, accept.V = accept.V, accept.beta0=accept.beta0, covNames.z=covNames.z, covNames.x=covNames.x)
            }else if(model == "restricted1")
            {
                ###
                p0 <- ncol(Xmat0)
                p1 <- ncol(Xmat1)
                p_adj <- ncol(Xmat.adj)
                
                p <- p1+p_adj
                p0 <- p1 <- p

                if(p_adj > 0){
                    Xmat0 <- cbind(Xmat0, Xmat.adj)
                    Xmat1 <- cbind(Xmat1, Xmat.adj)
                }
                
                covNames.z = c(colnames(Xmat0))
                covNames.x = c(colnames(Xmat1))
                
                ###
                gamma <- temp$B
                gamma[gamma !=0] <- 1
                
                if (p_adj > 0) {
                    for (i in 1:p_adj) {
                        gamma[p - i + 1, ] <- 1
                    }
                }
                temp$gamma <- gamma
                
                nu_t <- 7.3
                sigSq_t <- pi^2*(nu_t-2)/(3*nu_t)
                numPhi      <- 3
                temp$phi <- rgamma(n, nu_t/2, nu_t/2)
                
                if(!is.null(offset))
                {
                    offs <- offset
                }else
                {
                    offs <- rep(1, n)
                }
                
                ###
                hyperP  <- as.vector(c(hyperParams$muS, hyperParams$PsiS, hyperParams$mu_alpha0, hyperParams$mu_alpha, hyperParams$mu_beta0, hyperParams$mu_beta, hyperParams$a_alpha0, hyperParams$b_alpha0, hyperParams$a_alpha, hyperParams$b_alpha, hyperParams$a_beta0, hyperParams$b_beta0, hyperParams$a_beta, hyperParams$b_beta, nu_t, sigSq_t, hyperParams$rho0, hyperParams$Psi0, hyperParams$v, hyperParams$omega))
               
                ###
                mcmcP   <- as.vector(c(mcmcParams$tuning$beta0.prop.var, mcmcParams$tuning$beta.prop.var, mcmcParams$tuning$V.prop.var, mcmcParams$tuning$alpha0.prop.var, mcmcParams$tuning$alpha.prop.var, mcmcParams$tuning$W.prop.var, mcmcParams$tuning$S.prop.var, mcmcParams$tuning$rho.s))
                
                storeV <- mcmcParams$storage$storeV
                storeW <- mcmcParams$storage$storeW
                nPhi_save  <- numPhi
                
                store <- as.numeric(c(storeV, storeW))
                
                startV <- as.vector(c(temp$B, temp$A, temp$V, temp$W, temp$beta0, temp$alpha0, temp$R, temp$S, temp$sigSq_alpha0, temp$sigSq_alpha, temp$sigSq_beta0, temp$sigSq_beta, temp$phi, temp$gamma))
                
                mcmcRet     <- .C("mzip_restricted1_mcmc",
                Ymat            = as.double(as.matrix(Y)),
                Xmat0          = as.double(as.matrix(Xmat0)),
                Xmat1          = as.double(as.matrix(Xmat1)),
                offs            = as.double(offs),
                n                = as.integer(n),
                q                = as.integer(q),
                p0                = as.integer(p0),
                p1                = as.integer(p1),
                p                = as.integer(p),
                p_adj            = as.integer(p_adj),
                hyperP          = as.double(hyperP),
                mcmcP           = as.double(mcmcP),
                startValues     = as.double(startV),
                startGamma      = as.double(gamma),
                numReps            = as.integer(numReps),
                thin            = as.integer(thin),
                burninPerc      = as.double(burninPerc),
                nPhi_save        = as.integer(nPhi_save),
                store           = as.double(store),
                samples_beta0    = as.double(rep(0, nStore*q)),
                samples_B       = as.double(rep(0, nStore*p0*q)),
                samples_V       = as.double(rep(0, nStore*n*q)),
                samples_alpha0    = as.double(rep(0, nStore*q)),
                samples_A       = as.double(rep(0, nStore*p0*q)),
                samples_gamma   = as.double(rep(0, nStore*p*q)),
                samples_W       = as.double(rep(0, nStore*n*q)),
                samples_R        = as.double(rep(0, nStore*q*q)),
                samples_S          = as.double(rep(0, nStore*q)),
                samples_sigSq_alpha0 = as.double(rep(0, nStore*1)),
                samples_sigSq_beta0 = as.double(rep(0, nStore*1)),
                samples_sigSq_alpha = as.double(rep(0, nStore*p0)),
                samples_sigSq_beta = as.double(rep(0, nStore*p1)),
                samples_phi     = as.double(rep(0, nStore*numPhi)),
                samples_misc    = as.double(rep(0, p0*q+p1*q+n*q+n*q+q+q+1+q)))
                
                A.p                <- array(as.vector(mcmcRet$samples_A), c(p0, q, nStore))
                B.p                <- array(as.vector(mcmcRet$samples_B), c(p1, q, nStore))
                gamma.p                <- array(as.vector(mcmcRet$samples_gamma), c(p, q, nStore))
                if(storeV == TRUE)
                {
                    V.p                <- array(as.vector(mcmcRet$samples_V), c(n, q, nStore))
                }else
                {
                    V.p <- NULL
                }
                if(storeW == TRUE)
                {
                    W.p                <- array(as.vector(mcmcRet$samples_W), c(n, q, nStore))
                }else
                {
                    W.p <- NULL
                }
                
                R.p            <- array(as.vector(mcmcRet$samples_R), c(q, q, nStore))                
                for(i in 1:q) R.p[i,i,] <- 1
                
                S.p            <- matrix(as.vector(mcmcRet$samples_S), nrow=nStore, byrow=T)
                
                phi.p        <- matrix(as.vector(mcmcRet$samples_phi), nrow=nStore, byrow=T)
                
                alpha0.p        <- matrix(as.vector(mcmcRet$samples_alpha0), nrow=nStore, byrow=T)
                beta0.p         <- matrix(as.vector(mcmcRet$samples_beta0), nrow=nStore, byrow=T)
                
                sigSq_alpha0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha0), nrow=nStore, byrow=T)
                sigSq_beta0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta0), nrow=nStore, byrow=T)
                
                sigSq_alpha.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha), nrow=nStore, byrow=T)
                sigSq_beta.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta), nrow=nStore, byrow=T)
                
                accept.A        <- matrix(as.vector(mcmcRet$samples_misc[1:(p0*q)]), nrow = p0, byrow = FALSE)
                accept.B        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+1):(p0*q+p1*q)]), nrow = p1, byrow = FALSE)
                accept.V        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+p1*q+1):(p0*q+p1*q+n*q)]), nrow = n, byrow = FALSE)
                accept.W        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+1):(p0*q+p1*q+n*q+n*q)]), nrow = n, byrow = FALSE)
                accept.alpha0 <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+1):(p0*q+p1*q+n*q+n*q+q)])
                accept.beta0 <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+1):(p0*q+p1*q+n*q+n*q+q+q)])
                accept.R <- mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1)]
                accept.S <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1+1):(p0*q+p1*q+n*q+n*q+q+q+1+q)])
                
                ret[[nam]] <- list(A.p = A.p, B.p = B.p, gamma.p=gamma.p, W.p=W.p, V.p=V.p, alpha0.p=alpha0.p, beta0.p=beta0.p, R.p=R.p, S.p=S.p, sigSq_alpha0.p=sigSq_alpha0.p, sigSq_beta0.p=sigSq_beta0.p, sigSq_alpha.p=sigSq_alpha.p, sigSq_beta.p=sigSq_beta.p, accept.reg = accept.B, accept.V = accept.V, accept.beta0=accept.beta0, accept.Sigma=accept.R, covNames.z=covNames.z, covNames.x=covNames.x)
            }else if(model == "restricted2")
        {
            ###
            p0 <- ncol(Xmat0)
            p1 <- ncol(Xmat1)
            p_adj <- ncol(Xmat.adj)
            
            p <- p1+p_adj
            p0 <- p1 <- p
            
            if(p_adj > 0){
                Xmat0 <- cbind(Xmat0, Xmat.adj)
                Xmat1 <- cbind(Xmat1, Xmat.adj)
            }
            
            covNames.z = c(colnames(Xmat0))
            covNames.x = c(colnames(Xmat1))
            
            ###
            gamma_beta <- temp$B
            gamma_beta[gamma_beta !=0] <- 1
            
            gamma_alpha <- temp$A
            gamma_alpha[gamma_alpha !=0] <- 1
            
            if (p_adj > 0) {
                for (i in 1:p_adj) {
                    gamma_beta[p - i + 1, ] <- 1
                    gamma_alpha[p - i + 1, ] <- 1
                }
            }
            temp$gamma_beta <- gamma_beta
            temp$gamma_alpha <- gamma_alpha
            
            nu_t <- 7.3
            sigSq_t <- pi^2*(nu_t-2)/(3*nu_t)
            numPhi      <- 3
            temp$phi <- rgamma(n, nu_t/2, nu_t/2)
            
            if(!is.null(offset))
            {
                offs <- offset
            }else
            {
                offs <- rep(1, n)
            }
            
            ###
            hyperP  <- as.vector(c(hyperParams$muS, hyperParams$PsiS, hyperParams$mu_alpha0, hyperParams$mu_alpha, hyperParams$mu_beta0, hyperParams$mu_beta, hyperParams$a_alpha0, hyperParams$b_alpha0, hyperParams$a_alpha, hyperParams$b_alpha, hyperParams$a_beta0, hyperParams$b_beta0, hyperParams$a_beta, hyperParams$b_beta, nu_t, sigSq_t, hyperParams$rho0, hyperParams$Psi0, hyperParams$v_beta, hyperParams$omega_beta, hyperParams$v_alpha, hyperParams$omega_alpha))
            
            mcmcP   <- as.vector(c(mcmcParams$tuning$beta0.prop.var, mcmcParams$tuning$beta.prop.var, mcmcParams$tuning$V.prop.var, mcmcParams$tuning$alpha0.prop.var, mcmcParams$tuning$alpha.prop.var, mcmcParams$tuning$W.prop.var, mcmcParams$tuning$S.prop.var, mcmcParams$tuning$rho.s))
            
            storeV <- mcmcParams$storage$storeV
            storeW <- mcmcParams$storage$storeW
            nPhi_save  <- numPhi
            
            store <- as.numeric(c(storeV, storeW))
            
            startV <- as.vector(c(temp$B, temp$A, temp$V, temp$W, temp$beta0, temp$alpha0, temp$R, temp$S, temp$sigSq_alpha0, temp$sigSq_alpha, temp$sigSq_beta0, temp$sigSq_beta, temp$phi, gamma_beta, gamma_alpha))
            
            mcmcRet     <- .C("mzip_restricted2_mcmc",
            Ymat            = as.double(as.matrix(Y)),
            Xmat0          = as.double(as.matrix(Xmat0)),
            Xmat1          = as.double(as.matrix(Xmat1)),
            offs            = as.double(offs),
            n                = as.integer(n),
            q                = as.integer(q),
            p0                = as.integer(p0),
            p1                = as.integer(p1),
            p                = as.integer(p),
            p_adj            = as.integer(p_adj),
            hyperP          = as.double(hyperP),
            mcmcP           = as.double(mcmcP),
            startValues     = as.double(startV),
            startGamma_beta      = as.double(gamma_beta),
            startGamma_alpha      = as.double(gamma_alpha),
            numReps            = as.integer(numReps),
            thin            = as.integer(thin),
            burninPerc      = as.double(burninPerc),
            nPhi_save        = as.integer(nPhi_save),
            store           = as.double(store),
            samples_beta0    = as.double(rep(0, nStore*q)),
            samples_B       = as.double(rep(0, nStore*p0*q)),
            samples_V       = as.double(rep(0, nStore*n*q)),
            samples_alpha0    = as.double(rep(0, nStore*q)),
            samples_A       = as.double(rep(0, nStore*p0*q)),
            samples_gamma_beta   = as.double(rep(0, nStore*p*q)),
            samples_gamma_alpha   = as.double(rep(0, nStore*p*q)),
            samples_W       = as.double(rep(0, nStore*n*q)),
            samples_R        = as.double(rep(0, nStore*q*q)),
            samples_S          = as.double(rep(0, nStore*q)),
            samples_sigSq_alpha0 = as.double(rep(0, nStore*1)),
            samples_sigSq_beta0 = as.double(rep(0, nStore*1)),
            samples_sigSq_alpha = as.double(rep(0, nStore*p0)),
            samples_sigSq_beta = as.double(rep(0, nStore*p1)),
            samples_phi     = as.double(rep(0, nStore*numPhi)),
            samples_misc    = as.double(rep(0, p0*q+p1*q+n*q+n*q+q+q+1+q)))
            
            A.p                <- array(as.vector(mcmcRet$samples_A), c(p0, q, nStore))
            B.p                <- array(as.vector(mcmcRet$samples_B), c(p1, q, nStore))
            gamma_beta.p        <- array(as.vector(mcmcRet$samples_gamma_beta), c(p, q, nStore))
            gamma_alpha.p    <- array(as.vector(mcmcRet$samples_gamma_alpha), c(p, q, nStore))
            if(storeV == TRUE)
            {
                V.p                <- array(as.vector(mcmcRet$samples_V), c(n, q, nStore))
            }else
            {
                V.p <- NULL
            }
            if(storeW == TRUE)
            {
                W.p                <- array(as.vector(mcmcRet$samples_W), c(n, q, nStore))
            }else
            {
                W.p <- NULL
            }
            
            R.p            <- array(as.vector(mcmcRet$samples_R), c(q, q, nStore))
            for(i in 1:q) R.p[i,i,] <- 1
            
            S.p            <- matrix(as.vector(mcmcRet$samples_S), nrow=nStore, byrow=T)
            
            phi.p        <- matrix(as.vector(mcmcRet$samples_phi), nrow=nStore, byrow=T)
            
            alpha0.p        <- matrix(as.vector(mcmcRet$samples_alpha0), nrow=nStore, byrow=T)
            beta0.p         <- matrix(as.vector(mcmcRet$samples_beta0), nrow=nStore, byrow=T)
            
            sigSq_alpha0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha0), nrow=nStore, byrow=T)
            sigSq_beta0.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta0), nrow=nStore, byrow=T)
            
            sigSq_alpha.p        <- matrix(as.vector(mcmcRet$samples_sigSq_alpha), nrow=nStore, byrow=T)
            sigSq_beta.p        <- matrix(as.vector(mcmcRet$samples_sigSq_beta), nrow=nStore, byrow=T)
            
            accept.A        <- matrix(as.vector(mcmcRet$samples_misc[1:(p0*q)]), nrow = p0, byrow = FALSE)
            accept.B        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+1):(p0*q+p1*q)]), nrow = p1, byrow = FALSE)
            accept.V        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+p1*q+1):(p0*q+p1*q+n*q)]), nrow = n, byrow = FALSE)
            accept.W        <- matrix(as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+1):(p0*q+p1*q+n*q+n*q)]), nrow = n, byrow = FALSE)
            accept.alpha0 <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+1):(p0*q+p1*q+n*q+n*q+q)])
            accept.beta0 <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+1):(p0*q+p1*q+n*q+n*q+q+q)])
            accept.R <- mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1)]
            accept.S <- as.vector(mcmcRet$samples_misc[(p0*q+p1*q+n*q+n*q+q+q+1+1):(p0*q+p1*q+n*q+n*q+q+q+1+q)])
            
            ret[[nam]] <- list(A.p = A.p, B.p = B.p, gamma_beta.p=gamma_beta.p, gamma_alpha.p=gamma_alpha.p, W.p=W.p, V.p=V.p, alpha0.p=alpha0.p, beta0.p=beta0.p, R.p=R.p, S.p=S.p, sigSq_alpha0.p=sigSq_alpha0.p, sigSq_beta0.p=sigSq_beta0.p, sigSq_alpha.p=sigSq_alpha.p, sigSq_beta.p=sigSq_beta.p, accept.A = accept.A, accept.B = accept.B, accept.V = accept.V, accept.beta0=accept.beta0, accept.R=accept.R, accept.S=accept.S, covNames.z=covNames.z, covNames.x=covNames.x)
        }
            
            chain = chain + 1
        }
        
        ret[["setup"]]    <- list(hyperParams = hyperParams, startValues = startValues, mcmcParams = mcmcParams, numReps = numReps, thin = thin, burninPerc = burninPerc, model = model, nChain = nChain)
        
        if(model == "generalized")
        {
            class(ret) <- c("mzipBvs", "generalized")
        }else if(model == "restricted1")
        {
            class(ret) <- c("mzipBvs", "restricted1")
        }else if(model == "restricted2")
        {
            class(ret) <- c("mzipBvs", "restricted2")
        }
        
        return(ret)
    }
    else
    {
        print(" (numReps * burninPerc) must be divisible by (thin)")
    }

    return(ret)
}

