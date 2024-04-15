initiate_startValues <-
function(Formula, Y, data, model="MMZIP", B=NULL, beta0=NULL, V=NULL, SigmaV=NULL, gamma_beta=NULL, A=NULL, alpha0=NULL, W=NULL, m=NULL, gamma_alpha=NULL, sigSq_beta=NULL, sigSq_beta0=NULL, sigSq_alpha=NULL, sigSq_alpha0=NULL)
{

    ## MMZIP
    
    if(model == "MMZIP")
    {
        cat(paste("Starting values are initiated for ", model, sep = ""), cat("\n"))
        
        n = dim(Y)[1]
        q = dim(Y)[2]
        
        Xmat0 <- model.frame(Formula[[1]], data=data)
        Xmat1 <- model.frame(Formula[[2]], data=data)
        Xmat_adj <- model.frame(Formula[[3]], data=data)
        
        p_adj = ncol(Xmat_adj)
        p0 <- ncol(Xmat0) + p_adj
        p1 <- ncol(Xmat1) + p_adj
        
        if(is.null(B))
        {
            B <- matrix(rnorm(p1*q, 0, 0.1), p1, q)
        }
        if(is.null(beta0))
        {
            beta0 <- rnorm(q, 0, 0.1)
        }
        if(is.null(SigmaV))
        {
            SigmaV <- diag(runif(q, 0.3, 0.5), q)
        }
        if(is.null(V))
        {
            V <- matrix(NA, n, q)
            for(i in 1:n) V[i,] <- rmvnorm(1, mean = rep(0, q), sigma = SigmaV)
        }
        if(is.null(gamma_beta))
        {
            gamma_beta <- B
            gamma_beta[gamma_beta !=0] <- 1
        }
        
        if(is.null(A))
        {
            A <- matrix(rnorm(p0*q, 0, 0.1), p0, q)
        }
        if(is.null(alpha0))
        {
            alpha0 <- rnorm(q, 0, 0.1)
        }
        if(is.null(m))
        {
            m <- rnorm(q, 0, 0.1)
        }
        
        if(is.null(W))
        {
            RW <- cov2cor(m %*% t(m) + diag(1, q))
            W <- matrix(NA, n, q)
            for(i in 1:n) W[i,] <- rmvnorm(1, mean = rep(0, q), sigma = RW)
            W.lin   <- matrix(rep(alpha0, n), ncol = q, byrow = T) + as.matrix(cbind(Xmat0, Xmat_adj))%*%A
            W   <- W + W.lin
            for(i in 1:n)
            {
                for(j in 1:q)
                {
                    if(Y[i,j] != 0) W[i,j] <- abs(W[i,j])
                }
            }
        }
        
        if(is.null(gamma_alpha))
        {
            gamma_alpha <- A
            gamma_alpha[gamma_alpha !=0] <- 1
        }
        
        if(is.null(sigSq_beta))
        {
            sigSq_beta <- rep(0.5, p1) * runif(1, 0.8, 1.2)
        }
        if(is.null(sigSq_beta0))
        {
            sigSq_beta0 <- 0.5 * runif(1, 0.8, 1.2)
        }
        if(is.null(sigSq_alpha))
        {
            sigSq_alpha <- rep(0.5, p0) * runif(1, 0.8, 1.2)
        }
        if(is.null(sigSq_alpha0))
        {
            sigSq_alpha0 <- 0.5 * runif(1, 0.8, 1.2)
        }
        
        ret <- list(B=B, beta0=beta0, V=V, SigmaV=SigmaV, gamma_beta=gamma_beta, A=A, alpha0=alpha0, W=W, m=m, gamma_alpha=gamma_alpha, sigSq_beta=sigSq_beta, sigSq_beta0=sigSq_beta0, sigSq_alpha=sigSq_alpha, sigSq_alpha0=sigSq_alpha0)
        
    }
    
    ##
    return(ret)
}
