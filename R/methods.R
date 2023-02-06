

####
## PRINT METHOD
####
##

print.mvnBvs <- function(x, digits=3, ...)
{
    p <- dim(x$B.p)[1]
    q <- dim(x$B.p)[2]
    
    nS <- dim(x$B.p)[3]
    value <- list(model=class(x)[2])
    cov.names <- x$covNames
    out.names <- colnames(matrix(x$B.p[1,,1], 1, q), do.NULL=FALSE, prefix = "Outcome.")
    
    cat("\nMultivariate Bayesian Variable Selection \n")
    
    if(class(x)[2] == "factor-analytic")
    {
        ##
        cat("\nCovariance Structure: Factor-analytic \n")
    }
    if(class(x)[2] == "unstructured")
    {
        ##
        cat("\nCovariance Structure: Unstructured \n")
    }
    ##
    cat("Number of scans:     ", x$setup$numReps,"\n")
    ##
    cat("Thinning:            ", x$setup$thin,"\n")
    ##
    cat("Percentage of burnin: ", x$setup$burninPerc*100, "%\n", sep = "")
    
    cat("\n#####")
    
    #B and Gamma
    B <- array(NA, c(p,q, nS))
    Gamma <- array(NA, c(p,q, nS))
    B[,,1:nS] <- x$B.p
    Gamma[,,1:nS] <- x$gamma.p
    
    Gamma.Mean	<- apply(Gamma, c(1,2), mean)
    B.Mean <- matrix(NA, p, q)
    B.Sd <-	matrix(NA, p, q)
    for(k in 1:p)
    {
        for(j in 1:q)
        {
            if(any(B[k,j,]!=0))
            {
                B.Mean[k, j] <- mean(B[k,j,B[k,j,]!=0])
                B.Sd[k, j] <- sd(B[k,j,B[k,j,]!=0])
            }else
            {
                B.Mean[k, j] <- 0
                B.Sd[k, j] <- 0
            }
        }
    }
    rownames(Gamma.Mean) <- cov.names
    rownames(B.Mean) <- cov.names
    rownames(B.Sd) <- cov.names
    
    colnames(Gamma.Mean) <- out.names
    colnames(B.Mean) <- out.names
    colnames(B.Sd) <- out.names
    
    output.BG <- list()
    output.BG[[cov.names[1]]] <- cbind(B.Mean[1,], B.Sd[1,], Gamma.Mean[1,])
    colnames(output.BG[[cov.names[1]]]) <- c("beta|gamma=1", "SD", "gamma")
    
    if(p > 1){
        for(i in 2:p){
            nam <- cov.names[i]
            output.BG[[nam]] <- cbind(B.Mean[i,], B.Sd[i,], Gamma.Mean[i,])
            colnames(output.BG[[nam]]) <- c("beta|gamma=1", "SD", "gamma")
        }
    }
    
    ##
    cat("\nRegression Coefficients and Inclusion Probabilities \n")
    print(output.BG, digits=digits)
}




print.mzipBvs <- function(x, digits=3, ...)
{
    p_x <- dim(x$B.p)[1]
    p_z <- dim(x$A.p)[1]
    q <- dim(x$B.p)[2]
    
    nS <- dim(x$B.p)[3]
    value <- list(model=class(x)[2])
    cov.names.z <- x$covNames.z
    cov.names.x <- x$covNames.x
    out.names <- colnames(matrix(x$B.p[1,,1], 1, q), do.NULL=FALSE, prefix = "Outcome.")
    
    cat("\nMultivariate Bayesian Variable Selection \n")
    
    if(class(x)[2] == "generalized")
    {
        ##
        cat("\nModel: generalized \n")
    }
    if(class(x)[2] == "restricted1")
    {
        ##
        cat("\nModel: restricted1 \n")
    }
    if(class(x)[2] == "restricted2")
    {
        ##
        cat("\nModel: restricted2 \n")
    }
    
    ##
    cat("Number of scans:     ", x$setup$numReps,"\n")
    ##
    cat("Thinning:            ", x$setup$thin,"\n")
    ##
    cat("Percentage of burnin: ", x$setup$burninPerc*100, "%\n", sep = "")
    
    cat("\n#####")
    
    #B and Gamma
    B <- array(NA, c(p_x, q, nS))
    Gamma <- array(NA, c(p_x,q, nS))
    B[,,1:nS] <- x$B.p
    Gamma[,,1:nS] <- x$gamma_beta.p
    
    Gamma.Mean    <- apply(Gamma, c(1,2), mean)
    B.Mean <- matrix(NA, p_x, q)
    B.Sd <-    matrix(NA, p_x, q)
    for(k in 1:p_x)
    {
        for(j in 1:q)
        {
            if(any(B[k,j,]!=0))
            {
                B.Mean[k, j] <- mean(B[k,j,B[k,j,]!=0])
                B.Sd[k, j] <- sd(B[k,j,B[k,j,]!=0])
            }else
            {
                B.Mean[k, j] <- 0
                B.Sd[k, j] <- 0
            }
        }
    }
    rownames(Gamma.Mean) <- cov.names.x
    rownames(B.Mean) <- cov.names.x
    rownames(B.Sd) <- cov.names.x
    
    colnames(Gamma.Mean) <- out.names
    colnames(B.Mean) <- out.names
    colnames(B.Sd) <- out.names
    
    output.BG <- list()
    output.BG[[cov.names.x[1]]] <- cbind(B.Mean[1,], B.Sd[1,], Gamma.Mean[1,])
    colnames(output.BG[[cov.names.x[1]]]) <- c("beta|gamma=1", "SD", "gamma")
    
    if(p_x > 1){
        for(i in 2:p_x){
            nam <- cov.names.x[i]
            output.BG[[nam]] <- cbind(B.Mean[i,], B.Sd[i,], Gamma.Mean[i,])
            colnames(output.BG[[nam]]) <- c("beta|gamma=1", "SD", "gamma")
        }
    }
    
    ##
    cat("\nRegression Coefficients and Inclusion Probabilities for the Count Component\n")
    print(output.BG, digits=digits)
    
    cat("\n#####")
    
    #A and Gamma_alpha
    B <- array(NA, c(p_z, q, nS))
    Gamma <- array(NA, c(p_z,q, nS))
    B[,,1:nS] <- x$A.p
    Gamma[,,1:nS] <- x$gamma_alpha.p
    
    Gamma.Mean    <- apply(Gamma, c(1,2), mean)
    B.Mean <- matrix(NA, p_z, q)
    B.Sd <-    matrix(NA, p_z, q)
    for(k in 1:p_z)
    {
        for(j in 1:q)
        {
            if(any(B[k,j,]!=0))
            {
                B.Mean[k, j] <- mean(B[k,j,B[k,j,]!=0])
                B.Sd[k, j] <- sd(B[k,j,B[k,j,]!=0])
            }else
            {
                B.Mean[k, j] <- 0
                B.Sd[k, j] <- 0
            }
        }
    }
    rownames(Gamma.Mean) <- cov.names.z
    rownames(B.Mean) <- cov.names.z
    rownames(B.Sd) <- cov.names.z
    
    colnames(Gamma.Mean) <- out.names
    colnames(B.Mean) <- out.names
    colnames(B.Sd) <- out.names
    
    output.BG <- list()
    output.BG[[cov.names.z[1]]] <- cbind(B.Mean[1,], B.Sd[1,], Gamma.Mean[1,])
    colnames(output.BG[[cov.names.z[1]]]) <- c("alpha|gamma=1", "SD", "delta")
    
    if(p_z > 1){
        for(i in 2:p_z){
            nam <- cov.names.z[i]
            output.BG[[nam]] <- cbind(B.Mean[i,], B.Sd[i,], Gamma.Mean[i,])
            colnames(output.BG[[nam]]) <- c("alpha|delta=1", "SD", "delta")
        }
    }
    
    ##
    cat("\nRegression Coefficients and Inclusion Probabilities for the Binary Component\n")
    print(output.BG, digits=digits)
}




















####
## SUMMARY METHOD
####
##

summary.mvnBvs <- function(object, digits=3, ...)
{
    x <- object
    p <- dim(x$B.p)[1]
    q <- dim(x$B.p)[2]
    
    nS <- dim(x$B.p)[3]
    value <- list(model=class(x)[2])
    cov.names <- x$covNames
    out.names <- colnames(matrix(x$B.p[1,,1], 1, q), do.NULL=FALSE, prefix = "Outcome.")

    # estimates
    ##
    
    #B and Gamma
    B <- array(NA, c(p,q, nS))
    Gamma <- array(NA, c(p,q, nS))
    B[,,1:nS] <- x$B.p
    Gamma[,,1:nS] <- x$gamma.p
    
    Gamma.Mean	<- apply(Gamma, c(1,2), mean)
    B.Mean <- matrix(NA, p, q)
    B.Sd <-	matrix(NA, p, q)
    for(k in 1:p)
    {
        for(j in 1:q)
        {
            if(any(B[k,j,]!=0))
            {
                B.Mean[k, j] <- mean(B[k,j,B[k,j,]!=0])
                B.Sd[k, j] <- sd(B[k,j,B[k,j,]!=0])
            }else
            {
                B.Mean[k, j] <- 0
                B.Sd[k, j] <- 0
            }
        }
    }
    rownames(Gamma.Mean) <- cov.names
    rownames(B.Mean) <- cov.names
    rownames(B.Sd) <- cov.names
    
    colnames(Gamma.Mean) <- out.names
    colnames(B.Mean) <- out.names
    colnames(B.Sd) <- out.names
    
    output.BG <- list()
    output.BG[[cov.names[1]]] <- cbind(B.Mean[1,], B.Sd[1,], Gamma.Mean[1,])
    colnames(output.BG[[cov.names[1]]]) <- c("beta|gamma=1", "SD", "gamma")
    
    if(p > 1){
        for(i in 2:p){
            nam <- cov.names[i]
            output.BG[[nam]] <- cbind(B.Mean[i,], B.Sd[i,], Gamma.Mean[i,])
            colnames(output.BG[[nam]]) <- c("beta|gamma=1", "SD", "gamma")
        }
    }
    
    #beta0
    beta0 <- x$beta0.p

    beta0.Med <- apply(beta0, 2, median)
    beta0.Mean <- apply(beta0, 2, mean)
    beta0.Sd <- apply(beta0, 2, sd)
    beta0.Ub <- apply(beta0, 2, quantile, prob = 0.975)
    beta0.Lb <- apply(beta0, 2, quantile, prob = 0.025)
    
    output.beta0 <- cbind(beta0.Mean, beta0.Lb, beta0.Ub)
    dimnames(output.beta0) <- list(out.names, c("beta0", "LL", "UL"))
    
    value$BetaGamma <- output.BG
    value$beta0 <- output.beta0
    
    
    if(class(x)[2] == "unstructured")
    {
        
        #Sigma
        Sigma <- array(NA, c(q,q, nS))
        Sigma[,,1:nS] <- x$Sigma.p
        
        Sigma.Med <- apply(Sigma, c(1,2), median)
        Sigma.Sd <- apply(Sigma, c(1,2), sd)
        Sigma.Ub <- apply(Sigma, c(1,2), quantile, prob = 0.975)
        Sigma.Lb <- apply(Sigma, c(1,2), quantile, prob = 0.025)
        
        dimnames(Sigma.Med) <- list(c(rep("", q)), c("Sigma-PM", rep("", q-1)))
        dimnames(Sigma.Sd) <- list(c(rep("", q)), c("Sigma-SD", rep("", q-1)))
        dimnames(Sigma.Lb) <- list(c(rep("", q)), c("Sigma-LL", rep("", q-1)))
        dimnames(Sigma.Ub) <- list(c(rep("", q)), c("Sigma-UL", rep("", q-1)))
        
        value$Sigma.PM <- Sigma.Med
        value$Sigma.SD <- Sigma.Sd
        value$Sigma.LL <- Sigma.Lb
        value$Sigma.UL <- Sigma.Ub

    }
    
    if(class(x)[2] == "factor-analytic")
    {
        
        #lambda
        lambda <- x$lambda.p
        
        lambda.Med <- apply(lambda, 2, median)
        lambda.Mean <- apply(lambda, 2, mean)
        lambda.Sd <- apply(lambda, 2, sd)
        lambda.Ub <- apply(lambda, 2, quantile, prob = 0.975)
        lambda.Lb <- apply(lambda, 2, quantile, prob = 0.025)
        
        output.lambda <- cbind(lambda.Mean, lambda.Lb, lambda.Ub)
        dimnames(output.lambda) <- list(out.names, c("lambda", "LL", "UL"))
        
        value$lambda <- output.lambda
        
        sigSq.p <- x$sigSq.p
        
        sigSq.pMed <- apply(sigSq.p, 2, median)
        sigSq.pUb <- apply(sigSq.p, 2, quantile, prob = 0.975)
        sigSq.pLb <- apply(sigSq.p, 2, quantile, prob = 0.025)
        
        output.sigSq <- cbind(sigSq.pMed, sigSq.pLb, sigSq.pUb)
        dimnames(output.sigSq) <- list("", c( "sigmaSq", "LL", "UL"))
        
        value$sigmaSq <- output.sigSq
    }
    
    
    value$setup <- x$setup
    class(value) <- "summ.mvnBvs"
    return(value)
}


summary.mzipBvs <- function(object, digits=3, ...)
{
    x <- object
    p_x <- dim(x$B.p)[1]
    p_z <- dim(x$A.p)[1]
    q <- dim(x$B.p)[2]
    
    nS <- dim(x$B.p)[3]
    value <- list(model=class(x)[2])
    cov.names.z <- x$covNames.z
    cov.names.x <- x$covNames.x
    out.names <- colnames(matrix(x$B.p[1,,1], 1, q), do.NULL=FALSE, prefix = "Outcome.")

    # estimates
    ##
    
    #B and Gamma
    B <- array(NA, c(p_x, q, nS))
    Gamma <- array(NA, c(p_x,q, nS))
    B[,,1:nS] <- x$B.p
    Gamma[,,1:nS] <- x$gamma_beta.p
    
    Gamma.Mean    <- apply(Gamma, c(1,2), mean)
    B.Mean <- matrix(NA, p_x, q)
    B.Sd <-    matrix(NA, p_x, q)
    for(k in 1:p_x)
    {
        for(j in 1:q)
        {
            if(any(B[k,j,]!=0))
            {
                B.Mean[k, j] <- mean(B[k,j,B[k,j,]!=0])
                B.Sd[k, j] <- sd(B[k,j,B[k,j,]!=0])
            }else
            {
                B.Mean[k, j] <- 0
                B.Sd[k, j] <- 0
            }
        }
    }
    rownames(Gamma.Mean) <- cov.names.x
    rownames(B.Mean) <- cov.names.x
    rownames(B.Sd) <- cov.names.x
    
    colnames(Gamma.Mean) <- out.names
    colnames(B.Mean) <- out.names
    colnames(B.Sd) <- out.names
    
    output.BG <- list()
    output.BG[[cov.names.x[1]]] <- cbind(B.Mean[1,], B.Sd[1,], Gamma.Mean[1,])
    colnames(output.BG[[cov.names.x[1]]]) <- c("beta|gamma=1", "SD", "gamma")
    
    if(p_x > 1){
        for(i in 2:p_x){
            nam <- cov.names.x[i]
            output.BG[[nam]] <- cbind(B.Mean[i,], B.Sd[i,], Gamma.Mean[i,])
            colnames(output.BG[[nam]]) <- c("beta|gamma=1", "SD", "gamma")
        }
    }
    
    #beta0
    beta0 <- x$beta0.p
    
    beta0.Med <- apply(beta0, 2, median)
    beta0.Mean <- apply(beta0, 2, mean)
    beta0.Sd <- apply(beta0, 2, sd)
    beta0.Ub <- apply(beta0, 2, quantile, prob = 0.975)
    beta0.Lb <- apply(beta0, 2, quantile, prob = 0.025)
    
    output.beta0 <- cbind(beta0.Mean, beta0.Lb, beta0.Ub)
    dimnames(output.beta0) <- list(out.names, c("beta0", "LL", "UL"))
    
    value$BetaGamma <- output.BG
    value$beta0 <- output.beta0
    
    
    #A and delta
    B <- array(NA, c(p_z, q, nS))
    Gamma <- array(NA, c(p_z,q, nS))
    B[,,1:nS] <- x$A.p
    Gamma[,,1:nS] <- x$gamma_alpha.p
    
    Gamma.Mean    <- apply(Gamma, c(1,2), mean)
    B.Mean <- matrix(NA, p_z, q)
    B.Sd <-    matrix(NA, p_z, q)
    for(k in 1:p_z)
    {
        for(j in 1:q)
        {
            if(any(B[k,j,]!=0))
            {
                B.Mean[k, j] <- mean(B[k,j,B[k,j,]!=0])
                B.Sd[k, j] <- sd(B[k,j,B[k,j,]!=0])
            }else
            {
                B.Mean[k, j] <- 0
                B.Sd[k, j] <- 0
            }
        }
    }
    rownames(Gamma.Mean) <- cov.names.z
    rownames(B.Mean) <- cov.names.z
    rownames(B.Sd) <- cov.names.z
    
    colnames(Gamma.Mean) <- out.names
    colnames(B.Mean) <- out.names
    colnames(B.Sd) <- out.names
    
    output.BG <- list()
    output.BG[[cov.names.z[1]]] <- cbind(B.Mean[1,], B.Sd[1,], Gamma.Mean[1,])
    colnames(output.BG[[cov.names.z[1]]]) <- c("alpha|gamma=1", "SD", "delta")
    
    if(p_z > 1){
        for(i in 2:p_z){
            nam <- cov.names.z[i]
            output.BG[[nam]] <- cbind(B.Mean[i,], B.Sd[i,], Gamma.Mean[i,])
            colnames(output.BG[[nam]]) <- c("alpha|delta=1", "SD", "delta")
        }
    }
    
    #alpha0
    beta0 <- x$alpha0.p
    
    beta0.Med <- apply(beta0, 2, median)
    beta0.Mean <- apply(beta0, 2, mean)
    beta0.Sd <- apply(beta0, 2, sd)
    beta0.Ub <- apply(beta0, 2, quantile, prob = 0.975)
    beta0.Lb <- apply(beta0, 2, quantile, prob = 0.025)
    
    output.beta0 <- cbind(beta0.Mean, beta0.Lb, beta0.Ub)
    dimnames(output.beta0) <- list(out.names, c("alpha0", "LL", "UL"))
    
    value$AlphaDelta <- output.BG
    value$alpha0 <- output.beta0
    
    if(class(x)[2] == "generalized")
    {
        #Sigma_V
        Sigma <- array(NA, c(q,q, nS))
        Sigma[,,1:nS] <- x$Sigma_V.p
        
        Sigma.Med <- apply(Sigma, c(1,2), median)
        Sigma.Sd <- apply(Sigma, c(1,2), sd)
        Sigma.Ub <- apply(Sigma, c(1,2), quantile, prob = 0.975)
        Sigma.Lb <- apply(Sigma, c(1,2), quantile, prob = 0.025)
        
        dimnames(Sigma.Med) <- list(c(rep("", q)), c("SigmaV-PM", rep("", q-1)))
        dimnames(Sigma.Sd) <- list(c(rep("", q)), c("SigmaV-SD", rep("", q-1)))
        dimnames(Sigma.Lb) <- list(c(rep("", q)), c("SigmaV-LL", rep("", q-1)))
        dimnames(Sigma.Ub) <- list(c(rep("", q)), c("SigmaV-UL", rep("", q-1)))
        
        value$SigmaV.PM <- Sigma.Med
        value$SigmaV.SD <- Sigma.Sd
        value$SigmaV.LL <- Sigma.Lb
        value$SigmaV.UL <- Sigma.Ub
        
        #R
        Sigma <- array(NA, c(q,q, nS))
        Sigma[,,1:nS] <- x$R.p
        
        Sigma.Med <- apply(Sigma, c(1,2), median)
        Sigma.Sd <- apply(Sigma, c(1,2), sd)
        Sigma.Ub <- apply(Sigma, c(1,2), quantile, prob = 0.975)
        Sigma.Lb <- apply(Sigma, c(1,2), quantile, prob = 0.025)
        
        dimnames(Sigma.Med) <- list(c(rep("", q)), c("R-PM", rep("", q-1)))
        dimnames(Sigma.Sd) <- list(c(rep("", q)), c("R-SD", rep("", q-1)))
        dimnames(Sigma.Lb) <- list(c(rep("", q)), c("R-LL", rep("", q-1)))
        dimnames(Sigma.Ub) <- list(c(rep("", q)), c("R-UL", rep("", q-1)))
        
        value$R.PM <- Sigma.Med
        value$R.SD <- Sigma.Sd
        value$R.LL <- Sigma.Lb
        value$R.UL <- Sigma.Ub
    }

    value$setup <- x$setup
    class(value) <- "summ.mzipBvs"
    return(value)
}

####
## PRINT.SUMMARY METHOD
####
##


print.summ.mvnBvs <- function(x, digits=3, ...)
{
    
    cat("\nMultivariate Bayesian Variable Selection \n")
    
    if(x$model == "factor-analytic")
    {
        ##
        cat("\nCovariance Structure: Factor-analytic \n")
    }
    if(x$model == "unstructured")
    {
        ##
        cat("\nCovariance Structure: Unstructured \n")
    }
    cat("\n#####")
    
    ##
    cat("\nRegression Coefficients and Inclusion Probabilities \n")
    print(x$BetaGamma, digits=digits)
    
    cat("\n#####")
    
    ##
    cat("\nIntercepts \n")
    print(x$beta0, digits=digits)
    
    if(x$model == "factor-analytic")
    {
        cat("\n#####")
        ##
        cat("\nFactor Loadings \n")
        print(x$lambda, digits=digits)
        
        cat("\n#####")
        ##
        cat("\nResidual Variance \n")
        print(x$sigmaSq, digits=digits)
    }
    
}



print.summ.mzipBvs <- function(x, digits=3, ...)
{
    
    cat("\nMultivariate Bayesian Variable Selection \n")
    
    if(x$model == "generalized")
    {
        ##
        cat("\nModel: generalized \n")
    }
    if(x$model == "restricted1")
    {
        ##
        cat("\nModel: restricted1 \n")
    }
    if(x$model == "restricted2")
    {
        ##
        cat("\nModel: restricted2 \n")
    }
    cat("\n#####")
    
    ##
    cat("\nRegression Coefficients and Inclusion Probabilities for the Count Component\n")
    print(x$BetaGamma, digits=digits)
    
    cat("\n#####")
    
    ##
    cat("\nIntercepts for the Count Component\n")
    print(x$beta0, digits=digits)
    cat("\n#####")
    
    ##
    cat("\nRegression Coefficients and Inclusion Probabilities for the Binary Component\n")
    print(x$AlphaDelta, digits=digits)
    
    cat("\n#####")
    
    ##
    cat("\nIntercepts for the Binary Component\n")
    print(x$alpha0, digits=digits)
    
    
}








