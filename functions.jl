using LinearAlgebra
#import Pkg; Pkg.add("Distributions")
using Random, Distributions
import Plots as pl 



# creating the grid of fine points and the coarse observation points
function Create_Grid_and_Observation(x_gridsize,y_gridsize,N_coarse, plot::Bool )
    N_all=N_fine+N_coarse+1
    coord_fine=ones(x_gridsize*y_gridsize,2)
    for x in 0:(x_gridsize-1) 
        for y in 0:(y_gridsize-1)
            coord_fine[y+x*y_gridsize+1,1]=x/(x_gridsize-1)
            coord_fine[y+x*y_gridsize+1,2]=y/(y_gridsize-1)
        end
    end
    coord_coarse=rand(N_coarse,2)
    coord_x0=rand(1,2)
    coord_all=vcat(coord_fine, coord_coarse, coord_x0)

    if plot
        t=pl.scatter(coord_fine[:,1],coord_fine[:,2], label="Fine Grid to simulate on")
        pl.scatter!(coord_coarse[:,1],coord_coarse[:,2],label="Coarse Observations")
        pl.scatter!(coord_x0[:,1],coord_x0[:,2],label="Normalizing Observation")
        pl.title!("Observation and simulation points")
        display(t)
    end
    return coord_coarse, coord_fine, coord_x0
end

#l-pareto simulation for creating data
function simu_specfcts(coord, vario, chol_mat, alpha=1) 
    N=size(coord,1)
    shift=rand(1:N)
    trend=[vario(coord[i,:]-coord[shift,:]) for i in 1:N]
    res=chol_mat * rand(Normal(),N)
    res=exp.((res.-res[shift]-trend)) # W_ell (maybe without normalizing constant)
    res=res/mean(res) # W_ell/ l(w_ell)
    res*=(1/(1-rand()))^(1/alpha)
end

#variogram of the process
vario(x,beta,zeta,lambda_1,lambda_2)=sqrt(
                                            (x[1]*exp(lambda_1)*cos(zeta)+x[2]*exp(lambda_2)*sin(zeta))^2
                                            + (-x[1]*exp(lambda_1)*sin(zeta)+x[2]*exp(lambda_2)*cos(zeta))^2
                                            ).^beta


#cov and chol matrices for gaussian processes
function chol_mat(coord, vario)
    N=size(coord,1)
    cov_mat=ones(N,N)
    for i in 1:N
        for j in 1:N
            cov_mat[i,j]=vario(coord[i,:])+vario(coord[j,:])-vario(coord[i,:]-coord[j,:])
        end
    end
    cov_mat.+=1e-6 
    chol_mat=cholesky(cov_mat).L
end
function chol_mat_and_mu(coord,vario,coord_ref=coord_x0)
    N=size(coord,1)
    coord=coord.-coord_ref
    mu=-map(vario, eachrow(coord)) #hier 1/2 diag von covmat
    cov_mat=ones(N,N)
    for i in 1:N,j in 1:N
            cov_mat[i,j]=vario(coord[i,:])+vario(coord[j,:])-vario(coord[i,:]-coord[j,:])
    end
    result=Dict("cov_mat" => cov_mat, "chol_mat" => cholesky(Symmetric(cov_mat).+1e-6 ).L, "mu"=> mu)
end

function cov_mat_coarse_fine_fun(coord_fine,coord_coarse, vario,coord_ref=coord_x0)
    N_fine=size(coord_fine,1)
    N_coarse=size(coord_coarse,1)
    coord_fine=coord_fine.-coord_ref
    coord_coarse=coord_coarse.-coord_ref
    cov_mat_coarse_fine=ones(N_fine,N_coarse)
    for i in 1:N_fine,j in 1:N_coarse
            cov_mat_coarse_fine[i,j]=vario(coord_fine[i,:])+vario(coord_coarse[j,:])-vario(coord_fine[i,:]-coord_coarse[j,:])
    end
    cov_mat_coarse_fine
end

function chol_mat_fine_cond_coarse_fun(cov_mat_fine,cov_mat_coarse_fine,cov_mat_coarse_inv)
    chov_mat_fine_cond_coarse=cov_mat_fine - cov_mat_coarse_fine * cov_mat_coarse_inv * transpose(cov_mat_coarse_fine)
    cholesky(Symmetric(chov_mat_fine_cond_coarse) ).L
end

#conditional gaussian process simulation
function r_cond_gaussian(mu_fine,mu_coarse,cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse ,cond_values_coarse)
    N_fine=size(mu_fine,1)
    mu_fine_cond_coarse=mu_fine+cov_mat_coarse_fine * cov_mat_coarse_inv * (cond_values_coarse - mu_coarse)
    chol_mat_fine_cond_coarse * rand(Normal(),N_fine)+mu_fine_cond_coarse
end

#gaussian process simulation
function r_gaussian(mu,chol_mat)
    chol_mat * rand(Normal(),size(mu,1))+mu
end

#gaussian density
function d_gaussian(mu,cov_mat_inv,x)
    N=size(mu,1)
    exp(-0.5*transpose(x-mu) * cov_mat_inv * (x-mu) )*sqrt((2*pi)^(-N)) * sqrt(det(cov_mat_inv))
end

#log likelihood of log gaussian density of observations 
function l_1_fun(d_gaussian,mu_coarse,cov_mat_coarse_inv,log_observation)
    sum([log(d_gaussian(mu_coarse,cov_mat_coarse_inv,log_observation[i,:])) for i in 1:size(log_observation,1)])
end

#reciprocal mean estimation
# one estimate of c is: mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha)
#therefore we N_est_c many of them and take there mean, in addition we multiply with the number of threshold exccedances
function l_2_fun(number_of_excced,r_gaussian,mu_fine, chol_fine,alpha,N_est_c )
   -number_of_excced * log(mean([mean(exp.(r_gaussian(mu_fine,
            chol_fine)))^(alpha) for i in 1: N_est_c]))  
    # minus for 1/c_l (in log)
end

 #conditional mean estimator for intergal part
function l_3_fun(N_est_cond,alpha,log_observation,r_cond_gaussian,mu_fine,mu_coarse,cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse)    
    res_cond_int=rand(N_est_cond,size(log_observation,1))
    for row in 1:N_est_cond
        res_cond_int[row,:]=[(mean(
                            exp.(
                            r_cond_gaussian(mu_fine,mu_coarse,cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse ,log_observation[i,:])
                            ))^alpha) for i in 1:size(log_observation,1) ]
    end
    sum(log.(mean( res_cond_int, dims=1)))
end


function exceed_cond_sim(n_sim,num_obs,normalized_observation_data,observation_data_x0,threshold, alpha, r_cond_gaussian,
    mu_fine,mu_coarse, cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse )
   res_cond_sim=zeros(num_obs,N_fine)
   res_ell_X=[0.0 for i in 1:num_obs] 
for trial in 1:n_sim 
for i in 1:num_obs
   normalized_observation=normalized_observation_data[i,:]
   if (trial==1)
       old_value=r_cond_gaussian(mu_fine,mu_coarse,cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse ,log.(normalized_observation))
   else
       old_value=res_cond_sim[i,:]
   end
   proposal=r_cond_gaussian(mu_fine,mu_coarse,cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse ,log.(normalized_observation))
   acceptance_rate=min(1,mean(exp.(proposal))^alpha/mean(exp.(old_value))^alpha)   
   if (rand()< acceptance_rate)
       old_value=proposal
   end
   res_ell_X[i]=observation_data_x0[i]*mean(exp.(old_value))
   res_cond_sim[i,:]=old_value
end
end
if sum(res_ell_X.>threshold)==0
   println("not a single threshold exceedance")
else
   #likelihood calculation and param updates
   #find all threshold exccedances and calculate the log of them
log_observation=log.(normalized_observation_data[findall(res_ell_X.>threshold),:])
end
end






#MCMC algorithm

function MCMC(N_MCMC,N_coarse, coord_coarse, N_fine,coord_fine, N_est_c, N_est_cond,alpha,  par_beta,par_zeta,par_lambda_1,par_lambda_2,
    observation_data,vario,chol_mat_and_mu, cov_mat_coarse_fine_fun, chol_mat_fine_cond_coarse_fun, 
    r_cond_gaussian, r_gaussian, d_gaussian,l_1_fun,l_2_fun,l_3_fun,n_trial_print)

num_obs=size(observation_data,1)
#normalize observation
normalized_observation_data=reduce(hcat,[observation_data[i,1:N_coarse]./observation_data[i,N_coarse+1] for i in 1:num_obs])'

par_beta_vec  = repeat([par_beta],N_MCMC+1)
par_zeta_vec = [par_zeta for i=1:N_MCMC+1]
par_lambda_1_vec = [par_lambda_1 for i=1:N_MCMC+1]
par_lambda_2_vec = [par_lambda_2 for i=1:N_MCMC+1]

#initializing everything in the beginning
res_coarse=chol_mat_and_mu(coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
res_fine=chol_mat_and_mu(coord_fine,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_fine=cov_mat_coarse_fine_fun(coord_fine,coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_inv=inv(res_coarse["cov_mat"])
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_fun(res_fine["cov_mat"],cov_mat_coarse_fine,cov_mat_coarse_inv)


for trial in 1:N_MCMC

#conditional simulation to get threshold exceedances
log_observation=exceed_cond_sim(20,num_obs,normalized_observation_data,observation_data[:,N_coarse+1],threshold, alpha, r_cond_gaussian,
res_fine["mu"],res_coarse["mu"], cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse )
#log_observation=log.(normalized_observation_data)
#    println(log_observation[1,1])
#log likelihood of log gaussian density of observations 
l_1=l_1_fun(d_gaussian,res_coarse["mu"],cov_mat_coarse_inv,log_observation)
#reciprocal mean estimation
# one estimate of c is: mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha)
#therefore we N_est_c many of them and take there mean, in addition we multiply with the number of threshold exccedances
l_2=l_2_fun(size(log_observation,1),r_gaussian,res_fine["mu"], res_fine["chol_mat"],alpha,N_est_c )
#size(log_observation,1) * log(mean([mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha) for i in 1: N_est_c]))  
#conditional mean estimator for intergal part
l_3=l_3_fun(N_est_cond,alpha,log_observation,r_cond_gaussian,res_fine["mu"],res_coarse["mu"],cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse)

#res_cond_int=zeros(N_est_cond,size(log_observation,1))
#for row in 1:N_est_cond
#    res_cond_int[row,:]=[(mean(
#                        exp.(
#                        r_cond_gaussian(res_fine["mu"],res_coarse["mu"],cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse ,log_observation[i,:])
#                        ))^alpha) for i in 1:size(log_observation,1) ]
#end
#l_3= sum(log.(mean( res_cond_int, dims=1)))
log_likelihood_old=sum([l_1,l_2,l_3])
# println("Old likelihood: $log_likelihood_old, l_1=$l_1, l_2=$l_2, l_3=$l_3")
#     #now all old values are safed and then the new ones are sampled

res_coarse_old=res_coarse
res_fine_old=res_fine
cov_mat_coarse_fine_old=cov_mat_coarse_fine
cov_mat_coarse_inv_old=cov_mat_coarse_inv
chol_mat_fine_cond_coarse_old=chol_mat_fine_cond_coarse




#update par_lambda_1
##########################################################
##########################################################
par_lambda_1_old=par_lambda_1
par_lambda_1=rand(Normal(par_lambda_1_old,0.1))

#update vario
#vario(x,c=par_c,beta=par_beta)=c*sqrt(sum(x.^2)).^beta

#calculate cov_mat and co for new par_c
res_coarse=chol_mat_and_mu(coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
res_fine=chol_mat_and_mu(coord_fine,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_fine=cov_mat_coarse_fine_fun(coord_fine,coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_inv=inv(res_coarse["cov_mat"])
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_fun(res_fine["cov_mat"],cov_mat_coarse_fine,cov_mat_coarse_inv)
#calculate loglikelihood for new param  

#log likelihood of log gaussian density of observations 
l_1=l_1_fun(d_gaussian,res_coarse["mu"],cov_mat_coarse_inv,log_observation)
#sum([log(d_gaussian(res_coarse["mu"],cov_mat_coarse_inv,log_observation[i,:])) for i in 1:size(log_observation,1)])
#reciprocal mean estimation
# one estimate of c is: mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha)
#therefore we N_est_c many of them and take there mean, in addition we do this once for every threshold exccedance observation and sum the logs up
l_2=l_2_fun(size(log_observation,1),r_gaussian,res_fine["mu"], res_fine["chol_mat"],alpha,N_est_c )

#l_2=size(log_observation,1) * log(mean([mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha) for i in 1: N_est_c]))  
#conditional mean estimator for intergal part
l_3=l_3_fun(N_est_cond,alpha,log_observation,r_cond_gaussian,res_fine["mu"],res_coarse["mu"],cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse)
log_likelihood_new=sum([l_1,l_2,l_3])
#println("New likelihood: $log_likelihood_new, l_1=$l_1, l_2=$l_2, l_3=$l_3")    

if log_likelihood_new<log_likelihood_old
a=exp(log_likelihood_new-log_likelihood_old)
else
a=1
end
#println("Rate: $a")
u=rand()
#println("random var $u")
if (u<a)
#println("accepted")
log_likelihood_old=log_likelihood_new
res_coarse_old=res_coarse
res_fine_old=res_fine
cov_mat_coarse_fine_old=cov_mat_coarse_fine
cov_mat_coarse_inv_old=cov_mat_coarse_inv
chol_mat_fine_cond_coarse_old=chol_mat_fine_cond_coarse
else
par_lambda_1=par_lambda_1_old
res_coarse=res_coarse_old
res_fine=res_fine_old
cov_mat_coarse_fine=cov_mat_coarse_fine_old
cov_mat_coarse_inv=cov_mat_coarse_inv_old
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_old
#println("not accepted")
end
#println("safed c is $par_c")
par_lambda_1_vec[trial+1]=par_lambda_1   


#update par_lambda_1
##########################################################
##########################################################
par_lambda_2_old=par_lambda_2
par_lambda_2=rand(Normal(par_lambda_2_old,0.1))

#update vario
#vario(x,c=par_c,beta=par_beta)=c*sqrt(sum(x.^2)).^beta

#calculate cov_mat and co for new par_c
res_coarse=chol_mat_and_mu(coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
res_fine=chol_mat_and_mu(coord_fine,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_fine=cov_mat_coarse_fine_fun(coord_fine,coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_inv=inv(res_coarse["cov_mat"])
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_fun(res_fine["cov_mat"],cov_mat_coarse_fine,cov_mat_coarse_inv)
#calculate loglikelihood for new param  

#log likelihood of log gaussian density of observations 
l_1=l_1_fun(d_gaussian,res_coarse["mu"],cov_mat_coarse_inv,log_observation)
#sum([log(d_gaussian(res_coarse["mu"],cov_mat_coarse_inv,log_observation[i,:])) for i in 1:size(log_observation,1)])
#reciprocal mean estimation
# one estimate of c is: mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha)
#therefore we N_est_c many of them and take there mean, in addition we do this once for every threshold exccedance observation and sum the logs up
l_2=l_2_fun(size(log_observation,1),r_gaussian,res_fine["mu"], res_fine["chol_mat"],alpha,N_est_c )

#l_2=size(log_observation,1) * log(mean([mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha) for i in 1: N_est_c]))  
#conditional mean estimator for intergal part
l_3=l_3_fun(N_est_cond,alpha,log_observation,r_cond_gaussian,res_fine["mu"],res_coarse["mu"],cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse)
log_likelihood_new=sum([l_1,l_2,l_3])
#println("New likelihood: $log_likelihood_new, l_1=$l_1, l_2=$l_2, l_3=$l_3")    

if log_likelihood_new<log_likelihood_old
a=exp(log_likelihood_new-log_likelihood_old)
else
a=1
end
#println("Rate: $a")
u=rand()
#println("random var $u")
if (u<a)
#println("accepted")
log_likelihood_old=log_likelihood_new
res_coarse_old=res_coarse
res_fine_old=res_fine
cov_mat_coarse_fine_old=cov_mat_coarse_fine
cov_mat_coarse_inv_old=cov_mat_coarse_inv
chol_mat_fine_cond_coarse_old=chol_mat_fine_cond_coarse
else
par_lambda_2=par_lambda_2_old
res_coarse=res_coarse_old
res_fine=res_fine_old
cov_mat_coarse_fine=cov_mat_coarse_fine_old
cov_mat_coarse_inv=cov_mat_coarse_inv_old
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_old
#println("not accepted")
end
#println("safed c is $par_c")
par_lambda_2_vec[trial+1]=par_lambda_2   




#     #same thing for par_beta
##############################################################
##############################################################

par_beta_old=par_beta
eps_beta=0.1
if (par_beta>eps_beta && par_beta<2.0-eps_beta)
par_beta=rand(Uniform(par_beta-eps_beta,par_beta+eps_beta))
elseif (par_beta<eps_beta)
par_beta=rand(Uniform(0.0,2*eps_beta))
else
par_beta=rand(Uniform(2.0-2*eps_beta,2.0))
end

#update vario
#vario(x,c=par_c,beta=par_beta)=c*sqrt(sum(x.^2)).^beta

#calculate cov_mat and co for new par_c
res_coarse=chol_mat_and_mu(coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
res_fine=chol_mat_and_mu(coord_fine,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_fine=cov_mat_coarse_fine_fun(coord_fine,coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_inv=inv(res_coarse["cov_mat"])
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_fun(res_fine["cov_mat"],cov_mat_coarse_fine,cov_mat_coarse_inv)
#calculate loglikelihood for new param  

#log likelihood of log gaussian density of observations 
l_1=l_1_fun(d_gaussian,res_coarse["mu"],cov_mat_coarse_inv,log_observation)
#sum([log(d_gaussian(res_coarse["mu"],cov_mat_coarse_inv,log_observation[i,:])) for i in 1:size(log_observation,1)])
#reciprocal mean estimation
# one estimate of c is: mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha)
#therefore we N_est_c many of them and take there mean, in addition we do this once for every threshold exccedance observation and sum the logs up
l_2=l_2_fun(size(log_observation,1),r_gaussian,res_fine["mu"], res_fine["chol_mat"],alpha,N_est_c )
# l_2=size(log_observation,1) * log(mean([mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha) for i in 1: N_est_c]))  
#conditional mean estimator for intergal part
l_3=l_3_fun(N_est_cond,alpha,log_observation,r_cond_gaussian,res_fine["mu"],res_coarse["mu"],cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse)
log_likelihood_new=sum([l_1,l_2,l_3])

if log_likelihood_new<log_likelihood_old
a=exp(log_likelihood_new-log_likelihood_old)
else
a=1
end
if (rand()<a)
log_likelihood_old=log_likelihood_new
res_coarse_old=res_coarse
res_fine_old=res_fine
cov_mat_coarse_fine_old=cov_mat_coarse_fine
cov_mat_coarse_inv_old=cov_mat_coarse_inv
chol_mat_fine_cond_coarse_old=chol_mat_fine_cond_coarse
else
par_beta=par_beta_old
res_coarse=res_coarse_old
res_fine=res_fine_old
cov_mat_coarse_fine=cov_mat_coarse_fine_old
cov_mat_coarse_inv=cov_mat_coarse_inv_old
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_old

end
par_beta_vec[trial+1]=par_beta

#same with zeta
#########################################################
#########################################################
#     #same thing for par_beta
#     #update par_beta

par_zeta_old=par_zeta
eps_zeta=0.05
if (par_zeta>-pi/4+eps_zeta && par_zeta<pi/4-eps_zeta)
par_zeta=rand(Uniform(par_zeta-eps_zeta,par_zeta+eps_zeta))
elseif (par_zeta<-pi/4+eps_zeta)
par_zeta=rand(Uniform(-pi/4,-pi/4+2*eps_zeta))
else
par_zeta=rand(Uniform(pi/4-2*eps_zeta,pi/4))
end

#update vario
#vario(x,c=par_c,beta=par_beta)=c*sqrt(sum(x.^2)).^beta

#calculate cov_mat and co for new par_c
res_coarse=chol_mat_and_mu(coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
res_fine=chol_mat_and_mu(coord_fine,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_fine=cov_mat_coarse_fine_fun(coord_fine,coord_coarse,x->vario(x,par_beta,par_zeta,par_lambda_1,par_lambda_2))
cov_mat_coarse_inv=inv(res_coarse["cov_mat"])
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_fun(res_fine["cov_mat"],cov_mat_coarse_fine,cov_mat_coarse_inv)
#calculate loglikelihood for new param  

#log likelihood of log gaussian density of observations 
l_1=l_1_fun(d_gaussian,res_coarse["mu"],cov_mat_coarse_inv,log_observation)
#sum([log(d_gaussian(res_coarse["mu"],cov_mat_coarse_inv,log_observation[i,:])) for i in 1:size(log_observation,1)])
#reciprocal mean estimation
# one estimate of c is: mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha)
#therefore we N_est_c many of them and take there mean, in addition we do this once for every threshold exccedance observation and sum the logs up
l_2=l_2_fun(size(log_observation,1),r_gaussian,res_fine["mu"], res_fine["chol_mat"],alpha,N_est_c )
# l_2=size(log_observation,1) * log(mean([mean(exp.(r_gaussian(res_fine["mu"],res_fine["chol_mat"])))^(-alpha) for i in 1: N_est_c]))  
#conditional mean estimator for intergal part
l_3=l_3_fun(N_est_cond,alpha,log_observation,r_cond_gaussian,res_fine["mu"],res_coarse["mu"],cov_mat_coarse_fine,cov_mat_coarse_inv, chol_mat_fine_cond_coarse)
log_likelihood_new=sum([l_1,l_2,l_3])

if log_likelihood_new<log_likelihood_old
a=exp(log_likelihood_new-log_likelihood_old)
else
a=1
end
if (rand()<a)
log_likelihood_old=log_likelihood_new
res_coarse_old=res_coarse
res_fine_old=res_fine
cov_mat_coarse_fine_old=cov_mat_coarse_fine
cov_mat_coarse_inv_old=cov_mat_coarse_inv
chol_mat_fine_cond_coarse_old=chol_mat_fine_cond_coarse
else
par_zeta=par_zeta_old
res_coarse=res_coarse_old
res_fine=res_fine_old
cov_mat_coarse_fine=cov_mat_coarse_fine_old
cov_mat_coarse_inv=cov_mat_coarse_inv_old
chol_mat_fine_cond_coarse=chol_mat_fine_cond_coarse_old

end
par_zeta_vec[trial+1]=par_zeta

if trial%n_trial_print==0
println(trial)
end
end
Dict( "beta" => par_beta_vec, "zeta" => par_zeta_vec , "lambda_1" => par_lambda_1_vec, "lambda_2" => par_lambda_2_vec)
end
