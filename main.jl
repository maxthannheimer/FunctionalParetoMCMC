include("functions.jl")

#create grids

x_gridsize=5
y_gridsize=5
N_fine=x_gridsize*y_gridsize
N_coarse=8
#safe grid and observation points, alsp plots if last argument is true
(coord_coarse, coord_fine, coord_x0)=Create_Grid_and_Observation(x_gridsize,y_gridsize,N_coarse, true)
coord_all=vcat(coord_fine, coord_coarse, coord_x0)



#Simulate data on grid
#number of simulated realizations
num_sim=1000

#true params for simulation
beta_true=1.0
zeta_true= pi/4
lambda_1_true=log(0.5)
lambda_2_true=log(1)

#simulate data on all points and reduce it to observation data (  coarse observations)
cholmat=chol_mat(coord_all, x->vario(x,beta_true,zeta_true,lambda_1_true,lambda_2_true))
@time(sim_data= [simu_specfcts(coord_all, x->vario(x,beta_true,zeta_true,lambda_1_true,lambda_2_true), cholmat)  for i in 1:num_sim])
sim_data=reduce(hcat,sim_data)' #just make vector of vectors a matrix (same below for observations)
observation_data=reduce(hcat,[sim_data[i,N_fine+1:N_fine+N_coarse+1] for i in 1:num_sim])' #first argument is number of sim, second is coordinate

###################
#MCMC simulation
#starting params for mcmc simulation
alpha=1.0
par_beta=1.0#par_beta \in (0,2)
par_zeta=pi/4 #not zero!
par_lambda_1=log(0.5)
par_lambda_2=log(1)

#Number of random draws for estimation of c_l
N_est_c=1000
#Number of random draws for estimation of conditional integral
N_est_cond=500
#threshold as p quantile of alpha pareto distrib
p=0.8
threshold= (1-p)^(-1/alpha)
threshold= 0.0

### number of MCMC steps
N_MCMC=100
#every n_trial_print trial is printed to check for progress
n_trial_print=10
#timed MCMC function call
@time result=MCMC(N_MCMC,N_coarse, coord_coarse, N_fine,coord_fine, N_est_c, N_est_cond,alpha, par_beta,
                par_zeta,par_lambda_1,par_lambda_2, 
                observation_data,vario,chol_mat_and_mu, cov_mat_coarse_fine_fun, chol_mat_fine_cond_coarse_fun, 
                r_cond_gaussian, r_gaussian, d_gaussian,l_1_fun,l_2_fun,l_3_fun,n_trial_print)



####################################################
#results
#burn in
N_burn_in=20
#mean of params 
[mean(result["beta"]),mean(result["zeta"]),mean(result["lambda_1"]),mean(result["lambda_2"])]
#mean of params with burn in period canceled out
[mean(result["beta"][N_burn_in+1:N_MCMC+1]),mean(result["zeta"][N_burn_in+1:N_MCMC+1]),mean(result["lambda_1"][N_burn_in+1:N_MCMC+1]),mean(result["lambda_2"][N_burn_in+1:N_MCMC+1])]

#plot and hist for beta,zeta,lambda_1,lambda_2
pl.histogram(result["beta"][N_burn_in+1:N_MCMC+1],title="histogram for beta")
pl.scatter(1:N_MCMC+1,result["beta"],title="chain for beta")

pl.histogram(result["zeta"][N_burn_in+1:N_MCMC+1],title="histogram for zeta")
pl.scatter(1:N_MCMC+1,result["zeta"],title="chain for zeta")


pl.histogram(result["lambda_1"][N_burn_in+1:N_MCMC+1],title="histogram for lambda_1")
pl.scatter(1:N_MCMC+1,result["lambda_1"],title="chain for lambda_1")


pl.histogram(result["lambda_2"][N_burn_in+1:N_MCMC+1],title="histogram for lambda_2")
pl.scatter(1:N_MCMC+1,result["lambda_2"],title="chain for lambda_2")