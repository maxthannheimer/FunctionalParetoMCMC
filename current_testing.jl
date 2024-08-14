include("functions.jl")

#inputs
gridsize=5 #lenght of fine grid
N_fine=gridsize^2 #number of fine grid points
N_coarse=5 #number of weather stations/ conditioning points, obersavation points
num_sim=200 #number of simulated realizations

#true params for simulation
alpha_true = 1.0
beta_true=1.0
c_true=1.0
param=[ c_true , beta_true]
alpha=1.0

#Threshold definition as quantile
#p=0.0
#threshold= (1-p)^(-1/alpha)
threshold=0

#MCMC params
N_MCMC=20
param_start=[2.0,0.1]
N_est_c=1000
N_est_cond=5
N_burn_in=0

#create grids


#safe grid and observation points, also plots if last argument is true
#(coord_coarse, coord_fine, row_x0)=Create_Grid_and_Observation(gridsize,N_coarse, true)
(coord_coarse, coord_fine, row_x0)=Create_Grid_and_Observation_on_fine_grid(gridsize,N_coarse, true)

#gives the nearest fine gride coordinates for each coarse grid obsrvation 
#last argument is the coarse coordinates rounded to the nearest gridpoint coordinates
coord_cond_rows=get_common_rows_indices(coord_fine,floor.(coord_coarse.*gridsize)./gridsize)



#Simulate data on grid



#simulate data on all points and reduce it to observation data (  coarse observations)
#cholmat=chol_mat(vcat(coord_fine, coord_coarse), x->vario(x,param))
cholmat=chol_mat(coord_fine, x->vario(x,param))
#@time(sim_data= [simu_specfcts(vcat(coord_fine, coord_coarse), x->vario(x,param), cholmat, alpha_true)  for i in 1:num_sim])
@time(sim_data= [simu_specfcts(coord_fine, x->vario(x,param), cholmat, alpha_true)  for i in 1:num_sim])
sim_data=reduce(hcat,sim_data)' #just make vector of vectors a matrix (same below for observations)
#observation_data=reduce(hcat,[sim_data[i,N_fine+1:N_fine+N_coarse] for i in 1:num_sim])' #first argument is number of sim, second is coordinate
observation_data=reduce(hcat,[sim_data[i,coord_cond_rows] for i in 1:num_sim])' #first argument is number of sim, second is coordinate
observation_x0=reduce(hcat,[sim_data[i,row_x0] for i in 1:num_sim])'
#no normalization
#normalized_observation_data=reduce(hcat,[observation_data[i,1:N_coarse]./observation_x0[i,1] for i in 1:num_sim])'






# @time(vec(FBM_simu_fast(param,gridsize,10000)[1]'))
# @time(FBM_simu_fast_vec(param,gridsize,10000)[1])


# @time(
#     for i in 1:1000
#         r_log_gaussian(coord_fine,param,row_x0)
#     end)
# r_log_gaussian_vec(coord_fine,param,row_x0,1000)

# @time( for i in 1:10000
#         r_cond_log_gaussian(observation_data[1,:],observation_x0[1], coord_fine,coord_coarse,param,row_x0)
#         end) 
# #@time (r_cond_log_gaussian_vec(observation_data,observation_x0, coord_fine,coord_coarse,param,row_x0,100) )#coord_x0 (hier c egal)

# size(observation_data,1)



# @time(r_log_gaussian_vec(coord_fine,param,row_x0,1))
# #cov_mat_for_vectors(coord_fine, coord_coarse, param, coord_fine[row_x0,:])
# @time r_cond_log_gaussian(observation_data[1,:],observation_x0[1], coord_fine,coord_coarse,param,row_x0) #coord_x0 (hier c egal)

 #@time (modified_observation, modified_observation_x0) =exceed_cond_sim(10,num_sim,observation_data,observation_x0,threshold, alpha, coord_fine,coord_coarse,param,row_x0 )

 #@time exceed_cond_sim_with_all_info(10,num_sim,normalized_log_observation_data,observation_x0,threshold, alpha, coord_fine,coord_coarse,param,row_x0 )

# (modified_observation, modified_observation_x0) = exceed_cond_sim(10,num_sim,observation_data,observation_x0,threshold, alpha, coord_fine,coord_coarse,param,row_x0 )

#exceed_cond_sim_fast(10,num_sim,normalized_log_observation_data,observation_x0,threshold, alpha, coord_fine,coord_coarse,param,row_x0 )
                        

#N=size(coord_fine,1)
#uncond_rows=setdiff(1:gridsize*gridsize,coord_cond_rows)
#coord_cond=coord_fine[coord_cond_rows,:]
#coord_uncond=coord_fine[uncond_rows,:]
#cov_mat_coarse_inv_alt=inv(cov_mat_for_vectors(coord_cond,coord_cond, param,  coord_fine[row_x0,:])) 

#  @time l1=l_1_fun(coord_fine,coord_coarse,modified_observation,param, observation_x0, row_x0)

#  N_est_c=1000000
# @time l2= l_2_fun(coord_fine, param,row_x0, size(modified_observation,1),alpha,N_est_c)


#  N_est_cond=50

# @time l3=l_3_fun(coord_fine, coord_coarse, param, row_x0, modified_observation, observation_x0, alpha, N_est_cond)   

#  log_likelihood_old=sum([l1,l2,l3])















#some FBM testing
#Random.seed!(1234)
@time(FBM_simu_fast_vec(param, gridsize,1))
#Random.seed!(1234)
FBM_res=FBM_simu_fast_vec(param,gridsize,num_sim)
FBM_res[1][1]
r_log_gaussian(coord_fine,param,row_x0)
coord_fine






#empirical and true cov matrices to test FBM 
emp_cov_mat=zeros(N_fine,N_fine)
true_cov_mat=zeros(N_fine,N_fine)
mean_vec=zeros(N_fine)
for i in 1:N_fine
    mean_vec[i]=mean([FBM_res[rep][i] for rep in 1:num_sim])
end


for row in 1:N_fine
    for col in 1:N_fine
        emp_cov_mat[row,col]=1/(num_sim-1)*sum([(FBM_res[rep][row]-mean_vec[row])*(FBM_res[rep][col]-mean_vec[col]) for rep in 1:num_sim])
    end
end


for row in 1:N_fine
    for col in 1:N_fine
        true_cov_mat[row,col]=param[1]*(norm(coord_fine[row,:])^param[2]+norm(coord_fine[col,:])^param[2]-norm(coord_fine[row,:]-coord_fine[col,:])^param[2])
    end
end


maximum(
    abs.((emp_cov_mat-true_cov_mat)./(true_cov_mat.+10^(-8)))
    )