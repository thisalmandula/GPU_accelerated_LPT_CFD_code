######################################################################
######  THIS IS A CUDA PYTHON CODE FOR MEREL KOOI'S BIOFOULED   ######
#######          MICROPLASTIC PARTICLE TRACKING MODEL          #######
########  https://pubs.acs.org/doi/10.1021/acs.est.6b04702 ###########
######################################################################

import cupy as cp
from numba import cuda
import math
import time
import matplotlib.pyplot as plt

# Device kernel
@cuda.jit
def main_kernel(xfinal, d_pl, rho_pl1, itmax, deltat, N, tt): 
    k = cuda.grid(1)  # Global thread index
    if k < N:  # Check thread is within bounds
    ######################################################################
    #############  INITIALIZE DIMENSIONLESS PARAMETERS   #################
    ######################################################################

        r_pl = d_pl[k]/2        # calculate particle radius
        z0=10                   # characteristic space scale
        rho0=1000               # characteristic scale of all densities (water and plastic)
        t=0                     # initial time
        r0=r_pl                 # c. s. of all quantities related to plastic size
        nalg = 0                # Number of algae
        z = 0                   # Plastic particle depth

    ######################################################################
    ################  DEFINE CHARACTERISTIC SCALES   #####################
    ###################################################################### 

        rho0=1000                   #c. s. of all densities (water and plastic)
        z0=10                       #characteristic scale of depth
        t0=1*24*3600                #characteristic scale of time
        mu0=1e-3                    #characteristic scale of dynamic viscosity
        nu0=1e-6                    #characteristic scale of kinematic viscosity
        T0=20+273                   #characteristic scale of kelvin temperature
        na_amb0=1e-7                #c. s. of the inverse of ambient algae concentration
        dz=0                        #initialize dz
        dnalg=0                     #initialize dnalg
    
    #############  INITIALIZE DIMENSIONLESS PARAMETERS   #################      

        r_pl=r_pl/r0                #dimensioneless plastic radius
        rho_pl=rho_pl1[k]/rho0          #dimensionless plastic density

    ######################################################################
    ######################  DEFINE PARAMETERS   ##########################
    ######################################################################

        bc = 1.38064852e-23         # Boltzman constant

    ######################  LIGHT PARAMETERS   ###########################

        light_Im = 1.2e+8           # Surface light intensity at noon
        ext_water = 0.2             # Extinction coefficient water
        ext_algae = 0.02            # Extinction coefficient algae
        
    ###################  ALGAE GROWTH PARAMETERS   #######################

        mu_max = 1.85               # Maximum growth rate
        alpha = 0.12                # Initial slope in growth equation
        mort_alg = 0.39             # Mortality rate
        resp_alg = 0.1              # Respiration rate
        Q_10 = 2                    # Temperature coefficient respiration             
        temp_min = 0.2              # Minimum temperature algae growth
        temp_opt = 26.7             # Optimal temperature algae growth
        temp_max = 33.3             # Maximum temperature algae growth
        light_Iopt = 1.75392e+13    # Optimal light intensity algae growth
        
    ######################  ALGAE PROPERTIES   ###########################
        
        C_cell = 2726e-9            # mg carbon for a single cell
        vol_alg = 2.0e-16/r0**(3)   # dimensionless volume individual algae
        rho_bf = 1388.0/rho0        # dimensionless biofilm density            
        shear = 1.7e+5              # shear used in encounter rate

    ##############  CONVERSION FROM DAY^-1 TO SEC^-1   ###################

        sec_per_day = 86400                 # seconds per day
        light_Im=light_Im/sec_per_day       # light intensity per second
        mu_max=mu_max/sec_per_day           # max growth rate per second
        alpha=alpha/sec_per_day             # initial slope in growth eq. per second
        mort_alg=mort_alg/sec_per_day       # mortality rate per second
        resp_alg=resp_alg/sec_per_day       # respiration rate per second
        light_Iopt=light_Iopt/sec_per_day   # opt. lig. int. al. grw. per second
        shear=shear/sec_per_day             # shear in encounter rate per second

    ###########  CONVERSION TO DIMENSIONLESS PARAMETERS   ################

        light_Im=light_Im*t0        # light intensity dimensionalized
        mu_max=mu_max*t0            # max growth rate dimensionalized
        alpha=alpha*t0              # initial slope in growth eq. dimensionalized
        mort_alg=mort_alg*t0        # mortality rate dimensionalized
        resp_alg=resp_alg*t0        # respiration rate dimensionalized
        light_Iopt=light_Iopt*t0    # opt. lig. int. al. grw. dimensionalized
        shear=shear*t0              # shear in encounter rate dimensionalized
        
        for j in range(0,int(itmax)):                             # loop over all timesteps
    #############  CALCULATE CHLOROPHYLL OF THE NORTH PACIFIC OCEAN  ################

            z=z*z0                  # Convert z to dimensional for profiles

            chlazbase = 0.151
            Cb = 0.533
            s = 1.72e-3
            Cmax = 1.194
            zmax = 92.01
            Dz = 43.46

            chla = Cb-(s*(-z))+Cmax*math.exp(-((-z-zmax)/Dz)**2)
            chla = chla*chlazbase
            
    #############  CALCULATE TEMPERATURE OF THE NORTH PACIFIC OCEAN  ################

            p = 2
            zc = -300
            Tsurf = 25
            Tbot = 1.5

            temp=Tsurf+(Tbot-Tsurf)*(z**p)/(z**p+zc**p)

    ##############  CALCULATE SALINITY OF THE NORTH PACIFIC OCEAN  ##################

            zfix=-1000
            Sfix=34.6
            b1=9.9979979767e-17
            b2=1.0536246487e-12
            b3=3.9968286066e-09
            b4=6.5411526250e-06
            b5=4.1954014008e-03
            b6=3.5172984035e+01

            if z>zfix:
                salt=b1*z**5+b2*z**4+b3*z**3+b4*z**2+b5*z+b6
            else: 
                salt=Sfix

            z=z/z0                  # Convert z back to dimensionaless
            
    ######################################################################
    ###############  CALCULATE DENSITY AND VISCOSITY   ###################
    ################  FROM EQ. OF STATE (SEA WATER)   ####################
    ######  ATTENTION: convert salinity from g/kg (PSU) to kg/kg  ########
    ######################################################################

            salt=salt*1.0e-3
            a1=9.999e+2
            a2=2.034e-2
            a3=-6.162e-3
            a4=2.261e-5
            a5=-4.657e-8
            b1=8.020e+2
            b2=-2.001
            b3=1.677e-2
            b4=2.261e-5
            b5=-4.657e-5

    #####################  DENSITY CALCULATION   #########################

            density=a1+a2*temp+a3*(temp**2)+a4*(temp**3)+a5*(temp**4)+b1*salt+\
                    b2*salt*temp+b3*salt*(temp**2)+b4*salt*(temp**3)+b5*(salt**2)*(temp**2)
                    
            density=density/rho0        # dimensionless sea water density

    ####################  VISCOSITY CALCULATION   ########################

            viscosity=0.156*(temp+64.993)**2
            viscosity=4.2844e-5+1.0/(viscosity-91.296)

    ################  DYNAMIC VISCOSITY CALCULATION   ####################

            a1=1.541+1.998e-2*temp-9.52e-5*(temp**2)
            b1=7.974-7.561e-2*temp+4.724e-4*(temp**2)

            viscosity_dyn=viscosity*(1.0+a1*salt+b1*(salt**2))/mu0 #dimensionless 

    ###################  SURFACE LIGHT INTENSITY   #######################    
            
            # Light intensity at the surface 
            hour=math.fmod((t*t0)/3600,24)      
            # light_I0=light_Im*math.sin((hour-8)*math.pi/8.0)       #8 hours day : 8h-16h
            # light_I0=light_Im*math.sin((hour-6)*2.0*math.pi/24.0) #12 hours day : 6h-18h
            light_I0=light_Im*math.sin((hour-4)*math.pi/16.0)      #16 hours day : 4h-20h
            
    #####################  BOUNDARY CONDITIONS   #########################
                
            if light_I0<0: 
                light_I0 = 0


    ################# LIGHT INTENSITY AT FLOOR DEPTH #####################
        
            light_Iz=light_I0*math.exp((ext_water+ext_algae*chla)*z*z0)

    #################  GROWTH RATE UNDER OPT. TEMP.  #####################

            mu_opt=mu_max*light_Iz/(light_Iz+t0*(mu_max/alpha)*(light_Iz/light_Iopt-1)**2)
            
    ################# TEMP. INFLUENCE ON GROWTH RATE #####################

            temp_inf=((temp-temp_max)*(temp-temp_min)**2)/((temp_opt-temp_min)*\
                    ((temp_opt-temp_min)*(temp-temp_opt)-(temp_opt-temp_max)*\
                    (temp_opt+temp_min-2*temp)))

    ######################  ALGAE GROWTH RATE  ###########################

            mu_tot=mu_opt*temp_inf

    ##############    CONVERSION FACOR: CHLR TO CARBON    ################

            chla_C=0.003+0.0154*math.exp(0.05*temp)*\
                math.exp(-0.059*light_Iz/t0*1.0e-6*sec_per_day)*mu_tot/mu_max

    ################    AMBIENT ALGAE CONCENTRATION    ###################

            na_amb=na_amb0*(chla/chla_C)/C_cell 

    #####################  BOUNDARY CONDITIONS   #########################
        
            if light_Im*math.exp((ext_water+ext_algae*chla)*z*z0)<=0.01*light_Im:
                na_amb=0

    ####################  PARTICLE TOTAL RADIUS   ######################## 

            teta_pl=4*math.pi*(r_pl**2)
            v_pl=math.pi*(r_pl**3)*4/3
            v_bf=vol_alg*nalg*teta_pl  
            v_tot=v_bf+v_pl
            
            t_bf=(v_tot*3/(4*math.pi)) ** (1/3)

            if(t_bf<0.0): 
                t_bf = 0.0

            r_tot=(v_tot*3/(4*math.pi)) ** (1/3)
            r_alg=(vol_alg*3/(4*math.pi)) ** (1/3)
            
            rho_tot=(v_pl*rho_pl+v_bf*rho_bf)/v_tot

    ###############  DIMENSIONLESS PARTICLE DIAMETER   ###################

            viscosity_kin=viscosity_dyn/density                         # dimensionless kinematic viscosity        

            d_ast=(rho_tot-density)*9.81*r0**3/nu0**2*(2.0*r_tot)**3    # dimensionless particle diameter

            d_ast=math.fabs(d_ast/(density*viscosity_kin**2))

    ######################  SINKING VELOCITY   ###########################
            
            if d_ast<5.0e-2: 
                om_ast=(d_ast**2)/5832                                  # dimensionless particle velocity
            else:
                log_d=math.log10(d_ast)
                om_ast=10**(-3.76715+1.92944*log_d-0.09815*(log_d**2)-0.00575*(log_d**3)+0.00056*(log_d**4))

            w_sink=9.81*nu0*om_ast*viscosity_kin*(rho_tot-density)/density
            
            if w_sink < 0:
                w_sink=((-w_sink) ** (1/3)) *t0/z0  # Handle negative values
            else:
                w_sink=-(w_sink ** (1/3))*t0/z0   # Standard cube root for positive values

    #####################  BOUNDARY CONDITIONS   #########################

            if z*z0>=0 and w_sink>0:                                # Boundary condition at sea surface
                w_sink=0
                
            if z*z0<=-4000 and w_sink<0:                            # Boundary condition at ocean floor
                w_sink=0
            
            dz=w_sink  

            temp=(temp+273.16)/T0                                       # dimensionless kelvin temperature

    ####################  ENCOUNTER KERNEL RATE   ########################

            beta_alg_brownian=2/3*(temp/viscosity_dyn)\
                *(1/r_tot+1/r_alg)*(r_tot+r_alg);                       # dimensionless beta_alg_brownian

            beta_alg_settling=0.5*math.pi*r_tot**2*math.fabs(w_sink)         # dimensionless beta_alg_settling

            beta_alg_shear=4/3*shear*(r_tot+r_alg)**3                  # dimensionless beta_shear (Laminar)

    ##############  R.H.S. VALUES FOR ALGAE GROWTH EQU.   ###############
        
            collision=(beta_alg_brownian*(bc*T0*t0)/mu0+beta_alg_settling*z0*(r0**2)\
                    +beta_alg_shear*(r0**3))*na_amb/(na_amb0*teta_pl) # dimensionless collision term

            growth=mu_tot*nalg                                          # dimensionless growth term

            mortality=mort_alg*nalg                                     # dimensionless mortality term

            respiration=resp_alg*nalg*Q_10**((temp*T0-20-273.16)*0.1)   # dimensionless respiration term

            dnalg=collision+growth-mortality-respiration                # dimensionless dA/dt

            nalg = nalg + dnalg*deltat          # Find new algae value
            z = z + dz*deltat                   # Find new z value

    #######################  Boundary Conditions   ######################

            if nalg<0:                         # nalg<0 is not physical
                nalg=0
            
            if z>0:                            # Boundary condition at ocean surface
                z=0                            # Places particle on surface if it goes abovw

    ###########  SAVE TO MEMORY FOR TRANSFER BACK TO HOST   #############    

            t=t+deltat                         # move forward in time
            xfinal[j,k] = z                    # collect z values from each time step

            j=j+1                              # need to define matrix element to store time data
            if k == 1:
                tt[j] = t                          # collect time values from each time step

            #print("z : ", z)

# Host code
def main():

    # Constants
    N = 64*64                   # Number of particles
    densities = [840, 920, 960] # Densities of plastic ( 840 is Polypropene, 920 is Low density polyethylene, 960 is High density polyethylene)
    sizes = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]        # Plastic particle sizes
    tlim = 100                    # Total time in days
    timestep=100                # Timestep in seconds
    #blocksize = 1024               # Threads per block 1D
    blocksize = (16,16)         #Threads per block 2D
    t0=1.0*24.0*3600.0 		                            # Characteristic time scale 
    tlim=tlim*24.*3600. 		                        # Conversion from days to seconds
    itmax = tlim/timestep                               # calculation of total timesteps required
    tlim = tlim/timestep                                # Calculate total time non-dimensional
    deltat=timestep/t0 		                            # Define non-dimensional timestep

    cp.random.seed(17)                                  # Random seed
    rho_pl1 = cp.zeros(N, dtype=cp.float64)
    d_pl = cp.zeros(N, dtype=cp.float64)
    xfinal = cp.zeros((int(itmax),N) , dtype=float)         # initialize matrix to collect results of xCCD
    tt = cp.zeros(int(itmax), dtype=cp.float64)
    
    # Generate particles with random size and density
    for j in range(N):                                  # Populate arrays
        
        r1 = cp.random.random()                         # Generate a random number between 0 and 1
        random_index = int(r1 * 3)                      # Map to indices 0, 1, or 2
        rho_pl1[j] = densities[random_index]

        r2 = cp.random.random()                         # Generate another random number between 0 and 1
        random_index = int(r2 * 8)                      # Map to indices 0 to 7
        d_pl[j] = sizes[random_index]

    rho_pl1_d = cuda.to_device(rho_pl1)                 # Allocate device memory for densities
    d_pl_d = cuda.to_device(d_pl)                       # Allocate device memory for sizes
    xfinal_d = cuda.to_device(xfinal)                  # Allocate device memory for time array
    tt_d = cuda.to_device(tt)                         # Allocate device memory for particle locations
    
    threads_per_block = blocksize                       # Grid dimensions
    #blocks_per_grid = math.ceil(N / threads_per_block)  # Block dimensions 1D
    blocks_per_grid = (math.ceil(cp.sqrt(N) / threads_per_block[0]),math.ceil(cp.sqrt(N) / threads_per_block[1]))  # Block dimensions 2D

    start_time = time.time()                            # Launch kernel
    main_kernel[blocks_per_grid, threads_per_block](xfinal_d, d_pl_d, rho_pl1_d, itmax, deltat, N, tt_d)  # Call the kernel 

    cuda.synchronize()
    end_time = time.time()
    print("Simulation completed.")
    print("The time of execution of biofouling turbulence code is :", (end_time-start_time) , "s")  
    
    xfinal = xfinal_d.copy_to_host()                    # Copy results back to host
    tt = tt_d.copy_to_host()                    # Copy results back to host
    
    end_time = time.time()
    # Output results

    fig = plt.figure()
    ax = plt.axes()       
            
    plt.figure(1)        
    plt.plot(tt, xfinal[:,0], c=(0.25, 0.25, 1.00), lw=0.3) 

    ax.set_title('Trajectory', fontsize=10)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Depth (m)")
        
    plt.show()   
            
    #plt1_format = 'svg' # e.g .png, .svg, etc.
    #plt1_name = 'particle_trajectory.svg'

    #fig.savefig(plt1_name, format=plt1_format, dpi=1200) 

if __name__ == "__main__":
    main()
