import numpy as np

def Volume2OpticalPpt(VolArr):
    mass_of_ink_dil = VolArr[:,1]
    mass_of_pure_ink = mass_of_ink_dil*0.1
    mass_of_H20_in_ink_dil = mass_of_ink_dil*0.9
    volume_of_scatterer = VolArr[:,0]*0.227
    volume_of_H20_in_intralipid = VolArr[:,0]*(1-0.227)
    mass_of_scatterer = volume_of_scatterer*0.988
    mass_of_H20_in_intralipid_sol = volume_of_H20_in_intralipid
    mass_of_intralipid_sol = mass_of_scatterer + mass_of_H20_in_intralipid_sol
    conc_pure_ink = np.true_divide(mass_of_pure_ink,(mass_of_ink_dil+mass_of_intralipid_sol+1900))
    conc_scatter = np.true_divide(mass_of_scatterer,(mass_of_ink_dil+mass_of_intralipid_sol+1900))
    conc_H20 = np.true_divide((1900+mass_of_H20_in_ink_dil+mass_of_H20_in_intralipid_sol),(mass_of_ink_dil+mass_of_intralipid_sol+1900))
    mew_a_ink_690nm = (conc_pure_ink*143.978591)+0.002902422
    mew_a_H20_690nm = conc_H20*0.0005285160511
    mew_s_prime_690nm = conc_scatter*np.true_divide((-76.7+np.multiply(171000,np.power(690,-0.957))),10)
    ttl_msp_690 =  mew_s_prime_690nm
    ttl_mew_a_690 = mew_a_ink_690nm + mew_a_H20_690nm

    # Consrtuct an array to store the MewSPrime and MewA values 
    rtrnArr = np.zeros((VolArr.shape[0],2))
    rtrnArr[:,0] = ttl_msp_690[:]
    rtrnArr[:,1] = ttl_mew_a_690[:]
    return rtrnArr