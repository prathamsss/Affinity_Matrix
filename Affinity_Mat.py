import numpy as np

def cal_affinity_mat(x,sigma_d=1e2,sigma_g=1e-2):
    # FOR 2D IMG
    
    # Reshape into Col Vect
    xre = x.reshape((-1,1)) #

    # Cal similarity based on values 
    dot_prod = np.dot(xre,xre.T)        #nxn
    sq_xre = (xre*xre)                  #(n,1)
    d = sq_xre - 2*dot_prod + sq_xre.T
    s1 = np.exp(-1.0*d/sigma_g)

    # Calculate similarity based on positions
    c= np.arange(sq_xre.shape[0])
    c = np.expand_dims(c, axis=-1)
    c_dot_prod = np.dot(c,c.T)
    c_sq= c*c
    d_c = c_sq - 2*c_dot_prod + c_sq.T
    s2 = np.exp(-1.0*d_c/sigma_d)

    ans = s1*s2
    return ans



x= np.array([
[2,3,4],
[5,6,7]]
)

cal_affinity_mat(x,sigma_d=1e2,sigma_g=1e-2)
