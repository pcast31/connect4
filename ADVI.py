import autograd.numpy as np
from autograd import elementwise_grad, jacobian
import time


class ADVI():
    def __init__(self,inv_T,model,X, dependant=True):
        self.dependant = dependant
        self.inv_T = inv_T
        self.model = model
        self.X = X
        self.grad_log_distribution = elementwise_grad(self.distr_log)

        self.grad_inv_T = elementwise_grad(self.inv_T)

        self.log_jac_inv_T = self.log_jac
        self.grad_logjac_inv_T = elementwise_grad(self.log_jac_inv_T)
        self.theta_size = model.theta_size
    def log_jac(self,z):
        return np.log(np.abs(np.prod(elementwise_grad(self.inv_T)(z))))
        #return np.log(np.abs(np.linalg.det(jacobian(self.inv_T)(z))))

    def distr_log(self,theta):
        return self.model.log_distr(self.X,theta)

    def nu_to_zeta(self,nu_d):
        if self.dependant:
            return self.mu[-1]+self.omega[-1]@nu_d
        else:
            return self.mu[-1]+np.exp(self.omega[-1])*nu_d
            
    def zeta_to_theta(self,zeta):
        return self.inv_T(zeta)

    def compute_gradient_muomega(self,nu):
        self.grad_mu = np.zeros(self.mu[-1].shape)
        self.grad_omega = np.zeros(self.omega[-1].shape)
        self.ELBO = 0
        for nu_i in nu:
                      
            zeta_i = self.nu_to_zeta(nu_i)
            theta_i = self.zeta_to_theta(zeta_i)
            log_dis = self.grad_log_distribution(theta_i)
            grad_inv = self.grad_inv_T(zeta_i)
            grad_jac = self.grad_logjac_inv_T(zeta_i)

            self.ELBO = self.ELBO + self.distr_log(theta_i)+ self.log_jac_inv_T(zeta_i)
            self.grad_mu = self.grad_mu + log_dis*grad_inv+grad_jac
            if self.dependant:
                self.grad_omega = self.grad_omega+np.outer((log_dis*grad_inv+grad_jac),nu_i)
            else:
                self.grad_omega = self.grad_omega+(log_dis*grad_inv+grad_jac)*nu_i*np.exp(self.omega[-1])
        if self.dependant:
            meangrad_omega = self.grad_omega/len(nu) + np.linalg.inv(self.omega[-1]).T
            det_log = 0.5*np.log(np.linalg.det(self.omega[-1]@self.omega[-1].T))
        else: 
            det_log = np.sum(self.omega[-1])
            meangrad_omega = self.grad_omega/len(nu) + 1
        
        return self.grad_mu/len(nu), meangrad_omega,self.ELBO/len(nu)+det_log # + Constant




    def compute_step_size(self,iteration,lr,grad,s,eps=1e-4,tau=1,alpha=0.1):
        #TO DO : Implement formula (10) and (11)
        s = alpha*grad**2+(1-alpha)*s
        return lr * iteration**(-0.5+eps)/(tau+s**0.5),s

    def fit(self,eps=1e-4,max_step=1000,M=10, lr=1e-2, tau=1):
        P = self.theta_size
        N = len(self.X)
        iteration=1
        self.mu = [np.zeros(P)]
        if self.dependant:
            self.omega = [np.eye(P)]
        else: 
            self.omega = [np.ones(P)]
        ELBO=0
        delta_ELBO=2*eps
        #while iteration==1 or (np.linalg.norm(self.mu[-1]-self.mu[-2])+np.linalg.norm(self.omega[-1]-self.omega[-2])>eps and iteration<max_step):
        while delta_ELBO>eps:
            if self.dependant:
                nu = np.random.multivariate_normal(np.zeros(P),np.eye(P),M)
            else : 
                nu = np.random.normal(0,1,M*P)
                nu = nu.reshape((M,P))
        
            gradient_mu,gradient_omega, new_ELBO = self.compute_gradient_muomega(nu)
            delta_ELBO = np.abs(new_ELBO-ELBO)/np.abs(ELBO)
            ELBO=new_ELBO
            if iteration==1:
                
                self.s_mu = gradient_mu**2
                self.s_omega = gradient_omega**2
          #  gradient_omega = self.compute_gradient_omega(nu)
            
            self.step_mu, self.s_mu = self.compute_step_size(iteration,lr,gradient_mu,self.s_mu)
            self.step_omega, self.s_omega = self.compute_step_size(iteration,lr,gradient_omega,self.s_omega)
            self.mu.append(self.mu[-1]+self.step_mu*gradient_mu)
            self.omega.append(self.omega[-1]+self.step_omega*gradient_omega)
            iteration+=1
            if self.dependant:
                print(iteration,self.mu[-1],self.omega[-1]@self.omega[-1].T)
            else: 
                print(iteration,self.mu[-1],np.exp(self.omega[-1])**2)

            #print(np.linalg.norm(self.mu[-1]-self.mu[-2])+np.linalg.norm(self.omega[-1]-self.omega[-2]))
            print(ELBO,delta_ELBO)
        return self.mu[-1],self.omega[-1]



