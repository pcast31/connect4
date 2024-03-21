import autograd.numpy as np
from autograd import elementwise_grad, jacobian



class ADVI():
    def __init__(self,inv_T,model,X):
        self.inv_T = inv_T
        self.model = model
        self.X = X
        self.grad_log_distribution = elementwise_grad(self.grad_log)

        self.grad_inv_T = elementwise_grad(self.inv_T)

        self.log_jac_inv_T = self.log_jac
        self.grad_logjac_inv_T = elementwise_grad(self.log_jac_inv_T)

    def log_jac(self,z):
        return np.log(np.abs(np.linalg.det(jacobian(self.inv_T)(z))))

    def grad_log(self,theta):
        return self.model.log_distr(self.X,theta)

    def nu_to_zeta(self,nu_d):
        return self.mu[-1]+self.omega[-1]@nu_d

    def zeta_to_theta(self,zeta):
        return self.inv_T(zeta)

    def compute_gradient_muomega(self,nu):
        self.grad_mu = np.zeros(self.mu[-1].shape)
        self.grad_omega = np.zeros(self.omega[-1].shape)
        for nu_i in nu:
            zeta_i = self.nu_to_zeta(nu_i)
            theta_i = self.zeta_to_theta(zeta_i)

            log_dis = self.grad_log_distribution(theta_i)
            grad_inv = self.grad_inv_T(zeta_i)
            grad_jac = self.grad_logjac_inv_T(zeta_i)

            self.grad_mu = self.grad_mu + log_dis*grad_inv+grad_jac
            self.grad_omega = self.grad_omega+np.outer((log_dis*grad_inv+grad_jac),nu_i)

        return self.grad_mu/len(nu), self.grad_omega/len(nu)+np.linalg.inv(self.omega[-1]).T



    # def compute_gradient_omega(self,nu):
    #     self.grad_omega = np.zeros(self.omega[-1].shape)
    #     for nu_i in nu:
    #         zeta_i = self.nu_to_zeta(nu_i)
    #         theta_i = self.zeta_to_theta(zeta_i)
    #         self.grad_omega = self.grad_omega+np.outer((self.grad_log_distribution(theta_i)*self.grad_inv_T(zeta_i)+self.grad_logjac_inv_T(zeta_i)),nu_i)
    #     return self.grad_omega/len(nu)+np.linalg.inv(self.omega[-1]).T

    def compute_step_size(self,iteration,P,lr):
        #TO DO : Implement formula (10) and (11)
        return lr * np.ones(P)

    def fit(self,eps=1e-4,max_step=1000,M=10, lr=1e-2, tau=1):
        P = len(self.X[0])
        N = len(self.X)
        iteration=1
        self.mu = [np.zeros(P)]
        self.omega = [np.eye(P)]
        # TO DO :  Code to find optimal step size
        while iteration==1 or (np.linalg.norm(self.mu[-1]-self.mu[-2])+np.linalg.norm(self.omega[-1]-self.omega[-2])>eps and iteration<max_step):
            nu = np.random.multivariate_normal(np.zeros(P),np.eye(P),M)

            gradient_mu,gradient_omega = self.compute_gradient_muomega(nu)
          #  gradient_omega = self.compute_gradient_omega(nu)
            step_vect = self.compute_step_size(iteration,P,lr)*iteration**-0.5
            self.mu.append(self.mu[-1]+step_vect*gradient_mu)
            # new_omega = np.zeros((P,P))
            # print(self.omega[-1][0].shape)
            # for i in range(N):
            #     new_omega[i]=self.omega[-1][i]+step_vect*gradient_omega
            # self.omega.append(new_omega)
            self.omega.append(self.omega[-1]+step_vect*gradient_omega)
            iteration+=1
            print(iteration,self.mu[-1],self.omega[-1]@self.omega[-1].T)
            print(np.linalg.norm(self.mu[-1]-self.mu[-2])+np.linalg.norm(self.omega[-1]-self.omega[-2]))
            
        return self.mu[-1],self.omega[-1]



