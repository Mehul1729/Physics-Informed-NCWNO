if __name__ == '__main__':
    import os
    import wandb
    directory = os.path.abspath(os.path.join(os.path.dirname('PDE_Simulation_data'), '.'))
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from scipy.io import loadmat
    import scipy.stats as st # for confidence interval calculation
    from timeit import default_timer
    from utilities import *
    from modules import *

    import new_gradfree_fun2

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
    # ------------------------------------------------------------------------------------------------
    # Defining the Expert Block:
    class ExpertBlock(nn.Module):
        def __init__(self, width, size, expert_num, level):
            
            super().__init__()
            self.width = width
            self.expert_num = expert_num
            self.level = level
            
            
            wavelets = ["db"+str(i) for i in range(expert_num)]
            
            # wavelet experts inside the expert block:
            
            for i in range(expert_num):
                setattr(self, "Expert_layers" + str(i),WIB(self.width, self.width, self.level, size, wavelets[i]))
                
            # Defining the the the forward operation of the Expert WNO block (+ routing by the context_gate1d
        def forward(self, x, lambda_):
            out = 0
            for i in range(self.expert_num):
                out = out + lambda_[..., i:i+1] * getattr(self, "Expert_layers"+str(i))(x)
            return out
            
            

    # %%

    class NCWNO1d(nn.Module):
        def __init__(self, width, level, input_dim, num_layers, dom_bound,
                     expert_num, label_lifting, size, padding=0):
            super().__init__()

            self.level = level
            self.width = width # lifting dimensionof input 
            self.num_layers = num_layers
            self.dom_bound = dom_bound # right support of 1D x-domain: [0, dom_bound]
            self.padding = padding # pad the domain if input is non-periodic
            self.size = size
            self.expert_num = expert_num
            self.label_lifting = label_lifting
            self.conv_layers = nn.ModuleList() # expert blocks
            self.w_layers = nn.ModuleList() # linear transformations
            self.gate = nn.ModuleList() # context gates
            

            # Uplifting:
            self.p1 = nn.Conv1d(input_dim, self.width ,1)
            self.p2 = nn.Conv1d(self.width, self.width, 1)
            
            # Expert Block + lifting transformation layers:
            for i in range(self.num_layers):
                self.gate.append(Gate_context1d(width, width, expert_num, label_lifting, size))
                

                self.conv_layers.append(ExpertBlock(self.width, self.size, self.expert_num, self.level))
                self.w_layers.append(nn.Conv1d(self.width, self.width, 1))
                                        
            # Downlifting:
            self.q1 = nn.Conv1d(self.width, 128, 1)
            self.q2 = nn.Conv1d(128, 1, 1)
            # --- END CORRECTION ---

        def forward(self, x, label):

            
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=1)
            x=  self.p1(x)
            x = self.p2(x)
            if self.padding != 0:
                x = F.pad(x, [0, self.padding])

            lambda_ = []
            label = self.get_label(label, x.shape, x.device)
            for gates in self.gate:
                lambda_.append(gates(x, label))
                
            for wib, w0, lam in zip(self.conv_layers, self.w_layers, lambda_):
                x = wib(x, lam) + w0(x)
                x = F.mish(x)
                
                
            if self.padding != 0:
                x = x[..., :-self.padding] # removing padding when required
            x = self.q1(x)
            x = F.mish(x)
            x = self.q2(x)
            return x
        
        def get_grid(self, shape, device):
            # The grid of the solution
            batchsize, size_x = shape[0], shape[-1]
            gridx = torch.tensor(np.linspace(0, self.dom_bound, size_x), dtype=torch.float)  # dom_bound is the right support of 1D domain: [0, 1]
            gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])
            return gridx.to(device)
        
        
        def get_label(self, label, shape, device):
            # Adds batch and channel to the label
            batchsize, channel_size, size_x = shape

            label = label.view(1, 1, 1).repeat(batchsize, channel_size, 1).to(device)
            
            return label.float() 

# %%
    """ Model configurations """

    data_path = []
    data_path.append(r"C:\Users\mehul\a Folder\DataPhysicsHybrid\data\128\nagumo_dat_128.mat")
    data_path.append(r"C:\Users\mehul\a Folder\DataPhysicsHybrid\data\128\new_128burger_data.mat")

    case_len = len(data_path)
    data_label = torch.arange(1, case_len+1)
    case_names = ['Nagumo', 'Burgers']

    ntrain = 10
    ntest = 200 # since we need to test, we take mostly samples for testing 

    batch_size = 20

    T = 40
    T0 = 10
    step = 1
    S = 128
    prop = loadmat(data_path[0])['x']
    sub = len(prop) // S  

# %%
    """ Read data """
    data = []
    for path in data_path:
        print('Loading:',path)
        data.append( (MatReader(path).read_field('sol')[::sub,:,:]).permute(2,1,0) )

    train_a, train_u, test_a, test_u = ( [] for i in range(4) )
    for case in range(case_len):
        train_a.append( data[case][:ntrain, :T0, :] )
        train_u.append( data[case][:ntrain, T0:T0+T, :] )
        test_a.append( data[case][-ntest:, :T0, :] )
        test_u.append( data[case][-ntest:, T0:T0+T, :] )

    train_loader, test_loader = [], []
    for case in range(case_len):
        train_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a[case], train_u[case]), 
                                                      batch_size=batch_size, shuffle=True) )
        test_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a[case], test_u[case]), 
                                                     batch_size=batch_size, shuffle=False) )


    lb = np.array([0, 0])
    ub = np.array([1, 1])
    N_f = S # number of points in the mesh 
    xt1=lb[0] + (ub[0]-lb[0])*np.linspace(0,1,N_f) 
    Xt1 = np.meshgrid(xt1)
    X_f_train = np.array(Xt1).reshape(N_f,1)
    x_f_train = torch.tensor(X_f_train,dtype=torch.float).to(device)
    print(f"Shape o x_f_train:{x_f_train.shape}")
# %%
    """ Load the model to be tested """
    # Using the old pre-trained model for testing (change the path to test for a newly trained model by adding the model from the ) 
    model = torch.load(r"C:\Users\mehul\a Folder\DataPhysicsHybrid\data\models\7_experts\[num_experts=7]Foundational_phy_informed_[t=40]2025-10-05_18-49-52.pth", weights_only = False)
    print(count_params(model))

    myloss = LpLoss(size_average=False)

# %%
    """ Prediction and Accuracy Calculation """

    all_accuracies = []
    
    with torch.no_grad():
        print(f"\n--- Evaluating Model ---")
        
        for i, case_loader in enumerate(test_loader):

            case_accuracies_all_samples = []
            
            for xx, yy in case_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                
                batch_preds = []
                
                current_xx = xx.clone()

                for t in range(0, T, step):
                    im = model(current_xx, data_label[i])
                    batch_preds.append(im)
                    current_xx = torch.cat((current_xx[:, step:, ...], im), dim=1)
                

                batch_preds = torch.cat(batch_preds, dim=1)
 
                all_y = yy[:, 0:T:step, :]
                
                denominator = torch.norm(all_y, p=2, dim=2)
                
                denominator[denominator == 0] = 1e-8  # to avoid division by zero
                relative_errors = torch.norm(batch_preds - all_y, p=2, dim=2) / denominator
                

                accuracies = 1 - relative_errors
                
                case_accuracies_all_samples.append(accuracies.cpu().numpy())

            # combining the accuracies for all the batches 
            case_accuracies_all_samples = np.concatenate(case_accuracies_all_samples, axis=0)
            all_accuracies.append(case_accuracies_all_samples)
            print(f"Finished processing Case: {case_names[i]}. Accuracy matrix shape: {case_accuracies_all_samples.shape}")
            
            
            import pandas as pd 
            df = pd.DataFrame(case_accuracies_all_samples)
            df.to_csv(rf'C:\Users\mehul\a Folder\DataPhysicsHybrid\Debugged Nagumo\plots\accuracies_case_{case_names[i]}_T{T}.csv', index=False)
            
# %%
    """ Plotting Time Steps vs. Accuracy """
    
    print(f"\n--- Plotting Model Performance ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    axes = axes.flatten() 
    fig.suptitle(rf'Model Performance Over Time-time_steps_preicted={T}', fontsize=20)

    for case_idx, case_accs in enumerate(all_accuracies):
        
        mean_accuracy_per_step = np.mean(case_accs, axis=0)
        
        n_samples = case_accs.shape[0]
        confidence_interval = st.t.interval(0.95, df=n_samples-1, 
                                            loc=mean_accuracy_per_step, 
                                            scale=st.sem(case_accs, axis=0))
        
        time_steps = np.arange(0, T, step)
        
        ax = axes[case_idx]
        
        ax.plot(time_steps, mean_accuracy_per_step, label='Mean Accuracy', color='royalblue', lw=2)
        
        ax.fill_between(time_steps, confidence_interval[0], confidence_interval[1], 
                        color='cornflowerblue', alpha=0.3, label='95% Confidence Interval')
        
        ax.set_title(f'Case: {case_names[case_idx]}', fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Mean Sample Accuracy', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_ylim(bottom=min(0, np.min(confidence_interval[0])-0.1), top=1.05) 
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) 
    plt.savefig(rf'C:\Users\mehul\a Folder\DataPhysicsHybrid\Debugged Nagumo\plots\model_performance_plot{T}.png')
    plt.show()