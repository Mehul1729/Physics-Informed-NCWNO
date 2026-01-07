if __name__ == "__main__":

# This version has no Normalization of losses as scales of Nagumo and Burger data are similar
# corrected the Nagumo loss function bug 
    """

    This is the final code for combined foundational learning of 2 PDEs only:


    1. Nagumo

    2. Burger


    """

    # %%

    import wandb

    from datetime import datetime

    import os

    directory = os.path.abspath(os.path.join(os.getcwd(), '..'))

    import numpy as np

    import torch

    import torch.nn as nn

    import torch.nn.functional as F

    import matplotlib.pyplot as plt

    from scipy.io import loadmat

    from timeit import default_timer

    from utilities import *

    from modules import *
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


    import new_gradfree_fun_3 as new_gradfree_fun


    torch.manual_seed(0)

    np.random.seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    myloss = LpLoss(size_average= False)


    # %%

    # ------------------------------------------------------------------------------------------------

    # Defining the Expert Block:

    class ExpertBlock(nn.Module):

        def __init__(self, level, width, expert_num, size):

            

            super().__init__()

            self.width = width

            self.expert_num = expert_num

            self.level = level

            

            

            wavelets = ["db"+str(i+1) for i in range(expert_num)]

            

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

            self.dom_bound = dom_bound # right support of 1D x-domain: [0, space_len]

            self.padding = padding # pad the domain if input is non-periodic

            self.size = size

            self.expert_num = expert_num

            self.label_lifting = label_lifting

            self.conv_layers = nn.ModuleList() # expert blocks

            self.w_layers = nn.ModuleList() # linear transformations

            self.gate = nn.ModuleList() # context gates

            

            

            for num in range(self.num_layers):

                self.gate.append(Gate_context1d(width, width, expert_num, label_lifting, size))

            

            

                # Uplifting:

            self.p1 = nn.Conv1d(input_dim, self.width,1)

            self.p2 = nn.Conv1d(self.width, self.width, 1)

                # Expert Block + lifting transformation:

            for num in range(self.num_layers): # number of expert blocks

                self.conv_layers.append(ExpertBlock(self.level, self.width, self.expert_num, self.size))

                self.w_layers.append(nn.Conv1d(self.width, self.width, 1)) # linear transfrom inside expert block

            

                # Downlifting:

            self.q1 = nn.Conv1d(self.width, self.width, 1) # width = 128

            self.q2 = nn.Conv1d(self.width, 1, 1) # single dim out put for 1D PDEs

            

        def forward(self, x, label):


            # Input: 2 channel tesnor : (a(x), x)  ; shape = (batchsize * x=s * c=2)

            # Output: solution of later time step: u(x)  ;  shape=(batchsize * x=s * c=1)

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

            gridx = torch.tensor(np.linspace(0, self.dom_bound, size_x), dtype=torch.float)  # space_len is the right support of 1D domain: [0, 1]

            gridx = gridx.reshape(1, 1, size_x).repeat([batchsize, 1, 1])

            return gridx.to(device)

            

        def get_label(self, label, shape, device):

            # Adds batch and channel to the label

            batchsize, channel_size, size_x = shape

            label = label.repeat(batchsize, channel_size, 1).to(device)

            return label.float() 



    # %%

    """ Model configurations """

    # data_path.append(r"data/allen_cahn_stable.mat")

    # the data used earlier for the original training might not have the same coefficient as our CN loss has. So using the self-generated data

    

    data_path = []

    data_path.append(r"data/nagumo_data_256x1200.mat")


    data_path.append(r"data/burger_data_256x1200.mat")
    

    case_len = len(data_path) 

    data_label = torch.arange(1, case_len+1) 

    case_names = ['Nagumo', 'Burgers']



    ntrain = 400

    ntest = 100

    batch_size = 10

    epochs = 300

    # scheduler_step = 10

    # scheduler_gamma = 0.95

    level = 5
    

    learning_rate = 1e-5*2 # old : 1e-5 * 2

    width = 256

    lbda = 1000 # Weight for initial condition loss

    # data_loss_weight = 1.0 # NEW: Weight for the data-driven loss

    T = 40

    T0 = 10

    step = 1

    S = 128

    prop = loadmat(data_path[0])['x']

    sub = len(prop) // S

 

    run = wandb.init(

        project="256 Foundational runs",
 
        name = "Run-1 , 256 res.",
        
        config = {

            "n_train": ntrain,

            "n_test": ntest,

            "batch_size": batch_size,

            "epochs" : epochs,

            "lr" : learning_rate,
            
            "level" : level,

            "T" : T,

            "T0":T0,

            "S" : S,

            "num_pde":case_len,

            "expert_num" : 5,

            "optimizer" : "CosineAnnealingLR", 
            
            " learning rate" : learning_rate, 
            "min_lr" : 1e-5,
            
            "level" : level, 
            "Burger neigh" : 5,
            "Nagumo neigh" :5
            }
    )



    # %%

    """ Read data """

    data = []

    for path in data_path:

        print('Loading:',path)

        data.append( (MatReader(path).read_field('sol')[::sub,:,::]).permute(2,1,0) )


    train_a, train_u, test_a, test_u = ( [] for i in range(4) )

    for case in range(case_len):

        train_a.append( data[case][:ntrain, :T0, :] )

        train_u.append( data[case][:ntrain, T0:T0+T, :] )

        test_a.append( data[case][-ntest:, :T0, :] )

        test_u.append( data[case][-ntest:, T0:T0+T, :] )


    print(f"Train data shape: {train_a[0].shape}, {train_u[0].shape}")

    print(f"Test data shape: {test_a[0].shape}, {test_u[0].shape}")


    train_loader, test_loader = [], []

    for case in range(case_len):

        train_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a[case], train_u[case]), 

                                                batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4))

        test_loader.append( torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a[case], test_u[case]), 

                                                batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)) 


    # %%

    # creating the mesh for collocation points:

    lb = np.array([0, 0])

    ub = np.array([1, 1])

    N_f = S # number of points in the mesh 

    xt1=lb[0] + (ub[0]-lb[0])*np.linspace(0,1,N_f) 

    Xt1 = np.meshgrid(xt1)

    X_f_train = np.array(Xt1).reshape(N_f,1)

    x_f_train = torch.tensor(X_f_train,dtype=torch.float).to(device)

    print(f"Shape o x_f_train:{x_f_train.shape}")


    # defining grad_free functions for respec. PDEs

    gf_allen = new_gradfree_fun.allen_cahn_gradient_free()

    gf_nagumo = new_gradfree_fun.nagumo_gradient_free()

    gf_burger = new_gradfree_fun.burger_gradient_free()


    allen_p_index = gf_allen.neighbour_index(x_f_train)

    allen_invp_index = gf_allen.inverse_index(x_f_train)


    nagumo_p_index = gf_nagumo.neighbour_index(x_f_train)

    nagumo_invp_index = gf_nagumo.inverse_index(x_f_train)

    burger_p_index = gf_burger.neighbour_index(x_f_train)

    burger_invp_index = gf_burger.inverse_index(x_f_train)


    # %%

    """ The model definition """

    model = NCWNO1d(width=width, level=level, input_dim=T0+1, num_layers=4, dom_bound=1,

                    expert_num= 5, label_lifting=2**4, size=S).to(device)

    print(f"Number of model paramters:{count_params(model)}")

    pde_no = 2


    # %%

    """ Training and testing """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20,
                                            gamma=0.6)


    # # NEW: Define indices for uniformly sampled data points
    # num_data_points = 4
    # data_indices = torch.linspace(0, S - 1, num_data_points).long().to(device)


# helper function to log gradients and weights
    # def log_gradients_and_weights(model, step):
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             wandb.log({f"weights/{name}_norm": param.data.norm().item()}, step=step)
    #             if param.grad is not None:
    #                 wandb.log({f"grads/{name}_norm": param.grad.norm().item()}, step=step)
    try:

        for ep in range(epochs):

            model.train()

            t1 = default_timer()

            epoch_train_step = np.zeros( pde_no ) # loss for each PDE




            for i, case_loader in enumerate(train_loader[:pde_no]): # iterating over train-test data for first 2 PDEs

                print(f"Training started for PDE- {i}")

                case_train_step = 0 # loss for the current case PDE

                case_ic_loss = 0

                j = 0

                for xx, yy in case_loader:

                    loss_ic = 0 # total initial loss for this batch

                    j = j+1

                    print(f"Batch - {j}")

                    physics_loss = 0 # physics loss for current batch
                    # data_loss = 0 # NEW: data loss for current batch

                    xx = xx.to(device) # input x data

                    yy = yy.to(device) # output training data

                    

                    for t in range(0, T, step):

                    

                        y = yy[:, t:t + step, ...] # [20, 1, 128]

                        im = model(xx, data_label[i])    # input mesh + label
                        
                 

                        if t == 0:

                            loss_ic = F.mse_loss(im, yy[:, t:t+step, ...])

                            pred = im # shape : [20, 1, 128] :: Batch, Time step, Spatial Points

                        else:

                            pred = torch.cat((pred, im), 1)

                        # """Vectorized sample processing for Physics Loss"""

                        x_pf = xx[:, -1:, :].reshape(batch_size, N_f) # u_n for CN loss

                        y_pf = im.reshape(batch_size, N_f) # prediction for the current time step

                        y_pred = im.squeeze(1) # new shape : [batch_size, 128]
                        y_true = y.squeeze(1)

                        left_u = y_pred[:, 0]
                        right_u = y_pred[:, -1]

                        left_usol = y_true[:, 0]

                        right_usol = y_true[:, -1]

                    

                        all_u_train = torch.hstack([left_u, right_u ])

                        all_u_sol = torch.hstack([left_usol, right_usol])

                            

                        # collecting the physics losses for all the time steps here : 

                        match i:

                            case 0:

                                physics_loss += gf_nagumo.loss(all_u_train, all_u_sol, x_pf, x_f_train, y_pf, nagumo_p_index, nagumo_invp_index)

                            case 1:

                                physics_loss += gf_burger.loss(all_u_train, all_u_sol, x_pf, x_f_train, y_pf, burger_p_index, burger_invp_index)

                        

                        # Shifting the input window by one time step:

                        xx = torch.cat((xx[:, step:, ...], im.detach()), dim=1)

                        

                    case_train_step += physics_loss.item()
                    
                    total_loss = physics_loss + (lbda * loss_ic)
                    # total_loss = weights[i] * total_loss
 
                    print(f"loss for the batch:\n physics loss : {physics_loss } \n \n Initial loss : {loss_ic} {total_loss.item()}")

                    wandb.log({

                        f"PDE_{i}/batch_loss" : total_loss.item(),
                        f"PDE_{i}/physics_loss": physics_loss.item(),
                        f"PDE_{i}/initial_condition_loss": loss_ic.item(),

                        "epoch":ep

                    })

                    optimizer.zero_grad()

                    case_ic_loss = case_ic_loss + loss_ic

                    total_loss.backward() # losses are backpropagated for the single batch
                    # log_gradients_and_weights(model, step=ep * len(case_loader) + j)
                    optimizer.step() # batch loop ends

                    

                epoch_train_step[i] = case_train_step # pde loop ends

                match i:

                    case 0:
                        print(f"Total loss for PDE-{0} at epoch {ep}: {epoch_train_step[0]}")
                        wandb.log({

                            "Train_Loss/PDE_0": epoch_train_step[0],

                 

                            "epoch": ep,

                            "total initial loss": case_ic_loss})

# average loss for the current PDE

                    case 1:
                        print(f"Total loss for PDE-{1} at epoch {ep}: {epoch_train_step[1]}")

                        wandb.log({

                        "Train_Loss/PDE_1": epoch_train_step[1],

                        "epoch": ep,

                        "total initial loss": case_ic_loss

                    })


            # --- TESTING SECTION (UNCHANGED) ---
            epoch_test_step = np.zeros( pde_no )

            scheduler.step()

            model.eval()

            test_phyics_loss = 0

            scheduler.step()

            model.eval()

            print("\n--- ROBUST TESTING ---")

            

            # This will store the final mean accuracy for each PDE

            epoch_test_accuracy = np.zeros(pde_no)


            with torch.no_grad():

                for i, case_loader in enumerate(test_loader[:pde_no]):

                    print(f"Testing started for PDE: {case_names[i]}")

                    

                    # List to store accuracy matrices from each batch

                    case_accuracies_all_samples = []

                    

                    for xx, yy in case_loader: # iterating over the batches

                        xx = xx.to(device)

                        yy = yy.to(device)

                        

                        # Autoregressively predict for all T time steps

                        batch_preds = []

                        current_xx = xx.clone()

                        for t in range(0, T, step):

                            im = model(current_xx, data_label[i])

                            batch_preds.append(im)

                            current_xx = torch.cat((current_xx[:, step:, ...], im), dim=1)

                        

                        # Stack predictions along the time dimension

                        batch_preds = torch.cat(batch_preds, dim=1)

                        

                        # Get the ground truth for all corresponding time steps

                        all_y = yy[:, 0:T:step, :]


                        # Calculate relative error for each sample and each time step

                        numerator = torch.norm(batch_preds - all_y, p=2, dim=2)

                        denominator = torch.norm(all_y, p=2, dim=2)

                        denominator[denominator == 0] = 1e-8 # Avoid division by zero

                        

                        relative_errors = numerator / denominator

                        accuracies = 1 - relative_errors

                        

                        case_accuracies_all_samples.append(accuracies.cpu().numpy())


                    # Concatenate results from all batches for this case

                    # Final shape for this case: (ntest, T)

                    full_case_accuracies = np.concatenate(case_accuracies_all_samples, axis=0)

                    

                    # Calculate the overall mean accuracy for this PDE

                    mean_accuracy_for_case = np.mean(full_case_accuracies)

                    epoch_test_accuracy[i] = mean_accuracy_for_case

                    

                    # Log the robust metric to wandb

                    wandb.log({

                        f"Mean_Test_Accuracy/PDE_{i}_{case_names[i]}": mean_accuracy_for_case,

                        "epoch": ep

                    })


            t2 = default_timer()


            print('--- Epoch Summary ---')

            print('Epoch-{}, Time-{:0.4f}'.format(ep, t2-t1))

            print('Mean Test Accuracy: PDE_0-{:0.4f}, PDE_1-{:0.4f}'.format(

                  epoch_test_accuracy[0], epoch_test_accuracy[1]))

            print('---------------------\n')

            

        # Save the final model

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_name = f"Foudational Model-{timestamp}.pth"

        torch.save(model, rf"Models\{model_name}") 
        
        artifact = wandb.Artifact(name=f"model-foundational with 7 experts", type="model", metadata={"epochs_trained": ep + 1})
        artifact.add_file(rf"Models\{model_name}")
        wandb.log_artifact(artifact)
        
        print("All epochs run. Saved the final model.")


    except KeyboardInterrupt:

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_name = f"Foudational Model-{timestamp}.pth"

        torch.save(model, rf"Models\{model_name}")

        print(f"Training interrupted. Model saved at epoch {ep}.")
        
        
        
        
            #$ saving the model to wandb cloud :
        artifact = wandb.Artifact(name=f"model-foundational with 7 experts", type="model", metadata={"epochs_trained": ep + 1})
        artifact.add_file(rf"Models\{model_name}")
        wandb.log_artifact(artifact)
    wandb.finish()