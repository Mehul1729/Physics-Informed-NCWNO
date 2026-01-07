if __name__ == '__main__':
    import os
    import wandb
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.io import loadmat
    from timeit import default_timer
    from utilities import *
    from modules import *
    import new_gradfree_fun2

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ------------------- MODEL AND EXPERT BLOCK DEFINITIONS -------------------
    class ExpertBlock(nn.Module):
        def __init__(self, level,width, expert_num, size):
            super().__init__()
            self.width, self.expert_num, self.level = width, expert_num, level
            wavelets = ["db"+str(i) for i in range(expert_num)]
            for i in range(expert_num):
                setattr(self, "Expert_layers" + str(i), WIB(self.width, self.width, self.level, size, wavelets[i]))
        def forward(self, x, lambda_):
            out = 0
            for i in range(self.expert_num):
                out = out + lambda_[..., i:i+1] * getattr(self, "Expert_layers"+str(i))(x)
            return out

    class NCWNO1d(nn.Module):
        def __init__(self, width, level, input_dim, num_layers, dom_bound, expert_num, label_lifting, size, padding=0):
            super().__init__()
            self.level, self.width, self.num_layers, self.dom_bound, self.padding = level, width, num_layers, dom_bound, padding
            self.size, self.expert_num, self.label_lifting = size, expert_num, label_lifting
            self.conv_layers, self.w_layers, self.gate = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
            for _ in range(self.num_layers):
                self.gate.append(Gate_context1d(width, width, expert_num, label_lifting, size))
                self.conv_layers.append(ExpertBlock(self.level, self.width, self.expert_num, self.size))
                self.w_layers.append(nn.Conv1d(self.width, self.width, 1))
            self.p1, self.p2 = nn.Conv1d(input_dim, self.width, 1), nn.Conv1d(self.width, self.width, 1)
            self.q1, self.q2 = nn.Conv1d(self.width, 128, 1), nn.Conv1d(128, 1, 1)

        def forward(self, x, label):
            x = torch.cat((x, self.get_grid(x.shape, x.device)), dim=1)
            x = self.p2(self.p1(x))
            if self.padding != 0: x = F.pad(x, [0, self.padding])
            lambda_ = [g(x, self.get_label(label, x.shape, x.device)) for g in self.gate]
            for wib, w0, lam in zip(self.conv_layers, self.w_layers, lambda_):
                x = F.mish(wib(x, lam) + w0(x))
            if self.padding != 0: x = x[..., :-self.padding]
            return self.q2(F.mish(self.q1(x)))

        def get_grid(self, shape, device):
            return torch.tensor(np.linspace(0, self.dom_bound, shape[-1]), dtype=torch.float).reshape(1, 1, shape[-1]).repeat([shape[0], 1, 1]).to(device)

        def get_label(self, label, shape, device):
            return label.repeat(shape[0], shape[1], 1).to(device).float()

    # ------------------- PDE-SPECIFIC CONFIGURATIONS -------------------
    PDE_CASE_INDEX = 2
    PDE_NAME = "Heat"
    data_path = r"C:\Users\mehul\a Folder\DataPhysicsHybrid\data\128\heat_data.mat"
    data_label = torch.tensor(PDE_CASE_INDEX + 1)
    
    learning_rate, epochs, scheduler_step, scheduler_gamma = 0.001*5, 200, 50, 0.95
    ntrain, ntest, batch_size, lbda, T, T0, step, S = 300, 100, 50, 1000, 20, 10, 1, 128
    sub = len(loadmat(data_path)['x']) // S

    run = wandb.init(
        project="New Seq. Learning - Parallel", name=f"[run-2] Fine-tune {PDE_NAME}",
        config={"pde_name": PDE_NAME, "n_train": ntrain, "lr": learning_rate, "epochs": epochs}
    )

    # ------------------- DATA LOADING -------------------
    data = (MatReader(data_path).read_field('sol')[::sub,:,:]).permute(2,1,0)
    train_a, train_u = data[:ntrain, :T0, :], data[:ntrain, T0:T0+T, :]
    test_a, test_u = data[-ntest:, :T0, :], data[-ntest:, T0:T0+T, :]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True, pin_memory = True, num_workers  = 4 )
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False, pin_memory = True, num_workers = 4)
    
    x_f_train = torch.tensor(np.array(np.meshgrid(np.linspace(0,1,S)[0] + (1-0)*np.linspace(0,1,S))[0]).reshape(S,1), dtype=torch.float).to(device)
    gf_heat = new_gradfree_fun2.heat_gradient_free()
    heat_p_index, heat_invp_index = gf_heat.neighbour_index(x_f_train), gf_heat.inverse_index(x_f_train)

    # ------------------- MODEL PREPARATION -------------------
    model = torch.load(r"C:\Users\mehul\a Folder\DataPhysicsHybrid\data\models\nagumo debugged\1 Jan models\Foundational128, num_experts=7][t=40]2026-01-02_07-33-12.pth", weights_only=False)
    for param in model.parameters(): param.requires_grad = False
    for l in range(model.num_layers):
        for param in model.gate[l].parameters(): param.requires_grad = True
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    # ------------------- TRAINING & TESTING LOOP -------------------
    print(f"\nTraining Gate Parameters on PDE - {PDE_NAME}...")
    ep = 0
    try:
        for ep in range(epochs):
            model.train()
            t1 = default_timer()
            ep_loss, j = 0, 0
            for xx, yy in train_loader:
                loss, loss_ic, j = 0, 0, j + 1
                xx, yy = xx.to(device), yy.to(device)
                for t in range(0, T, step):
                    y, im = yy[:, t:t + step, ...], model(xx, data_label)
                    loss_ic = F.mse_loss(im, yy[:, 0:step, ...]) if t == 0 else loss_ic
                    x_pf, y_pf, y_pred, y_true = xx[:, -1:, :].reshape(batch_size, S), im.reshape(batch_size, S), im.squeeze(1), y.squeeze(1)
                    all_u_train = torch.hstack([y_pred[:, 0], y_pred[:, -1]])
                    all_u_sol = torch.hstack([y_true[:, 0], y_true[:, -1]])
                    loss += gf_heat.loss(all_u_train, all_u_sol, x_pf, x_f_train, y_pf, heat_p_index, heat_invp_index)
                    xx = torch.cat((xx[:, step:, ...], im), dim=1)
                
                total_loss = loss + lbda * loss_ic
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                ep_loss += total_loss.item()
            
            scheduler.step()
            
            # --- TESTING PER EPOCH ---
            model.eval()
            with torch.no_grad():
                case_accuracies_all_samples = []
                for xx, yy in test_loader:
                    xx, yy = xx.to(device), yy.to(device)
                    preds = []
                    for t in range(0, T, step):
                        im = model(xx, data_label)
                        preds.append(im)
                        xx = torch.cat((xx[:, step:, :], im), dim=1)
                    
                    batch_preds = torch.cat(preds, dim=1)
                    all_y = yy[:, 0:T:step, :]
                    
                    numerator = torch.norm(batch_preds - all_y, p=2, dim=2)
                    denominator = torch.norm(all_y, p=2, dim=2)
                    denominator[denominator == 0] = 1e-8
                    accuracies = 1 - (numerator / denominator)
                    case_accuracies_all_samples.append(accuracies.cpu().numpy())

                mean_accuracy_for_case = np.mean(np.concatenate(case_accuracies_all_samples, axis=0))
            
            t2 = default_timer()
            print(f"Epoch: {ep}, Train Loss: {ep_loss/j:.4f}, Test Accuracy: {mean_accuracy_for_case:.4f}, Time: {t2-t1:.2f}s")
            wandb.log({"train_loss": ep_loss / j, "test_accuracy": mean_accuracy_for_case, "epoch": ep})
    
    except KeyboardInterrupt:
        print(f"\n\n*** TRAINING INTERRUPTED AT EPOCH {ep}. SAVING MODEL... ***\n")

    # --- SAVE FINAL MODEL ---
    model_save_path = fr'C:\Users\mehul\a Folder\DataPhysicsHybrid\data\models\seq\run-2\finetune_{PDE_NAME}.pth'
    torch.save(model.state_dict(), model_save_path) 
    print(f"Model for {PDE_NAME} saved to {model_save_path}")
    
    artifact = wandb.Artifact(name=f"model-{PDE_NAME}", type="model", metadata={"epochs_trained": ep + 1})
    artifact.add_file(model_save_path)
    run.log_artifact(artifact)
    print("\n--- Fine-tuning finished ---")