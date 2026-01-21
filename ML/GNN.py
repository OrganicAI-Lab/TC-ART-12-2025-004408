import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import torch, json
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("/path/to/smiles/file/with/target/energies/")
smiles_list = df["SMILES"].tolist()

y_values = df[["Target1", "Target2"]].values
train_smiles, test_smiles, train_y, test_y = train_test_split(smiles_list, y_values, test_size=0.2, random_state=42)

def mol_to_graph(smiles, targets):
    mol = Chem.MolFromSmiles(smiles)
    atom_features = []
    edge_index = []
    edge_features = []

    #Atom features
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
        ])

    #Bond features
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
        edge_features.append([bond.GetBondTypeAsDouble()])
        edge_features.append([bond.GetBondTypeAsDouble()])

    #Convert to tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    y = torch.tensor(targets, dtype=torch.float).view(1,-1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

#Convert training dataset into graph format
train_dataset = [mol_to_graph(smiles, y) for smiles, y in zip(train_smiles, train_y)]
#train_dataset = np.array(train_dataset)  #Convert to NumPy array for KFold indexing

#Convert test dataset into graph format for final evaluation
test_dataset = [mol_to_graph(smiles, y) for smiles, y in zip(test_smiles, test_y)]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class GNNRegressionModel(torch.nn.Module):
    def __init__(self, node_feat_dim, fc_hidden_dim, hidden_dim, num_layers, use_gat, dropout_rate, activation_fn):
        super(GNNRegressionModel, self).__init__()
        conv_layer = GATConv if use_gat else GCNConv
        self.convs = torch.nn.ModuleList([conv_layer(node_feat_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
        
        self.dropout = torch.nn.Dropout(dropout_rate)

        #Activation Function
        if activation_fn == "ReLU":
            self.activation = torch.nn.ReLU()
        elif activation_fn == "LeakyReLU":
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = torch.nn.GELU()

        self.fc1 = torch.nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = torch.nn.Linear(fc_hidden_dim, 2)  #Two output values

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.activation(self.fc1(x))
        return self.fc2(x)  #Regression output

def objective(trial):
    #Hyperparameter search space
    fc_hidden_dim = trial.suggest_categorical("fc_hidden_dim", [32, 64, 128, 256])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 5)
    use_gat = trial.suggest_categorical("use_gat", [True, False])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.5])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    activation_fn = trial.suggest_categorical("activation_fn", ["ReLU", "LeakyReLU", "GELU"])

    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    total_rmse = 0


    for train_idx, val_idx in kf.split(train_dataset):
        train_data = [train_dataset[i] for i in train_idx] 
        val_data = [train_dataset[i] for i in val_idx] 

        #train_data = train_dataset[train_idx].tolist()
        #val_data = train_dataset[val_idx].tolist()

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        #Initialize model
        model = GNNRegressionModel(
            node_feat_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_gat=use_gat,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
            fc_hidden_dim=fc_hidden_dim
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = torch.nn.MSELoss()

        def train():
            model.train()
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

        def validate():
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    out = model(data)
                    all_preds.append(out.cpu().numpy())
                    all_targets.append(data.y.cpu().numpy())

            all_preds = np.vstack(all_preds)
            all_targets = np.vstack(all_targets)
            
            r2_t1 = r2_score(all_targets[:,0], all_preds[:,0])
            r2_t2 = r2_score(all_targets[:,1], all_preds[:,1])
            rmse_t1 = np.sqrt(mean_squared_error(all_targets[:, 0], all_preds[:, 0]))
            rmse_t2 = np.sqrt(mean_squared_error(all_targets[:, 1], all_preds[:, 1]))
            average_rmse=(rmse_t1 + rmse_t2) / 2
            r2_t1=float(r2_t1)
            rmse_t1=float(rmse_t1)
            r2_t2=float(r2_t2)
            rmse_t2=float(rmse_t2)
            average_rmse=float(average_rmse)
            
            with open("values.json", "w") as f:
                json.dump(rmse_t1, f, indent=4)
                json.dump(rmse_t2, f, indent=4)
                json.dump(r2_t1, f, indent=4)
                json.dump(r2_t2, f, indent=4)
            
            return (average_rmse, r2_t1, r2_t2, rmse_t1, rmse_t2) 
        
        for epoch in range(30):  #Reduce epochs for tuning speed
            train()

        avg_rmse,r2_t1, r2_t2, rmse_t1, rmse_t2 = validate()
        total_rmse += avg_rmse

    return total_rmse / k_folds 

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)  
best_params = study.best_params

with open("best_hyperparameters.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

with open("best_hyperparameters.json", "r") as f:
    best_params = json.load(f)

class GNNRegressionModel(torch.nn.Module):
    def __init__(self, node_feat_dim, fc_hidden_dim, hidden_dim, num_layers, use_gat, dropout_rate, activation_fn):
        super(GNNRegressionModel, self).__init__()
        conv_layer = GATConv if use_gat else GCNConv
        self.convs = torch.nn.ModuleList([conv_layer(node_feat_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(conv_layer(hidden_dim, hidden_dim))
        
        self.dropout = torch.nn.Dropout(dropout_rate)

        #Activation Function
        if activation_fn == "ReLU":
            self.activation = torch.nn.ReLU()
        elif activation_fn == "LeakyReLU":
            self.activation = torch.nn.LeakyReLU()
        else:
            self.activation = torch.nn.GELU()

        self.fc1 = torch.nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = torch.nn.Linear(fc_hidden_dim, 2)  #Two output values (Target1, Target2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.activation(self.fc1(x))
        return self.fc2(x)  #Regression output
#Recreate the model using best hyperparameters
best_model = GNNRegressionModel(node_feat_dim=3,hidden_dim=best_params["hidden_dim"],num_layers=best_params["num_layers"],use_gat=best_params["use_gat"],dropout_rate=best_params["dropout_rate"],activation_fn=best_params["activation_fn"],fc_hidden_dim=best_params["fc_hidden_dim"]).to(device)

criterion = torch.nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["learning_rate"], weight_decay=best_params["weight_decay"])
#Final Model
for epoch in range(50):  #Increase epochs for final training
    best_model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = best_model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}/50 - Training Loss: {loss.item():.4f}")

torch.save(best_model.state_dict(), "best_model.pth")

#Evaluate Final Model
test_loader = DataLoader(test_dataset, batch_size=best_params["batch_size"], shuffle=False)
best_model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = best_model(data)
        all_preds.append(out.cpu().numpy())
        all_targets.append(data.y.cpu().numpy())

#Convert to NumPy arrays
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

rmse_t1 = np.sqrt(mean_squared_error(all_targets[:, 0], all_preds[:, 0]))
rmse_t2 = np.sqrt(mean_squared_error(all_targets[:, 1], all_preds[:, 1]))
r2_t1 = r2_score(all_targets[:, 0], all_preds[:, 0])
r2_t2 = r2_score(all_targets[:, 1], all_preds[:, 1])

final_results = {"RMSE_Target1": float(rmse_t1),"RMSE_Target2": float(rmse_t2),"R2_Target1": float(r2_t1),"R2_Target2": float(r2_t2)}

with open("final_model_results.json", "w") as f:
    json.dump(final_results, f, indent=4)

print("\nFinal Results Saved in 'final_model_results.json'")
