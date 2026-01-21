import optuna, sys, json, os, os.path
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, make_scorer, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#%%Data
#json file and descriptor/model lists
default_json_path = os.path.join(os.getcwd(), "parameters.json")
#fixed parameters, project path, model and descriptor input list
def load_json(json_file=default_json_path):
    with open(json_file, "r") as file1:
        config=json.load(file1)
    column_target_1 = config.get("column_target_1")-1
    column_target_2 = config.get("column_target_2")-1
    feature_column_start = config.get("feature_column_start")-1
    project_path = config.get("project_path", os.getcwd())
    param_ranges = config["hyperparameters"]
    fixed_parameters = config.get("fixed_parameters", {})
    model_list_manual=config["model_list"]
    model_list_auto=list(config["hyperparameters"].keys())
    descriptor_input_list=config["descriptor_input_list"]
    target_list=config["target_list"]
    elec_property=config.get("elec_property")
    return elec_property,param_ranges, project_path, model_list_manual, descriptor_input_list, model_list_auto, fixed_parameters, column_target_1, column_target_2, feature_column_start, target_list

elec_property,param_ranges, project_path, model_list_manual, descriptor_input_list, model_list_auto, fixed_parameters, column_target_1, column_target_2, feature_column_start, target_list = load_json()

#dataset
def load_data(descriptor_input,file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, feature_column_start:].values
    if len(target_list)>1:
        y_hole = data.iloc[:, column_target_1].values 
        y_electron = data.iloc[:, column_target_2].values  
        y = np.column_stack((y_hole, y_electron))
    elif target_list == ["electron"]:
        y = data.iloc[:, column_target_2].values
    elif target_list == ["hole"]:
        y = data.iloc[:, column_target_1].values

    y=np.log10(y)

    if descriptor_input.startswith(("RDKit_descriptors")):
        descriptor_subgroup = "RDKit_descriptors"
    elif descriptor_input.startswith(("Descriptors")):
        descriptor_subgroup = "Descriptors"
    elif descriptor_input.startswith(("All_fp", "Daylight", "MACCS", "Morgan")):
        descriptor_subgroup = "fp"
    else:
        descriptor_subgroup = "other"

    return X,y, descriptor_subgroup

#%%Hyperparameter optimisation  
###results metrics###
def average_rmse(y_true, y_pred):
    if len(target_list) > 1:
        rmse_hole = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        rmse_electron = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        average_rmse = (rmse_hole+rmse_electron) / 2
    else:
        average_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    return average_rmse

average_rmse_scorer = make_scorer(average_rmse, greater_is_better=False)

def save_results(study, model_name, descriptor_input, descriptor_subgroup, target_list):
    if len(target_list)>1:
        descriptor_results_path = os.path.join(project_path, "results/multioutput")
    else:
        descriptor_results_path = os.path.join(project_path, f"results/{target_list[0]}")
    os.makedirs(descriptor_results_path, exist_ok=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    json_file = os.path.join(descriptor_results_path, "optimisation.json")

    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            all_results = json.load(file)
    else:
        all_results = {}
    
    #nesting
    all_results.setdefault(elec_property, {})
    all_results[elec_property].setdefault(descriptor_subgroup, {})
    all_results[elec_property][descriptor_subgroup].setdefault(descriptor_input, {})
    
    all_results[elec_property][descriptor_subgroup][descriptor_input][model_name] = {
        "best_avg_rmse": best_value,
        "best_parameters": best_params
    }

    with open(json_file, "w") as file:
        json.dump(all_results, file, indent=4, sort_keys=True)
        
    return best_params

###model parameters###
def get_model(trial, model_name):  
    params=param_ranges[model_name]
    if model_name == "knn":
        fixed_parameters_knn = fixed_parameters.get("knn", {})
        n_neighbors = trial.suggest_int("n_neighbors", params["n_neighbors"][0], params["n_neighbors"][1])
        weights = trial.suggest_categorical("weights", params["weights"])
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, **fixed_parameters_knn)   
    elif model_name == "xgb":
        fixed_parameters_xgb = fixed_parameters.get("xgb", {})
        gamma =trial.suggest_float("gamma", params["gamma"][0], params["gamma"][1])
        max_depth = trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1])
        learning_rate = trial.suggest_float("learning_rate", params["learning_rate"][0], params["learning_rate"][1])
        min_child_weight = trial.suggest_int("min_child_weight", params["min_child_weight"][0], params["min_child_weight"][1])
        model = XGBRegressor(gamma=gamma, max_depth=max_depth,learning_rate=learning_rate, min_child_weight=min_child_weight,**fixed_parameters_xgb)
    elif model_name == "svr_gaussian":
        fixed_parameters_svr_gaussian = fixed_parameters.get("svr_gaussian", {})
        C = trial.suggest_float("C", params["C"][0], params["C"][1], log=True)
        gamma = trial.suggest_float("gamma", params["gamma"][0], params["gamma"][1], log=True)
        model = SVR(C=C, gamma=gamma,**fixed_parameters_svr_gaussian)
    elif model_name == "svr_linear":
        fixed_parameters_svr_linear = fixed_parameters.get("svr_linear", {})
        C = trial.suggest_float("C", params["C"][0], params["C"][1], log=True)
        model = LinearSVR(C=C,**fixed_parameters_svr_linear)
    elif model_name == "lgbm":
        fixed_parameters_lgbm = fixed_parameters.get("lgbm", {})
        learning_rate = trial.suggest_float("learning_rate", params["learning_rate"][0], params["learning_rate"][1], log=True)  
        max_depth = trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1]) 
        num_leaves = trial.suggest_int("num_leaves", params["num_leaves"][0], params["num_leaves"][1])
        model = LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, learning_rate=learning_rate, **fixed_parameters_lgbm)
    elif model_name == "gbrt":
        fixed_parameters_gbrt = fixed_parameters.get("gbrt", {})
        n_estimators = trial.suggest_int("n_estimators", params["n_estimators"][0], params["n_estimators"][1])
        max_depth = trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1])
        learning_rate = trial.suggest_float("learning_rate", params["learning_rate"][0], params["learning_rate"][1])
        model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, **fixed_parameters_gbrt)
    elif model_name == "rf":
        fixed_parameters_rf = fixed_parameters.get("rf", {})
        n_estimators = trial.suggest_int("n_estimators", params["n_estimators"][0], params["n_estimators"][1])
        max_depth = trial.suggest_int("max_depth", params["max_depth"][0], params["max_depth"][1])
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, **fixed_parameters_rf)
    elif model_name == "adaboost":
        fixed_parameters_adaboost = fixed_parameters.get("adaboost", {})
        n_estimators = trial.suggest_int("n_estimators", params["n_estimators"][0], params["n_estimators"][1])
        learning_rate = trial.suggest_float("learning_rate", params["learning_rate"][0], params["learning_rate"][1])
        model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, **fixed_parameters_adaboost)
    elif model_name == "mlpregressor":
        fixed_parameters_mlpregressor = fixed_parameters.get("mlpregressor", {})
        solver = trial.suggest_categorical("solver", params["solver"])
        max_iter = trial.suggest_int("max_iter", params["max_iter"][0], params["max_iter"][1])
        hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", params["hidden_layer_sizes"][0], params["hidden_layer_sizes"][1])
        activation = trial.suggest_categorical("activation", params["activation"])
        alpha = trial.suggest_float("alpha", params["alpha"][0], params["alpha"][1], log=True)
        learning_rate_init = trial.suggest_float("learning_rate_init", params["learning_rate_init"][0], params["learning_rate_init"][1], log=True)
        batch_size = trial.suggest_categorical("batch_size", params["batch_size"])
        model = MLPRegressor(solver=solver,random_state=42,max_iter=max_iter,hidden_layer_sizes=hidden_layer_sizes,activation=activation,alpha=alpha, learning_rate_init=learning_rate_init, batch_size=batch_size, **fixed_parameters_mlpregressor)    
    else:
        raise ValueError("Invalid model name")
       
    if model_name in ["adaboost", "gbrt", "lgbm", "svr_gaussian", "svr_linear"]:
        if len(target_list)>1:
            model = MultiOutputRegressor((model))
    
    return model

###trial###    
def objective(trial, model_name, X_train, y_train):
    model = get_model(trial, model_name)
    
    scaler = StandardScaler() if model_name in ["svr_gaussian", "svr_linear", "lr"] else MinMaxScaler() if model_name in ["knn", "mlpregressor"] else None
    steps = [("model", model)]
    if scaler:
        steps.insert(0, ("scaler", scaler))
    pipeline = Pipeline(steps)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42) 
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_val_fold)
        fold_rmse = average_rmse(y_val_fold, y_pred)
        fold_scores.append(fold_rmse)
        
        # Print real-time fold progress
        sys.stdout.write(f'\r{" " * 100}')
        sys.stdout.write(f"\rTrial {trial.number+1}, Fold {fold}/5: Fold RMSE = {fold_rmse:.6f}")
        sys.stdout.flush()
    
    return np.mean(fold_scores)
    
def optimise_model(model_name, X_train, y_train, n_trials=50):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.create_study(direction="minimize")

    
    for trial in range(n_trials):       
        study.optimize(lambda t: objective(t, model_name, X_train, y_train), n_trials=1)

    sys.stdout.write(f"\rBest RMSE = {study.best_value:.6f}, Params = {study.best_params}")

    return study

#%%ML
###results###
def evaluate_model(y_test, y_pred):
    y_test = np.array(y_test) if isinstance(y_test, pd.DataFrame) else y_test
    y_pred = np.array(y_pred) if isinstance(y_pred, pd.DataFrame) else y_pred
    # Pearson correlation coefficient
    r2 = r2_score(y_test, y_pred)
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #MAE
    mae= mean_absolute_error(y_test, y_pred)

    return r2, rmse, mae

def results(descriptor_input,model_name,y_test,y_pred,descriptor_subgroup, hole_results_list, electron_results_list, results_list):
    #makefolders
    if len(target_list)>1:
        target_results_path=os.path.join(project_path, "results/multioutput")
    else:
        target_results_path=os.path.join(project_path, f"results/{target_list[0]}")
        
    descriptor_main_results_path = os.path.join(target_results_path, elec_property)
    os.makedirs(descriptor_main_results_path, exist_ok=True)
    
    descriptor_results_path = os.path.join(descriptor_main_results_path, f"{descriptor_input}")
    os.makedirs(descriptor_results_path, exist_ok=True)
    model_results_path = os.path.join(descriptor_results_path,str(model_name))
    os.makedirs(model_results_path, exist_ok=True)
    
    #define what y_pred and y_test is depending on target
    if len(target_list)>1:
        #predicted and actual values
        results_df = pd.DataFrame({"Actual_Hole": y_test[:, 0],"Predicted_Hole": y_pred[:, 0],"Difference_Hole": y_pred[:, 0] - y_test[:, 0],"Actual_Electron": y_test[:, 1],"Predicted_Electron": y_pred[:, 1],"Difference_Electron": y_pred[:, 1] - y_test[:, 1]})
        #predicted vs actual plot
        plot_results(y_test[:, 0], y_pred[:, 0], model_results_path, target="hole")
        plot_results(y_test[:, 1], y_pred[:, 1], model_results_path, target="electron")
        #r2, rmse, mae
        process_results(model_name, elec_property, descriptor_subgroup, descriptor_input, y_test[:, 0], y_pred[:, 0], hole_results_list, f"{project_path}/results/multioutput/hole_results.csv")
        process_results(model_name, elec_property, descriptor_subgroup, descriptor_input, y_test[:, 1], y_pred[:, 1], electron_results_list, f"{project_path}/results/multioutput/electron_results.csv")
    else:
        #predicted and actual values
        results_df = pd.DataFrame({"Actual": y_test,"Predicted": y_pred, "Difference": y_pred - y_test})
        #predicted vs actual plot
        plot_results(y_test, y_pred, model_results_path, target=f"{target_list[0]}")
        #r2, rmse, mae
        process_results(model_name, elec_property, descriptor_subgroup, descriptor_input, y_test, y_pred, results_list, project_path+ f"/results/{target_list[0]}/results.csv")
    results_df.to_csv(model_results_path+"/predicted_vs_actual.csv", index=False)            
        
def process_results(model_name, elec_property, descriptor_subgroup, descriptor_input, y_true, y_pred, results_list, file_name):
    y_true = np.array(y_true) if isinstance(y_true, pd.DataFrame) else y_true
    y_pred = np.array(y_pred) if isinstance(y_pred, pd.DataFrame) else y_pred

    r2, rmse, mae = evaluate_model(y_true, y_pred)
    results_list.append((model_name, elec_property, descriptor_subgroup, descriptor_input, r2, rmse, mae))

    new_row = pd.DataFrame([(model_name,elec_property, descriptor_subgroup, descriptor_input, r2, rmse, mae)],
                           columns=["Model","elec_property", "Descriptor_subgroup", "Descriptor", "r2", "rmse", "mae"])

    if os.path.exists(file_name):
        existing_df = pd.read_csv(file_name)
        full_df = pd.concat([existing_df, new_row], ignore_index=True)
    else:
        full_df = new_row

    #Sort the full DataFrame categorically
    full_df = full_df.sort_values(by=["Model", "elec_property", "Descriptor_subgroup", "Descriptor"],ascending=True)
    full_df.to_csv(file_name, index=False)

def plot_results(y_test, y_pred, model_results_path, target):
    sns.set_context("poster")
    plt.rcParams['font.family'] = 'Charter'
    #hole scatter
    if target == "hole":
        plt.figure(figsize=(9, 6))
        sns.scatterplot(x=y_test, y=y_pred, color="#e82729")
        plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")
        plt.margins(0)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        #axis 1.0, 2.0 etc, use a number slightly higher than axis limit to ensure tick
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.xlabel(r"Actual $\lambda_h$ / e.V.")
        plt.ylabel(r"Predicted $\lambda_h$ / e.V.")
        scatter_hole_path = os.path.join(model_results_path, "scatter_actual_vs_predicted_hole.png")
        
        plt.savefig(scatter_hole_path, dpi=900, bbox_inches="tight")
        plt.close()
        #electron scatter
    elif target == "electron":
        plt.figure(figsize=(9,6))
        sns.scatterplot(x=y_test, y=y_pred, color="#1f77b4")
        plt.plot([0, 1], [0, 1], color="black", linestyle="dashed")
        plt.margins(0)
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        #axis 1.0, 2.0 etc, use a number slightly higher than axis limit to ensure tick
        plt.yticks(np.arange(0, 1.1, 0.2))
        plt.xticks(np.arange(0, 1.1, 0.2))
        plt.xlabel(r"Actual $\lambda_e$ / e.V.")
        plt.ylabel(r"Predicted $\lambda_e$ / e.V.")
        
        scatter_electron_path = os.path.join(model_results_path, "scatter_actual_vs_predicted_electron.png")
        plt.savefig(scatter_electron_path, dpi=900, bbox_inches="tight")
        plt.close()
 
###model and scaling###
def model_scaling(target_list,model_name,model, X_train, X_test, y_train):
    if len(target_list)>1:
        if model_name in ["gbrt", "svr_gaussian", "lgbm", "adaboost", "svr_linear"]:
            model = MultiOutputRegressor(model)
            
    scaler = StandardScaler() if model_name in ["svr_gaussian", "svr_linear", "lr"] else MinMaxScaler() if model_name in ["knn", "mlpregressor"] else None
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
            
    y_scaler = StandardScaler() if model_name in ["svr_gaussian", "svr_linear", "lr"] else MinMaxScaler() if model_name in ["knn", "mlpregressor"] else None
    if y_scaler:
        if len(target_list)>1:
            y_train = y_scaler.fit_transform(y_train)
        else:
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
    return model, y_scaler, X_train, X_test, y_train

#%%main
def main():   
    global descriptor_input_list
    if not descriptor_input_list:     
        descriptor_input_list = [file[:-4] for file in os.listdir(f"{project_path}/Descriptors/{elec_property}/") if file.endswith(".csv")]
    
    model_list = model_list_manual if model_list_manual else model_list_auto
    
    model_mapping= {"lr" : LinearRegression,"knn" : KNeighborsRegressor,"mlpregressor" : MLPRegressor,"rf" : RandomForestRegressor,"gbrt" : GradientBoostingRegressor, 
        "svr_gaussian" : SVR, "svr_linear" : LinearSVR, "lgbm":LGBMRegressor, "adaboost" : AdaBoostRegressor, "xgb" : XGBRegressor}
    model_list_mapped = [model_mapping[model] for model in model_list]

    print("\nDescriptor Inpuputs found: "+str(descriptor_input_list))
    print("\nModels found: "+str(model_list))
    print("Target found: "+str(target_list))
    
    hole_results_list = []; electron_results_list = []; results_list = []
    
    for index,descriptor_input in enumerate(descriptor_input_list):
        print(f"\nCurrent descriptor_input: {descriptor_input} ({index+1}/{len(descriptor_input_list)})")
        
        descriptor_file_path = project_path+"/Descriptors/"+elec_property+"/"+str(descriptor_input)+".csv" 
        X,y, descriptor_subgroup = load_data(descriptor_input,descriptor_file_path)
        
        for model_index, (model_name, model) in enumerate(zip(model_list,model_list_mapped)):  
            print(f"\nCurrent model: {model_name} ({model_index+1}/{len(model_list)})")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_name == "lr":
                best_params = {}
            else:
                study = optimise_model(model_name, X_train, y_train, n_trials=50)
                best_params = save_results(study, model_name, descriptor_input, descriptor_subgroup,target_list)
            
            #%%final results
            final_parameters={**fixed_parameters.get(model_name, {}), **best_params}
            model = model(**final_parameters)
            model, y_scaler, X_train, X_test, y_train=model_scaling(target_list, model_name, model, X_train, X_test, y_train)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test) 
            if y_scaler:
                if len(target_list)>1:
                    y_pred = y_scaler.inverse_transform(y_pred)
                else:
                    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            y_pred = 10**y_pred
            y_test = 10**y_test

            results(descriptor_input,model_name,y_test,y_pred,descriptor_subgroup, hole_results_list, electron_results_list, results_list)
        print("-----")
if __name__ == "__main__":
    main()
