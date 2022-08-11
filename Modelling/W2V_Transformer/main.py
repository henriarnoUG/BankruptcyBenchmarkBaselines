# IMPORTS
# ______________________________________________________________________________________________________________________
from dataset import *
from NN import *

# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''
# indicate to use tensorboard or not
use_tensorboard = False
# location to store the tensorboard runs (if desired)
location_TB = ''
# name to create the results folder
name = 'bow'  # 'bow' or 'transformers'

# development or test setting
dev = True
# number of documents used for prediction
n = 1  # 1 or 3
# resample the training data or not
# if True, reweigh the instances according to the class distribution
resample = True

# the representation size
input_size = 100  # default 100 for w2v model and 768 for transformers

# general DL hyper parameters, not tuned
batch_size = 32
epochs = 50
output_size = 1

# tuned hyper parameters
# for tuning the these parameters, we used the Optuna package (https://optuna.org/)
# We start from coarse hyperparameter ranges and reduced this range iteratively
# the final range was as follows
"""
params = {
        'learning_rate': trial.suggest_uniform('learning_rate', 1e-5, 5e-3),
        'hidden_size': trial.suggest_int("hidden_1_size", 4, 64),
        'dropout_prob': trial.suggest_float('dropout_prob', 0, 0.5),
        'weight_decay': trial.suggest_float('weight_decay', 0, 1e-3),
    }
"""
# now we use some sensible values for demonstration
# if the hyperparameter optimisation code is desired, please ask so on the GitHub repo issue section
hidden_size = 22
dropout_p = 0.1
learning_rate = 0.010
weight_decay = 0.00035

# SCRIPT
# ______________________________________________________________________________________________________________________

# specify the location
checkpoint_dir = data_location + 'model_data/model_checkpoints/'
results_dir = data_location + 'results_' + name + '_NN/'
try:
    os.mkdir(results_dir)
    os.mkdir(checkpoint_dir)
except:
    pass

print('READING IN THE DATA')
# read in the firm-year dataset
if dev:
    train = pd.read_csv(data_location + 'model_data/bow/train_dev_bow_' + str(n) + '.csv', index_col=0)
    holdout = pd.read_csv(data_location + 'model_data/bow/dev_bow_' + str(n) + '.csv', index_col=0)
    # split up the holdout set
    holdout1 = holdout[holdout['holdout_year'] == 2017]
    holdout2 = holdout[holdout['holdout_year'] == 2018]

else:
    train = pd.read_csv(data_location + 'model_data/bow/train_full_bow_' + str(n) + '.csv', index_col=0)
    holdout = pd.read_csv(data_location + 'model_data/bow/holdout_bow_' + str(n) + '.csv', index_col=0)
    # split up the holdout set
    holdout1 = holdout[holdout['holdout_year'] == 2019]
    holdout2 = holdout[holdout['holdout_year'] == 2020]

# read in the encoded documents
tensors = np.load(data_location + 'model_data/' + name + '/tensors_healthy.npy', allow_pickle=True)
tensors_failed = np.load(data_location + 'model_data/' + name + '/tensors_failed.npy', allow_pickle=True)
tensors = np.concatenate((tensors, tensors_failed))

doc_id = np.load(data_location + 'model_data/' + name + '/doc_ids_healthy.npy', allow_pickle=True)
doc_id_failed = np.load(data_location + 'model_data/' + name + '/doc_ids_failed.npy', allow_pickle=True)
doc_id = np.concatenate((doc_id, doc_id_failed))

# sort tensors, the index of an encoding should correspond to the doc_id of that document
tensors = tensors[doc_id]

# ______________________________________________________________________________________________________________________

print('COMPUTING SAMPLE WEIGHTS ACCORDING TO FREQUENCY')
if resample:
    # compute sample weights according to frequency
    class_sample_count = np.array([len(np.where(train['label'] == t)[0]) for t in np.unique(train['label'])])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train['label']])
    # cast sample weights to tensor and create WeightedRandomSampler object
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

# Set device to gpu if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# Create TensorBoard writer
if use_tensorboard:
    writer = SummaryWriter(log_dir=location_TB)

# ______________________________________________________________________________________________________________________

print('CREATING DATASETS AND MODEL')
# create datasets
train = CustomDataset(dataframe=train.reset_index(drop=True), tensors=tensors, n=n, name=name)
holdout1 = CustomDataset(dataframe=holdout1.reset_index(drop=True), tensors=tensors, n=n, name=name)
holdout2 = CustomDataset(dataframe=holdout2.reset_index(drop=True), tensors=tensors, n=n, name=name)

# create model
model = NeuralNetwork(input_size=input_size, hidden_1_size=hidden_size, hidden_2_size=hidden_size,
                      output_size=output_size, n=n, dropout_p=dropout_p)

# specify loss function
loss_fn = nn.BCELoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# ______________________________________________________________________________________________________________________

print('START TRAINING')
# initialise a variable to track the best performing model checkpoint (in terms of AUC)
best_performance = 0

# loop over every epochs
for t in range(epochs):

    # Create train and testing dataloader, shuffle data
    if resample:
        train_loader = DataLoader(dataset=train, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    holdout_loader1 = DataLoader(dataset=holdout1, batch_size=batch_size, shuffle=True)
    holdout_loader2 = DataLoader(dataset=holdout2, batch_size=batch_size, shuffle=True)

    # track progress
    print(f"Epoch {t + 1}\n-------------------------------")

    # train NN in epoch
    train_loss = train_loop(dataloader=train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    # evaluate NN on holdout set in epoch
    loss1, auc1, (indices1, full_y1, full_pred1) = test_loop(dataloader=holdout_loader1, model=model, loss_fn=loss_fn)
    loss2, auc2, (indices2, full_y2, full_pred2) = test_loop(dataloader=holdout_loader2, model=model, loss_fn=loss_fn)

    # pass results to TB
    if use_tensorboard:
        writer.add_scalar('TRAIN LOSS', train_loss, t)
        writer.add_scalars('DEV LOSS', {'Loss 1': loss1, 'Loss 2:': loss2, 'Loss Train': train_loss}, t)
        writer.add_scalars('DEV AUC', {'AUC 1': auc1, 'AUC 2:': auc2}, t)

    # compute weight-adjusted mean AUC
    print(10 * '_')
    mean_auc = ((holdout1.__len__() * auc1) + (holdout2.__len__() * auc2)) / (
            holdout1.__len__() + holdout2.__len__())
    print(10 * '_')

    # save the model checkpoint if this is the best performing model
    if mean_auc > best_performance:
        torch.save(model.state_dict(), checkpoint_dir + 'best_model.pth')
        best_performance = mean_auc

# after training, load the best model checkpoint
model = NeuralNetwork(input_size=input_size, hidden_1_size=hidden_size, hidden_2_size=hidden_size,
                      output_size=output_size, n=n, dropout_p=dropout_p)
model.load_state_dict(torch.load(checkpoint_dir + 'best_model.pth'))

# evaluate final NN on holdout set
holdout_loader1 = DataLoader(dataset=holdout1, batch_size=batch_size, shuffle=True)
holdout_loader2 = DataLoader(dataset=holdout2, batch_size=batch_size, shuffle=True)
loss1, auc1, (indices1, full_y1, full_pred1) = test_loop(dataloader=holdout_loader1, model=model,
                                                         loss_fn=loss_fn)
loss2, auc2, (indices2, full_y2, full_pred2) = test_loop(dataloader=holdout_loader2, model=model,
                                                         loss_fn=loss_fn)

# report results
print('BEST MODEL: ')
print('AUC 1: ' + str(auc1))
print('AUC 2: ' + str(auc2))

# store results
pd.DataFrame({'indices': indices1, 'y': full_y1, 'pred': full_pred1}).to_csv(results_dir + 'preds_1.csv')
pd.DataFrame({'indices': indices2, 'y': full_y2, 'pred': full_pred2}).to_csv(results_dir + 'preds_2.csv')

# close TB writer
if use_tensorboard:
    writer.close()
