# IMPORTS
# ______________________________________________________________________________________________________________________
from encoding_library import *
# PARAMS
# ______________________________________________________________________________________________________________________
# location of the files created by the DSI code
data_location = ''

# the execution of this script does require a GPU as it otherwise takes a long time

# SCRIPT
# ______________________________________________________________________________________________________________________

print('READING THE DATA - PROCESSED DOCUMENTS - LOADING MODEL')
# specify location
path = data_location + 'intermediate_processed/'

# read in the data as iterator
healthy = pd.read_csv(path + 'healthy_transformers.csv', index_col=0, chunksize=100, header=0)
failed = pd.read_csv(path + 'failed_transformers.csv', index_col=0, chunksize=100, header=0)

# Set device to cuda if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# initialize the Huggingface model and tokenizer and push to device
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = model.to(device)


# perform the encoding and save the output
print('ENCODING HEALTHY DOCS')
encode_batch(healthy, tokenizer, model, data_location, 'healthy', device)
print('ENCODING FAILED DOCS')
encode_batch(failed, tokenizer, model, data_location, 'failed', device)


