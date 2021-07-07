import openke
import os
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/FB13/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)
print('Train data_loader done')
# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/FB13/", "link")
print('Test data_loader done')

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 300, 
	p_norm = 1, 
	norm_flag = True)
print('Transe model loaded')
transe.load_checkpoint('./checkpoint/mid499.ckpt')
# define the loss function
model = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size()
)

print('Training model defined')
# train the model

trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 500, save_steps=50, alpha = 1.0, use_gpu = False, checkpoint_dir='./checkpoint/checkpoint_300')
print('Start training')
trainer.run()
transe.save_checkpoint('./checkpoint/transe_300.ckpt')

# test the model
transe.load_checkpoint('./checkpoint/transe_300.ckpt')
tester = Tester(model = transe, data_loader = test_dataloader, use_gpu = False)
# tester.run_link_prediction(type_constrain = False)
tester.save_emb()