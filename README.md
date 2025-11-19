# HW3P2 README file
After completing all TODOS tasks in all classes and pass all the test here is what you will do to run the project, all codes are complete and all test are 100% passed, so here you go: 
#### Running Instruction.
to run these codes:
 - First, pull the codes from github but this folder already have these codes no need to pull
 - I used kaggle to run this Homework
 - import and run libraries cell
 - if you are using google colab you will need to mount your device but for me i didn't use google colab
 - for avoiding the crushed, i run my notebook on kaggle cloud
 - run all the cell in order from top to bottom
 - Github Token and WanDb api key was hidden for security purpose
### Highest  accuracy achieved:
the Character error rate (cer) achieved is 12.4 on public and 11.4 on private score on kaggle, and this noteebok is for that results
### Architectures Used:
in this homework i used many architectures but both of them has relate:
- tokenization type I used is 1k 
- in this notebook i used encoder-decoder Transformers
- I used  6 layers in encoder class and 5 layers of decoder class
- I used also 6 layers MLP in decoder class
### HyperParameters:
in this project (this notebook) i used different hyperparameters which are 19.20M in total and it's the combination which gave the best scores
- tokenization: 1k
- learning_rate: 0.0005
- number of epochs: 65 
- train_beam_width: 1
- Model dimension: 256 
- batch_size: 16
- dropout: 0.2
- Gradient accumulation step: 1
- time reduction: 3
- optimizer: AdamW
- weight decay: 0.0001 
- loss function: CTC loss
- Normalization: global mvn
- scheduler: CosineAnnealingLR
- Augmentations: [TimeMasking, FrequencyMasking]
##### in different runs I used different activations functions, different architectures, learning rates, optimzers and other hyperparameters.
other paramteres I tried in differents runs:
- learning rate:  0.0005
- scheduler: ReduceLROnPlateau
- batch size: 32
- Normalization: None, cepstral
- Dropu out: [0.3, 0.5]
- number of epochs: [60, 70,75]
### Experiments:
to achieve high cutoffs as our group we run many experiments, some of them had crashed, others ran succesfully but didn't achieve the distance we wanted

### Data Loading Scheme:
to load data both provided AudioDataSet class and AudioTestDataSet classes and Dataloaders was used.
### Wandb project Link:
[Visit WanDb](https://wandb.ai/tresoryvesIDL/hw4p2-asr_projects_2/workspace?nw=nwuserrdukunda)

### WanDb Screeshoot
##### the overall runs graphs screenshoot:
![alt text](<Screenshot 2025-04-30 233308.png>)
![alt text](<Screenshot 2025-04-30 233333.png>)
![alt text](<Screenshot 2025-04-30 233351.png>)
