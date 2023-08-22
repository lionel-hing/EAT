# Event Activation Transformer for Text-to-Video Retrieval

<a name="depend"/>

## Dependencies
Our model was developed and evaluated using the following package dependencies:
- PyTorch 1.8.1
- Transformers 4.6.1
- OpenCV 4.5.3

<a name="datasets"/>

## Datasets
We trained models on the MSR-VTT, DiDeMo and LSMDC datasets. To download the datasets, refer to this [repository](https://github.com/ArrowLuo/CLIP4Clip).

<a name="eval"/>

## Evaluation
The following commands can be used to reproduce the main results of our paper using the supplied checkpoint files for each dataset. The commands will by default generate results for text-to-video retrieval (t2v). For video-to-text retrieval (v2t) results, add the argument `--metric=v2t` to the command.

If the `outputs/` folder does not exist, first run `mkdir outputs` to create the directory. For each dataset, create a directory in `outputs/` and store the corresponding checkpoint file. For each command below, replace `{exp_name}` with the name of that directory.

Also, replace `{videos_dir}` with the path to the dataset's videos.

For evaluation, you can change the `batch_size` without affecting results.
  

<a name="eval-commands"/>

| Dataset | Command | 
|:-----------:|:-----------:|
|MSR-VTT-9k|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --huggingface --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=9k`| 
|DiDeMo|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32  --load_epoch=-1 --dataset_name=DeDeMo`
|LSMDC|`python test.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32  --load_epoch=-1 --dataset_name=LSMDC`|

<a name="train"/>

## Training
The following commands can be used to train our X-Pool *w/* EAT model for each dataset. Again, the evaluation is by default set to generate results for text-to-video retrieval (t2v). For video-to-text retrieval (v2t) results, add the argument `--metric=v2t` to the command.

For each command below, replace `{exp_name}` with your choice name of experiment. Also, replace `{videos_dir}` with the path to the dataset's videos.
  

<a name="train-commands"/>

| Dataset | Command |
|:-----------:|:-----------:|
|MSR-VTT-9k|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=3e-5 --transformer_dropout=0.3  --dataset_name=MSRVTT --msrvtt_train_file=9k`|
|DiDeMo|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=4e-5 --transformer_dropout=0.4  --dataset_name=DiDeMo`|
|LSMDC|`python train.py --exp_name={exp_name} --videos_dir={videos_dir} --batch_size=32 --noclip_lr=1e-5 --transformer_dropout=0.3  --dataset_name=LSMDC`|

<a name="train-commands"/>

