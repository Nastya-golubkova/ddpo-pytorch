
Создание отчета: [ноутбук](../../report/report.ipynb)

[Отчет](../../report/results_lora_ft.md)









```bash
python eval/benchmark/run_laion_benchmark.py \
  --pretrained runwayml/stable-diffusion-v1-5 \
  --use_training_prompts


python eval/benchmark/run_laion_benchmark.py \
  --pretrained runwayml/stable-diffusion-v1-5

python eval/benchmark/run_laion_benchmark.py \
  --pretrained runwayml/stable-diffusion-v1-5
  --run_dir eval/benchmark/runs/sd_base

python eval/benchmark/run_laion_benchmark.py \
  --pretrained runwayml/stable-diffusion-v1-5 \
  --lora_path /home/ayagolubkova/rl_project/ddpo-pytorch/logs/2026.03.18_15.47.31/checkpoints/checkpoint_9 \
  --run_dir eval/benchmark/runs/jpeg_compr_9


  python eval/benchmark/run_laion_benchmark.py \
  --pretrained runwayml/stable-diffusion-v1-5 \
  --lora_path /home/ayagolubkova/rl_project/ddpo-pytorch/logs/2026.03.19_04.13.50/checkpoints/checkpoint_9 \
  --run_dir eval/benchmark/runs/aesth_9

  python eval/benchmark/run_laion_benchmark.py \
  --pretrained runwayml/stable-diffusion-v1-5 \
  --lora_path /home/ayagolubkova/rl_project/ddpo-pytorch/logs/2026.03.19_17.09.16/checkpoints/checkpoint_9 \
  --run_dir eval/benchmark/runs/jp_and_aes_9


```