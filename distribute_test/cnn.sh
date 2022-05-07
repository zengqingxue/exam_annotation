bin/zsh

python distributed_minst.py --job_name=ps --task_index=0 --ps_hosts=127.0.0.1:22224 --worker_hosts=127.0.0.1:22225,127.0.0.1:22226

python distributed_minst.py --job_name=worker --task_index=0 --ps_hosts=127.0.0.1:22224 --worker_hosts=127.0.0.1:22225,127.0.0.1:22226

python distributed_minst.py --job_name=worker --task_index=1 --ps_hosts=127.0.0.1:22224 --worker_hosts=127.0.0.1:22225,127.0.0.1:22226

# 127.0.0.1
# 192.168.1.219