# family IDs
Autopilot / Navigation :	cf_autopilot
Search and Rescue : 	cf_search_and_rescue
Swarm Autopilot: 	cf_swarm_autopilot
Swarm Search and Rescue :	cf_swarm_sar
Interceptor :	cf_interceptor

# Submission CLIs

swarm doctor
swarm model test --source drohunter/ --family-id cf_autopilot
swarm model package --source my_agent/ --family-id cf_autopilot
swarm model verify --model Submission/submission.zip


swarm repo package \
  --repo-root https://github.com/viasamonte74/drohunter \
  --family-source cf_autopilot=./drohunter

swarm repo package \
  --repo-root ./publish/drohunter \
  --family-source cf_autopilot=./drohunter

  swarm repo package \
  --repo-root ./https://github.com/viasamonte74/drohunter.git \
  --family-source cf_autopilot=./drohunter

# Or update the artifact later
swarm repo package \
  --repo-root https://github.com/viasamonte74/drohunter.git \
  --source ./drohunter \
  --family-id cf_autopilot \
  --overwrite

swarm repo verify --repo-root https://github.com/viasamonte74/drohunter.git

swarm repo verify --repo-root ./publish/drohunter

# Run benchmark
# Default benchmark (3 seeds per environment group)
swarm benchmark --model Submission/submission.zip --workers 4

# Quick test (1 seed per environment type)
swarm benchmark --model Submission/submission.zip --seeds-per-group 1

swarm report

# RL train and local test
cd /root/work_pnj/Subnet/swarm
source miner_env/bin/activate   # if you used the miner setup
python3 RL/cf_interceptor/train.py

python3 RL/cf_interceptor/train.py --timesteps 5000 --seed 42

python3 RL/cf_interceptor/train.py --timesteps 2000000 --device cuda
python3 RL/cf_interceptor/train.py --resume RL/cf_interceptor/out/checkpoints/ppo_500000_steps.zip
python3 RL/cf_interceptor/train.py --no-package   # skip zip while iterating

python3 RL/test_RL.py \
  --model RL/cf_interceptor/out/submission.zip \
  --family_id cf_interceptor \
  --num-seeds 10

# same as default: 1100 cycling maps, validator template
python3 RL/cf_interceptor/train.py --timesteps 2000000 --map-seeds 1100
# new random map every episode (often better for unknown future validator seeds)
python3 RL/cf_interceptor/train.py --timesteps 2000000 --map-seeds 0
# if you later have a published epoch seed list
python3 RL/cf_interceptor/train.py --timesteps 2000000 --map-seed-file epoch_seeds.json


# start RL first checkpoint
python3 RL/cf_interceptor/train.py \
  --timesteps 2000000 \
  --device cuda \
  --map-seeds 1100 \
  --no-package

# is catch rate is still low then
python3 RL/cf_interceptor/train.py \
  --timesteps 8000000 \
  --device cuda \
  --map-seeds 1100 \
  --resume RL/cf_interceptor/out/checkpoints/ppo_500000_steps.zip \
  --no-package