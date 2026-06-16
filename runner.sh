source venv/bin/activate

# seeds:
for i in {0..9}
do
    num=$(($RANDOM % 100))
    python seed_ensemble.py --seed $num
# 
done