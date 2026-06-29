source venv/bin/activate

# seeds:
for i in {0..4}
do
    num=$(($RANDOM % 100))
    echo "Running analysis with seed $num"
    printf "1\n1\n" | python analysis.py --seed $num load npe nn pp save
# 
done