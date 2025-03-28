ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9
git pull
rm log*
rm out*
rm ERROR_LOG*
for ((i=$1; i<$2; i=i+1))
do
    touch "log_stats_proj_2_$i.txt"
    touch "log$i.txt"
    touch "ERROR_LOG_$i.txt"
    touch "out$i.txt"
    (sleep 1; python -u "trainer.py" $i $3 "geo-distributed" $4 $5 >"out$i.txt") &


done