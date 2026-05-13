#!/bin/sh

output="combined_lj.dat"
echo "#compound    plape-plape    plape-res    res-res" > "$output"

for d in */; do
    if [ -d "$d" ]; then
        file="$d/lj_interactions_avg.dat"
        if [ -f "$file" ]; then

            # extract values from the 3 LJ lines
            pp=$(grep "LJ-SR:plape-plape" "$file" | awk '{print $2}')
            pr=$(grep "LJ-SR:plape-res"   "$file" | awk '{print $2}')
            rr=$(grep "LJ-SR:res-res"     "$file" | awk '{print $2}')

            echo "$d    $pp    $pr    $rr" >> "$output"
        else
            echo "$d    MISSING_FILE    MISSING_FILE    MISSING_FILE" >> "$output"
        fi
    fi
done


