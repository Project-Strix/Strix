#! /bin/bash

if [ "$#" -lt "2" ] || [ "$1" != "--config" ]
then
echo "Please input: --config your_confg_file"
exit 0
fi

outdir="$(grep 'experiment_path:' "$2")"

IFS=" "
read -ra splits <<< "$outdir"
echo "${splits[1]}"

nsys profile --output "${splits[1]}" --force-overwrite true --trace-fork-before-exec true strix-train-from-cfg --config "$2"
