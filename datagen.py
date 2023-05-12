
xvfb-run -a -s "-screen 0 1400x900x24" python3 /vol/bitbucket/nm219/RLBench/tools/dataset_generator.py --save_path=/vol/bitbucket/nm219/Demos --tasks=[task names]|task name --image_size=x,y --renderer=opengl --episodes_per_task=n --variations=1 --processes=1

xvfb-run -a -s "-screen 0 1400x900x24" /vol/bitbucket/nm219/fyp/bin/python /homes/nm219/Final-Year-Project/models/policy_rollout.py

/vol/bitbucket/nm219/fyp/bin/python /homes/nm219/Final-Year-Project/data_mover.py /vol/bitbucket/nm219/data/<file> 

close_box
pick_and_lift
plug_charger_in_power_supply
take_off_weighing_scales