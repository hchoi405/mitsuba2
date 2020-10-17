import os
import subprocess

for dir in os.listdir('./scenes'):
    scene_dir = "./scenes/" + dir
    scene_file = scene_dir + "/scene.xml"

    if dir != "conference":
        continue

    print(scene_file)

    ### Replace work
    # file = open(scene_file, "r")
    # data = file.read()

    ## Replace indepedent -> $sampler
    # data = data.replace("independent", "$sampler")

    ## Replace sample number
    # target_str = '"sampleCount" value="'
    # start_ind = data.find(target_str) + len(target_str)
    # end_ind = start_ind + data[start_ind:start_ind+5].find('"')
    # data = "".join((data[:start_ind], "$samples", data[end_ind:]))

    # file.close()
    # file = open(scene_file, "w")
    # file.write(data)
    # file.close()
    # continue
    ###
    
    cmd_independent_scalar = "build/dist/mitsuba -Dsamples=128 -Dsampler=independent -mscalar_rgb -o %s %s" % (scene_dir + "/independent_scalar.exr", scene_file)
    cmd_independent_packet = "build/dist/mitsuba -Dsamples=128 -Dsampler=independent -mpacket_rgb -o %s %s" % (scene_dir + "/independent_packet.exr", scene_file)
    cmd_fixed_scalar = "build/dist/mitsuba -Dsamples=128 -Dsampler=fixed -mscalar_rgb -o %s %s" % (scene_dir + "/fixed_scalar.exr", scene_file)
    cmd_fixed_packet = "build/dist/mitsuba -Dsamples=128 -Dsampler=fixed -mpacket_rgb -o %s %s" % (scene_dir + "/fixed_packet.exr", scene_file)

    target_str = "finished. (took "

    print(cmd_independent_scalar)
    result = subprocess.check_output(cmd_independent_scalar, shell=True).decode("utf-8")
    start_ind = result.find(target_str) + len(target_str)
    took_time = result[start_ind:start_ind+20]
    took_time = took_time[0:took_time.find("\n") - 1]
    print(took_time)
    os.rename(scene_dir + "/independent_scalar.exr", scene_dir + "/independent_scalar_%s.exr" % took_time)

    print(cmd_independent_packet)
    result = subprocess.check_output(cmd_independent_packet, shell=True).decode("utf-8")
    start_ind = result.find(target_str) + len(target_str)
    took_time = result[start_ind:start_ind+20]
    took_time = took_time[0:took_time.find("\n") - 1]
    print(took_time)
    os.rename(scene_dir + "/independent_packet.exr", scene_dir + "/independent_packet_%s.exr" % took_time)

    print(cmd_fixed_scalar)
    result = subprocess.check_output(cmd_fixed_scalar, shell=True).decode("utf-8")
    start_ind = result.find(target_str) + len(target_str)
    took_time = result[start_ind:start_ind+20]
    took_time = took_time[0:took_time.find("\n") - 1]
    print(took_time)
    os.rename(scene_dir + "/fixed_scalar.exr", scene_dir + "/fixed_scalar_%s.exr" % took_time)

    print(cmd_fixed_packet)
    result = subprocess.check_output(cmd_fixed_packet, shell=True).decode("utf-8")
    start_ind = result.find(target_str) + len(target_str)
    took_time = result[start_ind:start_ind+20]
    took_time = took_time[0:took_time.find("\n") - 1]
    print(took_time)
    os.rename(scene_dir + "/fixed_packet.exr", scene_dir + "/fixed_packet_%s.exr" % took_time)


