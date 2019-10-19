
import json
import os
import config as cfg


def makemydir(enabler):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, enabler)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    return final_directory
    


def text(filename, istrain):
    if istrain:
        enabler = cfg.train_dir
    else:
        enabler = cfg.test_dir
    path = makemydir(enabler)
    with open(filename) as f:
        i = 0
        for line in f:
            j_content = json.loads(line)
            j_str = json.dumps(j_content)
            i = i + 1
            with open('{0}/rev_{1}.json'.format(enabler, i), 'w') as fout:
                fout.write(j_str)
    return path


                
