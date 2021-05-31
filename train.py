from datetime import datetime
import gpt_2_simple as gpt2
import sys

def main():
    filename = "/home/Lenovo/Desktop/poems.txt"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        message = 'Please add the path to the dataset file'
    gpt2.download_gpt2(model_name='124M')
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess, dataset=filename,
                  model_name='124M',
                  steps=1000,
                  restore_from='fresh',
                  run_name='run',
                  print_every=10,
                  sample_every=200,
                  save_every=500)


if __name__ == "__main__":
    main()
