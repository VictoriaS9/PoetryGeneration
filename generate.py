from datetime import datetime
import gpt_2_simple as gpt2
def main():
  sess=gpt2.start_tf_sess()
  gpt2.load_gpt2(sess, run_name='run')
  gpt2.generate(sess, run_name='run')
if __name__=='__main__':
  main()