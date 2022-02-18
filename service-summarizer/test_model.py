import sys
import hf_pegasus

print(hf_pegasus.summarize_model(sys.argv[1], sys.argv[2]))


