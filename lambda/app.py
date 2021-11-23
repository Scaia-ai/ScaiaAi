#!/usr/env/python

import sys
import hg_pegasus

def handler(event, context):

    
    text_to_summarize = event['text']
    sumarized_text = hf_pegasus.summarize(text_to_summarize)
    return summarized_text
