import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models import Encoder, Decoder
import unicodedata
import re

def initialize_chatbot():
    #print("initializing chatbot... \n")
    #print("Loading dictionary...")
    
    with open('model_data/vocab_dict.p', 'rb') as fp: # put in the path to vocab_dict
        vocab = pickle.load(fp)
    #print(f"Loaded {len(vocab)} words")

    #print("making sample embedding matrix...")
    sample_emb = tf.zeros((len(vocab), 100))

    """ ENCODER WORK """
    #print("Initializing Encoder...")
    encoder = Encoder(len(vocab), 100, 500,
                        128,  sample_emb,
                        num_layers=3,
                        drop_prob = 0.1)

    #print("Testing Encoder...")
    sample_hidden = encoder.initialize_hidden_state()
    ex_input_bt = tf.zeros((128,25))
    sample_output, sample_hidden = encoder(ex_input_bt, sample_hidden)
    assert  sample_output.shape == (128,25,500)
    assert sample_hidden.shape == (128,500)

    #print("Loading up encoder...")
    encoder.load_weights("model_data/encoder_gpu.h5") # put in the path to decoder weights

    """ DECODER WORK """
    #print("Initializing Decoder...")
    decoder = Decoder(len(vocab), 100, 500,
                        128,  sample_emb,
                        num_layers=3, 
                        drop_prob = 0.1)
    #print("Testing Decoder...")
    sample_decoder_output, _, _ = decoder(tf.random.uniform((128, 1)),
                                            sample_hidden, sample_output)
    assert sample_decoder_output.shape == (128, len(vocab))

    #print("Loading up decoder...")
    decoder.load_weights("model_data/decoder_gpu.h5") # put in the path to decoder weights

    # inverse vocabulary
    inv_vocab = {v:k for k,v in vocab.items()}

    """ Some variables"""
    pad_token = 0
    sos_token = 1
    eos_token = 2
    units = 500
    maxl = 25

    """Processing functions"""
    # Convert (or remove accents) sentence to non_accents sentence
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters
    def normalizeString(s):
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    return vocab, encoder, decoder, inv_vocab, pad_token, sos_token, eos_token, units, maxl, unicodeToAscii, normalizeString

""" Main reply function"""
def reply_to(sentence, vocab, encoder, decoder, inv_vocab, pad_token, sos_token, eos_token, units, maxl, unicodeToAscii, normalizeString ):

    try:
        inps = [vocab[word] for word in sentence.split(" ")]
    except KeyError:
        print("Missing word: " + word)
        #return "I didn't get you, try again"

    inps = pad_sequences([inps], maxl, padding='post')
    inps = tf.convert_to_tensor(inps)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inps, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([vocab["SOS"]], 0)

    for t in range(maxl):
        preds, dec_hidden, attn_wts = decoder(dec_input, dec_hidden, enc_out)

        pred_id = tf.argmax(preds[0]).numpy()

        if inv_vocab[pred_id] == "EOS":
            return result

        result += inv_vocab[pred_id] + " "

        dec_input = tf.expand_dims([pred_id], 0)

    return result














